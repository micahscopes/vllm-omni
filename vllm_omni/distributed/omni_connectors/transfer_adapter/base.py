# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import select as _select
import threading
from collections import deque
from typing import Any

from ..utils.logging import get_connector_logger
from ..utils.notify import NotifyReceiver

logger = get_connector_logger(__name__)


class OmniTransferAdapterBase:
    """Base class for managing data transfer via OmniConnector.

    This class handles the core loop logic and connector interactions, but
    leaves the specific data processing (chunks, KV cache, etc.) to subclasses.
    """

    def __init__(self, config: Any):
        self.config = config
        if not hasattr(self, "connector"):
            self.connector = None
        # Requests that are waiting to be polled
        self._pending_load_reqs = deque()
        # Requests that have successfully retrieved data
        self._finished_load_reqs = set()
        self._cancelled_load_reqs: set[str] = set()

        # Requests that are waiting to be saved
        self._pending_save_reqs = deque()
        # Requests that have successfully saved data
        self._finished_save_reqs = set()

        self.stop_event = threading.Event()
        self._recv_cond = threading.Condition()
        self._save_cond = threading.Condition()

        # Bind cross-process notify receiver for this stage's inbound edge.
        # Stage-0 has no upstream producer (see load_async), so skip. If
        # binding fails (permissions, non-Linux, etc.) NotifyReceiver sets
        # disabled=True and recv_loop falls back to pure polling.
        self._notify_recv: NotifyReceiver | None = None
        # In-process self-wake pipe: load_async and shutdown write one
        # byte here to break recv_loop out of select() immediately when
        # the cross-process notify path is active. Without this, a local
        # load_async enqueue would wait up to the select timeout.
        self._wake_r: int | None = None
        self._wake_w: int | None = None
        stage_id = getattr(getattr(self, "connector", None), "stage_id", -1)
        if stage_id is not None and stage_id > 0:
            try:
                self._notify_recv = NotifyReceiver(from_stage=stage_id - 1, to_stage=stage_id)
                if self._notify_recv.disabled:
                    self._notify_recv = None
            except Exception as e:  # pragma: no cover - best-effort
                logger.warning("notify: receiver init failed: %s; polling only", e)
                self._notify_recv = None
            if self._notify_recv is not None:
                try:
                    r, w = os.pipe()
                    os.set_blocking(r, False)
                    os.set_blocking(w, False)
                    self._wake_r, self._wake_w = r, w
                except OSError as e:
                    logger.warning("notify: self-wake pipe failed: %s; polling only", e)
                    self._notify_recv.close()
                    self._notify_recv = None

        self.recv_thread = threading.Thread(target=self.recv_loop, daemon=True)
        self.recv_thread.start()

        self.save_thread = threading.Thread(target=self.save_loop, daemon=True)
        self.save_thread.start()

    @classmethod
    def create_connector(cls, model_config: Any):
        raise NotImplementedError

    def _wake_recv_loop(self) -> None:
        """Wake the recv_loop. Subclasses call this from load_async after
        enqueueing a request. Safe when notify is disabled: it also does
        the classic Condition.notify() so the polling fallback wakes."""
        with self._recv_cond:
            self._recv_cond.notify()
        if self._wake_w is not None:
            try:
                os.write(self._wake_w, b"\x01")
            except (OSError, BlockingIOError):
                # pipe full means a wake is already pending — fine.
                pass

    def recv_loop(self):
        """Loop to poll for incoming data.

        Process each pending request exactly once per pass.  When no request
        made progress, back off 1 ms instead of tight-spinning on failed
        shm_open syscalls (which can burn a full CPU core).
        """
        while not self.stop_event.is_set():
            n = len(self._pending_load_reqs)
            any_success = False
            for _ in range(n):
                if not self._pending_load_reqs:
                    break
                request = self._pending_load_reqs.popleft()
                request_id = request.request_id
                if request_id in self._cancelled_load_reqs:
                    self._cancelled_load_reqs.discard(request_id)
                    continue
                self.request_ids_mapping[request_id] = request.external_req_id
                try:
                    is_success = self._poll_single_request(request)
                    if is_success:
                        any_success = True
                    else:
                        self._pending_load_reqs.append(request)
                except Exception as e:
                    self._pending_load_reqs.append(request)
                    logger.warning(f"Error receiving data for {request_id}: {e}")

            # Choose wait primitive:
            #   - If we have an active cross-process NotifyReceiver, block
            #     on select() over the notify fd + in-process self-wake
            #     pipe. The notify wakes us within ~100us of the peer's
            #     connector.put completing (the B1b target). The self-wake
            #     pipe preserves the old Condition.notify() semantics so
            #     load_async / shutdown still wake us immediately.
            #   - Otherwise fall back to the original Condition.wait path
            #     (1 ms poll cadence). Polling path is a strict fallback
            #     and remains functionally correct.
            if self.stop_event.is_set():
                continue
            # If nothing pending, wait long; if pending but no progress,
            # wait short — same cadence as the pre-notify code, giving the
            # polling-fallback path identical behavior.
            has_pending = bool(self._pending_load_reqs)
            timeout = 0.1 if not has_pending else 0.001
            if self._notify_recv is not None and not self._notify_recv.disabled:
                fds = []
                nfd = self._notify_recv.fileno()
                if nfd is not None:
                    fds.append(nfd)
                if self._wake_r is not None:
                    fds.append(self._wake_r)
                try:
                    r, _, _ = _select.select(fds, [], [], timeout)
                except (InterruptedError, OSError):
                    r = []
                # Drain whichever fds fired. We don't distinguish between
                # cross-process notifies and local self-wakes — either way
                # the next loop iteration handles the work.
                for fd in r:
                    if fd == nfd:
                        self._notify_recv.drain()
                    else:
                        try:
                            while True:
                                os.read(fd, 4096)
                        except BlockingIOError:
                            pass
                        except OSError:
                            pass
            else:
                # Timeout is the fallback for lock-free append/notify races.
                with self._recv_cond:
                    if not self._pending_load_reqs and not self.stop_event.is_set():
                        self._recv_cond.wait(timeout=0.1)
                    elif not any_success and not self.stop_event.is_set():
                        self._recv_cond.wait(timeout=0.001)

    def save_loop(self):
        """Loop to send outgoing data."""
        while not self.stop_event.is_set():
            while self._pending_save_reqs:
                task = self._pending_save_reqs.popleft()
                try:
                    self._send_single_request(task)
                except Exception as e:
                    logger.warning(f"Error saving data for {task.get('request_id')}: {e}")

            with self._save_cond:
                if not self._pending_save_reqs and not self.stop_event.is_set():
                    self._save_cond.wait(timeout=0.1)

    def _poll_single_request(self, *args, **kwargs):
        """Poll connector for a single request task.
        Subclasses should implement request-specific receive behavior."""
        raise NotImplementedError

    def _send_single_request(self, *args, **kwargs):
        """Send one pending save request task to the connector.
        Subclasses should implement task-specific handling logic."""
        raise NotImplementedError

    def load_async(self, *args, **kwargs):
        """Register a request to load data. To be implemented by subclasses."""
        raise NotImplementedError

    def save_async(self, *args, **kwargs):
        """Submit data to be saved. To be implemented by subclasses."""
        raise NotImplementedError

    def load(self, *args, **kwargs):
        """Load request data from connector synchronously. To be implemented by subclasses."""
        raise NotImplementedError

    def save(self, *args, **kwargs):
        """Save data to connector synchronously. To be implemented by subclasses."""
        raise NotImplementedError

    def get_finished_requests(self):
        """Get finished loaded or saved requests"""
        raise NotImplementedError

    def shutdown(self):
        """Stop background loops and close the connector."""
        self.stop_event.set()
        with self._recv_cond:
            self._recv_cond.notify_all()
        with self._save_cond:
            self._save_cond.notify_all()
        # Break recv_loop out of select() immediately if notify is active.
        if self._wake_w is not None:
            try:
                os.write(self._wake_w, b"\x01")
            except OSError:
                pass
        if self._notify_recv is not None:
            try:
                self._notify_recv.close()
            except Exception:
                pass
            self._notify_recv = None
        for fd in (self._wake_r, self._wake_w):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        self._wake_r = self._wake_w = None
        if self.connector is not None:
            try:
                self.connector.close()
            except Exception:
                pass
