# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Cross-process notify channel for SHM chunk transfer.

Problem: stage-0 writes a chunk into /dev/shm and stage-1's recv_loop only
discovers it via 1 ms polling (see OmniTransferAdapterBase.recv_loop).
Measured B1b sub-bucket on GB10: median 85 ms, up to 171 ms of pure
inter-process wake-up latency.

Fix: one AF_UNIX SOCK_DGRAM socket per directed stage pair. Sender fires a
1-byte datagram after every successful ``connector.put``. Receiver's
``recv_loop`` selects on the bound fd instead of timed-waiting on its
threading Condition, waking within ~100us of the put completing.

This module implements the notify channel. It is defensive by design:
any failure (EEXIST on bind, ECONNREFUSED on send, missing socket dir)
flips the channel to ``disabled = True`` and subsequent ops are no-ops.
Callers fall back to the existing polling path.

Linux-only. Not imported anywhere that runs on non-Linux.
"""

from __future__ import annotations

import errno
import os
import select
import socket
import threading

from .logging import get_connector_logger

logger = get_connector_logger(__name__)

# One directory for all notify sockets. Lives in /dev/shm (tmpfs) so it
# shares a lifetime with the SHM segments we're signaling about: when the
# host reboots or /dev/shm is purged, stale sockets go with it.
_NOTIFY_DIR = os.environ.get("VLLM_OMNI_NOTIFY_DIR", "/dev/shm/vllm_omni_notify")


def _socket_path(from_stage: int | str, to_stage: int | str) -> str:
    return os.path.join(_NOTIFY_DIR, f"s{from_stage}_to_s{to_stage}.sock")


def _ensure_dir() -> bool:
    try:
        os.makedirs(_NOTIFY_DIR, exist_ok=True)
        return True
    except OSError as e:
        logger.warning("notify: cannot create %s: %s; notify disabled", _NOTIFY_DIR, e)
        return False


class NotifyReceiver:
    """Bound AF_UNIX/DGRAM endpoint. Owns the socket file.

    ``wait(timeout)`` blocks up to timeout seconds or until at least one byte
    is ready, then drains all pending bytes non-blocking and returns.
    Safe to call from the recv_loop thread.
    """

    def __init__(self, from_stage: int | str, to_stage: int | str):
        self.path = _socket_path(from_stage, to_stage)
        self._sock: socket.socket | None = None
        self.disabled = False
        self._lock = threading.Lock()
        self._bind()

    def _bind(self) -> None:
        if not _ensure_dir():
            self.disabled = True
            return
        # Best-effort unlink of stale socket file from a previous crashed
        # process that owned this receiver slot.
        try:
            if os.path.exists(self.path):
                os.unlink(self.path)
        except OSError as e:
            logger.warning("notify: cannot unlink stale socket %s: %s", self.path, e)
            self.disabled = True
            return
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            s.setblocking(False)
            s.bind(self.path)
            # 1 MiB recv buffer is overkill for 1-byte datagrams but avoids
            # EAGAIN under bursty senders.
            try:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
            except OSError:
                pass
            self._sock = s
            logger.info("notify: receiver bound at %s", self.path)
        except OSError as e:
            logger.warning("notify: bind %s failed: %s; notify disabled", self.path, e)
            self.disabled = True

    def fileno(self) -> int | None:
        return self._sock.fileno() if self._sock is not None else None

    def drain(self) -> None:
        """Drain all pending datagrams, non-blocking. Safe to call after
        select() reports the fd readable."""
        if self.disabled or self._sock is None:
            return
        try:
            while True:
                self._sock.recv(4096)
        except BlockingIOError:
            pass
        except OSError:
            pass

    def wait(self, timeout: float) -> bool:
        """Block up to ``timeout`` seconds for at least one datagram.

        Returns True if we were woken by a notify, False on timeout. Drains
        any remaining datagrams non-blocking before returning. If the
        channel is disabled, returns False immediately (caller should use
        its own timed wait as a fallback).
        """
        if self.disabled or self._sock is None:
            return False
        try:
            r, _, _ = select.select([self._sock], [], [], timeout)
        except (InterruptedError, OSError) as e:
            logger.debug("notify: select error: %s", e)
            return False
        if not r:
            return False
        # Drain everything queued. We don't care how many there are; a
        # single notify is sufficient to wake the loop for a full pass.
        try:
            while True:
                self._sock.recv(4096)
        except BlockingIOError:
            pass
        except OSError as e:
            logger.debug("notify: drain error: %s", e)
        return True

    def close(self) -> None:
        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.close()
                except OSError:
                    pass
                self._sock = None
            try:
                if os.path.exists(self.path):
                    os.unlink(self.path)
            except OSError:
                pass


class NotifySender:
    """Unbound AF_UNIX/DGRAM endpoint. Sends to a receiver's socket file.

    Non-blocking ``send`` swallows errors (receiver not yet bound, socket
    purged) and flips ``disabled`` only on hard errors. Transient
    ECONNREFUSED / ENOENT are logged once at debug and the send is dropped;
    the polling fallback in recv_loop will still deliver the chunk (just
    with higher latency, i.e. the pre-notify behavior).
    """

    def __init__(self, from_stage: int | str, to_stage: int | str):
        self.path = _socket_path(from_stage, to_stage)
        self._sock: socket.socket | None = None
        self.disabled = False
        self._warned = False
        self._lock = threading.Lock()
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            s.setblocking(False)
            self._sock = s
        except OSError as e:
            logger.warning("notify: sender socket() failed: %s; notify disabled", e)
            self.disabled = True

    def send(self) -> None:
        """Fire one notify byte. Never raises; degrades to no-op on error."""
        if self.disabled or self._sock is None:
            return
        try:
            self._sock.sendto(b"\x01", self.path)
        except FileNotFoundError:
            # Receiver hasn't bound yet (startup race) or the socket file
            # was purged. Polling path will cover the miss.
            if not self._warned:
                logger.debug("notify: receiver socket %s absent; falling back to poll", self.path)
                self._warned = True
        except ConnectionRefusedError:
            # Receiver process died or hasn't reopened the socket. Same
            # fallback as above.
            if not self._warned:
                logger.debug("notify: %s refused; falling back to poll", self.path)
                self._warned = True
        except BlockingIOError:
            # Receiver's recv buffer is full. A pending notify is already
            # in flight and will wake them; dropping this one is fine.
            pass
        except OSError as e:
            if e.errno in (errno.ENOENT, errno.ECONNREFUSED, errno.EAGAIN):
                return
            logger.warning("notify: sendto failed (%s); disabling notify", e)
            self.disabled = True

    def close(self) -> None:
        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.close()
                except OSError:
                    pass
                self._sock = None
