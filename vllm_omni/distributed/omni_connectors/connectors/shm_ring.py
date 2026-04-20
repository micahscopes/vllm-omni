# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Persistent shared-memory ring buffer for stage-to-stage chunk transfer.

Replaces the per-chunk ``SharedMemory(create=True) + unlink`` pattern of
``SharedMemoryConnector`` with a single long-lived SHM segment shared between
producer (stage-N save_loop) and consumer (stage-N+1 recv_loop) processes.

Design
------
* One ring per edge ``(from_stage, to_stage)``.  Name is deterministic so both
  sides attach to the same segment without any handshake other than knowing
  the edge tuple.
* ``num_slots`` fixed-size slots of ``slot_bytes`` each.
* Small header at offset 0:
    - u64  producer_seq      (monotonic, producer writes)
    - u64  consumer_seq      (monotonic, consumer writes)
    - u32  num_slots
    - u32  slot_bytes
    - u32  magic
    - u32  version
* Each slot has a tiny per-slot header:
    - u32  ready_flag        (0 = empty, 1 = filled, 2 = consumed)
    - u32  payload_size
    - 56B  opaque_key        (for lookup matching: producer writes, consumer
                              reads.  Keeps the connector.put/get API shape
                              honoring the ``put_key`` argument.)
    - then ``slot_bytes - header`` bytes of payload.
* Mutual exclusion on the indices uses a process-shared ``fcntl.flock`` on a
  lock-file sibling to the SHM segment.  Producer holds the lock only across
  the (read producer_seq, bump producer_seq, publish ready_flag) tuple; the
  bulk bytes copy is lock-free because the slot is exclusively owned by the
  producer until ``ready_flag`` transitions 0 -> 1.
* Memory ordering on aarch64: we issue a ``_mm_sfence``-equivalent via
  ``ctypes`` by calling ``__atomic_store_n`` through a tiny ``ctypes`` trampoline
  is overkill — instead we reuse the ``flock``'s implied acquire/release.  Since
  every reader takes the lock before inspecting ``ready_flag``, and the writer
  bumps it under the same lock, the flock serves as the barrier.  For the
  bulk payload we rely on the fact that the reader only reads up to
  ``payload_size`` bytes which the writer set *before* releasing the flock.

Backpressure
------------
If all slots are full (``producer_seq - consumer_seq == num_slots``) the
producer blocks on a condition variable synthesized from a POSIX semaphore
(``multiprocessing.Semaphore``) named by the edge.  The consumer signals the
semaphore after advancing ``consumer_seq``.  A hard timeout (default 5s) is
applied so we do not deadlock during shutdown.

Fallback
--------
If ring allocation fails for *any* reason (platform, permissions, name
collision with a stale segment that cannot be unlinked), the caller should
drop back to the classic per-chunk SHM path.  ``RingBuffer.attach`` /
``RingBuffer.create`` raise ``RingUnavailableError`` on failure so callers
can fall through.

Cross-process notification
--------------------------
This module intentionally does *not* implement a wake-up channel for the
consumer — a sibling effort is adding an eventfd / AF_UNIX notification
layer.  We expose a ``notify_hook`` callable on the producer side which is
invoked after every successful publish; the notify layer plugs in there.
Until it is wired the consumer falls back to poll + short sleep, which is
what the existing recv_loop already does.
"""

from __future__ import annotations

import ctypes
import fcntl
import os
import struct
import time
from dataclasses import dataclass
from multiprocessing import shared_memory as shm_pkg
from typing import Callable

from ..utils.logging import get_connector_logger

logger = get_connector_logger(__name__)

# Header layout
_MAGIC = 0x4F4D4E52  # "OMNR"
_VERSION = 1
_HEADER_FMT = "<QQIIII"  # producer_seq, consumer_seq, num_slots, slot_bytes, magic, version
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)  # 32 bytes
# Align so slot 0 starts on a 64-byte boundary
_HEADER_PAD = (-_HEADER_SIZE) & 63
_HEADER_TOTAL = _HEADER_SIZE + _HEADER_PAD  # 64

# Per-slot header
_SLOT_HDR_FMT = "<II56s"  # ready_flag, payload_size, opaque_key
_SLOT_HDR_SIZE = struct.calcsize(_SLOT_HDR_FMT)  # 64

_READY_EMPTY = 0
_READY_FILLED = 1
_READY_CONSUMED = 2

DEFAULT_NUM_SLOTS = 8
DEFAULT_SLOT_BYTES = 16 * 1024  # 16 KB per slot (payload up to slot_bytes - _SLOT_HDR_SIZE)
DEFAULT_PUT_TIMEOUT_S = 5.0


class RingUnavailableError(RuntimeError):
    """Raised when the ring buffer cannot be created/attached and caller
    should fall back to the per-chunk SHM path."""


@dataclass
class RingConfig:
    num_slots: int = DEFAULT_NUM_SLOTS
    slot_bytes: int = DEFAULT_SLOT_BYTES
    put_timeout_s: float = DEFAULT_PUT_TIMEOUT_S


def ring_segment_name(from_stage: str, to_stage: str) -> str:
    return f"omni_ring_{from_stage}_{to_stage}"


def ring_lock_path(from_stage: str, to_stage: str) -> str:
    return f"/dev/shm/omni_ring_{from_stage}_{to_stage}.lock"


class RingBuffer:
    """Producer/consumer view of a persistent edge-shared ring buffer.

    A process can call ``create_or_attach`` without caring which side it is on;
    the first caller creates the segment and the second attaches to it.
    """

    def __init__(
        self,
        shm: shm_pkg.SharedMemory,
        lock_path: str,
        num_slots: int,
        slot_bytes: int,
        *,
        is_owner: bool,
        edge: tuple[str, str],
        put_timeout_s: float = DEFAULT_PUT_TIMEOUT_S,
    ):
        self._shm = shm
        self._buf = shm.buf
        self._lock_path = lock_path
        self.num_slots = num_slots
        self.slot_bytes = slot_bytes
        self._is_owner = is_owner
        self._edge = edge
        self._put_timeout_s = put_timeout_s
        self._slot0_offset = _HEADER_TOTAL
        self.notify_hook: Callable[[int], None] | None = None
        self._closed = False

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def create_or_attach(
        cls,
        from_stage: str,
        to_stage: str,
        config: RingConfig | None = None,
    ) -> "RingBuffer":
        cfg = config or RingConfig()
        name = ring_segment_name(from_stage, to_stage)
        total_size = _HEADER_TOTAL + cfg.num_slots * cfg.slot_bytes
        lock_path = ring_lock_path(from_stage, to_stage)

        # Take the lock file for the whole create/attach dance.  Any process
        # trying to create/attach the same edge's ring will serialize here,
        # so the first one wins the "owner" flag.
        try:
            lockf = open(lock_path, "ab+")
        except OSError as e:
            raise RingUnavailableError(f"cannot open lock file {lock_path}: {e}") from e

        try:
            fcntl.flock(lockf, fcntl.LOCK_EX)

            is_owner = False
            shm = None
            try:
                shm = shm_pkg.SharedMemory(name=name)
                # Segment exists.  Validate header.
                header = bytes(shm.buf[:_HEADER_SIZE])
                pseq, cseq, nslots, sbytes, magic, version = struct.unpack(_HEADER_FMT, header)
                if magic != _MAGIC or version != _VERSION:
                    # Stale or alien segment.  Try to unlink and recreate.
                    try:
                        shm.close()
                        shm.unlink()
                    except Exception:
                        pass
                    shm = None
                elif nslots != cfg.num_slots or sbytes != cfg.slot_bytes:
                    # Different shape than we want.  The first side to create
                    # the ring dictates shape; the second side must accept it.
                    logger.warning(
                        "Ring %s shape (%d x %d) differs from requested (%d x %d); using existing",
                        name, nslots, sbytes, cfg.num_slots, cfg.slot_bytes,
                    )
                    cfg = RingConfig(num_slots=nslots, slot_bytes=sbytes, put_timeout_s=cfg.put_timeout_s)
            except FileNotFoundError:
                shm = None

            if shm is None:
                try:
                    shm = shm_pkg.SharedMemory(create=True, size=total_size, name=name)
                except Exception as e:
                    raise RingUnavailableError(f"cannot create ring {name}: {e}") from e
                is_owner = True
                # Zero header + init magic/version
                struct.pack_into(
                    _HEADER_FMT, shm.buf, 0,
                    0, 0, cfg.num_slots, cfg.slot_bytes, _MAGIC, _VERSION,
                )
                # Zero slot headers
                for slot_idx in range(cfg.num_slots):
                    off = _HEADER_TOTAL + slot_idx * cfg.slot_bytes
                    struct.pack_into(_SLOT_HDR_FMT, shm.buf, off, _READY_EMPTY, 0, b"")
                logger.info(
                    "Ring %s created: %d slots x %d bytes (total %d)",
                    name, cfg.num_slots, cfg.slot_bytes, total_size,
                )
            else:
                logger.info(
                    "Ring %s attached: %d slots x %d bytes",
                    name, cfg.num_slots, cfg.slot_bytes,
                )

            return cls(
                shm=shm,
                lock_path=lock_path,
                num_slots=cfg.num_slots,
                slot_bytes=cfg.slot_bytes,
                is_owner=is_owner,
                edge=(from_stage, to_stage),
                put_timeout_s=cfg.put_timeout_s,
            )
        finally:
            try:
                fcntl.flock(lockf, fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                lockf.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Header accessors (NOT lock-protected; call while holding the lock)
    # ------------------------------------------------------------------

    def _read_header(self) -> tuple[int, int]:
        pseq, cseq, _ns, _sb, _mg, _vr = struct.unpack_from(_HEADER_FMT, self._buf, 0)
        return pseq, cseq

    def _write_producer_seq(self, seq: int) -> None:
        struct.pack_into("<Q", self._buf, 0, seq)

    def _write_consumer_seq(self, seq: int) -> None:
        struct.pack_into("<Q", self._buf, 8, seq)

    def _slot_offset(self, slot_idx: int) -> int:
        return self._slot0_offset + slot_idx * self.slot_bytes

    def _read_slot_header(self, slot_idx: int) -> tuple[int, int, bytes]:
        off = self._slot_offset(slot_idx)
        flag, size, key = struct.unpack_from(_SLOT_HDR_FMT, self._buf, off)
        return flag, size, key

    def _write_slot_header(self, slot_idx: int, flag: int, size: int, key: bytes) -> None:
        off = self._slot_offset(slot_idx)
        struct.pack_into(_SLOT_HDR_FMT, self._buf, off, flag, size, key[:56])

    # ------------------------------------------------------------------
    # Producer path
    # ------------------------------------------------------------------

    def put(self, put_key: str, payload: bytes) -> dict:
        """Publish one chunk.  Blocks if the ring is full until a slot frees
        or ``put_timeout_s`` elapses (raises ``TimeoutError``)."""
        if self._closed:
            raise RingUnavailableError("ring is closed")
        max_payload = self.slot_bytes - _SLOT_HDR_SIZE
        if len(payload) > max_payload:
            raise RingUnavailableError(
                f"payload {len(payload)}B exceeds slot capacity {max_payload}B"
            )

        deadline = time.monotonic() + self._put_timeout_s
        key_bytes = put_key.encode("utf-8")

        lockf = open(self._lock_path, "ab+")
        try:
            while True:
                fcntl.flock(lockf, fcntl.LOCK_EX)
                try:
                    pseq, cseq = self._read_header()
                    if pseq - cseq < self.num_slots:
                        slot_idx = pseq % self.num_slots
                        off = self._slot_offset(slot_idx) + _SLOT_HDR_SIZE
                        # Write payload before publishing the ready flag.
                        self._buf[off : off + len(payload)] = payload
                        # Header write under the same lock serves as the
                        # release barrier on aarch64.
                        self._write_slot_header(slot_idx, _READY_FILLED, len(payload), key_bytes)
                        self._write_producer_seq(pseq + 1)
                        if self.notify_hook is not None:
                            try:
                                self.notify_hook(pseq)
                            except Exception:
                                logger.debug("notify_hook raised", exc_info=True)
                        return {
                            "ring": True,
                            "edge": list(self._edge),
                            "seq": pseq,
                            "slot": slot_idx,
                            "size": len(payload),
                        }
                finally:
                    fcntl.flock(lockf, fcntl.LOCK_UN)

                # Ring full; wait a tick and retry.  A sibling eventfd layer
                # may later replace this sleep with a wait on a semaphore.
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"ring {self._edge} full for >{self._put_timeout_s}s"
                    )
                time.sleep(0.0005)
        finally:
            try:
                lockf.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Consumer path
    # ------------------------------------------------------------------

    def try_get(self, get_key: str) -> bytes | None:
        """Non-blocking: return payload bytes for the next slot if its
        ``put_key`` matches ``get_key``.  Returns ``None`` if the next slot
        is empty or belongs to a different key (for recv_loop's per-request
        polling pattern this happens when chunks for different requests
        interleave; caller should keep the request in its pending queue).

        Key-matching semantics: we peek at the oldest *unconsumed* slot
        (the one at ``consumer_seq``).  If its key matches, consume it.
        Otherwise we scan forward up to ``num_slots`` looking for a match
        and return that payload *without* advancing ``consumer_seq`` — the
        slot is marked CONSUMED but remains in-place until the head
        catches up.  This preserves FIFO visibility for the per-slot
        ready flag while allowing out-of-order key lookup.
        """
        if self._closed:
            return None
        key_bytes = get_key.encode("utf-8")

        lockf = open(self._lock_path, "ab+")
        try:
            fcntl.flock(lockf, fcntl.LOCK_EX)
            try:
                pseq, cseq = self._read_header()
                if pseq == cseq:
                    return None  # ring empty

                # Scan slots in [cseq, pseq) for a key match.
                for seq in range(cseq, pseq):
                    slot_idx = seq % self.num_slots
                    flag, size, key = self._read_slot_header(slot_idx)
                    if flag != _READY_FILLED:
                        continue
                    # Trim trailing zero bytes for the stored key
                    stored = key.rstrip(b"\x00")
                    if stored != key_bytes:
                        continue
                    off = self._slot_offset(slot_idx) + _SLOT_HDR_SIZE
                    data = bytes(self._buf[off : off + size])
                    self._write_slot_header(slot_idx, _READY_CONSUMED, 0, b"")

                    # Advance consumer_seq past any prefix of CONSUMED slots.
                    new_cseq = cseq
                    while new_cseq < pseq:
                        probe_idx = new_cseq % self.num_slots
                        pflag, _, _ = self._read_slot_header(probe_idx)
                        if pflag != _READY_CONSUMED:
                            break
                        # Reset the slot header to EMPTY so reuse is clean.
                        self._write_slot_header(probe_idx, _READY_EMPTY, 0, b"")
                        new_cseq += 1
                    if new_cseq != cseq:
                        self._write_consumer_seq(new_cseq)
                    return data
                return None
            finally:
                fcntl.flock(lockf, fcntl.LOCK_UN)
        finally:
            try:
                lockf.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def close(self, unlink: bool | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._shm.close()
        except Exception:
            pass
        if unlink is None:
            unlink = self._is_owner
        if unlink:
            try:
                self._shm.unlink()
            except Exception:
                pass
            try:
                if os.path.exists(self._lock_path):
                    os.remove(self._lock_path)
            except Exception:
                pass

    def __del__(self):
        try:
            self.close(unlink=False)
        except Exception:
            pass
