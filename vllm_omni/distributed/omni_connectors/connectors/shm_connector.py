# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import fcntl
import os
from multiprocessing import shared_memory as shm_pkg
from typing import Any

from vllm_omni.entrypoints.stage_utils import shm_read_bytes, shm_write_bytes

from ..utils.logging import get_connector_logger
from .base import OmniConnectorBase
from .shm_ring import (
    DEFAULT_NUM_SLOTS,
    DEFAULT_SLOT_BYTES,
    RingBuffer,
    RingConfig,
    RingUnavailableError,
)

logger = get_connector_logger(__name__)


class SharedMemoryConnector(OmniConnectorBase):
    """Key-addressed local shared-memory connector.

    SHM is a local-only transport: it reads/writes POSIX shared memory
    segments identified purely by *key*.  It does **not** understand
    remote-transport metadata such as ``source_host`` / ``source_port``
    (that is the RDMA connector's job).  When such metadata is passed in,
    the connector silently falls back to key-based lookup.

    When ``use_ring_buffer`` is set in the config (default True), small
    payloads take a persistent ring-buffer fast path keyed by the edge
    ``(from_stage, to_stage)`` instead of creating a fresh SHM segment
    per chunk.  Payloads that exceed the slot capacity, or ring-allocation
    failures, fall back to the legacy per-chunk SHM path transparently.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.stage_id = config.get("stage_id", -1)
        self.device = config.get("device", "cuda:0")
        self.threshold = int(config.get("shm_threshold_bytes", 65536))
        # Ring-buffer settings
        self._use_ring = bool(config.get("use_ring_buffer", True))
        self._ring_num_slots = int(config.get("ring_num_slots", DEFAULT_NUM_SLOTS))
        self._ring_slot_bytes = int(config.get("ring_slot_bytes", DEFAULT_SLOT_BYTES))
        self._rings: dict[tuple[str, str], RingBuffer] = {}
        self._ring_blacklist: set[tuple[str, str]] = set()
        self._pending_keys: set[str] = set()
        self._metrics = {
            "puts": 0,
            "gets": 0,
            "bytes_transferred": 0,
            "shm_writes": 0,
            "inline_writes": 0,
            "ring_writes": 0,
            "ring_reads": 0,
            "ring_fallbacks": 0,
        }

    def _get_ring(self, from_stage: str, to_stage: str) -> RingBuffer | None:
        """Return an attached ring for this edge, or ``None`` if ring is
        disabled or attach previously failed."""
        if not self._use_ring:
            return None
        edge = (str(from_stage), str(to_stage))
        if edge in self._ring_blacklist:
            return None
        ring = self._rings.get(edge)
        if ring is not None:
            return ring
        try:
            ring = RingBuffer.create_or_attach(
                edge[0], edge[1],
                config=RingConfig(
                    num_slots=self._ring_num_slots,
                    slot_bytes=self._ring_slot_bytes,
                ),
            )
            self._rings[edge] = ring
            return ring
        except RingUnavailableError as e:
            logger.warning(
                "Ring buffer unavailable for edge %s, falling back to per-chunk SHM: %s",
                edge, e,
            )
            self._ring_blacklist.add(edge)
            return None
        except Exception as e:
            logger.warning(
                "Unexpected ring attach failure for edge %s, falling back: %s",
                edge, e,
            )
            self._ring_blacklist.add(edge)
            return None

    def put(
        self,
        from_stage: str,
        to_stage: str,
        put_key: str,
        data: Any,
    ) -> tuple[bool, int, dict[str, Any] | None]:
        try:
            # Serialize once; both paths need the bytes.
            payload = self.serialize_obj(data)
            size = len(payload)

            # Fast path: try the persistent ring buffer if it fits.
            ring = self._get_ring(from_stage, to_stage)
            if ring is not None and size <= (ring.slot_bytes - 64):
                try:
                    ring_meta = ring.put(put_key, payload)
                    metadata = {"ring": ring_meta, "size": size}
                    self._metrics["ring_writes"] += 1
                    self._metrics["puts"] += 1
                    self._metrics["bytes_transferred"] += size
                    return True, size, metadata
                except TimeoutError as e:
                    logger.warning(
                        "Ring put timed out for edge (%s,%s), falling back: %s",
                        from_stage, to_stage, e,
                    )
                    self._metrics["ring_fallbacks"] += 1
                except RingUnavailableError as e:
                    logger.warning(
                        "Ring put rejected for edge (%s,%s), falling back: %s",
                        from_stage, to_stage, e,
                    )
                    self._metrics["ring_fallbacks"] += 1

            # Fallback / legacy path: per-chunk SHM segment.
            lock_file = f"/dev/shm/shm_{put_key}_lockfile.lock"
            with open(lock_file, "wb+") as lockf:
                fcntl.flock(lockf, fcntl.LOCK_EX)
                meta = shm_write_bytes(payload, name=put_key)
                fcntl.flock(lockf, fcntl.LOCK_UN)

            metadata = {"shm": meta, "size": size}
            self._pending_keys.add(put_key)
            self._metrics["shm_writes"] += 1
            self._metrics["puts"] += 1
            self._metrics["bytes_transferred"] += size

            return True, size, metadata

        except Exception as e:
            logger.error(f"SharedMemoryConnector put failed for req {put_key}: {e}")
            return False, 0, None

    def _get_data_with_lock(self, lock_file: str, shm_handle: dict):
        obj = None
        try:
            with open(lock_file, "rb+") as lockf:
                fcntl.flock(lockf, fcntl.LOCK_EX)
                data_bytes = shm_read_bytes(shm_handle)
                fcntl.flock(lockf, fcntl.LOCK_UN)
            obj = self.deserialize_obj(data_bytes)
            return obj, int(shm_handle.get("size", 0))
        except Exception as e:
            logger.error(f"SharedMemoryConnector shm get failed for req : {e}")
            return None
        finally:
            # If data has been received, delete lock_file.
            if obj and os.path.exists(lock_file):
                os.remove(lock_file)

    def _try_get_from_ring(
        self, from_stage: str, to_stage: str, get_key: str
    ) -> tuple[Any, int] | None:
        ring = self._get_ring(from_stage, to_stage)
        if ring is None:
            return None
        try:
            raw = ring.try_get(get_key)
        except Exception:
            logger.debug("ring try_get failed for %s", get_key, exc_info=True)
            return None
        if raw is None:
            return None
        try:
            obj = self.deserialize_obj(raw)
        except Exception as e:
            logger.error("Ring deserialization failed for key %s: %s", get_key, e)
            return None
        self._metrics["ring_reads"] += 1
        return obj, len(raw)

    def _get_by_key(self, get_key: str) -> tuple[Any, int] | None:
        """Read a SHM segment addressed purely by *get_key*."""
        shm = None
        try:
            shm = shm_pkg.SharedMemory(name=get_key)
            if shm is None or shm.size == 0:
                return None
            lock_file = f"/dev/shm/shm_{get_key}_lockfile.lock"
            shm_handle = {"name": get_key, "size": shm.size}
            result = self._get_data_with_lock(lock_file, shm_handle)
            if result is not None:
                self._pending_keys.discard(get_key)
            return result
        except FileNotFoundError:
            return None
        except Exception:
            logger.debug("_get_by_key: unexpected error reading SHM segment %s", get_key, exc_info=True)
            return None
        finally:
            if shm:
                shm.close()

    def get(
        self,
        from_stage: str,
        to_stage: str,
        get_key: str,
        metadata=None,
    ) -> tuple[Any, int] | None:
        if metadata is not None:
            if isinstance(metadata, dict) and get_key in metadata:
                metadata = metadata.get(get_key)

            if not isinstance(metadata, dict):
                result = self._try_get_from_ring(from_stage, to_stage, get_key)
                if result is not None:
                    return result
                return self._get_by_key(get_key)

            if "ring" in metadata:
                result = self._try_get_from_ring(from_stage, to_stage, get_key)
                if result is not None:
                    return result
                # The sender wrote to ring but the entry is not (yet) present
                # in this reader's view.  Return None and let the caller poll.
                return None

            if "inline_bytes" in metadata:
                try:
                    obj = self.deserialize_obj(metadata["inline_bytes"])
                    self._pending_keys.discard(get_key)
                    return obj, int(metadata.get("size", 0))
                except Exception as e:
                    logger.error(f"SharedMemoryConnector inline get failed for req {get_key}: {e}")
                    return None

            if "shm" in metadata:
                shm_handle = metadata["shm"]
                lock_file = f"/dev/shm/shm_{shm_handle['name']}_lockfile.lock"
                result = self._get_data_with_lock(lock_file, shm_handle)
                if result is not None:
                    self._pending_keys.discard(get_key)
                return result

            # Metadata is a dict but has no SHM-specific handle (e.g. RDMA-
            # style source_host/source_port).  Try ring then fall back to
            # key-based read.
            result = self._try_get_from_ring(from_stage, to_stage, get_key)
            if result is not None:
                return result
            return self._get_by_key(get_key)

        # No metadata: try ring first (current recv_loop call pattern) then
        # fall back to key-based SHM read.
        result = self._try_get_from_ring(from_stage, to_stage, get_key)
        if result is not None:
            return result
        return self._get_by_key(get_key)

    def cleanup(self, request_id: str) -> None:
        """Best-effort cleanup of unconsumed SHM segments for *request_id*.

        Matches pending keys where *request_id* appears as the full key,
        as a ``_``-delimited prefix, or as a ``_``-delimited suffix.
        If ``get()`` was never called, we unlink it here so /dev/shm
        doesn't leak.
        """
        stale = [
            k
            for k in self._pending_keys
            if k == request_id or k.startswith(request_id + "_") or k.endswith("_" + request_id)
        ]
        for key in stale:
            self._pending_keys.discard(key)
            try:
                seg = shm_pkg.SharedMemory(name=key)
                seg.close()
                seg.unlink()
                logger.debug("cleanup: unlinked unconsumed SHM segment %s", key)
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.debug("cleanup: failed to unlink SHM segment %s: %s", key, e)
            lock_file = f"/dev/shm/shm_{key}_lockfile.lock"
            if os.path.exists(lock_file):
                try:
                    os.remove(lock_file)
                except OSError:
                    pass

    def close(self) -> None:
        """Unlink all remaining tracked SHM segments and close rings."""
        for key in list(self._pending_keys):
            try:
                seg = shm_pkg.SharedMemory(name=key)
                seg.close()
                seg.unlink()
            except Exception:
                pass
            lock_file = f"/dev/shm/shm_{key}_lockfile.lock"
            if os.path.exists(lock_file):
                try:
                    os.remove(lock_file)
                except OSError:
                    pass
        self._pending_keys.clear()

        # Close (and, if we own them, unlink) the per-edge rings.
        for edge, ring in list(self._rings.items()):
            try:
                ring.close()
            except Exception:
                logger.debug("ring close failed for edge %s", edge, exc_info=True)
        self._rings.clear()

    def health(self) -> dict[str, Any]:
        return {"status": "healthy", "threshold": self.threshold, **self._metrics}
