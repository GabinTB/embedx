"""On-demand multi-model registry with TTL eviction.

Importable without torch: every torch touch is lazy and inside a function,
enforced by the task-06 ast guard which walks all of `src/embedx`.

The registry owns model lifetime, not device discovery — it is handed the
ranked device list and never calls `discover_devices` or `rank_devices`.

Holding a model
---------------
`acquire()` is the API, not `get_or_load()`:

    with registry.acquire("intfloat/e5-base", pooling=Pooling.MEAN) as engine:
        vectors = engine.embed(texts)

The context manager is the chosen answer to "when does a model stop being
used". `__enter__` resolves (loading if needed) and increments `ref_count`;
`__exit__` decrements it and stamps `last_used`, in a `finally`, so an
exception still releases. That means idle time is measured from the end of
the last *use*, not the last request for a handle, without the registry
having to wrap or proxy `Engine.embed`. While `ref_count > 0` the reaper
will not evict, however long the idle timer says.

`get_or_load()` is the same resolution without the reference count: it
refreshes `last_used` and hands back the Engine, so a caller that holds one
across a reaper tick can have it evicted underneath them. It exists because
the spec names it; prefer `acquire()`.

Why a thread and not an asyncio task
------------------------------------
The reaper is a plain daemon `threading.Thread`. Eviction calls
`torch.cuda.empty_cache()` and drops model tensors, both of which are
blocking C calls that would stall an event loop; and the registry is
already thread-based because `Engine` dispatches backends across threads
and the API layer reaches it through `run_in_threadpool`. An asyncio task
would have to hop back into a thread to do the only work it does. It is a
daemon so a killed server never hangs on it, and `stop()` uses an
`Event`, so shutdown does not wait out a full interval.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from embedx.backend.base import EmbeddingBackend
from embedx.backend.factory import (
    BackendFactory,
    engine_from_backends,
    hf_backend_factory,
    new_length_cache,
)
from embedx.config import Dtype, Pooling, Settings
from embedx.engine.engine import Engine
from embedx.gpu.discovery import DeviceInfo

logger = logging.getLogger("embedx.registry")

# Until task 15 wires `default_keep_alive_s` into Settings, an unspecified
# keep_alive means this many idle seconds before eviction.
DEFAULT_KEEP_ALIVE_S = 300.0

_SAFETENSORS_SUFFIX = ".safetensors"
# Formats that go through `torch.load`, i.e. through pickle. Loading one
# executes whatever the file says to execute.
_LEGACY_WEIGHT_SUFFIXES = (".bin", ".pt", ".pth", ".ckpt", ".pkl")


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class RegistryError(Exception):
    """Base for every registry failure.

    One base so the API layer (task 14) maps these to status codes by type
    rather than by matching message strings.
    """


class PoolingRequiredError(RegistryError):
    """A model was requested for first load without a pooling strategy."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        super().__init__(
            f"{model_id} is not loaded and no pooling was given: pooling is never "
            "inferred, because a wrong one produces plausible but garbage vectors "
            "with no error anywhere. Specify one of: "
            + ", ".join(member.value for member in Pooling)
        )


class PoolingConflictError(RegistryError):
    """A loaded model was requested with a different pooling strategy."""

    def __init__(self, model_id: str, requested: Pooling, resolved: Pooling) -> None:
        self.model_id = model_id
        self.requested = requested
        self.resolved = resolved
        super().__init__(
            f"{model_id} is already loaded with pooling={resolved.value}, but "
            f"pooling={requested.value} was requested. Serving the request would "
            "silently return vectors from the wrong pooling; reloading would change "
            "what every other in-flight caller gets. Neither is done."
        )


class ModelPlacementError(RegistryError):
    """No device could hold the model."""

    def __init__(self, model_id: str, failures: list[tuple[int, str]]) -> None:
        self.model_id = model_id
        self.failures = list(failures)
        detail = "; ".join(f"device {index}: {reason}" for index, reason in failures)
        super().__init__(
            f"{model_id} did not fit on any of the {len(failures)} device(s) tried "
            f"({detail or 'no devices available'})"
        )


class UnsupportedWeightFormatError(RegistryError):
    """The checkpoint offers only pickle-based weights."""

    def __init__(self, model_id: str, files: list[str]) -> None:
        self.model_id = model_id
        self.files = list(files)
        super().__init__(
            f"{model_id} ships no .safetensors weights; the only weight file(s) "
            f"found are {', '.join(files)}, which load through pickle and can "
            "execute arbitrary code. embedx will not load them. Ask the publisher "
            "for a safetensors conversion, or convert it yourself."
        )


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelStatus:
    """One loaded model, in plain types only — no torch types, no Engine."""

    model_id: str
    pooling: Pooling
    dtype: Dtype
    device_indices: tuple[int, ...]
    # SPEC-GAP: task 13 says "per-device idle time", but idle time is a
    # property of the model, not of one of its devices — every device it
    # sits on goes idle and is evicted together, so a per-device mapping
    # would be the same number repeated. Reported once per model, which is
    # also the singular "its idle time" task 14's /info asks for.
    idle_s: float
    last_used_epoch_s: float
    ref_count: int


@dataclass
class _Entry:
    """Registry-internal bookkeeping for one loaded model."""

    engine: Engine
    backends: list[EmbeddingBackend]
    devices: list[DeviceInfo]
    pooling: Pooling
    dtype: Dtype
    max_seq_len: int | None
    keep_alive: float
    loaded_at: float
    last_used: float  # monotonic: TTL arithmetic
    last_used_epoch: float  # wall clock: what /info shows a human
    ref_count: int = 0
    length_cache: Any = field(default=None, repr=False)


# --------------------------------------------------------------------------- #
# Lazy torch helpers
# --------------------------------------------------------------------------- #


def _import_torch() -> Any:
    try:
        import torch
    except ImportError:
        return None
    return torch


def _is_out_of_memory(exc: BaseException) -> bool:
    """True for a CUDA OOM, and only that.

    Any other construction failure (a bad model id, a missing config) would
    fail identically on every device, so it is re-raised rather than turned
    into a placement report that blames the hardware.

    Matched by type name as well as by isinstance because torch is absent
    entirely in CPU CI, which is also where the placement logic is tested.
    """
    torch = _import_torch()
    if torch is not None and isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return type(exc).__name__ == "OutOfMemoryError"


def _empty_cache(device_index: int) -> None:
    """Release cached blocks on one device; no-op without torch or CUDA.

    Called after a failed placement so a partial allocation does not poison
    the next device's attempt, and after eviction so the freed model's
    memory is actually returned to the driver.
    """
    torch = _import_torch()
    if torch is None or not torch.cuda.is_available():
        return
    with torch.cuda.device(device_index):
        torch.cuda.empty_cache()


def _default_weight_files(model_id: str, revision: str | None) -> list[str]:
    """Filenames `from_pretrained` could choose from, without downloading."""
    local = Path(model_id)
    if local.is_dir():
        return [str(path.relative_to(local)) for path in local.rglob("*") if path.is_file()]
    from huggingface_hub import list_repo_files  # lazy: not needed for local paths

    return list(list_repo_files(model_id, revision=revision))


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


class ModelRegistry:
    """Loads models on demand, one Engine each, and evicts them when idle."""

    def __init__(
        self,
        devices: list[DeviceInfo],
        reaper_interval_s: float = 30.0,
        *,
        backend_factory: BackendFactory | None = None,
        weight_file_lister: Callable[[str, str | None], list[str]] | None = None,
    ) -> None:
        """`devices` is the ranked list from `rank_devices`, fastest first.

        `backend_factory` and `weight_file_lister` are the test seams: the
        defaults are the real HFBackend constructor and a hub/local file
        listing, so production callers pass neither.
        """
        self.devices = list(devices)
        self.reaper_interval_s = reaper_interval_s
        self._backend_factory: BackendFactory = backend_factory or hf_backend_factory
        self._weight_file_lister = weight_file_lister or _default_weight_files

        self._entries: dict[str, _Entry] = {}
        # Guards `_entries` and every ref_count/last_used mutation. Held
        # only for dict-sized work, never across a model load.
        self._entries_lock = threading.RLock()
        # One lock per model_id, so a slow load of A does not block B. The
        # outer lock guards only the creation of the inner ones.
        self._load_locks: dict[str, threading.Lock] = {}
        self._load_locks_lock = threading.Lock()

        self._reaper: threading.Thread | None = None
        self._stop = threading.Event()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def get_or_load(
        self,
        model_id: str,
        pooling: Pooling | None = None,
        dtype: Dtype = Dtype.AUTO,
        max_seq_len: int | None = None,
        keep_alive: float | None = None,
    ) -> Engine:
        """Resolve `model_id` to an Engine, loading it if necessary.

        On a cache hit, `dtype`, `max_seq_len` and `keep_alive` are IGNORED:
        the model is already resident with the values it was loaded under,
        and only `pooling` is conflict-checked, because only pooling can
        silently produce wrong vectors rather than merely different
        performance. `pooling=None` means "whatever it was loaded with".

        Does not take a reference — see `acquire`, which does.
        """
        return self._entry_for(model_id, pooling, dtype, max_seq_len, keep_alive, hold=False).engine

    @contextmanager
    def acquire(
        self,
        model_id: str,
        pooling: Pooling | None = None,
        dtype: Dtype = Dtype.AUTO,
        max_seq_len: int | None = None,
        keep_alive: float | None = None,
    ) -> Iterator[Engine]:
        """Hold `model_id` against eviction for the duration of the block.

        Same resolution rules as `get_or_load`. The reference is released in
        a `finally`, so an exception inside the block still releases it and
        still stamps `last_used`.
        """
        entry = self._entry_for(model_id, pooling, dtype, max_seq_len, keep_alive, hold=True)
        try:
            yield entry.engine
        finally:
            with self._entries_lock:
                entry.ref_count -= 1
                entry.last_used = time.monotonic()
                entry.last_used_epoch = time.time()
                # keep_alive <= 0 means "do not stay resident": tear down as
                # soon as the request that triggered the load finishes,
                # rather than leaving the model on the card until whenever
                # the reaper next happens to wake (up to reaper_interval_s
                # later). The reaper keeps the same rule as a backstop for
                # entries that were never acquired.
                expire_now = (
                    entry.ref_count == 0
                    and entry.keep_alive <= 0
                    and self._entries.get(model_id) is entry
                )
                if expire_now:
                    del self._entries[model_id]
            if expire_now:
                self._evict(model_id, entry)

    def list_loaded(self) -> list[ModelStatus]:
        """Every resident model, sorted by model_id for stable output."""
        now = time.monotonic()
        with self._entries_lock:
            return sorted(
                (
                    ModelStatus(
                        model_id=model_id,
                        pooling=entry.pooling,
                        dtype=entry.dtype,
                        device_indices=tuple(device.index for device in entry.devices),
                        # In use is not idle, whatever the clock says.
                        idle_s=0.0 if entry.ref_count > 0 else max(0.0, now - entry.last_used),
                        last_used_epoch_s=entry.last_used_epoch,
                        ref_count=entry.ref_count,
                    )
                    for model_id, entry in self._entries.items()
                ),
                key=lambda status: status.model_id,
            )

    def start(self) -> None:
        """Start the reaper thread; idempotent."""
        if self._reaper is not None and self._reaper.is_alive():
            return
        self._stop.clear()
        self._reaper = threading.Thread(
            target=self._reap_loop, name="embedx-model-reaper", daemon=True
        )
        self._reaper.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the reaper thread and wait for it; idempotent."""
        self._stop.set()
        thread, self._reaper = self._reaper, None
        if thread is not None:
            thread.join(timeout)

    def evict_all(self) -> list[str]:
        """Tear every model down regardless of idle time or references.

        For shutdown. Returns the model_ids evicted.
        """
        with self._entries_lock:
            evicted = list(self._entries)
            entries = [self._entries.pop(model_id) for model_id in evicted]
        for model_id, entry in zip(evicted, entries, strict=True):
            self._evict(model_id, entry)
        return evicted

    # ------------------------------------------------------------------ #
    # Resolution and loading
    # ------------------------------------------------------------------ #

    def _entry_for(
        self,
        model_id: str,
        pooling: Pooling | None,
        dtype: Dtype,
        max_seq_len: int | None,
        keep_alive: float | None,
        *,
        hold: bool,
    ) -> _Entry:
        # Fast path: already resident, no load lock needed at all.
        with self._entries_lock:
            entry = self._entries.get(model_id)
            if entry is not None:
                return self._touch(entry, model_id, pooling, hold=hold)

        # Not resident. Serialise on this model_id only: a concurrent first
        # load of a DIFFERENT model takes a different lock and proceeds.
        with self._load_lock(model_id):
            # Someone may have finished loading while we waited here; that
            # is the whole point of the second check.
            with self._entries_lock:
                entry = self._entries.get(model_id)
                if entry is not None:
                    return self._touch(entry, model_id, pooling, hold=hold)

            if pooling is None:
                raise PoolingRequiredError(model_id)

            settings = self._settings_for(model_id, pooling, dtype, max_seq_len)
            # Before anything constructs a backend, and therefore before
            # anything hands a file to torch.load.
            self._check_weight_format(model_id, settings.revision)
            entry = self._load(model_id, settings, keep_alive)

            with self._entries_lock:
                self._entries[model_id] = entry
                if hold:
                    entry.ref_count += 1
                entry.last_used = time.monotonic()
            return entry

    def _touch(
        self, entry: _Entry, model_id: str, pooling: Pooling | None, *, hold: bool
    ) -> _Entry:
        """Conflict-check, refresh, and optionally reference a live entry.

        Caller must hold `_entries_lock`.
        """
        if pooling is not None and pooling is not entry.pooling:
            # WARNING, not just an exception: the caller sees the 409, but
            # the operator is the one who needs to know two clients disagree
            # about how this model should be pooled.
            logger.warning(
                "pooling conflict for %s: loaded as %s, requested %s; refusing "
                "rather than reapplying or reloading",
                model_id,
                entry.pooling.value,
                pooling.value,
            )
            raise PoolingConflictError(model_id, pooling, entry.pooling)
        entry.last_used = time.monotonic()
        entry.last_used_epoch = time.time()
        if hold:
            entry.ref_count += 1
        return entry

    def _load_lock(self, model_id: str) -> threading.Lock:
        with self._load_locks_lock:
            lock = self._load_locks.get(model_id)
            if lock is None:
                lock = threading.Lock()
                self._load_locks[model_id] = lock
            return lock

    def _settings_for(
        self, model_id: str, pooling: Pooling, dtype: Dtype, max_seq_len: int | None
    ) -> Settings:
        """Per-model Settings carrying the load parameters.

        The Engine takes its batching knobs (`max_batch_tokens`,
        `device_batch_tokens`, `max_batch_items`) plus `normalize` and
        `revision` from a Settings, so one is built per load from the
        parameters given, with everything else at its default (or its
        `EMBEDX_*` environment value). Task 15 replaces this with the real
        Settings threaded through from the server.
        """
        return Settings(
            model_id=model_id,
            pooling=pooling,
            dtype=dtype,
            max_seq_len=max_seq_len,
        )

    def _check_weight_format(self, model_id: str, revision: str | None) -> None:
        """Refuse a checkpoint whose only weights go through pickle.

        Safetensors alongside a legacy file is normal and fine: HF's own
        `from_pretrained` prefers safetensors, so that case passes silently.
        Only a checkpoint with no safetensors at all is refused.
        """
        try:
            files = self._weight_file_lister(model_id, revision)
        except Exception as exc:
            # A cached model behind an unreachable hub must still serve, so
            # this fails open — loudly. Same call as _is_st_checkpoint's.
            logger.warning(
                "could not list files for %s (%s: %s); loading without the weight-format check",
                model_id,
                type(exc).__name__,
                exc,
            )
            return
        if any(name.endswith(_SAFETENSORS_SUFFIX) for name in files):
            return
        legacy = sorted(name for name in files if name.endswith(_LEGACY_WEIGHT_SUFFIXES))
        if legacy:
            raise UnsupportedWeightFormatError(model_id, legacy)
        # No weight file of either kind: not this check's business. Let
        # from_pretrained say what it is actually missing.

    def _load(self, model_id: str, settings: Settings, keep_alive: float | None) -> _Entry:
        """Place the model on every device it fits on, fastest first.

        No memory estimation: an estimate that is wrong in the optimistic
        direction OOMs anyway, and one that is wrong pessimistically refuses
        a model that would have fitted. Trying is the measurement.
        """
        length_cache = new_length_cache()
        backends: list[EmbeddingBackend] = []
        placed: list[DeviceInfo] = []
        failures: list[tuple[int, str]] = []

        for device in self.devices:
            try:
                backend = self._backend_factory(
                    model_id=model_id,
                    device_index=device.index,
                    pooling=settings.pooling,
                    normalize=settings.normalize,
                    dtype=settings.dtype,
                    max_seq_length=settings.max_seq_len,
                    revision=settings.revision,
                    length_cache=length_cache,
                )
            except Exception as exc:
                if not _is_out_of_memory(exc):
                    raise
                failures.append((device.index, f"{type(exc).__name__}: {exc}"))
                # Whatever was allocated before the OOM is still held; the
                # next device starts clean only if it is released now.
                _empty_cache(device.index)
                logger.info(
                    "%s did not fit on device %d (%s); trying the next device",
                    model_id,
                    device.index,
                    type(exc).__name__,
                )
                continue
            backends.append(backend)
            placed.append(device)

        if not backends:
            raise ModelPlacementError(model_id, failures)

        now = time.monotonic()
        # WARNING, not INFO: this is the moment a model's pooling is fixed
        # for as long as it stays resident, and a wrong one produces
        # plausible garbage with no error anywhere downstream. It belongs in
        # the log at a level nobody filters out.
        logger.warning(
            "pooling for %s resolved to %s on first load (dtype=%s, device(s) %s); "
            "later requests naming a different pooling will be refused",
            model_id,
            settings.pooling.value,
            settings.dtype.value,
            [device.index for device in placed],
        )
        return _Entry(
            engine=engine_from_backends(backends, placed, settings),
            backends=backends,
            devices=placed,
            pooling=settings.pooling,
            dtype=settings.dtype,
            max_seq_len=settings.max_seq_len,
            keep_alive=DEFAULT_KEEP_ALIVE_S if keep_alive is None else keep_alive,
            loaded_at=now,
            last_used=now,
            last_used_epoch=time.time(),
            length_cache=length_cache,
        )

    # ------------------------------------------------------------------ #
    # Eviction
    # ------------------------------------------------------------------ #

    def _reap_loop(self) -> None:
        # Event.wait, not sleep: stop() returns promptly instead of after a
        # full interval.
        while not self._stop.wait(self.reaper_interval_s):
            try:
                self._reap_once()
            except Exception:  # a reaper that dies silently is worse
                logger.exception("reaper pass failed")

    def _reap_once(self) -> list[str]:
        """One eviction pass. Public loop calls it; tests call it directly."""
        now = time.monotonic()
        with self._entries_lock:
            doomed = [
                model_id
                for model_id, entry in self._entries.items()
                if entry.ref_count == 0 and self._is_expired(entry, now)
            ]
            # Removed from the map before teardown, still under the lock, so
            # no caller can acquire a half-torn-down entry. A request that
            # arrives after this just reloads.
            entries = [self._entries.pop(model_id) for model_id in doomed]
        for model_id, entry in zip(doomed, entries, strict=True):
            self._evict(model_id, entry)
        return doomed

    @staticmethod
    def _is_expired(entry: _Entry, now: float) -> bool:
        # keep_alive <= 0 means "do not linger": evict on the very next pass
        # once nothing holds it, without consulting the clock at all. Same
        # branch as the timed case so there is one eviction rule, not two.
        if entry.keep_alive <= 0:
            return True
        return now - entry.last_used >= entry.keep_alive

    def _evict(self, model_id: str, entry: _Entry) -> None:
        resident_s = time.monotonic() - entry.loaded_at
        for backend in entry.backends:
            close = getattr(backend, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.exception("closing backend for %s failed", model_id)
        # Order matters. CUDA memory returns only when the last reference to
        # the model's tensors is gone, and the Engine holds one per backend,
        # so both it and the entry release before empty_cache is asked for
        # the blocks back. Doing it the other way round frees nothing.
        entry.engine.close()
        entry.backends.clear()
        entry.length_cache = None
        for device in entry.devices:
            _empty_cache(device.index)
        logger.info(
            "evicted %s from device(s) %s after %.1fs resident",
            model_id,
            [device.index for device in entry.devices],
            resident_s,
        )
