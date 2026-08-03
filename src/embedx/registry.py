"""On-demand multi-model registry with TTL eviction.

Importable without torch: every torch touch is lazy and inside a function,
enforced by the task-06 ast guard which walks all of `src/embedx`. Device
memory is released through the `Accelerator` seam, so nothing here names a
GPU vendor.

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
The reaper is a plain daemon `threading.Thread`. Eviction calls the
accelerator's `empty_cache` and drops model tensors, both of which are
blocking C calls that would stall an event loop; and the registry is
already thread-based because `Engine` dispatches backends across threads
and the API layer reaches it through `run_in_threadpool`. An asyncio task
would have to hop back into a thread to do the only work it does. It is a
daemon so a killed server never hangs on it, and `stop()` uses an
`Event`, so shutdown does not wait out a full interval.
"""

from __future__ import annotations

import gc
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
from embedx.config import DEFAULT_KEEP_ALIVE_S, Dtype, Pooling, Settings
from embedx.engine.engine import Engine
from embedx.gpu.discovery import DeviceInfo
from embedx.gpu.vendor import Accelerator, get_accelerator

logger = logging.getLogger("embedx.registry")

# DEFAULT_KEEP_ALIVE_S is re-exported from config, where it now lives: the
# registry cannot own it and also be configured by Settings, and config
# cannot import the registry (the dependency runs the other way). Settings
# is the source of truth; this is the fallback for a registry built without
# one.
__all__ = [
    "DEFAULT_KEEP_ALIVE_S",
    "ModelCapacityError",
    "ModelPlacementError",
    "ModelRegistry",
    "ModelStatus",
    "PoolingConflictError",
    "PoolingRequiredError",
    "RegistryError",
    "SmokeTestFailedError",
    "UnsupportedWeightFormatError",
    "WeightFormatUnverifiableError",
]

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


class ModelCapacityError(RegistryError):
    """`max_loaded_models` is reached and every resident model is in use.

    Distinct from `ModelPlacementError`: the hardware is fine, the cap is
    administrative, and the condition clears as soon as an in-flight
    request finishes. Evicting a model another request is mid-way through
    would be the one thing worse than making this one wait.
    """

    def __init__(self, model_id: str, resident: list[str], cap: int) -> None:
        self.model_id = model_id
        self.resident = list(resident)
        self.cap = cap
        super().__init__(
            f"cannot load {model_id}: max_loaded_models={cap} is reached and all "
            f"{len(resident)} resident model(s) ({', '.join(resident)}) are serving "
            "requests, so none can be evicted to make room. Retry shortly, or raise "
            "EMBEDX_MAX_LOADED_MODELS."
        )


class SmokeTestFailedError(RegistryError):
    """The model loaded onto the device(s) but could not produce a vector.

    The failure this exists for: placement succeeding while inference is
    broken. torch JIT-compiles Triton kernels at first inference, not at
    load, so on a host without a C toolchain a model loads clean, registers
    as resident, and then raises on every request while `/info` reports it
    healthy. Task 17 hit exactly that with Qwen3.

    Its cause is always chained (`raise ... from exc`): the real reason —
    a Triton compile failure, a missing compiler, a malformed backend
    return — reaches the server log even though the client sees only the
    envelope.
    """

    def __init__(self, model_id: str, devices: list[int], reason: str) -> None:
        self.model_id = model_id
        self.devices = list(devices)
        self.reason = reason
        super().__init__(
            f"{model_id} loaded onto device(s) {devices} but failed its smoke test "
            f"({reason}). The model is NOT resident: a model that cannot embed one "
            "short string would have failed every request instead. This usually means "
            "the runtime is missing something inference needs rather than something "
            "loading needs — a C compiler for torch's Triton JIT is the common one."
        )


class WeightFormatUnverifiableError(RegistryError):
    """The weight format could not be determined, so the load is refused.

    Deliberately not a subclass of `UnsupportedWeightFormatError`: that one
    is a permanent property of the checkpoint (it ships pickle weights and
    always will, so retrying is pointless), this one is a transient state
    of the server (the hub is unreachable and the model is not cached, so
    retrying is exactly right). Task 14 can only map them to different
    status codes if the types differ.
    """

    def __init__(self, model_id: str, cause: BaseException) -> None:
        self.model_id = model_id
        self.cause = cause
        super().__init__(
            f"cannot verify the weight format of {model_id}: listing its files "
            f"failed ({type(cause).__name__}: {cause}). The load is refused rather "
            "than attempted, because an unverified checkpoint may be pickle-based, "
            "and pickle executes code on deserialization. Retry when the hub is "
            "reachable, or point embedx at a local path."
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
    # Summed over every backend behind this model, so a model sharded
    # across devices reports one total rather than a number per card.
    # Truncation is silent everywhere else — an over-long input is cut and
    # embedded with no error — so this counter is the only way an operator
    # sees it happening.
    truncated_count: int = 0


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
# Placement helpers
# --------------------------------------------------------------------------- #


def _is_out_of_memory(exc: BaseException, accelerator: Accelerator) -> bool:
    """True for an out-of-memory failure, and only that.

    Any other construction failure (a bad model id, a missing config) would
    fail identically on every device, so it is re-raised rather than turned
    into a placement report that blames the hardware.

    Matched by type name as well as by isinstance because the GPU runtime is
    absent entirely in CPU CI, which is also where the placement logic is
    tested — so the exception classes cannot be imported to compare against.
    The names come from the accelerator rather than being hardcoded here,
    which keeps "OutOfMemoryError" a CUDA fact rather than a core one.
    """
    types = accelerator.oom_error_types()
    if types and isinstance(exc, types):
        return True
    return type(exc).__name__ in accelerator.oom_error_names()


def _cached_snapshot_files(model_id: str, revision: str | None) -> list[str] | None:
    """What is already on disk for `model_id`, or None if nothing is.

    `local_files_only=True` makes this a cache lookup, never a download.
    """
    from huggingface_hub import snapshot_download  # lazy: not needed for local paths

    try:
        root = Path(snapshot_download(model_id, revision=revision, local_files_only=True))
    except Exception:
        return None
    return [str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()]


def _default_weight_files(model_id: str, revision: str | None) -> list[str]:
    """Filenames `from_pretrained` could choose from, without downloading.

    Three sources, in order of authority: a local directory, the hub's file
    listing, and finally whatever is already in the local hub cache.

    That last fallback is not an optimisation. `list_repo_files` is a pure
    API call — it never looks at the cache — so a model that is fully
    downloaded and would load perfectly well offline still raises here on
    an unreachable hub. Since the caller refuses any load it cannot verify,
    without this fallback an air-gapped host could not serve a single
    cached model, safetensors or not.
    """
    local = Path(model_id)
    if local.is_dir():
        return [str(path.relative_to(local)) for path in local.rglob("*") if path.is_file()]

    from huggingface_hub import list_repo_files  # lazy: not needed for local paths

    try:
        return list(list_repo_files(model_id, revision=revision))
    except Exception as hub_exc:
        cached = _cached_snapshot_files(model_id, revision)
        if cached is None:
            raise  # nothing on disk either: the caller decides what to do
        logger.info(
            "hub listing for %s failed (%s: %s); verifying the weight format "
            "against the cached snapshot instead",
            model_id,
            type(hub_exc).__name__,
            hub_exc,
        )
        return cached


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
        settings: Settings | None = None,
        backend_factory: BackendFactory | None = None,
        weight_file_lister: Callable[[str, str | None], list[str]] | None = None,
        accelerator: Accelerator | None = None,
    ) -> None:
        """`devices` is the ranked list from `rank_devices`, fastest first.

        `settings` supplies everything server-wide that a loaded model
        needs and a request cannot say: the Engine's batching budgets,
        `normalize`, `revision`, the default keep-alive and the residency
        cap. It defaults to `Settings()` so a registry is still
        constructible on its own, which is what the tests rely on; `serve`
        passes the real one.

        `backend_factory`, `weight_file_lister` and `accelerator` are the
        test seams: the defaults are the real HFBackend constructor, a
        hub/local file listing and the process's real accelerator, so
        production callers pass none of them.
        """
        self.devices = list(devices)
        self.reaper_interval_s = reaper_interval_s
        self._settings = settings if settings is not None else Settings()
        self.default_keep_alive_s = self._settings.default_keep_alive_s
        self.max_loaded_models = self._settings.max_loaded_models
        self._backend_factory: BackendFactory = backend_factory or hf_backend_factory
        self._weight_file_lister = weight_file_lister or _default_weight_files
        self._accelerator = accelerator if accelerator is not None else get_accelerator()

        self._entries: dict[str, _Entry] = {}
        # Guards `_entries` and every ref_count/last_used mutation. Held
        # only for dict-sized work, never across a model load.
        self._entries_lock = threading.RLock()
        # One lock per model_id, so a slow load of A does not block B. The
        # outer lock guards only the creation of the inner ones.
        self._load_locks: dict[str, threading.Lock] = {}
        self._load_locks_lock = threading.Lock()

        # Cold loads only. This lives in the registry rather than being
        # injected because the registry is the only place that knows which
        # acquisitions are cold: the HTTP layer sees one `acquire()` call
        # and cannot tell a 9-second load from a dictionary lookup. It is
        # taken deep inside `_entry_for`, past both resident checks, so a
        # warm acquire never touches it -- if it did, this would silently
        # become a second request cap and warm requests would queue behind
        # cold loads, which is exactly what the separate caps prevent.
        #
        # No timeout on acquiring it: the request semaphore in the API layer
        # is what bounds how long a caller waits, and it also bounds how
        # many callers can be queued here at all.
        self._load_slots = threading.Semaphore(self._settings.max_concurrent_loads)
        self._in_flight_loads = 0
        self._loads_lock = threading.Lock()

        self._reaper: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def max_concurrent_loads(self) -> int:
        """The configured cold-load cap; read by /info."""
        return self._settings.max_concurrent_loads

    @property
    def in_flight_loads(self) -> int:
        """Cold loads running right now; read by /info."""
        return self._in_flight_loads

    @contextmanager
    def _cold_load_slot(self, model_id: str) -> Iterator[None]:
        """Hold one of the cold-load slots. Reached only on a real load."""
        waiting = self._load_slots.acquire(blocking=False)
        if not waiting:
            logger.info(
                "%s is waiting for a cold-load slot (%d/%d in use)",
                model_id,
                self._in_flight_loads,
                self._settings.max_concurrent_loads,
            )
            self._load_slots.acquire()
        with self._loads_lock:
            self._in_flight_loads += 1
        try:
            yield
        finally:
            with self._loads_lock:
                self._in_flight_loads -= 1
            self._load_slots.release()

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
                        truncated_count=sum(
                            int(getattr(backend, "truncated_count", 0))
                            for backend in entry.backends
                        ),
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

            # Before anything constructs a backend, and therefore before
            # anything hands a file to torch.load. Deliberately OUTSIDE the
            # cold-load slot below: this is a metadata listing that contends
            # for neither VRAM nor PCIe, and holding a scarce slot through a
            # network round-trip would bound the wrong thing.
            self._check_weight_format(model_id, self._settings.revision)

            # Everything that touches device memory is inside the slot.
            with self._cold_load_slot(model_id):
                self._make_room_for(model_id)
                entry = self._load(model_id, pooling, dtype, max_seq_len, keep_alive)

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

    def _check_weight_format(self, model_id: str, revision: str | None) -> None:
        """Refuse a checkpoint whose only weights go through pickle.

        Safetensors alongside a legacy file is normal and fine: HF's own
        `from_pretrained` prefers safetensors, so that case passes silently.
        Only a checkpoint with no safetensors at all is refused.

        Fails CLOSED. If the listing itself cannot be obtained, the check
        has not run, and a check that silently does not run in the one
        situation it was written for is not a check. The listing already
        falls back to the local cache (see `_default_weight_files`), so
        reaching this branch means embedx genuinely does not know what it
        would be deserializing.
        """
        try:
            files = self._weight_file_lister(model_id, revision)
        except Exception as exc:
            logger.warning(
                "refusing to load %s: could not list its files (%s: %s), so the "
                "safetensors check cannot run",
                model_id,
                type(exc).__name__,
                exc,
            )
            raise WeightFormatUnverifiableError(model_id, exc) from exc
        if any(name.endswith(_SAFETENSORS_SUFFIX) for name in files):
            return
        legacy = sorted(name for name in files if name.endswith(_LEGACY_WEIGHT_SUFFIXES))
        if legacy:
            raise UnsupportedWeightFormatError(model_id, legacy)
        # No weight file of either kind: not this check's business. Let
        # from_pretrained say what it is actually missing.

    def _make_room_for(self, model_id: str) -> None:
        """Enforce `max_loaded_models` by evicting the least-recently-used.

        Refusing the request at the cap would make the cap a wall rather
        than a budget; evicting is what the operator asked for by setting a
        number. Only unreferenced models are candidates -- a model with a
        request in flight is never taken out from under it, and if every
        resident model is busy the load fails instead.

        Called with this model's load lock held and the entry not yet in
        the map, so the arithmetic is "one more than what is resident".
        """
        cap = self.max_loaded_models
        if cap is None:
            return
        while True:
            with self._entries_lock:
                if len(self._entries) < cap:
                    return
                idle = [
                    (entry.last_used, name)
                    for name, entry in self._entries.items()
                    if entry.ref_count == 0
                ]
                if not idle:
                    raise ModelCapacityError(model_id, sorted(self._entries), cap)
                _, victim = min(idle)
                entry = self._entries.pop(victim)
            logger.info(
                "evicting %s (least recently used) to stay within max_loaded_models=%d "
                "while loading %s",
                victim,
                cap,
                model_id,
            )
            self._evict(victim, entry)

    def _load(
        self,
        model_id: str,
        pooling: Pooling,
        dtype: Dtype,
        max_seq_len: int | None,
        keep_alive: float | None,
    ) -> _Entry:
        """Place the model on every device it fits on, fastest first.

        No memory estimation: an estimate that is wrong in the optimistic
        direction OOMs anyway, and one that is wrong pessimistically refuses
        a model that would have fitted. Trying is the measurement.
        """
        length_cache = new_length_cache()
        backends: list[EmbeddingBackend] = []
        placed: list[DeviceInfo] = []
        failures: list[tuple[int, str]] = []
        engine: Engine | None = None

        try:
            for device in self.devices:
                try:
                    backend = self._backend_factory(
                        model_id=model_id,
                        device_index=device.index,
                        pooling=pooling,
                        normalize=self._settings.normalize,
                        dtype=dtype,
                        max_seq_length=max_seq_len,
                        revision=self._settings.revision,
                        length_cache=length_cache,
                    )
                except Exception as exc:
                    if not _is_out_of_memory(exc, self._accelerator):
                        raise
                    failures.append((device.index, f"{type(exc).__name__}: {exc}"))
                    # Whatever was allocated before the OOM is still held; the
                    # next device starts clean only if it is released now.
                    #
                    # And it is `exc` that holds it. Its traceback pins the
                    # frame of the constructor that raised, and that frame
                    # pins the half-built backend with however many layers
                    # reached the card before it ran out. Python drops `exc`
                    # at the end of this block — one line too late, since
                    # `empty_cache` runs inside it and frees only what is
                    # already unreachable. The message above is a string and
                    # keeps nothing, so the traceback can go now.
                    exc.__traceback__ = None
                    # And the same collection `_release_backends` explains:
                    # a half-built transformer is as cyclic as a whole one,
                    # so unreachable is not yet freed. This is the path a
                    # client retry loop hammers, which makes it the one
                    # place where not reclaiming compounds.
                    gc.collect()
                    self._accelerator.empty_cache(device.index)
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

            engine = engine_from_backends(backends, placed, self._settings)
            if self._settings.smoke_test_on_load:
                self._smoke_test(model_id, engine, placed)
        except BaseException:
            # A load that fails PART WAY THROUGH is the dangerous one: the
            # devices that already succeeded hold a full model copy each,
            # and `backends` is a local of a frame that is about to unwind,
            # so nothing downstream ever sees them to free them. Retry, and
            # the card fills a model at a time.
            #
            # BaseException, not Exception: a KeyboardInterrupt or a
            # cancellation part way through a load strands VRAM exactly as
            # thoroughly as an error does.
            self._release_backends(model_id, engine, backends, placed)
            raise

        now = time.monotonic()
        # WARNING, not INFO: this is the moment a model's pooling is fixed
        # for as long as it stays resident, and a wrong one produces
        # plausible garbage with no error anywhere downstream. It belongs in
        # the log at a level nobody filters out.
        logger.warning(
            "pooling for %s resolved to %s on first load (dtype=%s, device(s) %s); "
            "later requests naming a different pooling will be refused",
            model_id,
            pooling.value,
            dtype.value,
            [device.index for device in placed],
        )
        return _Entry(
            engine=engine,
            backends=backends,
            devices=placed,
            pooling=pooling,
            dtype=dtype,
            max_seq_len=max_seq_len,
            keep_alive=(self.default_keep_alive_s if keep_alive is None else keep_alive),
            loaded_at=now,
            last_used=now,
            last_used_epoch=time.time(),
            length_cache=length_cache,
        )

    # The string is fixed and deliberately trivial: this proves the pipeline
    # runs at all, not that it is correct. Correctness against a reference
    # implementation is what the GPU pooling tests are for.
    _SMOKE_TEXT = "embedx smoke test"

    def _smoke_test(
        self,
        model_id: str,
        engine: Engine,
        placed: list[DeviceInfo],
    ) -> None:
        """Embed one string before the model can become resident.

        NOT OVERHEAD, AND NOT SAFE TO DELETE AS SUCH. This does not add a
        cost, it moves one: torch JIT-compiles Triton kernels on first
        inference, and task 17 measured that warmup at 2.29s for Qwen3-4B.
        Somebody pays it either way. The only question is whether it is paid
        by the load, where it is expected and attributable, or by whichever
        user happens to send the first request.

        What it buys is that a broken environment fails here — loudly, once,
        with the model refused — instead of producing a resident model that
        `/info` calls healthy and that raises on every request forever.

        Runs BEFORE `_entries[model_id]` is set, so a failure leaves nothing
        visible to `list_loaded()`. Releasing the backends on failure is
        `_load`'s job, not this method's: it wraps the whole placement in
        one handler, so a refused model gives its VRAM back by the same
        route whether the smoke test failed, a device OOMed, or the
        constructor raised something else entirely. One release path, not
        three that can drift.
        """
        # SPEC-GAP: one item exercises the Engine plus whichever worker wins
        # the claim, not every device. The scheduler is work-stealing, so
        # even N items would not guarantee one per backend; per-device
        # coverage would mean bypassing the Engine and calling each backend
        # directly, which would stop testing the wiring this is here to test.
        # A fault specific to a second device therefore still surfaces on a
        # real request.
        try:
            vectors = engine.embed([self._SMOKE_TEXT])
            if vectors.ndim != 2 or vectors.shape[0] != 1:
                raise ValueError(f"expected one 2-D row, got array of shape {vectors.shape}")
            if vectors.shape[1] <= 0:
                raise ValueError(f"backend returned zero-width vectors: shape {vectors.shape}")
        except Exception as exc:
            logger.exception(
                "smoke test failed for %s on device(s) %s; releasing it and refusing the load",
                model_id,
                [device.index for device in placed],
            )
            raise SmokeTestFailedError(
                model_id,
                [device.index for device in placed],
                f"{type(exc).__name__}: {exc}",
            ) from exc
        logger.info(
            "smoke test passed for %s on device(s) %s (dim=%d)",
            model_id,
            [device.index for device in placed],
            vectors.shape[1],
        )

    # ------------------------------------------------------------------ #
    # Eviction
    # ------------------------------------------------------------------ #

    def _release_backends(
        self,
        model_id: str,
        engine: Engine | None,
        backends: list[EmbeddingBackend],
        devices: list[DeviceInfo],
    ) -> None:
        """Drop a model's backends and hand its device memory back.

        Shared by eviction and by every load-failure path, so all of them
        follow the same release ORDER — which is the part that is easy to
        get wrong, and which this function got wrong in 0.3.0. Device memory
        returns only when the last Python reference to the model's tensors
        is gone, and `empty_cache` returns only blocks the allocator already
        considers free, so anything still reachable when it runs is simply
        not freed. There is no error and no warning; `/info` reports no
        models and `nvidia-smi` reports the full weight, indefinitely.

        Three references had to die before `empty_cache`. Only the caller's
        list released; the other two are the leak:

        1. The backend's own tensors. `HFBackend` had no `close`, so the
           `getattr` below found nothing and this loop was a no-op against
           the real backend. It now has one, and that alone makes the
           release robust: the weights go whether or not some frame is
           still holding the backend object.
        2. `Engine._length_fn`, a bound method of `backends[0]`, which
           `Engine.close()` did not reset. Device 0 only — which is exactly
           the shape the production report had.
        3. `backend`, the loop variable BELOW. A `for` loop leaves its last
           value bound in this frame, and this frame is alive four lines
           later when `empty_cache` runs. Hence `while backends: pop()`,
           which looks fussier than it is: the list must empty AND no name
           here may outlive it.

        And then a fourth thing, which none of the above would have caught:
        a loaded transformer is CYCLIC. Making it unreachable is not the
        same as freeing it — reference counting cannot collect a cycle, so
        the weights sit in the allocator as uncollected garbage and
        `empty_cache` walks past them. Measured on this repo's own
        benchmark model: dropping every reference freed 0 bytes, and the
        `gc.collect()` below then freed all 43.9 MiB of it. That is why the
        collection is not optional and not a belt-and-braces flourish; it
        is the step that does the work.

        The collection is a full stop-the-world pass and it is not free
        (tens of milliseconds on a large heap). It is affordable here
        because eviction is rare by construction — a model has been idle
        for `keep_alive` seconds, or the process is shutting down — and it
        is the difference between returning a model's VRAM and not.

        `engine` is optional because a load can fail before there is one.
        """
        while backends:
            backend = backends.pop()
            close = getattr(backend, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    # Logged, not raised: teardown continues for every other
                    # device, and a backend that failed to close is a worse
                    # reason than most to leave the rest of them loaded.
                    logger.exception("closing backend for %s failed", model_id)
            # See (3) above. `close` is a bound method and pins the backend
            # just as hard as `backend` does.
            del close, backend
        if engine is not None:
            engine.close()
        # Between "nothing references it" and "it is freed" — see (4) above.
        gc.collect()
        for device in devices:
            self._accelerator.empty_cache(device.index)

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
        self._release_backends(model_id, entry.engine, entry.backends, entry.devices)
        entry.length_cache = None
        logger.info(
            "evicted %s from device(s) %s after %.1fs resident",
            model_id,
            [device.index for device in entry.devices],
            resident_s,
        )
