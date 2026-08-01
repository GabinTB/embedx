"""Multi-worker embedding engine over the converging scheduler.

Each request builds one `Scheduler` and starts one thread per backend.
Workers claim budget-sized batches straight from the scheduler — `claim`
already sized them with `pack_count` against the device's own budget, so
there is no re-batching here. Every worker accumulates `(indices, vectors)`
pairs locally; the output is allocated once after all threads join and each
block is scattered to its original indices, so input order is preserved by
construction rather than by discipline.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence

import numpy as np

from embedx.backend.base import EmbeddingBackend
from embedx.config import Settings
from embedx.engine.scheduling import Scheduler, assign_sides
from embedx.gpu.budgets import device_budgets
from embedx.gpu.discovery import DeviceInfo


class Engine:
    """N backend workers draining one converging scheduler per request.

    `backends[i]` serves `devices[i]`, and `devices` is already ranked
    fastest-first by `rank_devices`, so `assign_sides` hands the faster
    half the LONG (expensive) end of the queue.
    """

    def __init__(
        self,
        backends: Sequence[EmbeddingBackend],
        devices: Sequence[DeviceInfo],
        settings: Settings,
        length_fn: Callable[[str], int] = len,
    ) -> None:
        if not backends:
            raise ValueError("Engine requires at least one backend")
        if len(backends) != len(devices):
            raise ValueError(
                f"backends ({len(backends)}) and devices ({len(devices)}) must have equal length"
            )
        self._backends = list(backends)
        self._devices = list(devices)
        budgets = device_budgets(
            self._devices, settings.max_batch_tokens, settings.device_batch_tokens
        )
        self._budgets = [budgets[device.index] for device in self._devices]
        self._sides = assign_sides(len(self._backends))
        self._max_items = settings.max_batch_items
        # Budgets are named in tokens, so the scheduler must measure inputs
        # in the same unit. The factory injects the tokenizer-based length;
        # the default `len` (characters) is for CPU tests and fakes.
        self._length_fn = length_fn
        # One lock per backend, held across backend.embed: FastAPI calls
        # `embed` concurrently, and one CUDA device cannot run two batches
        # at once without doubling its activation memory. Serialising per
        # backend makes concurrent requests interleave batch by batch per
        # device instead of oversubscribing VRAM.
        self._backend_locks = [threading.Lock() for _ in self._backends]
        self._closed = False

    @property
    def devices_with_budgets(self) -> list[tuple[DeviceInfo, int]]:
        """(device, token budget) per worker in rank order; used by /info."""
        return list(zip(self._devices, self._budgets, strict=True))

    @property
    def truncated_counts(self) -> list[int]:
        """Per-worker truncation counters; 0 for backends without one."""
        return [int(getattr(backend, "truncated_count", 0)) for backend in self._backends]

    def close(self) -> None:
        """Drop the backends, making this engine permanently unusable.

        The registry needs this for eviction: CUDA memory comes back only
        when the last Python reference to a model's tensors is gone, and the
        engine holds one per backend. Clearing the entry's own list is not
        enough — `torch.cuda.empty_cache()` would run while the engine still
        pinned every weight. Idempotent.
        """
        self._backends = []
        self._backend_locks = []
        self._budgets = []
        self._closed = True

    def token_count(self, texts: list[str]) -> int:
        """Total input length in the engine's own unit.

        Real tokens when the factory injected the tokenizer-based
        `length_fn`; characters with the default `len` (fakes, CPU tests).
        """
        return sum(self._length_fn(text) for text in texts)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed `texts`; row i of the result corresponds to `texts[i]`."""
        if self._closed:
            # Reachable only by holding an Engine across its eviction, which
            # is why the registry hands out references through `acquire`.
            raise RuntimeError("engine is closed: its model was evicted and its backends released")
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        scheduler = Scheduler(enumerate(texts), length_fn=self._length_fn)
        results: list[list[tuple[list[int], np.ndarray]]] = [[] for _ in self._backends]
        failures: list[Exception | None] = [None] * len(self._backends)

        def worker(w: int) -> None:
            backend, side = self._backends[w], self._sides[w]
            budget, lock = self._budgets[w], self._backend_locks[w]
            try:
                while (batch := scheduler.claim(side, budget, self._max_items)) is not None:
                    with lock:
                        vectors = backend.embed([text for _, text in batch])
                    results[w].append(([index for index, _ in batch], vectors))
            except Exception as exc:
                failures[w] = exc

        threads = [threading.Thread(target=worker, args=(w,)) for w in range(len(self._backends))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # A failed worker must never yield a partially filled (zeroed) array:
        # raise before any output is allocated.
        for w, failure in enumerate(failures):
            if failure is not None:
                raise RuntimeError(
                    f"embedding failed on device {self._devices[w].index}: {failure}"
                ) from failure

        # The backend does not declare its dimension up front; take it from
        # the first returned block and hold every other block to it.
        dim: int | None = None
        for w, worker_results in enumerate(results):
            for indices, vectors in worker_results:
                if vectors.ndim != 2 or vectors.shape[0] != len(indices):
                    raise ValueError(
                        f"backend for device {self._devices[w].index} returned shape "
                        f"{vectors.shape} for a batch of {len(indices)} texts"
                    )
                if dim is None:
                    dim = int(vectors.shape[1])
                elif int(vectors.shape[1]) != dim:
                    raise ValueError(
                        f"backend for device {self._devices[w].index} returned dimension "
                        f"{vectors.shape[1]}, expected {dim}"
                    )
        assert dim is not None  # texts is non-empty, so at least one batch ran

        out = np.empty((len(texts), dim), dtype=np.float32)
        for worker_results in results:
            for indices, vectors in worker_results:
                out[indices] = vectors
        return out
