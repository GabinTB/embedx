# Task 13 — On-demand multi-model registry and lifecycle

## Scope

Replace the single implicit model with an on-demand registry. Any embedding
request may name a model that is not yet resident; the registry loads it,
places it on whichever devices it fits, and evicts it after an idle timeout.
Concurrent requests for different models never block each other. This does
NOT change how a model already spans multiple devices — that's the existing
`Engine` + converging `Scheduler` from 0.1.0, reused per model, not replaced.

Design decisions already made, do not relitigate them here:

- Fully dynamic loading: any HF hub id or local path named in a request may
  be loaded. No allowlist. (Single-user server; revisit if that changes.)
- Device placement is decided by attempting the load, not by estimating
  memory footprint up front. No benchmarking, per the project's existing
  no-benchmark philosophy (task 06) — this is a capacity/OOM check, not a
  throughput benchmark.
- Pooling is never silently inferred, per the project's core rule. See the
  resolution rule below; it's the load-bearing part of this task.
- Safetensors only for dynamically loaded weights. Pickle/`.bin` formats are
  a deserialization risk on a server that will load whatever a request names;
  refuse before attempting to deserialize.

## Do

New module `src/embedx/registry.py`. Central type: `ModelRegistry`.

**Loading.** Given a model id (hub id or local path) plus optional
`pooling` / `dtype` / `max_seq_len` overrides:

- Resolve weights format before touching torch. If the checkpoint's weight
  files are not safetensors (`.bin`, `.pt`, or anything pickle-based),
  refuse and name the offending file in the error. Do this check before any
  deserialization is attempted, not as a try/except around a failed load.
- Device placement: attempt to construct an `HFBackend` on the
  highest-ranked free device first (reuse `rank_devices` from task 06),
  catch CUDA OOM, move to the next device, repeat. A model that fits on N
  devices gets N backends and one `Engine` spanning them — this is the
  existing multi-device behavior, unchanged.
- A model that fits nowhere raises, naming the model id and every device
  tried with the reason it was rejected on each.

**Pooling resolution — the rule that keeps this safe.** The first successful
load of a given model id requires an explicit `pooling` in the load call;
refuse the load otherwise, naming the model id, and say why (pooling is
never inferred). Once resolved, store it on the registry entry for that
model id. A later load call for the same model id with no `pooling` reuses
the stored value. A later call with a *different* `pooling` than what's
stored is rejected as a conflict — never silently reapplied, never silently
ignored. Log the first resolution and any conflict at WARNING.

**TTL idle eviction.** Global default (`Settings.default_keep_alive_s`,
wired in task 15), overridable per call (`keep_alive` parameter, threaded
through from the API in task 14). A background reaper (thread or asyncio
task, your call, document which and why) periodically evicts models past
their idle TTL: tear down the backend(s), free CUDA memory
(`torch.cuda.empty_cache()` on the relevant devices), remove from the
registry. `keep_alive <= 0` evicts immediately after the triggering request
completes. `keep_alive` absent uses the default.

**Concurrency.** A per-model-id lock guards only the *load* path: two
concurrent first-requests for the same not-yet-loaded model id must result
in exactly one load, with the second caller waiting on the first and then
reusing the result — never a duplicate load. Requests for *different* model
ids must never contend on this lock; a slow cold load for model A must not
delay a request to already-loaded model B. Requests to an already-loaded
model go straight to its `Engine` with no additional locking beyond what
`Engine` already does internally (task 07a's per-backend locks).

**Eviction race.** A model must never be evicted while a request is
actively using it. Reference-count in-flight requests per model id, or use
a read/write lock (read during `embed()`, write during eviction) — pick one
approach, document the choice in a module docstring, and prove it under
concurrency in tests, not just by inspection.

**Introspection.** `list_loaded() -> list[ModelStatus]`: model id, resolved
pooling, resolved dtype, the devices it's on, per-device idle time, last-used
timestamp. This is what `/info` (task 14) reads; do not duplicate this
bookkeeping in the API layer.

## Files

`src/embedx/registry.py`, `tests/test_registry.py`

## Tests to add

- Load-on-demand: requesting an unloaded model loads it; a second request
  for the same id reuses the loaded instance without reloading (assert with
  a load-counting stub).
- Concurrent first-requests for the same unloaded model id: exactly one load
  happens; both callers get correct results.
- Concurrent requests for two different model ids proceed in parallel, not
  serially — assert with a stub load that sleeps, timing both.
- Device placement: a stub backend factory that OOMs on the first device is
  skipped, the model lands on the second; both devices OOM raises, naming
  both and the reasons.
- Non-safetensors weights are refused before any load attempt is made,
  naming the specific file.
- Pooling: first load with no `pooling` is refused, naming the model id.
  First load with `pooling=mean` succeeds and is stored. A later call for
  the same model with `pooling=mean` succeeds (reuse). A later call with
  `pooling=cls` is refused as a conflict; the already-loaded instance is
  untouched.
- TTL eviction: a model idle past its TTL is evicted (assert real teardown
  via a stub, not just a registry-entry removal). A model with an in-flight
  request is never evicted even past its TTL. `keep_alive=0` evicts right
  after the triggering request completes.
- `list_loaded()` accurately reflects state after loads and evictions.

## Gate

`uv run ruff check .`, `uv run ruff format --check .`,
`uv run mypy src/embedx`, `uv run pytest -m "not gpu"`.

Commit: `feat(registry): on-demand multi-model loading with TTL eviction`
