# Task 18 — Vendor seam: abstract the CUDA-specific surface

## Scope

embedx is CUDA-only and will stay CUDA-only in this task. The goal is to
isolate every CUDA-specific call behind one small interface, so that adding
ROCm (AMD) or XPU (Intel) later is *a new module implementing an existing
Protocol* rather than a rewrite spanning five files.

**Do not implement ROCm or XPU here.** Building a backend for hardware that
cannot be tested is how projects end up with a broken vendor path that looks
supported. Ship CUDA, ship the seam, ship a fake vendor that proves the seam
is real.

This mirrors a decision already made at task 01: `EmbeddingBackend` +
`FakeBackend` made all scheduling logic testable without a GPU. The same
pattern applied one layer down.

## What is actually CUDA-specific today

The agent should verify this list against the code rather than trusting it,
but it is the expected surface:

- `gpu/discovery.py` — `torch.cuda.is_available()`, `torch.cuda.device_count()`,
  `torch.cuda.get_device_properties(i)`, and `ARCH_FACTOR` keyed on CUDA
  compute-capability major. Compute capability has no meaning on ROCm or XPU;
  that table is the most vendor-bound thing in the codebase.
- `backend/hf.py` — the `cuda:{index}` device string, `torch.cuda.OutOfMemoryError`,
  and the dtype-resolution rule (`bf16` when capability major >= 8).
- `registry.py` — `torch.cuda.empty_cache()`, and the OOM type caught during
  device placement.
- `cli.py` / `backend/factory.py` — "no CUDA device found" error paths.
- `gpu/budgets.py` — expected to be vendor-neutral already (it scales on
  `total_memory_bytes`). Confirm rather than assume.

## Do

New module `src/embedx/gpu/vendor.py` defining an `Accelerator` Protocol.
The no-torch-at-import rule applies as everywhere else — the ast guard test
from task 06 covers this file too.

The Protocol needs, at minimum:

- `name: str` — "cuda", for logs and `/info`.
- `is_available() -> bool`
- `device_count() -> int`
- `device_info(index) -> DeviceInfo` — the existing vendor-neutral dataclass.
  If `capability` is too CUDA-shaped to survive, replace it in `DeviceInfo`
  with something a vendor can populate meaningfully and keep the CUDA
  implementation mapping compute capability into it. Say which choice was
  made and why.
- `device_string(index) -> str` — `"cuda:0"` today.
- `oom_error_types() -> tuple[type[BaseException], ...]` — what to catch
  during placement. A tuple, not a single type: vendors raise different
  things and some raise more than one.
- `empty_cache(index) -> None`
- `supports_bfloat16(info) -> bool` — replaces the inline `capability.major >= 8`
  rule, which is a CUDA fact, not a universal one.
- `arch_score(info) -> float` — the per-SM throughput proxy. The CUDA
  implementation keeps the existing `ARCH_FACTOR` table verbatim, including
  its nearest-lower-major fallback. Do not change the scoring behaviour in
  this task; a refactor that also changes numbers is unreviewable.

`CudaAccelerator` implements it, lazily importing torch inside methods.

A module-level resolver (`get_accelerator()`) returns the CUDA implementation
and is the single place a future vendor gets selected. It may read a config
field or an env var, but with only one implementation, keep it simple and
leave a comment on where vendor selection will go.

Then rewrite `discovery.py`, `hf.py`, `registry.py` and the factory/CLI error
paths to go through the `Accelerator` rather than calling `torch.cuda.*`
directly.

**Behaviour must not change.** This is a pure refactor. Every existing test,
including the 8 GPU tests, must pass unmodified. If a test needs changing to
accommodate the seam, that is a signal the seam is leaking — stop and explain
rather than editing the test.

## The test that proves the seam is real

A `FakeAccelerator` in the test suite: reports N devices with configurable
memory and scores, raises a custom OOM type, and never touches torch. Then:

- Device discovery, ranking and budget computation run end to end against
  `FakeAccelerator` with no torch installed — this is the one that proves the
  vendor surface is genuinely closed, because anything that leaked would fail
  on the missing import.
- Placement retries and OOM handling work against the fake's own OOM type,
  not `torch.cuda.OutOfMemoryError`. If the registry still catches the CUDA
  type by name, this test fails, which is the point.
- `supports_bfloat16` is consulted rather than a capability comparison being
  inlined anywhere.
- Extend the task-06 ast guard: no `torch.cuda` attribute access anywhere
  under `src/embedx/` outside `gpu/vendor.py`. That is the durable form of
  this task's whole result — without it, the next feature quietly reintroduces
  a direct call and the seam rots.

## Files

`src/embedx/gpu/vendor.py`, `src/embedx/gpu/discovery.py`,
`src/embedx/backend/hf.py`, `src/embedx/registry.py`,
`src/embedx/backend/factory.py`, `src/embedx/cli.py`,
`tests/test_vendor.py`, plus the existing guard test.

## Notes for later, not to build now

When ROCm or XPU is eventually added, the remaining work is: an `Accelerator`
implementation, a vendor-specific score table (with the honest admission that
it is uncalibrated until someone measures on that hardware), a GPU test run
on that vendor, and a separate Docker build target (task 19 makes the base
image and torch index build arguments precisely so this is not a Dockerfile
rewrite). Note in `vendor.py`'s docstring that Apple MPS is a fourth backend
with a different memory model (unified memory, no discrete VRAM budget), so
it does not fit this Protocol cleanly and should not be forced into it.

## Gate

`uv run ruff check .`, `uv run ruff format --check .`,
`uv run mypy src/embedx`, `uv run pytest -m "not gpu"`, and
`uv run pytest -m gpu` on the GPU host — the GPU run matters more than usual
here, since the whole claim is that behaviour is unchanged.

Commit: `refactor(gpu): isolate CUDA behind an Accelerator seam`
