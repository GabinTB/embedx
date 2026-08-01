# Task 19 — Docker deployment

## Scope

An additional deployment path, not a replacement. The systemd unit and its
docs stay valid and untouched.

The reason this image exists, and the thing to get right: a public repo
without an image means every user re-fights the Python / CUDA-wheel / Triton
battle. The image is the answer to "how do I run this".

Honest scope, state it in the README: **Linux + NVIDIA is first class.
Windows + NVIDIA via WSL2 works with friction. macOS is not supported at all**
(Apple dropped NVIDIA drivers after 10.13; Docker Desktop on Mac has no GPU
passthrough). Do not write "cross-platform" without that qualification.

## Verify sm_120 first, before anything else

Task 18 landed the `Accelerator` seam, but a code abstraction does not change
which kernels a wheel was compiled with. Before building the compose file or
anything downstream, confirm inside the built image that
`torch.cuda.get_arch_list()` contains `sm_120`. A cu128 + cp314 wheel missing
it produces exactly the "no kernel image is available for execution on the
device" failure this project has already hit once on the host. Check it early
in the build, not after everything else works.

## Version decisions, already researched — verify, do not re-derive

- **sm_120 (Blackwell) requires CUDA 12.8 or newer.** 12.8 is therefore the
  floor, and the base image should be the *oldest* CUDA that supports it, not
  the newest available. A container's CUDA runtime needs a host driver at
  least as new: CUDA 12.8 runs on driver 525+ through minor-version
  compatibility, while CUDA 13.x needs 580+. Matching this host's 13.3 would
  exclude most potential users for no benefit.
- **Python 3.14 CUDA wheels (cp314) exist from PyTorch 2.10 onward.** This
  host already runs torch 2.13.0+cu130 on Python 3.14, so the combination is
  known good. Verify the cp314 wheel exists on whichever CUDA index is
  chosen before building everything around it — PyTorch drops older CUDA
  variants over time and cu128 + cp314 may or may not both be present for the
  torch version selected. If they are not, report the conflict rather than
  silently bumping CUDA.

## The Triton trap — this is the main reason the image exists

torch routes RoPE-family models (Qwen3) through a Triton kernel that
**JIT-compiles a C helper at first inference**, not at load. It needs
`Python.h` *and a C compiler* present at runtime. Consequences:

> **Corrected after implementation.** Three things are required, not two: a
> C compiler, **libc headers**, and Python headers. Point 1 below is wrong
> about which to install. In the image, `Python.h` is already supplied by
> uv's standalone CPython, so **no Python dev package is needed** (and none
> exists for 3.14 on Ubuntu 24.04 — that base has only `python3.12-dev`).
> What the runtime stage actually needs is `gcc` **and `libc6-dev`**: gcc
> alone fails with `fatal error: stdlib.h: No such file or directory`.
> The bare-metal host failed the other way round — it had gcc and libc, and
> was missing `Python.h` because its venv uses the distro Python. Which of
> the three bites depends on the environment; `DEPLOY.md` has the full list.

1. A slim `nvidia/cuda:*-runtime-*` base has no gcc and no Python headers, so
   it fails exactly the way this project's host failed: model loads clean,
   registers as resident, every request raises. Install the Python dev
   headers and a C toolchain in the **runtime** stage, not just the builder,
   and comment why or someone will strip them as bloat.
2. Triton caches compiled kernels. Set `TRITON_CACHE_DIR` to a persisted,
   writable path, or every container start re-JITs — and for a non-root user
   with an unwritable cache dir it fails outright. This interacts with the
   non-root user below; get the ownership right.

**Verify the fix in the build or document how to verify it.** The image is
worthless if the headers are present but Triton still cannot compile. Either
a build-stage check that a trivial Triton JIT succeeds, or a documented
post-build command running real inference against a RoPE model. Do not claim
it is fixed without something that would fail if it were not.

## Dockerfile

Multi-stage. `uv` for the install, matching project tooling. Install with the
`gpu` extra. Non-root runtime user.

**Base image and torch index must be build arguments** (`ARG CUDA_IMAGE`,
`ARG TORCH_INDEX_URL`) with CUDA defaults, not hardcoded. Task 18 abstracted
the code side behind the `Accelerator` Protocol, so these two build args are
now genuinely the whole story for adding a ROCm or Intel XPU build target
later — their runtimes cannot share an image, and adding one should be a new
build target rather than a Dockerfile rewrite. Do not add non-CUDA targets
now.

**Do not bake model weights into the image.** Task 17 measured disk read as
the dominant cold-load stage (~61% for Qwen3-4B, 6.38 s of a ~10.4 s load).
The HF cache is a mounted volume; losing it on every container recreate
re-pays the single largest cost. Set `HF_HOME` to a path meant to be mounted
and document it.

Expect 8–12 GB before anything project-specific. That is normal for a
torch + CUDA image and is why CI must not build it.

## compose file

`docker-compose.yml` with:

- GPU passthrough. Use whichever form the installed Compose version actually
  supports (`deploy.resources.reservations.devices`, or `runtime: nvidia`)
  and say which in a comment.
- The HF cache volume as a **bind mount to host storage**, not a named
  volume in the overlay filesystem — this is the disk-performance point
  above, and it is the one place containerization can genuinely cost
  measurable time.
- The Triton cache path, persisted.
- An env file.
- Port bound to `127.0.0.1` by default, never `0.0.0.0`.
- Restart policy, and a healthcheck hitting `/health` (it touches no model
  state, which is exactly what makes it a valid healthcheck).
- A comment noting Ollama runs on this host and both allocate VRAM on the
  same devices, so a resident Ollama model shrinks what embedx can batch.

## .dockerignore

Exclude `dev/`, `tests/`, `.git`, notebooks, and anything else with no
business in the image.

## DEPLOY.md

A Docker section alongside the existing systemd instructions, not replacing
them. Cover: build, run, the cache volumes and why they matter (reference the
measured stage-1 cost), NVIDIA Container Toolkit as a prerequisite, the env
file, and how `CUDA_VISIBLE_DEVICES`, container device selection and
`EMBEDX_DEVICES` interact — that triple is how people end up serving the
wrong card.

State plainly which path suits what: systemd for a host that already has the
right Python and dev headers; Docker when you want the Python / CUDA-wheel /
Triton dependency solved in the image.

Record the minimum host driver version implied by the chosen CUDA base, in
both `DEPLOY.md` and the README.

## Performance note for the docs

GPU compute overhead under the NVIDIA Container Toolkit is roughly 1–2% and
not worth mentioning as a caveat. Disk I/O is the part that can actually
regress, and only if the cache is misconfigured. Say that once, plainly, so
nobody assumes containerization is costing them throughput.

## Tests

- The Dockerfile and compose file parse (hadolint if available, otherwise
  `docker compose config` validation).
- **Do not build the image in CI.** CUDA base layers make it enormous. Note
  that decision in the CI config or this task's docs so it reads as
  deliberate.

## Manual verification after this lands (on the GPU host)

1. A RoPE model (Qwen3-Embedding-4B) serves a real request — proves the
   Triton fix, which unit tests cannot.
2. The cache volume survives `docker compose down && up`, and the second cold
   load skips the disk stage — proves the volume is doing its job.
3. `torch.cuda.get_arch_list()` inside the running container contains
   `sm_120`.

## Gate

`uv run ruff check .`, `uv run ruff format --check .`,
`uv run mypy src/embedx`, `uv run pytest -m "not gpu"`.

Commit: `feat(deploy): Dockerfile and compose for GPU deployment`
