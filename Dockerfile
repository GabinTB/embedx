# syntax=docker/dockerfile:1
#
# embedx GPU image. Linux + NVIDIA only; see README for the platform matrix.
#
# Why this image exists: the Python / CUDA-wheel / Triton combination is the
# hard part of running embedx, not the code. Everything awkward about it is
# solved here so users do not re-fight it.
#
# Both vendor-specific decisions are build arguments. Task 18 put the code
# side behind the `Accelerator` Protocol, so CUDA_IMAGE + TORCH_INDEX_URL are
# genuinely the whole story for adding a ROCm or XPU target later: their
# runtimes cannot share an image, so that would be a new build target, not a
# rewrite of this file. No non-CUDA target is provided today.

# CUDA 12.8 is the FLOOR for sm_120 (Blackwell), and deliberately the oldest
# CUDA that clears it rather than the newest available. A container's CUDA
# runtime needs a host driver at least as new: 12.8 runs on driver 525+
# through minor-version compatibility, while CUDA 13.x needs 580+. Matching a
# 13.x host would exclude most users for no benefit.
#
# The `base` variant, not `runtime` or `cudnn-runtime`: torch's pip wheels
# bundle their own cuBLAS/cuDNN/cuFFT, so the fatter bases ship a second copy
# of libraries torch will not load. Measured: 12.3 GB on `base` against
# 19.2 GB on `cudnn-runtime`, with Triton JIT and real Qwen3 inference
# verified identical on both.
ARG CUDA_IMAGE=nvidia/cuda:12.8.1-base-ubuntu24.04
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
# 2.11.0 is the newest cu128 build: 2.12 and 2.13 ship cu130 only. cp314
# wheels exist on cu128 from 2.9.0 onward. Verified against the index, and
# `docs/` records the arch list this wheel actually carries.
ARG TORCH_VERSION=2.11.0
ARG PYTHON_VERSION=3.14
ARG UV_VERSION=0.12.1


# --------------------------------------------------------------------------- #
# Builder: resolve and install into a self-contained /opt/venv
# --------------------------------------------------------------------------- #
FROM ${CUDA_IMAGE} AS builder

ARG TORCH_INDEX_URL
ARG TORCH_VERSION
ARG PYTHON_VERSION

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /usr/local/bin/uv

# uv's standalone CPython, not the distro's: it ships its own headers (see the
# runtime stage), and it makes the Python version a build argument rather than
# a property of the base image.
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy
RUN uv python install "${PYTHON_VERSION}" \
    && uv venv --python "${PYTHON_VERSION}" /opt/venv

# torch first, pinned, from the CUDA index. Then the project from PyPI, which
# finds torch already satisfied. Doing it in one step would need
# --index-strategy unsafe-best-match across both indexes, which is how
# dependency-confusion substitutions get in.
RUN uv pip install --python /opt/venv/bin/python \
        --index-url "${TORCH_INDEX_URL}" \
        "torch==${TORCH_VERSION}"

WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --python /opt/venv/bin/python ".[gpu]"

# Fail the BUILD, not the first request, if the wheel lacks Blackwell kernels.
# A code abstraction cannot change which kernels a wheel was compiled with, and
# a missing sm_120 surfaces as "no kernel image is available for execution on
# the device" at first inference -- long after deploy.
#
# Reads the private _cuda_getArchFlags(), not the public get_arch_list(),
# on purpose: the public one is gated on cuda.is_available() and so returns
# [] wherever there is no driver -- which is every `docker build`. It would
# have passed this check vacuously. The private symbol reads the flags
# compiled into the wheel and needs no device. get_arch_list() is literally
# this string split on whitespace.
RUN /opt/venv/bin/python -c "\
import torch, sys; \
archs = torch._C._cuda_getArchFlags().split(); \
print('torch', torch.__version__, 'cuda', torch.version.cuda, archs); \
sys.exit(0 if 'sm_120' in archs else 'FATAL: wheel has no sm_120 (Blackwell) kernels')"


# --------------------------------------------------------------------------- #
# Runtime
# --------------------------------------------------------------------------- #
FROM ${CUDA_IMAGE} AS runtime

# DO NOT STRIP gcc AND libc6-dev AS BUILD BLOAT. They are runtime dependencies.
#
# torch routes RoPE-family models (Qwen3) through a Triton kernel that
# JIT-compiles a C helper at FIRST INFERENCE, not at load. Without a compiler
# the model loads clean, registers as resident, and every request raises --
# which is exactly how this project's bare-metal host failed once already.
#
# Triton's compile needs three things: a C compiler, libc headers, and
# Python headers. Measured in this image, not assumed:
#   - Python.h is ALREADY PRESENT here, from uv's standalone CPython at
#     /opt/python/.../include/python3.14. No python3-dev is needed, and for
#     3.14 the distro package does not exist on Ubuntu 24.04 anyway (that
#     base carries only python3.12-dev). This is a property of the uv Python,
#     NOT a general one: a venv built on a distro interpreter does need the
#     matching pythonX.Y-dev, which is how the bare-metal host failed.
#   - gcc alone is NOT enough: with --no-install-recommends it pulls no libc
#     headers, and the compile dies on `fatal error: stdlib.h: No such file`.
#     libc6-dev is what fixes it.
#   - Triton's driver.c also needs cuda.h and libcuda.so.1. Both arrive
#     without help: Triton bundles the CUDA headers, and the NVIDIA Container
#     Toolkit injects the driver library at run time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        gcc \
        libc6-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/python /opt/python
COPY --from=builder /opt/venv /opt/venv

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Cache roots, meant to be mounted. Task 17 measured disk read as the dominant
# cold-load stage -- 6.38s of a 9.36s Qwen3-4B load, 68% -- so an unmounted
# HF_HOME re-pays the single largest cost on every container recreate.
# TRITON_CACHE_DIR must be writable or Triton cannot even start; see below.
ENV HF_HOME=/cache/hf \
    TRITON_CACHE_DIR=/cache/triton

# HOME must be writable by WHATEVER uid this runs as, not just by `embedx`:
# compose overrides `user:` so the container uid can match the host owner of
# the bind-mounted caches. /tmp is the one path that is always writable.
ENV HOME=/tmp

# Listens on all interfaces INSIDE the container -- the network namespace is
# the boundary, and compose publishes only to 127.0.0.1 on the host. Binding
# 127.0.0.1 here would make the published port unreachable.
ENV EMBEDX_HOST=0.0.0.0 \
    EMBEDX_PORT=8477

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin embedx \
    && mkdir -p /cache/hf /cache/triton \
    && chown -R 10001:10001 /cache
USER 10001:10001

EXPOSE 8477

# No model weights are baked in. `serve` starts with nothing loaded and pulls
# on first request; the HF cache volume is what makes that cheap.
ENTRYPOINT ["embedx"]
CMD ["serve"]
