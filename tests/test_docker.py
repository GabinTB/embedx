"""Tests for the Docker deployment files (task 19).

These are static checks, deliberately. Building the image pulls ~12 GB of
torch and CUDA layers, so CI must never do it — see `.github/workflows/ci.yml`.
What is asserted here is the set of properties that would silently rot: the
runtime C toolchain, the cache paths, the loopback publish, and the two
build arguments the vendor seam depends on.

The claims these guard were verified against a real build and a real GPU;
DEPLOY.md documents how to re-verify them on a host that has Docker.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DOCKERFILE = REPO / "Dockerfile"
COMPOSE = REPO / "docker-compose.yml"
DOCKERIGNORE = REPO / ".dockerignore"
ENV_EXAMPLE = REPO / ".env.example"


@pytest.fixture(scope="module")
def dockerfile() -> str:
    return DOCKERFILE.read_text()


@pytest.fixture(scope="module")
def compose() -> str:
    return COMPOSE.read_text()


def test_deployment_files_exist() -> None:
    for path in (DOCKERFILE, COMPOSE, DOCKERIGNORE, ENV_EXAMPLE):
        assert path.is_file(), f"{path.name} is missing"


# --------------------------------------------------------------------------- #
# Dockerfile
# --------------------------------------------------------------------------- #


def test_base_image_and_torch_index_are_build_args(dockerfile: str) -> None:
    """The whole story for adding a ROCm/XPU target later (task 18)."""
    assert "ARG CUDA_IMAGE=" in dockerfile
    assert "ARG TORCH_INDEX_URL=" in dockerfile
    assert "FROM ${CUDA_IMAGE}" in dockerfile


def test_cuda_base_is_12_8(dockerfile: str) -> None:
    """12.8 is the sm_120 floor AND the oldest CUDA that clears it.

    Bumping to 13.x raises the minimum host driver from 525+ to 580+, which
    is a user-facing breaking change disguised as a version bump.
    """
    assert "nvidia/cuda:12.8" in dockerfile
    assert "download.pytorch.org/whl/cu128" in dockerfile


def test_runtime_stage_has_a_c_compiler(dockerfile: str) -> None:
    """The reason this image exists.

    torch JIT-compiles a Triton C helper at FIRST INFERENCE for RoPE models.
    Without gcc the model loads clean and every request then raises. gcc
    alone is not enough either: with --no-install-recommends it pulls no
    libc headers and the compile dies on a missing stdlib.h.
    """
    runtime = dockerfile.split("AS runtime", 1)[1]
    assert "gcc" in runtime, "no C compiler in the runtime stage"
    assert "libc6-dev" in runtime, "gcc without libc6-dev cannot find stdlib.h"


def test_build_fails_if_the_wheel_lacks_blackwell_kernels(dockerfile: str) -> None:
    """Checked at build time, not discovered at first inference.

    Must use _cuda_getArchFlags, not get_arch_list: the public helper is
    gated on cuda.is_available() and returns [] wherever there is no driver,
    which is every `docker build`. It would pass vacuously.
    """
    assert "_cuda_getArchFlags" in dockerfile
    assert "sm_120" in dockerfile
    assert "torch.cuda.get_arch_list()" not in dockerfile.split("AS runtime")[0]


def test_image_builds_with_the_http_layer_included(dockerfile: str) -> None:
    """The image is the server; a library-only build would be silently useless.

    `[tool.hatch.build.targets.wheel] exclude` drops `embedx.api` so the PyPI
    wheel is the library alone (task 21), and `pip install .` builds that same
    target. Both halves are needed here and neither implies the other: the
    `server` extra supplies the HTTP dependencies, `EMBEDX_BUILD_SERVER=1`
    supplies the modules. Without the flag the image would build, install and
    start with no `embedx serve` at all.
    """
    assert "EMBEDX_BUILD_SERVER=1" in dockerfile
    assert '".[gpu,server]"' in dockerfile
    # hatch_build.py implements the re-inclusion, so it must reach the context.
    assert "hatch_build.py" in dockerfile
    # And the build must prove it landed rather than trusting the flag.
    assert "import embedx.api" in dockerfile


def test_no_model_weights_are_baked_in(dockerfile: str) -> None:
    """Weights live in the mounted cache; task 17 measured why."""
    for pattern in ("snapshot_download", "huggingface-cli download", "hf download"):
        assert pattern not in dockerfile
    assert "HF_HOME=/cache/hf" in dockerfile


def test_runs_as_a_non_root_user(dockerfile: str) -> None:
    assert "USER 10001:10001" in dockerfile
    body = dockerfile.split("AS runtime", 1)[1]
    assert body.rstrip().index("USER 10001") < body.rstrip().index("ENTRYPOINT")


def test_cache_paths_are_owned_by_the_runtime_user(dockerfile: str) -> None:
    """Docker creates a missing bind source as root; an unwritable Triton
    cache fails at first inference, not at start."""
    assert "chown -R 10001:10001 /cache" in dockerfile
    assert "TRITON_CACHE_DIR=/cache/triton" in dockerfile


def test_binds_all_interfaces_inside_the_container(dockerfile: str) -> None:
    """0.0.0.0 in the container is correct: the network namespace is the
    boundary and compose publishes to 127.0.0.1. Binding loopback here would
    make the published port unreachable."""
    assert "EMBEDX_HOST=0.0.0.0" in dockerfile


# --------------------------------------------------------------------------- #
# compose
# --------------------------------------------------------------------------- #


def test_compose_publishes_to_loopback_only(compose: str) -> None:
    assert "127.0.0.1:${EMBEDX_PUBLISH_PORT:-8477}:8477" in compose
    assert '"0.0.0.0:' not in compose


def test_compose_requests_gpus(compose: str) -> None:
    assert "driver: nvidia" in compose
    assert "capabilities: [gpu]" in compose


def test_compose_caches_are_bind_mounts_not_named_volumes(compose: str) -> None:
    """Named volumes live in the overlay filesystem. Task 17 measured the
    disk read as 68% of a cold load, so this is the one place
    containerization can cost measurable time."""
    assert "/cache/hf" in compose
    assert "/cache/triton" in compose
    # A top-level `volumes:` key would mean named volumes were declared.
    assert not any(line.rstrip() == "volumes:" for line in compose.splitlines())


def test_compose_healthchecks_health(compose: str) -> None:
    """/health touches no model state, which is what makes it valid."""
    assert "healthcheck:" in compose
    assert "/health" in compose
    assert "restart: unless-stopped" in compose


def test_compose_uses_an_env_file(compose: str) -> None:
    assert "env_file:" in compose
    assert ".env" in compose


def test_compose_runs_as_a_configurable_uid(compose: str) -> None:
    """Must match the host owner of the bind-mounted caches."""
    assert "${EMBEDX_UID:" in compose
    assert "${EMBEDX_GID:" in compose
    assert "EMBEDX_UID" in ENV_EXAMPLE.read_text()


# --------------------------------------------------------------------------- #
# .dockerignore
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("excluded", ["dev/", "tests/", ".git", ".venv/", "*.ipynb"])
def test_dockerignore_excludes_the_heavy_things(excluded: str) -> None:
    assert excluded in DOCKERIGNORE.read_text().splitlines()


def test_env_is_ignored_but_the_example_is_not() -> None:
    lines = DOCKERIGNORE.read_text().splitlines()
    assert ".env" in lines
    assert "!.env.example" in lines


# --------------------------------------------------------------------------- #
# Real parse, when the tooling happens to be present
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker not installed")
def test_compose_file_actually_parses() -> None:
    result = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE), "config", "--quiet"],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(shutil.which("hadolint") is None, reason="hadolint not installed")
def test_dockerfile_lints() -> None:
    result = subprocess.run(
        ["hadolint", "--failure-threshold", "error", str(DOCKERFILE)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout
