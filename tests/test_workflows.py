"""Static guards on the release workflows (task: publish to PyPI and GHCR).

Text assertions rather than a YAML parse, matching `test_docker.py`: PyYAML is
not a declared dependency and these are string-level facts anyway.

What is worth guarding here is narrow but sharp. A wheel cannot be
un-published and a tag cannot be re-pointed, so the two properties that must
never quietly disappear are the tag/version assertion and the library-wheel
assertion — plus the fact that both workflows run the SAME copy of the latter.
"""

from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CI = REPO / ".github" / "workflows" / "ci.yml"
PUBLISH = REPO / ".github" / "workflows" / "publish.yml"
CHECK_SCRIPT = REPO / "scripts" / "check_library_wheel.py"
MAKEFILE = REPO / "Makefile"


@pytest.fixture(scope="module")
def publish() -> str:
    return PUBLISH.read_text()


def _load_check_script():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("check_library_wheel", CHECK_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# The shared wheel check
# --------------------------------------------------------------------------- #


def test_both_workflows_run_the_same_wheel_check() -> None:
    """One copy of the rule, not two.

    The publish path's copy is the one that must not drift, because PyPI
    uploads are permanent. Inlining it in either workflow is how the two
    diverge.
    """
    invocation = "scripts/check_library_wheel.py"
    assert CHECK_SCRIPT.is_file()
    assert invocation in CI.read_text(), "ci.yml no longer runs the shared wheel check"
    assert invocation in PUBLISH.read_text(), "publish.yml no longer runs the shared wheel check"


def _wheel(tmp_path: Path, names: list[str]) -> Path:
    dist = tmp_path / "dist"
    dist.mkdir()
    path = dist / "embedx_inference-0.0.0-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as zf:
        for name in names:
            zf.writestr(name, "")
    return dist


LIBRARY = [
    "embedx/__init__.py",
    "embedx/cli.py",
    "embedx/config.py",
    "embedx/registry.py",
    "embedx/engine/engine.py",
    "embedx/engine/scheduling.py",
    "embedx/gpu/vendor.py",
]


def test_check_accepts_a_library_wheel(tmp_path: Path) -> None:
    assert _load_check_script().check(_wheel(tmp_path, LIBRARY)) == []


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        (["embedx/api/app.py"], "HTTP layer"),
        (["embedx/service/embedx.service"], "systemd unit"),
    ],
)
def test_check_rejects_a_server_wheel(tmp_path: Path, extra: list[str], expected: str) -> None:
    """The assertion must actually fire. A guard that only ever passes is not one."""
    problems = _load_check_script().check(_wheel(tmp_path, LIBRARY + extra))
    assert any(expected in problem for problem in problems), problems


def test_check_rejects_an_over_broad_exclusion(tmp_path: Path) -> None:
    """An empty wheel satisfies "contains no server"; it must still fail."""
    problems = _load_check_script().check(_wheel(tmp_path, ["embedx/__init__.py"]))
    assert any("took the library with it" in problem for problem in problems), problems


# --------------------------------------------------------------------------- #
# publish.yml
# --------------------------------------------------------------------------- #


def test_publishes_on_version_tags_with_oidc(publish: str) -> None:
    assert 'tags: ["v*"]' in publish
    # Trusted Publishing: the token is minted per-run, never stored.
    assert "id-token: write" in publish
    assert "pypa/gh-action-pypi-publish" in publish


def test_environment_name_matches_the_registered_publisher(publish: str) -> None:
    """`pypi` is half of the OIDC claim; renaming it breaks publishing.

    The other half is this file's NAME, which is why the workflow lives at
    .github/workflows/publish.yml and cannot be renamed casually either.
    """
    assert "name: pypi" in publish
    assert PUBLISH.name == "publish.yml"


def test_tag_and_version_must_agree(publish: str) -> None:
    assert "GITHUB_REF_NAME" in publish
    assert 'f"v{version}"' in publish, "publish.yml no longer compares the tag to the version"


def test_publish_builds_the_library_wheel_not_a_server_build(publish: str) -> None:
    """No EMBEDX_BUILD_SERVER on the publish path, ever.

    Setting it would produce a wheel containing `embedx.api` and the unit
    file, and PyPI would take it permanently.
    """
    assert "uv build" in publish
    # Comments stripped: the workflow explains in prose why the flag is absent,
    # and that explanation must not be what satisfies this assertion.
    executable = "\n".join(
        line for line in publish.splitlines() if not line.lstrip().startswith("#")
    )
    assert "EMBEDX_BUILD_SERVER" not in executable, (
        "the publish path sets EMBEDX_BUILD_SERVER; that wheel would carry the "
        "HTTP layer and the unit file, and PyPI would keep it permanently"
    )


def test_testpypi_dry_run_exists_and_is_manual(publish: str) -> None:
    assert "workflow_dispatch" in publish
    assert "https://test.pypi.org/legacy/" in publish
    assert "name: testpypi" in publish


# --------------------------------------------------------------------------- #
# The image release path
# --------------------------------------------------------------------------- #


def test_image_is_released_from_the_makefile_not_a_workflow() -> None:
    """The image is pushed from a GPU host; CI cannot verify what it builds."""
    makefile = MAKEFILE.read_text()
    assert "image-push:" in makefile
    # The push must be gated on the image actually having a server in it.
    assert "image-verify" in makefile
    assert "serve --help" in makefile
    # And the flag stays in the Dockerfile, where no caller can forget it.
    recipes = makefile.split("\nimage:", 1)[1]
    assert "EMBEDX_BUILD_SERVER" not in recipes


def test_image_goes_to_both_registries() -> None:
    """Docker Hub and GHCR, from one build.

    Named separately from the gate test because the failure modes differ: this
    one catches a registry silently dropping out of the release, which nobody
    would notice until someone's pull 404s.
    """
    makefile = MAKEFILE.read_text()
    assert "IMAGE_REPO ?= gabintb/embedx" in makefile
    for registry in ("docker.io/$(IMAGE_REPO)", "ghcr.io/$(IMAGE_REPO)"):
        assert registry in makefile, f"{registry} is no longer a push destination"


def test_verification_gates_every_registry_not_just_the_first() -> None:
    """Two destinations must not mean a weaker gate.

    `image-verify` loops over REGISTRIES rather than checking one tag, so a
    typo or a stale tag fails before the push instead of after it.
    """
    makefile = MAKEFILE.read_text()
    verify = makefile.split("image-verify:", 1)[1].split("\nimage-push:", 1)[0]
    assert "for reg in $(REGISTRIES)" in verify
    assert "serve --help" in verify
    # image-push must depend on both building and verifying, in that order.
    assert "image-push: image image-verify" in makefile
