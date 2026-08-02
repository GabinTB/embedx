"""Tests for the packaged systemd unit and serve preflight (task 10)."""

from __future__ import annotations

import configparser
import logging
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

import embedx.cli as cli
from embedx.backend import FakeBackend
from embedx.gpu.discovery import DeviceInfo

runner = CliRunner()
GIB = 2**30

REPO = Path(__file__).resolve().parent.parent
UNIT = REPO / "src" / "embedx" / "service" / "embedx.service"


class FakeRegistry:
    """Registry double for serve: starts, stops, loads nothing."""

    in_flight_loads = 0
    max_concurrent_loads = 2

    def start(self) -> None: ...

    def stop(self, timeout: float = 5.0) -> None: ...

    def evict_all(self) -> list[str]:
        return []

    def list_loaded(self) -> list[Any]:
        return []


class FakeEngine:
    def __init__(self) -> None:
        self.inner = FakeBackend(dim=8)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.inner.embed(texts)

    @property
    def devices_with_budgets(self) -> list[tuple[DeviceInfo, int]]:
        device = DeviceInfo(
            index=0,
            name="Fake GPU 0",
            total_memory_bytes=16 * GIB,
            multi_processor_count=100,
            capability=(8, 0),
        )
        return [(device, 16384)]


# --------------------------------------------------------------------------- #
# The packaged unit file
# --------------------------------------------------------------------------- #


def _parsed_unit() -> configparser.RawConfigParser:
    # strict=False: unit files legitimately repeat keys (DeviceAllow=...);
    # optionxform=str: systemd keys are case-sensitive.
    parser = configparser.RawConfigParser(strict=False)
    parser.optionxform = str  # type: ignore[assignment]
    parser.read_string(UNIT.read_text())
    return parser


def test_unit_file_exists_in_the_source_tree() -> None:
    """Resolved from the repo, NOT `resources.files("embedx")`.

    It used to come through the installed package, so that a missing
    package-data entry failed this test. That stopped being a coherent check
    once the unit became server-side: it is legitimately absent from a
    library wheel, so resolving through the package means the content
    assertions below either fail for a correct build or get skipped for one.

    Skipping was the other option and is worse. A skip triggered by "the file
    is not here" is indistinguishable from a skip triggered by "the
    force-include silently broke on a server build", so the suite would go
    green on exactly the regression it exists to catch. Reading from the
    source tree instead means the content assertions ALWAYS run and can never
    silently pass, and the packaging question is asked separately below --
    statically here, and against the real artifacts in the CI packaging job,
    which builds both ways.
    """
    assert UNIT.is_file(), f"{UNIT} is missing from the source tree"


def test_unit_is_packaged_only_for_server_builds() -> None:
    """The unit ships with the server half, under one flag, not on its own.

    This is the packaging assertion the source-tree read gives up. It is
    static -- the built artifacts are checked in CI -- but it catches the
    realistic regression: someone restores the unconditional force-include,
    or edits one of the two files and not the other.
    """
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text())
    wheel = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]

    assert "src/embedx/service" in wheel["exclude"], (
        "the unit must be excluded from the default wheel: its ExecStart runs "
        "`embedx serve`, which a library install does not have"
    )
    # An unconditional force-include would put it back in EVERY wheel and make
    # the exclude above decorative. That is how it was before this change.
    assert "force-include" not in wheel, (
        "force-include is unconditional; the unit belongs in hatch_build.py's flagged path instead"
    )
    assert wheel["hooks"]["custom"]["path"] == "hatch_build.py"

    # ...and the hook must actually re-include it, or a server build ships
    # without the unit and nothing here would notice.
    hook = (REPO / "hatch_build.py").read_text()
    assert '"*.service"' in hook, "hatch_build.py does not re-include the unit file"
    for excluded in wheel["exclude"]:
        directory = excluded.rsplit("/", 1)[-1]
        assert f'"{directory}"' in hook, (
            f"pyproject excludes {excluded} but hatch_build.py never puts it "
            "back: that directory would be missing from server builds too"
        )


def test_unit_file_parses_with_expected_sections() -> None:
    parser = _parsed_unit()
    assert set(parser.sections()) == {"Unit", "Service", "Install"}


def test_unit_exec_lines_and_basics() -> None:
    service = _parsed_unit()["Service"]
    assert "embedx serve" in service["ExecStart"]
    assert "embedx check" in service["ExecStartPre"]
    assert service["Restart"] == "on-failure"
    assert service["EnvironmentFile"] == "/etc/embedx/embedx.env"
    assert _parsed_unit()["Install"]["WantedBy"] == "multi-user.target"


def test_gpu_breaking_hardening_options_absent() -> None:
    # The assertion that stops a future "hardening improvement" from
    # silently removing GPU access or breaking torch. These are checked on
    # the PARSED options: both strings do appear in explanatory comments.
    service = _parsed_unit()["Service"]
    assert "MemoryDenyWriteExecute" not in service  # breaks torch
    assert "PrivateDevices" not in service  # hides /dev/nvidia*


def test_hardening_block_present_with_device_allowlist() -> None:
    service = _parsed_unit()["Service"]
    assert service["NoNewPrivileges"] == "yes"
    assert service["ProtectSystem"] == "strict"
    assert service["DevicePolicy"] == "closed"
    # RawConfigParser keeps the last duplicate; presence of the key proves
    # the allowlist form is used.
    assert "DeviceAllow" in service


# --------------------------------------------------------------------------- #
# serve preflight exposure warning
# --------------------------------------------------------------------------- #


@pytest.fixture()
def quiet_serve(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_ranked_devices", lambda settings: [])
    monkeypatch.setattr(cli, "_build_registry", lambda settings, ranked: FakeRegistry())
    monkeypatch.setattr(cli, "_run_uvicorn", lambda application, settings: None)


BASE_ARGS: list[str] = []


def test_serve_preflight_warns_on_exposed_bind(
    quiet_serve: None, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        result = runner.invoke(cli.app, ["serve", *BASE_ARGS, "--host", "0.0.0.0"])
    assert result.exit_code == 0, result.output
    assert "without an API key" in caplog.text


def test_serve_preflight_no_warning_on_loopback(
    quiet_serve: None, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        result = runner.invoke(cli.app, ["serve", *BASE_ARGS])
    assert result.exit_code == 0, result.output
    assert "without an API key" not in caplog.text
