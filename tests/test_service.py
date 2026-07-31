"""Tests for the packaged systemd unit and serve preflight (task 10)."""

from __future__ import annotations

import configparser
import logging
from importlib import resources
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

import embedx.cli as cli
from embedx.backend import FakeBackend
from embedx.gpu.discovery import DeviceInfo

runner = CliRunner()
GIB = 2**30


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


def _unit_resource() -> Any:
    # Through the installed package, never a relative path: a missing
    # package-data entry makes this test fail, which is the point.
    return resources.files("embedx") / "service" / "embedx.service"


def _parsed_unit() -> configparser.RawConfigParser:
    # strict=False: unit files legitimately repeat keys (DeviceAllow=...);
    # optionxform=str: systemd keys are case-sensitive.
    parser = configparser.RawConfigParser(strict=False)
    parser.optionxform = str  # type: ignore[assignment]
    parser.read_string(_unit_resource().read_text())
    return parser


def test_unit_file_resolves_through_installed_package() -> None:
    assert _unit_resource().is_file()


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
    monkeypatch.setattr(cli, "build_engine", lambda settings: FakeEngine())
    monkeypatch.setattr(cli, "_run_uvicorn", lambda application, settings: None)


BASE_ARGS = ["--model-id", "test-model", "--pooling", "mean"]


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
