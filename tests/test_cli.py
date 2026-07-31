"""Tests for the CLI (task 09). In-process, no real ports, no real models."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

import embedx.cli as cli
from embedx.backend import FakeBackend
from embedx.config import Settings
from embedx.gpu.discovery import DeviceInfo

runner = CliRunner()
GIB = 2**30

BASE_ARGS = ["--model-id", "test-model", "--pooling", "mean"]


def make_device(index: int, name: str = "Fake GPU") -> DeviceInfo:
    return DeviceInfo(
        index=index,
        name=f"{name} {index}",
        total_memory_bytes=16 * GIB,
        multi_processor_count=100,
        capability=(8, 0),
    )


class FakeEngine:
    def __init__(self) -> None:
        self.inner = FakeBackend(dim=8)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.inner.embed(texts)

    @property
    def devices_with_budgets(self) -> list[tuple[DeviceInfo, int]]:
        return [(make_device(0), 16384)]


def output_of(result: Any) -> str:
    # click >= 8.2 separates stderr; older versions mix it into output.
    try:
        return str(result.output) + str(result.stderr)
    except (ValueError, AttributeError):
        return str(result.output)


@pytest.fixture()
def serve_capture(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Neutralize engine build + uvicorn; capture what serve resolves."""
    captured: dict[str, Any] = {}

    def fake_build_engine(settings: Settings) -> FakeEngine:
        captured["build_settings"] = settings
        return FakeEngine()

    def fake_run_uvicorn(application: Any, settings: Settings) -> None:
        captured["application"] = application
        captured["host"] = settings.host
        captured["port"] = settings.port

    monkeypatch.setattr(cli, "build_engine", fake_build_engine)
    monkeypatch.setattr(cli, "_run_uvicorn", fake_run_uvicorn)
    return captured


# --------------------------------------------------------------------------- #
# The precedence trap — regression test first
# --------------------------------------------------------------------------- #


def test_env_port_wins_when_cli_option_not_passed(
    monkeypatch: pytest.MonkeyPatch, serve_capture: dict[str, Any]
) -> None:
    monkeypatch.setenv("EMBEDX_MODEL_ID", "env-model")
    monkeypatch.setenv("EMBEDX_POOLING", "mean")
    monkeypatch.setenv("EMBEDX_PORT", "9005")

    result = runner.invoke(cli.app, ["serve"])
    assert result.exit_code == 0, output_of(result)
    # No --port on the command line: the env value must survive. If typer
    # defaults were forwarded to Settings, this would be 8477.
    assert serve_capture["port"] == 9005

    result = runner.invoke(cli.app, ["serve", "--port", "9006"])
    assert result.exit_code == 0, output_of(result)
    assert serve_capture["port"] == 9006  # explicit CLI wins over env


# --------------------------------------------------------------------------- #
# serve
# --------------------------------------------------------------------------- #


def test_serve_wires_engine_app_and_uvicorn(serve_capture: dict[str, Any]) -> None:
    from fastapi import FastAPI

    result = runner.invoke(cli.app, ["serve", *BASE_ARGS, "--port", "9100"])
    assert result.exit_code == 0, output_of(result)
    assert isinstance(serve_capture["application"], FastAPI)
    assert serve_capture["host"] == "127.0.0.1"
    assert serve_capture["port"] == 9100
    assert serve_capture["build_settings"].model_id == "test-model"


def test_api_key_never_appears_in_output_or_logs(
    serve_capture: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    secret = "sekret123-do-not-print"
    with caplog.at_level(logging.DEBUG):
        result = runner.invoke(cli.app, ["serve", *BASE_ARGS, "--api-key", secret])
    assert result.exit_code == 0, output_of(result)
    assert secret not in output_of(result)
    assert secret not in caplog.text
    assert "api key: set" in caplog.text  # presence is logged, value never


# --------------------------------------------------------------------------- #
# info
# --------------------------------------------------------------------------- #


def test_info_prints_device_table_and_budgets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli, "discover_devices", lambda requested=None: [make_device(0), make_device(1)]
    )
    result = runner.invoke(cli.app, ["info", *BASE_ARGS, "--max-batch-tokens", "16384"])
    assert result.exit_code == 0, output_of(result)
    out = output_of(result)
    assert "model_id:          test-model" in out
    assert "pooling:           mean" in out
    assert "Fake GPU 0" in out
    assert "Fake GPU 1" in out
    assert "max_batch_tokens=16384" in out


def test_info_without_cuda_exits_zero_and_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "discover_devices", lambda requested=None: [])
    result = runner.invoke(cli.app, ["info", *BASE_ARGS])
    assert result.exit_code == 0, output_of(result)
    assert "none visible" in output_of(result)


# --------------------------------------------------------------------------- #
# check
# --------------------------------------------------------------------------- #


def test_check_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "discover_devices", lambda requested=None: [make_device(0)])
    result = runner.invoke(cli.app, ["check", *BASE_ARGS])
    assert result.exit_code == 0, output_of(result)
    assert "check ok" in output_of(result)


def test_check_fails_on_missing_device_index(monkeypatch: pytest.MonkeyPatch) -> None:
    def raising(requested: list[int] | None = None) -> list[DeviceInfo]:
        raise ValueError("requested device indices [3] not present; visible devices: [0, 1]")

    monkeypatch.setattr(cli, "discover_devices", raising)
    result = runner.invoke(cli.app, ["check", *BASE_ARGS, "--devices", "0,3"])
    assert result.exit_code == 1
    assert "[3] not present" in output_of(result)


def test_check_fails_on_no_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "discover_devices", lambda requested=None: [])
    result = runner.invoke(cli.app, ["check", *BASE_ARGS])
    assert result.exit_code == 1
    assert "no CUDA device available" in output_of(result)


def test_check_fails_on_invalid_config() -> None:
    # Pooling given, model_id missing entirely: the cause must be named.
    result = runner.invoke(cli.app, ["check", "--pooling", "mean"])
    assert result.exit_code == 2
    assert "model_id" in output_of(result)


def test_check_exposure_is_warning_not_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "discover_devices", lambda requested=None: [make_device(0)])
    result = runner.invoke(cli.app, ["check", *BASE_ARGS, "--host", "0.0.0.0"])
    assert result.exit_code == 0, output_of(result)
    out = output_of(result)
    assert "warning" in out
    assert "API key" in out
    assert "check ok" in out


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #


def test_invalid_log_level_fails_listing_valid_levels() -> None:
    result = runner.invoke(cli.app, ["info", *BASE_ARGS, "--log-level", "CHATTY"])
    assert result.exit_code == 2
    out = output_of(result)
    assert "CHATTY" in out
    for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        assert level in out
