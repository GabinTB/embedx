"""Tests for the CLI (task 09). In-process, no real ports, no real models."""

from __future__ import annotations

import ast
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

import embedx.cli as cli
from embedx.backend import FakeBackend
from embedx.config import Pooling, Settings
from embedx.gpu.discovery import DeviceInfo

runner = CliRunner()
GIB = 2**30

# No model arguments any more: a server is started without naming one.
BASE_ARGS: list[str] = []


def test_cli_does_not_import_the_http_layer_at_module_level() -> None:
    """The whole CLI must survive a wheel that ships no `embedx.api`.

    The console script is `embedx.cli:app`, so a module-level
    `from embedx.api import ...` would make `--help`, `info` and `check` fail
    on a library install too -- not just `serve`, the one command that
    actually needs it. This is a guard, not a style rule: the import was at
    module level until task 21, and nothing else in the suite would notice it
    coming back.
    """
    tree = ast.parse(Path(cli.__file__).read_text(), filename=cli.__file__)
    offenders = [
        node.module or ""
        for node in tree.body  # module level only; inside `serve` is fine
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("embedx.api")
    ] + [
        alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name.startswith("embedx.api")
    ]
    assert not offenders, f"cli.py imports the HTTP layer at module level: {offenders}"


def test_serve_is_registered_when_the_http_layer_is_present() -> None:
    """The other half of the conditional: present here, absent on a wheel.

    CI asserts the absent case against the real artifact; this asserts the
    source checkout still offers the command it is supposed to.
    """
    assert cli._api_available()
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0, output_of(result)
    assert "serve" in output_of(result)


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


class FakeRegistry:
    """Registry double: records what was warmed, loads nothing real."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings
        self.warmed: list[tuple[str, Any]] = []
        self.resident: list[Any] = []
        self.started = False
        self.stopped = False
        self.evicted = False
        self.in_flight_loads = 0
        self.max_concurrent_loads = 2

    @contextmanager
    def acquire(self, model_id: str, pooling: Any = None, **kwargs: Any) -> Iterator[FakeEngine]:
        self.warmed.append((model_id, pooling))
        self.kwargs = kwargs
        yield FakeEngine()

    def list_loaded(self) -> list[Any]:
        return list(self.resident)

    def start(self) -> None:
        self.started = True

    def stop(self, timeout: float = 5.0) -> None:
        self.stopped = True

    def evict_all(self) -> list[str]:
        self.evicted = True
        return []


def output_of(result: Any) -> str:
    # click >= 8.2 separates stderr; older versions mix it into output.
    try:
        return str(result.output) + str(result.stderr)
    except (ValueError, AttributeError):
        return str(result.output)


@pytest.fixture()
def serve_capture(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Neutralize device discovery, registry and uvicorn; capture settings."""
    captured: dict[str, Any] = {}

    def fake_registry(settings: Settings, ranked: Any) -> FakeRegistry:
        captured["build_settings"] = settings
        registry = FakeRegistry(settings)
        captured["registry"] = registry
        return registry

    def fake_run_uvicorn(application: Any, settings: Settings) -> None:
        captured["application"] = application
        captured["host"] = settings.host
        captured["port"] = settings.port

    monkeypatch.setattr(cli, "_ranked_devices", lambda settings: [make_device(0)])
    monkeypatch.setattr(cli, "_build_registry", fake_registry)
    monkeypatch.setattr(cli, "_run_uvicorn", fake_run_uvicorn)
    return captured


# --------------------------------------------------------------------------- #
# The precedence trap — regression test first
# --------------------------------------------------------------------------- #


def test_env_port_wins_when_cli_option_not_passed(
    monkeypatch: pytest.MonkeyPatch, serve_capture: dict[str, Any]
) -> None:
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
    # Nothing is loaded at startup; the registry is built and started empty.
    registry = serve_capture["registry"]
    assert registry.started is True
    assert registry.list_loaded() == []


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
    assert "Fake GPU 0" in out
    assert "Fake GPU 1" in out
    assert "max_batch_tokens=16384" in out
    # Removed fields must not linger in the output.
    assert "model_id" not in out
    assert "pooling:" not in out
    # A fresh `info` process is not the server; its empty list must not be
    # readable as "the running server has nothing loaded".
    assert "not the running server" in out
    assert "none - this command loads nothing" in out


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
    # A value that fails validation must name the field it came from.
    result = runner.invoke(cli.app, ["check", "--default-keep-alive-s", "0"])
    assert result.exit_code == 2
    assert "default_keep_alive_s" in output_of(result)


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


# --------------------------------------------------------------------------- #
# check --warm
# --------------------------------------------------------------------------- #


@pytest.fixture()
def warm_capture(monkeypatch: pytest.MonkeyPatch) -> FakeRegistry:
    registry = FakeRegistry()
    monkeypatch.setattr(cli, "_ranked_devices", lambda settings: [make_device(0)])
    monkeypatch.setattr(cli, "_build_registry", lambda settings, ranked: registry)
    return registry


def test_check_without_warm_loads_nothing(warm_capture: FakeRegistry) -> None:
    result = runner.invoke(cli.app, ["check"])
    assert result.exit_code == 0, output_of(result)
    assert warm_capture.warmed == []
    assert "check ok" in output_of(result)


def test_check_warm_loads_then_leaves_nothing_resident(warm_capture: FakeRegistry) -> None:
    result = runner.invoke(cli.app, ["check", "--warm", "org/model", "--warm-pooling", "mean"])
    assert result.exit_code == 0, output_of(result)
    assert warm_capture.warmed == [("org/model", Pooling.MEAN)]
    # keep_alive=0 is what makes this a load-then-evict rather than a load.
    assert warm_capture.kwargs["keep_alive"] == 0
    assert warm_capture.list_loaded() == []
    out = output_of(result)
    assert "warm ok" in out and "nothing left resident" in out


def test_check_warm_fails_if_the_model_stays_resident(
    warm_capture: FakeRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    # If the keep_alive=0 contract ever broke, a preflight that silently
    # left a model on the GPU would be worse than one that failed.
    warm_capture.resident = ["org/model"]
    result = runner.invoke(cli.app, ["check", "--warm", "org/model", "--warm-pooling", "mean"])
    assert result.exit_code == 1
    assert "still resident" in output_of(result)
    assert warm_capture.evicted is True


def test_warm_without_pooling_is_refused_and_explains_why(
    warm_capture: FakeRegistry,
) -> None:
    result = runner.invoke(cli.app, ["check", "--warm", "org/model"])
    assert result.exit_code == 2
    out = output_of(result)
    assert "--warm-pooling" in out
    assert "never inferred" in out
    assert warm_capture.warmed == [], "nothing may load without a pooling"


def test_warm_pooling_without_warm_is_refused(warm_capture: FakeRegistry) -> None:
    result = runner.invoke(cli.app, ["check", "--warm-pooling", "mean"])
    assert result.exit_code == 2
    assert "only means something with --warm" in output_of(result)


def test_check_warm_reports_a_load_failure_as_a_failed_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from embedx.registry import UnsupportedWeightFormatError

    class RefusingRegistry(FakeRegistry):
        @contextmanager
        def acquire(self, model_id: str, pooling: Any = None, **kwargs: Any) -> Iterator[Any]:
            raise UnsupportedWeightFormatError(model_id, ["pytorch_model.bin"])
            yield  # pragma: no cover - unreachable, keeps this a generator

    monkeypatch.setattr(cli, "_ranked_devices", lambda settings: [make_device(0)])
    monkeypatch.setattr(cli, "_build_registry", lambda settings, ranked: RefusingRegistry())
    result = runner.invoke(cli.app, ["check", "--warm", "org/bad", "--warm-pooling", "mean"])
    assert result.exit_code == 1
    out = output_of(result)
    assert "check failed" in out and "pytorch_model.bin" in out
