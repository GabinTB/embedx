"""Tests for Settings: sources, parsing, validation (task 05)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic_settings import SettingsError

from embedx.config import Dtype, Pooling, Settings

# Env isolation for these tests lives in conftest.py (_clean_embedx_env).


def make_settings(**overrides: object) -> Settings:
    kwargs: dict[str, object] = {"model_id": "test-model", "pooling": "mean"}
    kwargs.update(overrides)
    return Settings(**kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Sources and precedence
# --------------------------------------------------------------------------- #


def test_precedence_cli_over_env_over_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = tmp_path / "embedx.toml"
    config.write_text('model_id = "from-file"\npooling = "cls"\nport = 9001\n')
    monkeypatch.setenv("EMBEDX_CONFIG", str(config))

    # File only.
    settings = Settings()
    assert settings.model_id == "from-file"
    assert settings.port == 9001
    assert settings.pooling is Pooling.CLS

    # Env beats file.
    monkeypatch.setenv("EMBEDX_MODEL_ID", "from-env")
    monkeypatch.setenv("EMBEDX_PORT", "9002")
    settings = Settings()
    assert settings.model_id == "from-env"
    assert settings.port == 9002

    # Init kwarg (the CLI path) beats env; unset keys still fall through.
    settings = Settings(model_id="from-init", port=9003)
    assert settings.model_id == "from-init"
    assert settings.port == 9003
    assert settings.pooling is Pooling.CLS  # from file
    assert settings.dtype is Dtype.AUTO  # default


def test_absent_config_env_var_is_not_an_error() -> None:
    assert make_settings().model_id == "test-model"


def test_nonexistent_config_path_is_empty_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDX_CONFIG", "/definitely/not/a/real/path.toml")
    assert make_settings().model_id == "test-model"


def test_malformed_config_file_names_the_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = tmp_path / "broken.toml"
    config.write_text("model_id = [unclosed\n")
    monkeypatch.setenv("EMBEDX_CONFIG", str(config))
    with pytest.raises(SettingsError, match=re.escape(str(config))):
        Settings()


# --------------------------------------------------------------------------- #
# Required fields
# --------------------------------------------------------------------------- #


def test_missing_model_id_names_the_field() -> None:
    with pytest.raises(ValidationError, match="model_id"):
        Settings(pooling="mean")  # type: ignore[call-arg]


def test_empty_model_id_rejected() -> None:
    with pytest.raises(ValidationError, match="model_id"):
        make_settings(model_id="")


def test_missing_pooling_explains_why_it_is_never_inferred() -> None:
    with pytest.raises(ValidationError, match="never inferred"):
        Settings(model_id="m")  # type: ignore[call-arg]


# --------------------------------------------------------------------------- #
# Override strings
# --------------------------------------------------------------------------- #


def test_weight_override_string_and_mapping_are_equivalent() -> None:
    from_string = make_settings(device_weights="0=1.0, 1=0.35")
    from_mapping = make_settings(device_weights={0: 1.0, 1: 0.35})
    assert from_string.device_weights == from_mapping.device_weights == {0: 1.0, 1: 0.35}


def test_batch_tokens_override_string() -> None:
    settings = make_settings(device_batch_tokens="0=16384,1=4096")
    assert settings.device_batch_tokens == {0: 16384, 1: 4096}


def test_empty_override_string_is_empty_dict() -> None:
    settings = make_settings(device_weights="", device_batch_tokens="")
    assert settings.device_weights == {}
    assert settings.device_batch_tokens == {}


@pytest.mark.parametrize(
    ("raw", "offending"),
    [
        ("0:1.0", "0:1.0"),  # missing =
        ("a=1.0", "'a'"),  # non-integer device index
        ("0=abc", "'abc'"),  # non-numeric value
        ("0=1.0,0=2.0", "0=2.0"),  # duplicate index
        ("-1=1.0", "-1=1.0"),  # negative index
        ("0=0", "0=0"),  # weight <= 0
        ("0=-2", "0=-2"),  # weight <= 0 (negative)
    ],
)
def test_malformed_weight_override_names_the_token(raw: str, offending: str) -> None:
    with pytest.raises(ValidationError, match=re.escape(offending)):
        make_settings(device_weights=raw)


@pytest.mark.parametrize(
    ("raw", "offending"),
    [
        ("0=4096.5", "'4096.5'"),  # not an integer budget
        ("0=0", "0=0"),  # budget <= 0
    ],
)
def test_malformed_batch_tokens_override_names_the_token(raw: str, offending: str) -> None:
    with pytest.raises(ValidationError, match=re.escape(offending)):
        make_settings(device_batch_tokens=raw)


def test_mapping_form_still_validated() -> None:
    with pytest.raises(ValidationError, match="must be > 0"):
        make_settings(device_weights={0: -1.0})
    with pytest.raises(ValidationError, match="must be > 0"):
        make_settings(device_batch_tokens={0: 0})


# --------------------------------------------------------------------------- #
# devices
# --------------------------------------------------------------------------- #


def test_devices_string_and_list_forms_are_equivalent() -> None:
    assert make_settings(devices="0,1").devices == make_settings(devices=[0, 1]).devices == [0, 1]


def test_devices_must_be_unique_and_non_negative() -> None:
    with pytest.raises(ValidationError, match="duplicate device index 1"):
        make_settings(devices=[0, 1, 1])
    with pytest.raises(ValidationError, match="device index -1"):
        make_settings(devices="0,-1")


def test_override_key_not_in_devices_is_rejected() -> None:
    with pytest.raises(ValidationError, match="device_weights"):
        make_settings(devices=[0], device_weights="1=0.5")
    with pytest.raises(ValidationError, match="device_batch_tokens"):
        make_settings(devices=[0, 1], device_batch_tokens={2: 4096})
    # Same keys are fine when devices covers them (or is unset).
    make_settings(devices=[0, 1], device_weights={1: 0.5})
    make_settings(device_weights={3: 0.5})


# --------------------------------------------------------------------------- #
# Exposure
# --------------------------------------------------------------------------- #


def test_exposure_flag() -> None:
    assert make_settings().is_exposed_without_auth is False  # loopback default
    assert make_settings(host="localhost").is_exposed_without_auth is False
    assert make_settings(host="").is_exposed_without_auth is False
    assert make_settings(host="0.0.0.0").is_exposed_without_auth is True
    assert make_settings(host="0.0.0.0", api_key="secret").is_exposed_without_auth is False


def test_exposure_warning_names_the_bind_address() -> None:
    settings = make_settings(host="0.0.0.0", port=9000)
    message = settings.exposure_warning()
    assert "0.0.0.0" in message
    assert "9000" in message
    assert "API key" in message


# --------------------------------------------------------------------------- #
# Bounds
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "overrides",
    [
        {"port": 80},
        {"port": 70000},
        {"max_batch_tokens": 0},
        {"max_batch_items": 0},
        {"max_request_items": 0},
        {"max_request_bytes": -1},
        {"max_seq_len": 0},
    ],
)
def test_out_of_bounds_values_rejected(overrides: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        make_settings(**overrides)


def test_valid_bounds_accepted() -> None:
    settings = make_settings(port=1024, max_seq_len=512, max_batch_items=32)
    assert settings.port == 1024
    assert settings.max_seq_len == 512
    assert settings.max_batch_items == 32


def test_settings_are_frozen_and_reject_unknown_fields() -> None:
    settings = make_settings()
    with pytest.raises(ValidationError):
        settings.port = 9999  # type: ignore[misc]
    with pytest.raises(ValidationError):
        make_settings(bogus_field=1)
