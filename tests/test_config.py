"""Tests for Settings: sources, parsing, validation (task 05)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic_settings import SettingsError

from embedx.config import DEFAULT_KEEP_ALIVE_S, Settings

# Env isolation for these tests lives in conftest.py (_clean_embedx_env).


def make_settings(**overrides: object) -> Settings:
    return Settings(**overrides)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Sources and precedence
# --------------------------------------------------------------------------- #


def test_precedence_cli_over_env_over_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = tmp_path / "embedx.toml"
    config.write_text('host = "10.0.0.1"\nlog_level = "DEBUG"\nport = 9001\n')
    monkeypatch.setenv("EMBEDX_CONFIG", str(config))

    # File only.
    settings = Settings()
    assert settings.host == "10.0.0.1"
    assert settings.port == 9001
    assert settings.log_level == "DEBUG"

    # Env beats file.
    monkeypatch.setenv("EMBEDX_HOST", "10.0.0.2")
    monkeypatch.setenv("EMBEDX_PORT", "9002")
    settings = Settings()
    assert settings.host == "10.0.0.2"
    assert settings.port == 9002

    # Init kwarg (the CLI path) beats env; unset keys still fall through.
    settings = Settings(host="10.0.0.3", port=9003)
    assert settings.host == "10.0.0.3"
    assert settings.port == 9003
    assert settings.log_level == "DEBUG"  # from file
    assert settings.normalize is True  # default


def test_absent_config_env_var_is_not_an_error() -> None:
    assert make_settings().port == 8477


def test_nonexistent_config_path_is_empty_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDX_CONFIG", "/definitely/not/a/real/path.toml")
    assert make_settings().port == 8477


def test_malformed_config_file_names_the_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = tmp_path / "broken.toml"
    config.write_text("port = [unclosed\n")
    monkeypatch.setenv("EMBEDX_CONFIG", str(config))
    with pytest.raises(SettingsError, match=re.escape(str(config))):
        Settings()


# --------------------------------------------------------------------------- #
# No required fields: a server is configured without naming a checkpoint
# --------------------------------------------------------------------------- #


def test_settings_construct_with_nothing_set() -> None:
    # Replaces the task-05 tests that asserted model_id and pooling were
    # required. They are not fields any more: `model` is per request, and
    # pooling is required on a model's FIRST LOAD (registry), not at
    # configuration time. The rule moved; it did not go away.
    settings = Settings()
    assert settings.port == 8477
    assert settings.normalize is True


@pytest.mark.parametrize("removed", ["model_id", "pooling", "dtype", "max_seq_len"])
def test_single_model_fields_are_gone_and_rejected(removed: str) -> None:
    assert removed not in Settings.model_fields
    # extra="forbid" turns a stale env-file or deployment into a loud
    # failure rather than a silently ignored setting.
    with pytest.raises(ValidationError):
        Settings(**{removed: "mean"})  # type: ignore[arg-type]


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
        {"default_keep_alive_s": 0},
        {"max_loaded_models": 0},
    ],
)
def test_out_of_bounds_values_rejected(overrides: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        make_settings(**overrides)


def test_valid_bounds_accepted() -> None:
    settings = make_settings(port=1024, max_batch_items=32, default_keep_alive_s=30)
    assert settings.port == 1024
    assert settings.max_batch_items == 32
    assert settings.default_keep_alive_s == 30


def test_settings_are_frozen_and_reject_unknown_fields() -> None:
    settings = make_settings()
    with pytest.raises(ValidationError):
        settings.port = 9999  # type: ignore[misc]
    with pytest.raises(ValidationError):
        make_settings(bogus_field=1)


# --------------------------------------------------------------------------- #
# Wrapping template
# --------------------------------------------------------------------------- #


def test_wrapping_defaults_to_none_and_wrap_is_identity() -> None:
    settings = make_settings()
    assert settings.wrapping is None
    assert settings.wrap("hello {text} world") == "hello {text} world"


def test_wrapping_valid_template_substitutes() -> None:
    settings = make_settings(wrapping="Q: {text}")
    assert settings.wrapping == "Q: {text}"
    assert settings.wrap("banana") == "Q: banana"


@pytest.mark.parametrize("value", ["no placeholder here", "", "{}", "{other}"])
def test_wrapping_without_the_placeholder_is_rejected(value: str) -> None:
    # Zero placeholders would embed one constant string for every request.
    with pytest.raises(ValidationError) as excinfo:
        make_settings(wrapping=value)
    message = str(excinfo.value)
    assert repr(value) in message, "the error must name the offending value"
    assert "{text}" in message


@pytest.mark.parametrize("value", ["{text} {text}", "a{text}b{text}c{text}"])
def test_wrapping_with_repeated_placeholder_is_rejected(value: str) -> None:
    # Two is ambiguous, not a licence to guess which one was meant.
    with pytest.raises(ValidationError) as excinfo:
        make_settings(wrapping=value)
    message = str(excinfo.value)
    assert repr(value) in message, "the error must name the offending value"
    assert "ambiguous" in message


def test_wrapping_with_an_extra_placeholder_is_rejected() -> None:
    # Would raise KeyError per request instead of at startup.
    with pytest.raises(ValidationError) as excinfo:
        make_settings(wrapping="{context}: {text}")
    assert repr("{context}: {text}") in str(excinfo.value)


def test_wrapping_escaped_placeholder_is_rejected() -> None:
    # Contains the substring, formats to a literal, drops the input.
    with pytest.raises(ValidationError) as excinfo:
        make_settings(wrapping="{{text}}")
    assert "escaped" in str(excinfo.value)


def test_wrapping_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDX_WRAPPING", "passage: {text}")
    assert make_settings().wrapping == "passage: {text}"


# --------------------------------------------------------------------------- #
# Model residency
# --------------------------------------------------------------------------- #


def test_default_keep_alive_has_a_positive_default() -> None:
    assert make_settings().default_keep_alive_s == DEFAULT_KEEP_ALIVE_S
    assert DEFAULT_KEEP_ALIVE_S > 0


@pytest.mark.parametrize("value", [0, -1, -0.5])
def test_default_keep_alive_must_be_positive(value: float) -> None:
    # Zero here would mean every model unloads the instant its request
    # finishes, i.e. a reload per request. Per-request keep_alive=0 asks for
    # exactly that deliberately; the server-wide default must not.
    with pytest.raises(ValidationError, match="default_keep_alive_s"):
        make_settings(default_keep_alive_s=value)


def test_default_keep_alive_accepts_a_float() -> None:
    assert make_settings(default_keep_alive_s=90.5).default_keep_alive_s == 90.5


def test_max_loaded_models_is_optional_and_unset_means_no_cap() -> None:
    assert make_settings().max_loaded_models is None


@pytest.mark.parametrize("value", [0, -1])
def test_max_loaded_models_must_be_positive_when_set(value: int) -> None:
    with pytest.raises(ValidationError, match="max_loaded_models"):
        make_settings(max_loaded_models=value)


def test_max_loaded_models_accepts_a_positive_cap() -> None:
    assert make_settings(max_loaded_models=2).max_loaded_models == 2


def test_residency_settings_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDX_DEFAULT_KEEP_ALIVE_S", "45")
    monkeypatch.setenv("EMBEDX_MAX_LOADED_MODELS", "4")
    settings = make_settings()
    assert settings.default_keep_alive_s == 45
    assert settings.max_loaded_models == 4


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("EMBEDX_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2"),
        ("EMBEDX_POOLING", "mean"),
        ("EMBEDX_DTYPE", "float16"),
        ("EMBEDX_MAX_SEQ_LEN", "512"),
    ],
)
def test_removed_env_vars_fail_loudly_instead_of_being_ignored(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str
) -> None:
    # pydantic-settings ignores an EMBEDX_* variable matching no field, even
    # with extra="forbid" — that silence is the migration hazard. An upgraded
    # box whose env file still pins EMBEDX_MODEL_ID would start happily and
    # serve whatever requests name, while its operator believed otherwise.
    monkeypatch.setenv(variable, value)
    with pytest.raises(ValidationError) as excinfo:
        Settings()
    message = str(excinfo.value)
    assert variable in message
    assert "removed" in message


def test_removed_keys_in_a_config_file_are_also_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = tmp_path / "embedx.toml"
    config.write_text('model_id = "org/model"\n')
    monkeypatch.setenv("EMBEDX_CONFIG", str(config))
    with pytest.raises(ValidationError, match="model_id"):
        Settings()


def test_a_clean_environment_still_constructs() -> None:
    assert Settings().port == 8477


# --------------------------------------------------------------------------- #
# Concurrency caps
# --------------------------------------------------------------------------- #


def test_concurrency_defaults_are_positive_and_consistent() -> None:
    settings = make_settings()
    assert settings.max_concurrent_requests > 0
    assert settings.request_queue_timeout_s > 0
    assert settings.max_concurrent_loads > 0
    assert settings.max_concurrent_loads <= settings.max_concurrent_requests


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_concurrent_requests": 0},
        {"max_concurrent_requests": -1},
        {"request_queue_timeout_s": 0},
        {"request_queue_timeout_s": -0.5},
        {"max_concurrent_loads": 0},
        {"max_concurrent_loads": -1},
    ],
)
def test_concurrency_bounds_are_enforced(overrides: dict[str, float]) -> None:
    with pytest.raises(ValidationError, match=next(iter(overrides))):
        make_settings(**overrides)


def test_a_load_cap_above_the_request_cap_is_rejected() -> None:
    # It could never bind: a load runs inside a request and holds a request
    # slot for its whole duration. Accepting it would leave an operator
    # believing they had raised a limit that does nothing.
    with pytest.raises(ValidationError, match="never bind"):
        make_settings(max_concurrent_requests=2, max_concurrent_loads=3)


def test_a_load_cap_equal_to_the_request_cap_is_allowed() -> None:
    settings = make_settings(max_concurrent_requests=3, max_concurrent_loads=3)
    assert settings.max_concurrent_loads == 3


def test_raising_the_request_cap_makes_a_larger_load_cap_valid() -> None:
    settings = make_settings(max_concurrent_requests=16, max_concurrent_loads=4)
    assert (settings.max_concurrent_requests, settings.max_concurrent_loads) == (16, 4)


def test_concurrency_settings_come_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDX_MAX_CONCURRENT_REQUESTS", "12")
    monkeypatch.setenv("EMBEDX_REQUEST_QUEUE_TIMEOUT_S", "2.5")
    monkeypatch.setenv("EMBEDX_MAX_CONCURRENT_LOADS", "3")
    settings = make_settings()
    assert settings.max_concurrent_requests == 12
    assert settings.request_queue_timeout_s == 2.5
    assert settings.max_concurrent_loads == 3
