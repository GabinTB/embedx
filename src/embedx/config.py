"""Configuration: holds and validates values, nothing else.

Importable without torch — no CUDA calls and no device enumeration here.
Precedence is CLI (init kwargs) > environment (`EMBEDX_*`) > config file
(TOML at `$EMBEDX_CONFIG`) > defaults.
"""

from __future__ import annotations

import ipaddress
import os
import tomllib
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field, ValidationInfo, field_validator, model_validator
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    NoDecode,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    SettingsError,
)

CONFIG_PATH_ENV_VAR = "EMBEDX_CONFIG"


class Pooling(StrEnum):
    """How token embeddings are reduced to one vector per text."""

    CLS = "cls"
    MEAN = "mean"
    LAST_TOKEN = "last_token"


class Dtype(StrEnum):
    """Model compute dtype; AUTO defers to the checkpoint."""

    AUTO = "auto"
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _parse_override_string(
    raw: str, cast: Callable[[str], float], value_desc: str
) -> dict[int, float]:
    """Parse an `"0=1.0,1=0.35"`-style override string into a dict."""
    overrides: dict[int, float] = {}
    if not raw.strip():
        return overrides
    for raw_token in raw.split(","):
        token = raw_token.strip()
        if "=" not in token:
            raise ValueError(f"bad override token {token!r}: expected 'index=value'")
        index_part, _, value_part = token.partition("=")
        try:
            index = int(index_part)
        except ValueError:
            raise ValueError(
                f"bad override token {token!r}: device index {index_part!r} is not an integer"
            ) from None
        if index < 0:
            raise ValueError(f"bad override token {token!r}: device index must be >= 0")
        if index in overrides:
            raise ValueError(f"bad override token {token!r}: duplicate device index {index}")
        try:
            value = cast(value_part)
        except ValueError:
            raise ValueError(
                f"bad override token {token!r}: {value_part!r} is not a valid {value_desc}"
            ) from None
        if value <= 0:
            raise ValueError(f"bad override token {token!r}: {value_desc} must be > 0")
        overrides[index] = value
    return overrides


class _EnvConfigFileSource(PydanticBaseSettingsSource):
    """TOML settings source at `$EMBEDX_CONFIG`.

    An unset variable or a path that does not exist is an empty source, not
    an error. A file that exists but fails to parse raises with the path.
    """

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        return None, field_name, False  # unused: __call__ is overridden wholesale

    def __call__(self) -> dict[str, Any]:
        raw_path = os.environ.get(CONFIG_PATH_ENV_VAR)
        if not raw_path:
            return {}
        path = Path(raw_path)
        if not path.is_file():
            return {}
        try:
            with path.open("rb") as fh:
                return tomllib.load(fh)
        except tomllib.TOMLDecodeError as exc:
            raise SettingsError(f"malformed config file {path}: {exc}") from exc


class Settings(BaseSettings):
    """embedx runtime configuration."""

    model_config = SettingsConfigDict(
        env_prefix="EMBEDX_",
        case_sensitive=False,
        frozen=True,
        extra="forbid",
    )

    # Model. Field descriptions are user-facing: the README config table is
    # rendered from them (tests/test_readme.py keeps it in sync).
    model_id: str = Field(min_length=1, description="Hugging Face model id or local path.")
    revision: str | None = Field(
        default=None, description="Model revision: branch, tag, or commit."
    )
    pooling: Pooling = Field(
        description="Pooling strategy. Required on purpose: a wrong pooling produces "
        "plausible garbage vectors, so it is never inferred."
    )
    normalize: bool = Field(default=True, description="L2-normalize embeddings after pooling.")
    dtype: Dtype = Field(
        default=Dtype.AUTO,
        description="Compute dtype; auto resolves by device capability (bf16/fp16/fp32).",
    )
    max_seq_len: int | None = Field(
        default=None,
        gt=0,
        description="Max sequence length in tokens; longer inputs are truncated and counted.",
    )

    # Server
    host: str = Field(default="127.0.0.1", description="Bind address; loopback by default.")
    port: int = Field(default=8477, ge=1024, le=65535, description="Bind port (1024-65535).")
    api_key: str | None = Field(
        default=None, description="Bearer API key; unset disables auth entirely."
    )
    max_request_items: int = Field(
        default=2048, gt=0, description="Maximum number of inputs per request."
    )
    max_request_bytes: int = Field(
        default=32_000_000, gt=0, description="Maximum request body size in bytes."
    )
    log_level: str = Field(
        default="INFO", description="Log level: DEBUG, INFO, WARNING, ERROR, or CRITICAL."
    )

    # Devices and batching
    devices: Annotated[list[int] | None, NoDecode] = Field(
        default=None, description='CUDA device indices (list or "0,1"); default: all visible.'
    )
    max_batch_tokens: int = Field(
        default=16384, gt=0, description="Default per-batch padded-token budget."
    )
    max_batch_items: int | None = Field(
        default=None, gt=0, description="Per-batch item-count cap (bounds zero-cost batches)."
    )
    device_weights: Annotated[dict[int, float], NoDecode] = Field(
        default_factory=dict,
        description='Per-device speed-weight overrides, e.g. "0=1.0,1=0.35".',
    )
    device_batch_tokens: Annotated[dict[int, int], NoDecode] = Field(
        default_factory=dict,
        description='Per-device token-budget overrides, e.g. "0=16384,1=4096".',
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Earlier wins: CLI kwargs > env > file; defaults apply last implicitly.
        return (init_settings, env_settings, _EnvConfigFileSource(settings_cls))

    @model_validator(mode="before")
    @classmethod
    def _require_pooling(cls, data: Any) -> Any:
        if isinstance(data, dict) and not data.get("pooling"):
            raise ValueError(
                "pooling is required and is never inferred: a wrong pooling produces "
                "plausible but garbage vectors with no error. "
                "Set pooling to one of: cls, mean, last_token."
            )
        return data

    @field_validator("devices", mode="before")
    @classmethod
    def _parse_devices(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        parsed = []
        for raw_token in value.split(","):
            token = raw_token.strip()
            if not token:
                continue
            try:
                parsed.append(int(token))
            except ValueError:
                raise ValueError(f"device index {token!r} is not an integer") from None
        return parsed

    @field_validator("devices")
    @classmethod
    def _check_devices(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return value
        seen: set[int] = set()
        for index in value:
            if index < 0:
                raise ValueError(f"device index {index} must be >= 0")
            if index in seen:
                raise ValueError(f"duplicate device index {index}")
            seen.add(index)
        return value

    @field_validator("device_weights", "device_batch_tokens", mode="before")
    @classmethod
    def _parse_override(cls, value: object, info: ValidationInfo) -> object:
        if not isinstance(value, str):
            return value
        if info.field_name == "device_weights":
            return _parse_override_string(value, float, "device weight")
        return _parse_override_string(value, int, "token budget")

    @field_validator("device_weights")
    @classmethod
    def _check_weights(cls, value: dict[int, float]) -> dict[int, float]:
        for index, weight in value.items():
            if index < 0:
                raise ValueError(f"device weight index {index} must be >= 0")
            if weight <= 0:
                raise ValueError(f"device weight for index {index} must be > 0, got {weight}")
        return value

    @field_validator("device_batch_tokens")
    @classmethod
    def _check_batch_tokens(cls, value: dict[int, int]) -> dict[int, int]:
        for index, budget in value.items():
            if index < 0:
                raise ValueError(f"token budget index {index} must be >= 0")
            if budget <= 0:
                raise ValueError(f"token budget for index {index} must be > 0, got {budget}")
        return value

    @model_validator(mode="after")
    def _check_override_keys_in_devices(self) -> Settings:
        if self.devices is None:
            return self
        allowed = set(self.devices)
        for name in ("device_weights", "device_batch_tokens"):
            unknown = sorted(set(getattr(self, name)) - allowed)
            if unknown:
                raise ValueError(
                    f"{name} refers to device indices {unknown} not present in "
                    f"devices {sorted(allowed)}: the override would silently do nothing"
                )
        return self

    @property
    def is_exposed_without_auth(self) -> bool:
        """True when reachable beyond loopback with no API key. Never raises."""
        return self.api_key is None and self.host != "" and not _is_loopback(self.host)

    def exposure_warning(self) -> str:
        """The warning line for `is_exposed_without_auth`; CLI and tests share it."""
        return (
            f"embedx is listening on {self.host}:{self.port} without an API key; "
            "anyone who can reach this address can submit requests. "
            "Set EMBEDX_API_KEY or bind to a loopback address."
        )
