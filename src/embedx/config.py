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

# Idle seconds before a model is unloaded. Ollama, the closest prior art for
# keep-alive on a local model server, defaults to 5 minutes and documents
# roughly 10-15 minutes as the sane range for keeping a model warm without
# hoarding VRAM; 600s sits at the bottom of that range, which suits a box
# whose GPUs are shared with other work. Task 17 recalibrates this against
# measured load latency -- the right value is "long enough that a reload
# costs less than the memory held", and that number is not yet measured.
DEFAULT_KEEP_ALIVE_S = 600.0

# Settings removed in the multi-model migration. `extra="forbid"` catches
# these as init kwargs and as config-file keys, but NOT as environment
# variables: pydantic-settings simply ignores an EMBEDX_* variable that
# matches no field. That silence is the dangerous case -- an upgraded box
# whose env file still pins EMBEDX_MODEL_ID starts happily and serves
# whatever requests name, while its operator believes it is pinned. So the
# variables are detected and rejected explicitly.
_REMOVED_SETTINGS: dict[str, str] = {
    "model_id": "send `model` in each request instead",
    "pooling": "send `pooling` on a model's first request instead",
    "dtype": "send `dtype` on a model's first request instead",
    "max_seq_len": "send `max_seq_len` on a model's first request instead",
}

WRAPPING_PLACEHOLDER = "{text}"
# Sentinel used only to prove the template really substitutes: a value no
# plausible template contains literally.
_WRAPPING_PROBE = "\x00embedx-probe\x00"


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

    # Model handling. Field descriptions are user-facing: the README config
    # table is rendered from them (tests/test_readme.py keeps it in sync).
    #
    # model_id / pooling / dtype / max_seq_len used to live here. They are
    # per-request now (task 14), resolved once per model id on first load
    # (task 13), because one server serves many checkpoints.
    revision: str | None = Field(
        default=None, description="Model revision: branch, tag, or commit."
    )
    # DELIBERATE, not an oversight: `normalize` and `revision` are arguably
    # per-model properties, exactly like pooling. A caller asking for
    # unnormalized vectors from one model and normalized from another cannot
    # say so, and a server-wide `revision` pinned to one checkpoint's commit
    # is close to meaningless across many. Both stay server-wide here because
    # moving them is a request-schema change this task does not cover; they
    # are the obvious candidates if per-model handling is revisited.
    normalize: bool = Field(default=True, description="L2-normalize embeddings after pooling.")
    wrapping: str | None = Field(
        default=None,
        description='Template wrapping every input, e.g. "Q: {text}"; must contain '
        "'{text}' exactly once. Unset means inputs are passed through untouched.",
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

    # Model residency
    default_keep_alive_s: float = Field(
        default=DEFAULT_KEEP_ALIVE_S,
        gt=0,
        description="Seconds an idle model stays resident before it is unloaded.",
    )
    max_loaded_models: int | None = Field(
        default=None,
        gt=0,
        description="Max models resident at once; unset means no cap. At the cap, a new "
        "load evicts the least-recently-used model that no request is using, rather "
        "than refusing; if every resident model is in use, the load fails instead.",
    )

    # Concurrency. Two separate caps, deliberately: see the comment on
    # max_concurrent_loads for why sharing one would starve warm requests.
    #
    # 8 is not scaled to device count, though that would be the obvious
    # move, because Settings is constructed before any device is discovered
    # and must stay importable without torch -- scaling here would mean
    # either a CUDA call in config or a default that lies until something
    # else corrects it. The number is chosen against what concurrency can
    # actually buy: Engine holds a per-backend lock across embed, so real
    # GPU parallelism is capped at the device count no matter how many
    # requests are admitted. Everything above that only overlaps CPU work
    # (tokenizing, assembling the output array) with GPU work, and a couple
    # of requests' worth of overlap saturates that. 8 covers the 1-4 device
    # boxes this targets with room to spare, while bounding both the thread
    # count (requests x devices worker threads) and the memory held by
    # in-flight inputs, which at max_request_items=2048 each is not small.
    max_concurrent_requests: int = Field(
        default=8,
        gt=0,
        description="Max embedding requests in flight at once; the rest queue.",
    )
    # 30s, not a few seconds: a queued request may be waiting behind one
    # that is doing a cold load, measured at ~9.4s for MiniLM on the
    # reference two-GPU host. A timeout below that would turn a normal cold
    # start into a 503 storm. It is a backpressure valve, not a latency
    # target -- it exists so a client gets a definite answer instead of an
    # unbounded queue, and 30s leaves room for a couple of loads ahead.
    request_queue_timeout_s: float = Field(
        default=30.0,
        gt=0,
        description="Seconds a queued request waits for a slot before returning 503.",
    )
    # Cold loads get their own, smaller cap and their own semaphore. They
    # must not share the request cap: N slow cold loads would fill it and
    # starve every request to an already-resident model, which is the
    # opposite of what a cap is for. A request to a warm model should never
    # wait behind a cold load of a different one.
    #
    # 2 is a starting point, not a measurement: it allows one load to
    # overlap another's download/CPU phase while keeping at most two
    # simultaneous claims on VRAM and PCIe bandwidth. Task 17's benchmark
    # gives the real number -- how much a second simultaneous load degrades
    # an in-flight one on this hardware -- and this should be revisited
    # then, the same loop as default_keep_alive_s.
    max_concurrent_loads: int = Field(
        default=2,
        gt=0,
        description="Max cold model loads running at once; warm requests never queue here.",
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
    def _reject_removed_settings(cls, data: Any) -> Any:
        found = [
            (f"{cls.model_config.get('env_prefix', '')}{name}".upper(), advice)
            for name, advice in _REMOVED_SETTINGS.items()
            if f"{cls.model_config.get('env_prefix', '')}{name}".upper() in os.environ
        ]
        if found:
            detail = "; ".join(f"{name} -> {advice}" for name, advice in found)
            raise ValueError(
                "these settings were removed when embedx became multi-model, and are "
                f"no longer read: {detail}. Remove them from the environment or env "
                "file. Leaving them set would look like configuration that is being "
                "honoured when it is being ignored."
            )
        return data

    @field_validator("wrapping")
    @classmethod
    def _check_wrapping(cls, value: str | None) -> str | None:
        """Reject templates that would drop or duplicate the caller's input.

        Both failure modes are silent at request time -- zero placeholders
        embeds one constant string for every request, two embeds the input
        twice -- so neither is guessed around. They are configuration
        errors, raised here with the offending value in the message.
        """
        if value is None:
            return value
        found = value.count(WRAPPING_PLACEHOLDER)
        if found != 1:
            reason = (
                "zero occurrences would drop the input and embed a constant string"
                if found == 0
                else f"{found} occurrences are ambiguous"
            )
            raise ValueError(
                f"wrapping must contain {WRAPPING_PLACEHOLDER!r} exactly once "
                f"({reason}); got {value!r}"
            )
        try:
            rendered = value.format(text=_WRAPPING_PROBE)
        except (IndexError, KeyError, ValueError) as exc:
            raise ValueError(
                f"wrapping {value!r} is not a usable format template "
                f"({type(exc).__name__}: {exc}): {WRAPPING_PLACEHOLDER!r} must be the "
                "only placeholder, and any other brace must be doubled"
            ) from None
        if _WRAPPING_PROBE not in rendered:
            # "{{text}}" contains the substring but formats to a literal, so
            # the count above passes while the input still goes nowhere.
            raise ValueError(
                f"wrapping {value!r} does not substitute the input: "
                f"{WRAPPING_PLACEHOLDER!r} is escaped rather than a placeholder"
            )
        return value

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
    def _check_load_cap_can_bind(self) -> Settings:
        # A cold load happens inside a request and holds a request slot for
        # its whole duration, so in-flight loads can never exceed in-flight
        # requests. A larger load cap is unreachable by construction:
        # accepting it silently would leave an operator believing they had
        # raised a limit that does nothing.
        if self.max_concurrent_loads > self.max_concurrent_requests:
            raise ValueError(
                f"max_concurrent_loads={self.max_concurrent_loads} exceeds "
                f"max_concurrent_requests={self.max_concurrent_requests}, so it can "
                "never bind: every load runs inside a request and holds a request "
                "slot while it does. Lower it, or raise max_concurrent_requests."
            )
        return self

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

    def wrap(self, text: str) -> str:
        """Apply the configured wrapping template; identity when unset.

        Validated at construction, so `format` cannot fail here.
        """
        if self.wrapping is None:
            return text
        return self.wrapping.format(text=text)

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
