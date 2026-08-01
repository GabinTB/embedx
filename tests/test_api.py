"""Tests for the OpenAI-compatible API layer (task 07b). Fake engine only."""

from __future__ import annotations

import base64
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

import embedx
from embedx.api import create_app
from embedx.backend import FakeBackend
from embedx.config import Dtype, Pooling, Settings
from embedx.gpu.discovery import DeviceInfo
from embedx.registry import (
    DEFAULT_KEEP_ALIVE_S,
    ModelPlacementError,
    ModelRegistry,
    ModelStatus,
    PoolingConflictError,
    PoolingRequiredError,
    UnsupportedWeightFormatError,
    WeightFormatUnverifiableError,
)

GIB = 2**30


class FakeEngine:
    """Deterministic engine double: FakeBackend vectors, no threads."""

    def __init__(self, dim: int = 8) -> None:
        self.inner = FakeBackend(dim=dim)
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls.append(texts)
        return self.inner.embed(texts)

    def token_count(self, texts: list[str]) -> int:
        # Mirrors the real Engine with the default `len` length function.
        return sum(len(text) for text in texts)

    @property
    def truncated_counts(self) -> list[int]:
        return [0]

    @property
    def devices_with_budgets(self) -> list[tuple[DeviceInfo, int]]:
        device = DeviceInfo(
            index=0,
            name="Fake GPU 0",
            total_memory_bytes=16 * GIB,
            multi_processor_count=100,
            capability=(8, 0),
            score=160.0,
            weight=1.0,
        )
        return [(device, 16384)]


class ExplodingEngine(FakeEngine):
    def embed(self, texts: list[str]) -> np.ndarray:
        raise RuntimeError("cuda exploded: secret internal details")


class TokenCountingEngine(FakeEngine):
    """Length function that is deliberately NOT len: 1000 per text."""

    def token_count(self, texts: list[str]) -> int:
        return 1000 * len(texts)


class WordCountingEngine(FakeEngine):
    """Length function that is not len, but still depends on the text.

    Wrapping has to change this number, which a fixed-per-item count could
    not detect.
    """

    def token_count(self, texts: list[str]) -> int:
        return sum(len(text.split()) for text in texts)


class WrongShapeEngine(FakeEngine):
    def embed(self, texts: list[str]) -> np.ndarray:
        return np.zeros(len(texts), dtype=np.float32)  # 1-D, not (n, dim)


class TruncatingEngine(FakeEngine):
    """Two devices behind one model, each having truncated some inputs."""

    @property
    def truncated_counts(self) -> list[int]:
        return [7, 5]


def make_settings(**overrides: object) -> Settings:
    kwargs: dict[str, object] = {"model_id": "test-model", "pooling": "mean"}
    kwargs.update(overrides)
    return Settings(**kwargs)  # type: ignore[arg-type]


def make_client(engine: FakeEngine | None = None, **settings_overrides: object) -> TestClient:
    engine = engine or FakeEngine()
    app = create_app(make_settings(**settings_overrides), engine)  # type: ignore[arg-type]
    return TestClient(app, raise_server_exceptions=False)


def _reference(texts: list[str], dim: int = 8) -> np.ndarray:
    return FakeBackend(dim=dim).embed(texts)


def _assert_envelope(body: dict) -> None:
    assert set(body) == {"error"}
    assert set(body["error"]) == {"message", "type", "param", "code"}


# --------------------------------------------------------------------------- #
# /v1/embeddings shape
# --------------------------------------------------------------------------- #


def test_openai_shape_and_index_order() -> None:
    client = make_client()
    texts = ["banana", "a", "some longer text here", "zz"]
    response = client.post("/v1/embeddings", json={"input": texts, "model": "my-model"})
    assert response.status_code == 200
    body = response.json()

    assert body["object"] == "list"
    assert body["model"] == "my-model"
    chars = sum(len(t) for t in texts)
    assert body["usage"] == {"prompt_tokens": chars, "total_tokens": chars}

    reference = _reference(texts)
    assert len(body["data"]) == len(texts)
    for position, item in enumerate(body["data"]):
        assert item["object"] == "embedding"
        assert item["index"] == position
        np.testing.assert_array_equal(
            np.asarray(item["embedding"], dtype=np.float32), reference[position]
        )


def test_usage_uses_engine_length_function_not_characters() -> None:
    # A regression back to character counting fails here: the fake engine's
    # length function is deliberately not len.
    client = make_client(TokenCountingEngine())
    texts = ["hello world", "abc"]
    body = client.post("/v1/embeddings", json={"input": texts, "model": "m"}).json()
    assert body["usage"] == {"prompt_tokens": 2000, "total_tokens": 2000}
    assert body["usage"]["prompt_tokens"] != sum(len(t) for t in texts)


def test_single_string_equals_single_item_list() -> None:
    client = make_client()
    as_string = client.post("/v1/embeddings", json={"input": "hello", "model": "m"}).json()
    as_list = client.post("/v1/embeddings", json={"input": ["hello"], "model": "m"}).json()
    assert as_string["data"] == as_list["data"]
    assert len(as_string["data"]) == 1


def test_base64_round_trips_bit_for_bit() -> None:
    client = make_client()
    texts = ["alpha", "beta"]
    float_body = client.post(
        "/v1/embeddings", json={"input": texts, "model": "m", "encoding_format": "float"}
    ).json()
    b64_body = client.post(
        "/v1/embeddings", json={"input": texts, "model": "m", "encoding_format": "base64"}
    ).json()

    for float_item, b64_item in zip(float_body["data"], b64_body["data"], strict=True):
        assert isinstance(b64_item["embedding"], str)
        decoded = np.frombuffer(base64.b64decode(b64_item["embedding"]), dtype="<f4")
        expected = np.asarray(float_item["embedding"], dtype=np.float32)
        assert decoded.tobytes() == expected.tobytes()  # bit for bit


# --------------------------------------------------------------------------- #
# /embed
# --------------------------------------------------------------------------- #


def test_embed_returns_bare_vectors() -> None:
    client = make_client()
    texts = ["one", "two"]
    response = client.post("/embed", json={"inputs": texts, "model": "m"})
    assert response.status_code == 200
    assert response.json() == _reference(texts).tolist()


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #


def test_auth_off_needs_no_header() -> None:
    client = make_client()
    assert client.post("/v1/embeddings", json={"input": "x", "model": "m"}).status_code == 200


def test_auth_on() -> None:
    client = make_client(api_key="sekret")
    request = {"input": "x", "model": "m"}

    ok = client.post("/v1/embeddings", json=request, headers={"Authorization": "Bearer sekret"})
    assert ok.status_code == 200

    for headers in ({}, {"Authorization": "Bearer wrong"}):
        denied = client.post("/v1/embeddings", json=request, headers=headers)
        assert denied.status_code == 401
        _assert_envelope(denied.json())
        assert denied.json()["error"]["code"] == "invalid_api_key"

    # /health stays open either way.
    assert client.get("/health").status_code == 200
    # /info is protected.
    assert client.get("/info").status_code == 401


# --------------------------------------------------------------------------- #
# Limits and rejection
# --------------------------------------------------------------------------- #


def test_too_many_items_is_400_naming_the_limit() -> None:
    client = make_client(max_request_items=3)
    response = client.post("/v1/embeddings", json={"input": ["a"] * 4, "model": "m"})
    assert response.status_code == 400
    _assert_envelope(response.json())
    assert "max_request_items=3" in response.json()["error"]["message"]


def test_oversized_body_is_413_before_parsing() -> None:
    client = make_client(max_request_bytes=64)
    response = client.post("/v1/embeddings", json={"input": ["x" * 500], "model": "m"})
    assert response.status_code == 413
    _assert_envelope(response.json())
    assert "max_request_bytes" in response.json()["error"]["message"]


def _chunked_post(client: TestClient, payload: dict, path: str = "/v1/embeddings") -> object:
    # httpx sends an iterator body as Transfer-Encoding: chunked, no
    # Content-Length — the path that bypasses the header fast path.
    import json

    raw = json.dumps(payload).encode()

    def chunks() -> object:
        for start in range(0, len(raw), 64):
            yield raw[start : start + 64]

    return client.post(path, content=chunks(), headers={"Content-Type": "application/json"})


def test_chunked_body_over_limit_is_413() -> None:
    client = make_client(max_request_bytes=200)
    response = _chunked_post(client, {"input": ["x" * 500], "model": "m"})
    assert response.status_code == 413  # type: ignore[attr-defined]
    _assert_envelope(response.json())  # type: ignore[attr-defined]
    assert "max_request_bytes" in response.json()["error"]["message"]  # type: ignore[attr-defined]


def test_chunked_body_under_limit_succeeds() -> None:
    client = make_client(max_request_bytes=10_000)
    response = _chunked_post(client, {"input": ["hello"], "model": "m"})
    assert response.status_code == 200  # type: ignore[attr-defined]
    assert response.json()["data"][0]["index"] == 0  # type: ignore[attr-defined]


def test_wrong_width_engine_fails_loudly_not_serialized() -> None:
    # Response-model validation: a 1-D array cannot quietly serialize into
    # the OpenAI shape; it must surface as the generic 500 envelope.
    client = make_client(WrongShapeEngine())
    response = client.post("/v1/embeddings", json={"input": ["a", "b"], "model": "m"})
    assert response.status_code == 500
    _assert_envelope(response.json())
    assert response.json()["error"]["message"] == "internal server error"


def test_empty_input_list_is_400() -> None:
    client = make_client()
    response = client.post("/v1/embeddings", json={"input": [], "model": "m"})
    assert response.status_code == 400
    _assert_envelope(response.json())


@pytest.mark.parametrize("tokens", [[1, 2, 3], [[1, 2], [3]]])
def test_pretokenized_input_rejected_clearly(tokens: list) -> None:
    client = make_client()
    response = client.post("/v1/embeddings", json={"input": tokens, "model": "m"})
    assert response.status_code == 400
    _assert_envelope(response.json())
    assert "pre-tokenized" in response.json()["error"]["message"]


def test_dimensions_validated_against_output_width() -> None:
    client = make_client(FakeEngine(dim=8))
    request = {"input": "x", "model": "m", "dimensions": 16}
    mismatch = client.post("/v1/embeddings", json=request)
    assert mismatch.status_code == 400
    assert "dimensions=16" in mismatch.json()["error"]["message"]

    request["dimensions"] = 8
    assert client.post("/v1/embeddings", json=request).status_code == 200


# --------------------------------------------------------------------------- #
# Failure path
# --------------------------------------------------------------------------- #


def test_engine_failure_is_generic_envelope_without_internals(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = make_client(ExplodingEngine())
    with caplog.at_level(logging.ERROR, logger="embedx.api"):
        response = client.post("/v1/embeddings", json={"input": "x", "model": "m"})
    assert response.status_code == 500
    _assert_envelope(response.json())
    assert response.json()["error"]["message"] == "internal server error"
    assert response.json()["error"]["type"] == "server_error"
    assert "cuda exploded" not in response.text
    assert "secret internal" not in response.text
    # The traceback went to the server log instead.
    assert any(record.exc_info for record in caplog.records)


# --------------------------------------------------------------------------- #
# /health and /info
# --------------------------------------------------------------------------- #


def test_health() -> None:
    client = make_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_info_reports_server_config_and_the_legacy_fixed_model() -> None:
    # The single-model startup path still serves until task 15 removes it,
    # and it reports itself through the same per-model shape.
    client = make_client(model_id="org/model", pooling="cls", normalize=False, dtype="float16")
    body = client.get("/info").json()
    assert body["version"] == embedx.__version__
    assert body["normalize"] is False
    assert body["default_keep_alive_s"] == DEFAULT_KEEP_ALIVE_S
    assert body["models"] == [
        {
            "model_id": "org/model",
            "pooling": "cls",
            "dtype": "float16",
            "devices": [0],
            "idle_s": 0.0,
            "last_used_epoch_s": body["models"][0]["last_used_epoch_s"],
            "ref_count": 0,
            "truncated_count": 0,
        }
    ]


# --------------------------------------------------------------------------- #
# wrapping template
# --------------------------------------------------------------------------- #


def test_wrapping_applied_to_every_input() -> None:
    engine = FakeEngine()
    client = make_client(engine, wrapping="Q: {text}")
    texts = ["banana", "a longer piece of text", "", "unicode: café"]
    response = client.post("/v1/embeddings", json={"input": texts, "model": "m"})
    assert response.status_code == 200
    # The engine records exactly what it was handed.
    assert engine.calls == [["Q: " + text for text in texts]]


def test_wrapping_applies_to_tei_endpoint_too() -> None:
    engine = FakeEngine()
    client = make_client(engine, wrapping="Q: {text}")
    body = {"inputs": ["one", "two"], "model": "m"}
    assert client.post("/embed", json=body).status_code == 200
    assert engine.calls == [["Q: one", "Q: two"]]


def test_no_wrapping_by_default_passes_text_through_byte_for_byte() -> None:
    # Regression guard for every deployment that never sets wrapping: the
    # bytes the caller sent are the bytes the model sees.
    engine = FakeEngine()
    client = make_client(engine)
    texts = ["hello world", "  leading and trailing  ", "{text}", "café — emdash", ""]
    assert client.post("/v1/embeddings", json={"input": texts, "model": "m"}).status_code == 200
    assert engine.calls == [texts]
    for sent, received in zip(texts, engine.calls[0], strict=True):
        assert received.encode() == sent.encode()


def test_wrapping_is_counted_in_usage_prompt_tokens() -> None:
    # The length function is word count, not len: usage must be measured on
    # the wrapped string, because the same value drives the token budgets
    # the scheduler batches against.
    engine = WordCountingEngine()
    client = make_client(engine, wrapping="Question: {text}\nAnswer:")
    texts = ["hello world", "a b c"]
    body = client.post("/v1/embeddings", json={"input": texts, "model": "m"}).json()
    # "Question: hello world\nAnswer:" is 4 words, "Question: a b c\nAnswer:"
    # is 5; the unwrapped inputs are 2 and 3.
    wrapped = 9
    assert body["usage"] == {"prompt_tokens": wrapped, "total_tokens": wrapped}
    assert body["usage"]["prompt_tokens"] != sum(len(text.split()) for text in texts)


def test_info_reports_wrapping_template() -> None:
    body = make_client(wrapping="Q: {text}").get("/info").json()
    assert body["wrapping"] == "Q: {text}"


def test_info_reports_null_wrapping_when_unset() -> None:
    assert make_client().get("/info").json()["wrapping"] is None


# --------------------------------------------------------------------------- #
# Routing through the model registry
# --------------------------------------------------------------------------- #


class FakeRegistry:
    """Registry double recording acquire calls and enter/exit separately.

    Enter and exit are counted apart on purpose: a reference that is taken
    and never released still looks fine if you only count calls.
    """

    def __init__(
        self,
        engine: FakeEngine | None = None,
        loaded: list[ModelStatus] | None = None,
        raises: Exception | None = None,
    ) -> None:
        self.engine = engine or FakeEngine()
        self.raises = raises
        self.statuses = loaded or []
        self.calls: list[dict[str, Any]] = []
        self.entered = 0
        self.exited = 0
        self.loads = 0
        self.list_loaded_calls = 0
        self._resident: set[str] = {status.model_id for status in self.statuses}

    @contextmanager
    def acquire(
        self,
        model_id: str,
        pooling: Pooling | None = None,
        dtype: Dtype = Dtype.AUTO,
        max_seq_len: int | None = None,
        keep_alive: float | None = None,
    ) -> Iterator[FakeEngine]:
        self.calls.append(
            {
                "model_id": model_id,
                "pooling": pooling,
                "dtype": dtype,
                "max_seq_len": max_seq_len,
                "keep_alive": keep_alive,
            }
        )
        if self.raises is not None:
            raise self.raises
        if model_id not in self._resident:
            self._resident.add(model_id)
            self.loads += 1
        self.entered += 1
        try:
            yield self.engine
        finally:
            self.exited += 1

    def list_loaded(self) -> list[ModelStatus]:
        self.list_loaded_calls += 1
        return list(self.statuses)


def make_status(model_id: str, **overrides: Any) -> ModelStatus:
    fields: dict[str, Any] = {
        "model_id": model_id,
        "pooling": Pooling.MEAN,
        "dtype": Dtype.AUTO,
        "device_indices": (0,),
        "idle_s": 1.5,
        "last_used_epoch_s": 1_700_000_000.0,
        "ref_count": 0,
        "truncated_count": 0,
    }
    fields.update(overrides)
    return ModelStatus(**fields)


def make_registry_client(
    registry: FakeRegistry | None = None, **settings_overrides: object
) -> tuple[TestClient, FakeRegistry]:
    registry = registry or FakeRegistry()
    app = create_app(make_settings(**settings_overrides), registry=registry)  # type: ignore[arg-type]
    return TestClient(app, raise_server_exceptions=False), registry


def test_a_new_model_is_acquired_once_and_returns_correct_vectors() -> None:
    client, registry = make_registry_client()
    texts = ["banana", "a longer piece of text"]
    response = client.post("/v1/embeddings", json={"input": texts, "model": "org/new"})

    assert response.status_code == 200
    assert registry.loads == 1
    assert len(registry.calls) == 1
    assert registry.calls[0]["model_id"] == "org/new"
    assert registry.entered == 1 and registry.exited == 1

    reference = _reference(texts)
    for position, item in enumerate(response.json()["data"]):
        np.testing.assert_array_equal(
            np.asarray(item["embedding"], dtype=np.float32), reference[position]
        )


def test_an_already_loaded_model_is_still_acquired_but_not_reloaded() -> None:
    # It must still go through acquire: that is what holds the reference
    # against the reaper for the duration of the request.
    client, registry = make_registry_client(FakeRegistry(loaded=[make_status("org/hot")]))
    for _ in range(3):
        assert (
            client.post("/v1/embeddings", json={"input": "x", "model": "org/hot"}).status_code
            == 200
        )
    assert registry.loads == 0, "a resident model must not be reloaded"
    assert len(registry.calls) == 3
    assert registry.entered == 3 and registry.exited == 3


def test_embed_endpoint_routes_through_the_registry_too() -> None:
    client, registry = make_registry_client()
    response = client.post("/embed", json={"inputs": ["one", "two"], "model": "org/tei"})
    assert response.status_code == 200
    assert response.json() == _reference(["one", "two"]).tolist()
    assert registry.calls[0]["model_id"] == "org/tei"
    assert registry.entered == 1 and registry.exited == 1


def test_the_reference_is_released_when_embedding_raises() -> None:
    # The test that actually proves the reference is not leaked: the engine
    # blows up inside the `with`, and exit must still have run.
    client, registry = make_registry_client(FakeRegistry(engine=ExplodingEngine()))
    response = client.post("/v1/embeddings", json={"input": "x", "model": "org/boom"})

    assert response.status_code == 500
    assert registry.entered == 1
    assert registry.exited == 1, "reference leaked: the model could never be evicted"
    body = response.json()
    _assert_envelope(body)
    assert body["error"]["message"] == "internal server error"
    assert "cuda exploded" not in str(body), "internals must not reach the client"
    assert "Traceback" not in str(body)


def test_load_options_reach_acquire_including_keep_alive() -> None:
    client, registry = make_registry_client()
    response = client.post(
        "/v1/embeddings",
        json={
            "input": "x",
            "model": "org/new",
            "pooling": "cls",
            "dtype": "float16",
            "max_seq_len": 128,
            "keep_alive": 42.5,
        },
    )
    assert response.status_code == 200
    assert registry.calls == [
        {
            "model_id": "org/new",
            "pooling": Pooling.CLS,
            "dtype": Dtype.FLOAT16,
            "max_seq_len": 128,
            "keep_alive": 42.5,
        }
    ]


def test_keep_alive_zero_is_forwarded_not_treated_as_unset() -> None:
    # 0 means "unload as soon as this finishes"; a falsy-check bug here
    # would silently turn that into the server default.
    client, registry = make_registry_client()
    client.post("/v1/embeddings", json={"input": "x", "model": "org/once", "keep_alive": 0})
    assert registry.calls[0]["keep_alive"] == 0


def test_omitted_load_options_are_passed_as_unset() -> None:
    client, registry = make_registry_client()
    client.post("/v1/embeddings", json={"input": "x", "model": "org/new"})
    call = registry.calls[0]
    assert call["pooling"] is None and call["max_seq_len"] is None and call["keep_alive"] is None
    # The registry's own "unset" for dtype is AUTO, not None.
    assert call["dtype"] is Dtype.AUTO


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (PoolingRequiredError("org/x"), 400),
        (PoolingConflictError("org/x", Pooling.CLS, Pooling.MEAN), 409),
        (ModelPlacementError("org/x", [(0, "OutOfMemoryError: no room")]), 503),
        (UnsupportedWeightFormatError("org/x", ["pytorch_model.bin"]), 400),
        (WeightFormatUnverifiableError("org/x", ConnectionError("hub down")), 503),
    ],
)
def test_registry_errors_map_to_status_in_the_standard_envelope(
    error: Exception, status: int
) -> None:
    client, _ = make_registry_client(FakeRegistry(raises=error))
    response = client.post("/v1/embeddings", json={"input": "x", "model": "org/x"})

    assert response.status_code == status
    body = response.json()
    _assert_envelope(body)
    assert "org/x" in body["error"]["message"]
    assert "Traceback" not in str(body)
    expected_type = "invalid_request_error" if status < 500 else "server_error"
    assert body["error"]["type"] == expected_type


def test_registry_errors_map_the_same_way_on_the_tei_endpoint() -> None:
    client, _ = make_registry_client(
        FakeRegistry(raises=PoolingConflictError("org/x", Pooling.CLS, Pooling.MEAN))
    )
    response = client.post("/embed", json={"inputs": "x", "model": "org/x"})
    assert response.status_code == 409
    _assert_envelope(response.json())


def test_an_unmapped_registry_error_is_not_leaked_to_the_caller() -> None:
    from embedx.registry import RegistryError

    client, _ = make_registry_client(FakeRegistry(raises=RegistryError("secret internals")))
    response = client.post("/v1/embeddings", json={"input": "x", "model": "org/x"})
    assert response.status_code == 500
    assert response.json()["error"]["message"] == "internal server error"
    assert "secret internals" not in str(response.json())


# --------------------------------------------------------------------------- #
# /info and /health against the registry
# --------------------------------------------------------------------------- #


def test_info_with_no_models_loaded_is_an_empty_list_not_an_error() -> None:
    client, _ = make_registry_client()
    response = client.get("/info")
    assert response.status_code == 200
    assert response.json()["models"] == []
    assert response.json()["default_keep_alive_s"] == DEFAULT_KEEP_ALIVE_S


def test_info_renders_one_and_many_loaded_models() -> None:
    statuses = [
        make_status("a/model", pooling=Pooling.CLS, device_indices=(0, 1), ref_count=2),
        make_status("b/model", dtype=Dtype.BFLOAT16, idle_s=9.0),
    ]
    client, _ = make_registry_client(FakeRegistry(loaded=statuses[:1]))
    assert client.get("/info").json()["models"] == [
        {
            "model_id": "a/model",
            "pooling": "cls",
            "dtype": "auto",
            "devices": [0, 1],
            "idle_s": 1.5,
            "last_used_epoch_s": 1_700_000_000.0,
            "ref_count": 2,
            "truncated_count": 0,
        }
    ]

    client, _ = make_registry_client(FakeRegistry(loaded=statuses))
    body = client.get("/info").json()
    assert [entry["model_id"] for entry in body["models"]] == ["a/model", "b/model"]
    assert body["models"][1]["dtype"] == "bfloat16"
    assert body["models"][1]["idle_s"] == 9.0


def test_info_lists_without_acquiring_a_reference() -> None:
    # Polling /info must not pin models against the reaper.
    client, registry = make_registry_client(FakeRegistry(loaded=[make_status("a/model")]))
    assert client.get("/info").status_code == 200
    assert registry.list_loaded_calls == 1
    assert registry.calls == [], "/info must never call acquire()"
    assert registry.entered == 0


def test_health_touches_no_registry_state() -> None:
    client, registry = make_registry_client(FakeRegistry(loaded=[make_status("a/model")]))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert registry.calls == []
    assert registry.list_loaded_calls == 0
    assert registry.entered == 0


def test_create_app_requires_exactly_one_model_source() -> None:
    settings = make_settings()
    with pytest.raises(ValueError, match="exactly one"):
        create_app(settings)
    with pytest.raises(ValueError, match="exactly one"):
        create_app(settings, FakeEngine(), registry=FakeRegistry())  # type: ignore[arg-type]


def test_end_to_end_against_a_real_registry() -> None:
    """The fake above could drift from ModelRegistry; this catches that.

    A real registry with a stubbed backend factory, wired through
    create_app: proves the signatures line up, that a request holds and
    then releases a real reference, and that a real PoolingConflictError
    reaches the caller as a 409.
    """
    # tests/ is not a package; pytest puts the test directory on sys.path.
    from test_registry import StubFactory, make_devices, safetensors_listing

    factory = StubFactory()
    registry = ModelRegistry(
        make_devices(2),
        backend_factory=factory,
        weight_file_lister=safetensors_listing,
    )
    app = create_app(make_settings(), registry=registry)
    client = TestClient(app, raise_server_exceptions=False)

    assert client.get("/info").json()["models"] == []

    ok = client.post(
        "/v1/embeddings", json={"input": ["a", "b"], "model": "org/real", "pooling": "mean"}
    )
    assert ok.status_code == 200
    assert factory.model_loads == ["org/real"]

    loaded = client.get("/info").json()["models"]
    assert [entry["model_id"] for entry in loaded] == ["org/real"]
    assert loaded[0]["pooling"] == "mean"
    assert loaded[0]["devices"] == [0, 1]
    assert loaded[0]["ref_count"] == 0, "the request must have released its reference"

    # First load with no pooling would have been 400; this one is resident,
    # so omitting pooling reuses it and a different pooling is a 409.
    reuse = client.post("/v1/embeddings", json={"input": "x", "model": "org/real"})
    assert reuse.status_code == 200
    conflict = client.post(
        "/v1/embeddings", json={"input": "x", "model": "org/real", "pooling": "cls"}
    )
    assert conflict.status_code == 409
    _assert_envelope(conflict.json())

    missing_pooling = client.post("/v1/embeddings", json={"input": "x", "model": "org/other"})
    assert missing_pooling.status_code == 400
    assert factory.model_loads == ["org/real"], "the refused load must not have happened"


def test_info_reports_truncation_summed_across_a_model_devices() -> None:
    # Equivalent in spirit to the old per-device assertion, against the
    # per-model shape: one model on two devices reports 7 + 5, not [7, 5].
    # Truncation is silent everywhere else, so /info is where it surfaces.
    client = make_client(TruncatingEngine())
    body = client.get("/info").json()
    assert body["models"][0]["truncated_count"] == 12


def test_info_reports_truncation_from_a_real_registry() -> None:
    from test_registry import StubFactory, make_devices, safetensors_listing

    factory = StubFactory()
    registry = ModelRegistry(
        make_devices(2), backend_factory=factory, weight_file_lister=safetensors_listing
    )
    app = create_app(make_settings(), registry=registry)
    client = TestClient(app, raise_server_exceptions=False)

    client.post("/v1/embeddings", json={"input": "x", "model": "org/real", "pooling": "mean"})
    assert client.get("/info").json()["models"][0]["truncated_count"] == 0

    # Two backends, one per device, each having truncated some inputs.
    for backend, truncated in zip(factory.created, (3, 4), strict=True):
        backend.truncated_count = truncated
    assert client.get("/info").json()["models"][0]["truncated_count"] == 7


def test_unverifiable_weight_format_is_503_not_400() -> None:
    # The distinction that justifies the separate exception type: pickle-only
    # weights will never load (400), but an unreachable hub may well clear,
    # and 4xx would tell the caller not to bother retrying.
    client, _ = make_registry_client(
        FakeRegistry(raises=WeightFormatUnverifiableError("org/x", ConnectionError("hub down")))
    )
    unverifiable = client.post("/v1/embeddings", json={"input": "x", "model": "org/x"})
    assert unverifiable.status_code == 503

    client, _ = make_registry_client(
        FakeRegistry(raises=UnsupportedWeightFormatError("org/x", ["pytorch_model.bin"]))
    )
    pickled = client.post("/v1/embeddings", json={"input": "x", "model": "org/x"})
    assert pickled.status_code == 400
