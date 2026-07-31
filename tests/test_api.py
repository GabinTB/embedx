"""Tests for the OpenAI-compatible API layer (task 07b). Fake engine only."""

from __future__ import annotations

import base64
import logging

import numpy as np
import pytest
from fastapi.testclient import TestClient

import embedx
from embedx.api import create_app
from embedx.backend import FakeBackend
from embedx.config import Settings
from embedx.gpu.discovery import DeviceInfo

GIB = 2**30


class FakeEngine:
    """Deterministic engine double: FakeBackend vectors, no threads."""

    def __init__(self, dim: int = 8) -> None:
        self.inner = FakeBackend(dim=dim)
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls.append(texts)
        return self.inner.embed(texts)

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
    response = client.post("/embed", json={"inputs": texts})
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


def test_info_reflects_settings_and_devices() -> None:
    client = make_client(model_id="org/model", pooling="cls", normalize=False, dtype="float16")
    body = client.get("/info").json()
    assert body["model_id"] == "org/model"
    assert body["pooling"] == "cls"
    assert body["normalize"] is False
    assert body["dtype"] == "float16"
    assert body["version"] == embedx.__version__
    assert body["devices"] == [
        {"index": 0, "name": "Fake GPU 0", "weight": 1.0, "max_batch_tokens": 16384}
    ]
