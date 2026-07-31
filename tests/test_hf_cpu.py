"""CPU-safe tests for hf.py internals — no torch, no gpu marker.

HFBackend.__init__ needs CUDA, so these build a bare instance with
`__new__` and stub only the attributes the code under test touches.
"""

from __future__ import annotations

import logging
import sys
import types
from typing import Any

import pytest

from embedx.backend.hf import HFBackend, _is_st_checkpoint


class CountingTokenizer:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, text: Any, truncation: bool = False, padding: bool = False) -> dict:
        assert isinstance(text, str), "batch re-tokenization would be a regression too"
        self.calls += 1
        return {"input_ids": [0] * len(text)}  # 1 token per character


def make_stub_backend(max_seq_length: int = 8) -> HFBackend:
    backend = HFBackend.__new__(HFBackend)  # skip __init__: no torch here
    backend._tokenizer = CountingTokenizer()
    backend._length_cache = {}
    backend.max_seq_length = max_seq_length
    backend.truncated_count = 0
    backend.device_index = 0
    return backend


def test_each_text_tokenized_once_beyond_encode() -> None:
    backend = make_stub_backend()
    texts = ["hello", "hi", "a much longer text"]
    for text in texts:
        backend.length_fn(text)  # what the scheduler does per request
    backend._count_truncations(texts)  # what embed does before encoding
    assert backend._tokenizer.calls == len(texts)  # type: ignore[attr-defined]


def test_truncation_semantics_preserved() -> None:
    backend = make_stub_backend(max_seq_length=8)
    assert backend.length_fn("x" * 20) == 8  # clamped
    backend._count_truncations(["short", "x" * 20, "y" * 30])
    assert backend.truncated_count == 2
    backend._count_truncations(["z" * 40])
    assert backend.truncated_count == 3


def test_st_detection_fallback_warns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def unreachable(*args: Any, **kwargs: Any) -> bool:
        raise ConnectionError("hub unreachable")

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", types.SimpleNamespace(file_exists=unreachable)
    )
    with caplog.at_level(logging.WARNING, logger="embedx.backend.hf"):
        assert _is_st_checkpoint("org/some-model", None) is False
    assert "org/some-model" in caplog.text
    assert "AutoModel" in caplog.text
    assert "hub unreachable" in caplog.text
