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

from embedx.backend.hf import HFBackend, TokenLengthCache, _is_st_checkpoint


class CountingTokenizer:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, text: Any, truncation: bool = False, padding: bool = False) -> dict:
        assert isinstance(text, str), "batch re-tokenization would be a regression too"
        self.calls += 1
        return {"input_ids": [0] * len(text)}  # 1 token per character


def make_stub_backend(
    max_seq_length: int = 8,
    cache: TokenLengthCache | None = None,
    tokenizer: CountingTokenizer | None = None,
) -> HFBackend:
    backend = HFBackend.__new__(HFBackend)  # skip __init__: no torch here
    backend._tokenizer = tokenizer if tokenizer is not None else CountingTokenizer()
    backend._length_cache = cache if cache is not None else TokenLengthCache()
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


def test_cache_shared_across_backends() -> None:
    # The batcher only calls backends[0].length_fn; the other device's
    # _count_truncations must read the same entries, not re-tokenize.
    # Without the shared cache this asserts 2 * len(texts).
    cache = TokenLengthCache()
    tokenizer = CountingTokenizer()
    first = make_stub_backend(cache=cache, tokenizer=tokenizer)
    second = make_stub_backend(cache=cache, tokenizer=tokenizer)

    texts = ["hello", "hi", "a longer text"]
    for text in texts:
        first.length_fn(text)
    second._count_truncations(texts)
    assert tokenizer.calls == len(texts)


def test_byte_cap_evicts_before_entry_cap() -> None:
    cache = TokenLengthCache(max_entries=100, max_bytes=50)
    cache.put("a" * 30, 30)
    assert cache.get("a" * 30) == 30
    # 30 + 30 bytes crosses max_bytes long before 100 entries: evict.
    cache.put("b" * 30, 30)
    assert cache.get("a" * 30) is None
    assert cache.get("b" * 30) == 30


def test_entry_cap_still_applies() -> None:
    cache = TokenLengthCache(max_entries=2, max_bytes=10_000)
    cache.put("a", 1)
    cache.put("b", 1)
    cache.put("c", 1)  # third entry hits the count cap: evict
    assert cache.get("a") is None
    assert cache.get("c") == 1


def test_backend_without_explicit_cache_still_works() -> None:
    import inspect

    # __init__ needs CUDA, so the default is checked on the signature and
    # the private-cache behavior on a stub.
    assert inspect.signature(HFBackend.__init__).parameters["length_cache"].default is None
    backend = make_stub_backend()
    assert backend.length_fn("hello") == 5
    assert backend.length_fn("hello") == 5  # second read from private cache
    assert backend._tokenizer.calls == 1  # type: ignore[attr-defined]


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
