"""CPU-safe tests for hf.py internals — no torch, no gpu marker.

HFBackend.__init__ needs CUDA, so these build a bare instance with
`__new__` and stub only the attributes the code under test touches.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
import types
from typing import Any

import numpy as np
import pytest

from embedx.backend.hf import HFBackend, TokenLengthCache, _is_st_checkpoint


class CountingTokenizer:
    def __init__(self, counter: list[int] | None = None) -> None:
        self._counter = counter if counter is not None else [0]

    @property
    def calls(self) -> int:
        return self._counter[0]

    def __deepcopy__(self, memo: dict) -> CountingTokenizer:
        # `_install_tokenizer_guards` copies the tokenizer for the length
        # path, so a plain deepcopy would split the counter in two and hide
        # exactly the re-tokenization these tests exist to catch. The clone
        # is an independent object sharing ONE counter.
        return CountingTokenizer(self._counter)

    def __call__(self, text: Any, truncation: bool = False, padding: bool = False) -> dict:
        assert isinstance(text, str), "batch re-tokenization would be a regression too"
        self._counter[0] += 1
        return {"input_ids": [0] * len(text)}  # 1 token per character


def make_stub_backend(
    max_seq_length: int = 8,
    cache: TokenLengthCache | None = None,
    tokenizer: Any = None,
    st_model: Any = None,
) -> HFBackend:
    backend = HFBackend.__new__(HFBackend)  # skip __init__: no torch here
    backend._tokenizer = tokenizer if tokenizer is not None else CountingTokenizer()
    backend._length_cache = cache if cache is not None else TokenLengthCache()
    backend.max_seq_length = max_seq_length
    backend.truncated_count = 0
    backend.device_index = 0
    backend.model_id = "stub/model"
    backend._st_model = st_model
    backend.normalize = False
    backend.dim = 8
    # The real thing, not a hand-set pair of locks: the wiring is what is
    # under test in the thread-safety cases below.
    backend._install_tokenizer_guards()
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


# --------------------------------------------------------------------------- #
# Tokenizer thread safety
#
# The FakeBackend the engine's concurrency tests use has no Rust tokenizer,
# so `RuntimeError: Already borrowed` is structurally invisible to them. These
# reproduce the borrow rule itself on CPU.
# --------------------------------------------------------------------------- #


class BorrowCheckedTokenizer:
    """A stub with a fast tokenizer's borrow rule and nothing else.

    HF fast tokenizers wrap a Rust object with interior mutability: a call
    mutates truncation/padding state, so a second call that overlaps it on
    the same instance fails the borrow check. `_flag_lock` guards only the
    flag, never the call, so overlap is DETECTED rather than serialised —
    the opposite of what the code under test must do.
    """

    def __init__(self) -> None:
        self._borrowed = False
        self._flag_lock = threading.Lock()

    def __deepcopy__(self, memo: dict) -> BorrowCheckedTokenizer:
        return BorrowCheckedTokenizer()  # an independent Rust object

    def __call__(self, text: Any, **kwargs: Any) -> dict:
        with self._flag_lock:
            if self._borrowed:
                raise RuntimeError("Already borrowed")
            self._borrowed = True
        try:
            time.sleep(0.001)  # widen the window; without it the race is rare
            payload = text if isinstance(text, str) else "".join(text)
            return {"input_ids": [0] * len(payload)}
        finally:
            with self._flag_lock:
                self._borrowed = False


class UncopyableTokenizer(BorrowCheckedTokenizer):
    """Refuses to be copied, like any tokenizer holding an unpicklable ref."""

    def __deepcopy__(self, memo: dict) -> BorrowCheckedTokenizer:
        raise TypeError("cannot pickle this tokenizer")


class FakeSTModel:
    """Stands in for SentenceTransformer: tokenizes inside `encode`.

    That is the whole reason the ST path locks around the forward pass —
    there is no seam between the tokenization and the compute.
    """

    def __init__(self, tokenizer: Any, dim: int = 8) -> None:
        self._tokenizer = tokenizer
        self._dim = dim

    def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        self._tokenizer(texts)
        return np.zeros((len(texts), self._dim), dtype=np.float32)


def _hammer(backend: HFBackend, workers: int = 8, rounds: int = 12) -> list[BaseException]:
    """Concurrent `embed` and `length_fn`, the two contending call paths.

    Texts are unique per call so the shared length cache never short-circuits
    the tokenizer, which is what has to be exercised.
    """
    errors: list[BaseException] = []
    start = threading.Barrier(2 * workers)

    def embedder(w: int) -> None:
        start.wait()
        try:
            for r in range(rounds):
                backend.embed([f"embed {w} round {r}"])
        except BaseException as exc:  # recorded, then re-asserted at the join
            errors.append(exc)

    def measurer(w: int) -> None:
        start.wait()
        try:
            for r in range(rounds):
                backend.length_fn(f"length {w} round {r}")
        except BaseException as exc:  # recorded, then re-asserted at the join
            errors.append(exc)

    threads = [threading.Thread(target=embedder, args=(w,)) for w in range(workers)]
    threads += [threading.Thread(target=measurer, args=(w,)) for w in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return errors


def test_concurrent_embed_and_length_fn_never_borrow_one_tokenizer() -> None:
    # Fails without the guards with `RuntimeError: Already borrowed`, which
    # is the production symptom this reproduces.
    tokenizer = BorrowCheckedTokenizer()
    backend = make_stub_backend(max_seq_length=1000, tokenizer=tokenizer)
    backend._st_model = FakeSTModel(backend._tokenizer)
    assert _hammer(backend) == []


def test_length_path_holds_a_tokenizer_of_its_own() -> None:
    # The property that keeps batching off the lock that spans inference.
    backend = make_stub_backend(tokenizer=BorrowCheckedTokenizer())
    assert backend._length_tokenizer is not backend._tokenizer
    assert backend._length_lock is not backend._tokenizer_lock


def test_uncopyable_tokenizer_degrades_to_one_lock_but_stays_correct(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="embedx.backend.hf"):
        backend = make_stub_backend(max_seq_length=1000, tokenizer=UncopyableTokenizer())
    assert "stub/model" in caplog.text
    assert "cannot pickle this tokenizer" in caplog.text

    # One tokenizer, so necessarily one lock — sharing the instance under two
    # locks would guard nothing.
    assert backend._length_tokenizer is backend._tokenizer
    assert backend._length_lock is backend._tokenizer_lock

    # And the fallback must not deadlock: `embed` calls `_count_truncations`,
    # which takes that same non-reentrant lock.
    backend._st_model = FakeSTModel(backend._tokenizer)
    assert _hammer(backend, workers=4, rounds=6) == []


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
