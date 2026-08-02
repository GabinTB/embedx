"""GPU tests for HFBackend (task 08). All gpu-marked, skipped in CPU CI.

Heavy imports (torch, transformers, sentence-transformers) stay inside
fixtures and tests so collection works without the gpu extra installed.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import pytest

from embedx.config import Dtype, Pooling, Settings

pytestmark = pytest.mark.gpu

ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# A plain (non-sentence-transformers) checkpoint, so this exercises the
# AutoModel path and embedx's own pooling rather than the ST module stack.
# NOT prajjwal1/bert-tiny: its config.json predates the `model_type` key, so
# transformers >=5 cannot resolve it through AutoConfig/AutoModel/AutoTokenizer
# at all. This is the Google checkpoint that one is a re-upload of.
PLAIN_MODEL = "google/bert_uncased_L-2_H-128_A-2"
MAX_LEN = 128


@pytest.fixture(scope="module")
def torch() -> Any:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    return torch


def make_backend(model_id: str, pooling: Pooling, **overrides: Any) -> Any:
    from embedx.backend.hf import HFBackend

    kwargs: dict[str, Any] = {
        "device_index": 0,
        "normalize": False,
        # FLOAT32 keeps reference comparisons tight; AUTO is covered by use.
        "dtype": Dtype.FLOAT32,
        "max_seq_length": MAX_LEN,
    }
    kwargs.update(overrides)
    return HFBackend(model_id, pooling=pooling, **kwargs)


def _reference_hidden(torch: Any, texts: list[str]) -> tuple[Any, Any]:
    """last_hidden_state and attention mask from the plain model itself."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(PLAIN_MODEL)
    model = AutoModel.from_pretrained(PLAIN_MODEL).to("cuda:0").eval()
    encoded = tokenizer(
        texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt"
    ).to("cuda:0")
    with torch.inference_mode():
        hidden = model(**encoded).last_hidden_state
    return hidden, encoded["attention_mask"]


def test_st_model_loads_and_dim_matches_config(torch: Any) -> None:
    backend = make_backend(ST_MODEL, Pooling.MEAN)
    assert backend.dim == 384  # MiniLM-L6-v2 hidden size
    out = backend.embed(["hello", "world"])
    assert out.shape == (2, 384)
    assert out.dtype == np.float32


@pytest.mark.parametrize("pooling", [Pooling.MEAN, Pooling.CLS, Pooling.LAST_TOKEN])
def test_pooling_matches_hand_computed_reference(torch: Any, pooling: Pooling) -> None:
    # Different lengths on purpose: padding is where wrong pooling hides.
    texts = ["a tiny sentence", "a much longer sentence with many more tokens inside it"]
    ours = make_backend(PLAIN_MODEL, pooling).embed(texts)

    hidden, mask = _reference_hidden(torch, texts)
    if pooling is Pooling.CLS:
        expected = hidden[:, 0]
    elif pooling is Pooling.MEAN:
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        expected = (hidden * weights).sum(dim=1) / weights.sum(dim=1)
    else:  # LAST_TOKEN: last non-padding position per row
        positions = torch.arange(mask.shape[1], device=mask.device)
        last = (mask * positions).argmax(dim=1)
        expected = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
    np.testing.assert_allclose(ours, expected.float().cpu().numpy(), atol=1e-4)


def test_masked_mean_differs_from_naive_unmasked_mean(torch: Any) -> None:
    # The short row gets heavy padding; a naive .mean(dim=1) averages it in.
    texts = ["hi", "a considerably longer sentence that forces real padding onto the first row"]
    ours = make_backend(PLAIN_MODEL, Pooling.MEAN).embed(texts)

    hidden, _mask = _reference_hidden(torch, texts)
    naive = hidden.mean(dim=1).float().cpu().numpy()
    assert not np.allclose(ours[0], naive[0], atol=1e-3)  # the mask is the point


def test_matches_sentence_transformers_reference(torch: Any) -> None:
    sentence_transformers = pytest.importorskip("sentence_transformers")
    texts = ["a cat sat on the mat", "quantum computing in finance", "short"]
    ours = make_backend(ST_MODEL, Pooling.MEAN, normalize=True).embed(texts)
    reference = sentence_transformers.SentenceTransformer(ST_MODEL, device="cuda:0").encode(
        texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )
    np.testing.assert_allclose(ours, reference, atol=1e-4)


def test_truncated_count(torch: Any) -> None:
    backend = make_backend(ST_MODEL, Pooling.MEAN, max_seq_length=16)
    backend.embed(["short text"])
    assert backend.truncated_count == 0
    backend.embed(["word " * 200])
    assert backend.truncated_count == 1
    backend.embed(["word " * 200, "again " * 300, "ok"])
    assert backend.truncated_count == 3


def test_concurrent_requests_do_not_raise_already_borrowed(torch: Any) -> None:
    """Regression test for `RuntimeError: Already borrowed` under load.

    Found in production with 8 concurrent clients against this checkpoint.
    HF fast tokenizers are Rust objects with interior mutability, and the
    engine's per-backend lock does not cover them: `Scheduler.__init__`
    calls `length_fn` on the requesting thread, outside that lock, while
    another request's worker is inside `embed` on the same backend.

    Driven through a real `Engine` rather than the backend alone, because
    that interleaving is the bug — a backend hammered directly would not
    reproduce the call site that escapes the lock.
    """
    from embedx.backend.factory import engine_from_backends
    from embedx.gpu.discovery import discover_devices

    devices = [device for device in discover_devices() if device.index == 0]
    engine = engine_from_backends([make_backend(ST_MODEL, Pooling.MEAN)], devices, Settings())

    clients = 8
    errors: list[BaseException] = []
    start = threading.Barrier(clients)

    def client(c: int) -> None:
        start.wait()
        try:
            for r in range(4):
                # Varied, unique lengths: distinct texts defeat the shared
                # length cache, so every request really does tokenize, and
                # mixed lengths keep batches from lining up identically.
                texts = [f"client {c} round {r} item {i} " + "word " * (i % 7) for i in range(16)]
                out = engine.embed(texts)
                assert out.shape == (len(texts), 384)
        except BaseException as exc:  # recorded, then re-asserted at the join
            errors.append(exc)

    threads = [threading.Thread(target=client, args=(c,)) for c in range(clients)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []


def test_length_fn_tokenizer_lengths_clamped_and_monotonic(torch: Any) -> None:
    backend = make_backend(ST_MODEL, Pooling.MEAN, max_seq_length=32)
    raw = len(backend._tokenizer("hello there", truncation=False, padding=False)["input_ids"])
    assert backend.length_fn("hello there") == raw
    assert backend.length_fn("word " * 500) == 32  # clamped to max_seq_length
    lengths = [backend.length_fn("word " * n) for n in (1, 3, 6, 12, 1000)]
    assert lengths == sorted(lengths)
