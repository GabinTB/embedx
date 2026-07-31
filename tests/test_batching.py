"""Tests for length-sorted token-budget batching (task 03)."""

from __future__ import annotations

import random

import pytest

from embedx.engine import make_batches


def _indexed(texts: list[str]) -> list[tuple[int, str]]:
    return list(enumerate(texts))


def _padded_cost(batch: list[tuple[int, str]]) -> int:
    return len(batch) * max(len(text) for _, text in batch)


def _assert_invariants(texts: list[str], budget: int) -> None:
    batches = list(make_batches(_indexed(texts), budget))

    for batch in batches:
        assert batch, "empty batch emitted"
        oversized_alone = len(batch) == 1 and len(batch[0][1]) > budget
        assert _padded_cost(batch) <= budget or oversized_alone

    flat = [pair for batch in batches for pair in batch]
    assert sorted(index for index, _ in flat) == list(range(len(texts)))
    for index, text in flat:
        assert text == texts[index], "index/text pairing broken"


def test_no_batch_exceeds_budget() -> None:
    texts = ["a" * n for n in [1, 4, 4, 7, 2, 12, 3]]
    for batch in make_batches(_indexed(texts), 12):
        assert _padded_cost(batch) <= 12


def test_oversized_item_forms_batch_of_one() -> None:
    texts = ["ab", "x" * 50, "cd"]
    batches = list(make_batches(_indexed(texts), 10))
    oversized = [batch for batch in batches if any(len(t) > 10 for _, t in batch)]
    assert oversized == [[(1, "x" * 50)]]


def test_every_index_exactly_once_pairing_preserved() -> None:
    texts = ["hello", "hi", "a much longer sentence", "", "mid-size text"]
    _assert_invariants(texts, 20)


def test_stable_for_equal_lengths() -> None:
    # Equal-length texts must keep input order (stable sort).
    texts = ["aa", "bb", "cc", "dd"]
    batches = list(make_batches(_indexed(texts), 4))
    assert batches == [[(0, "aa"), (1, "bb")], [(2, "cc"), (3, "dd")]]


def test_hand_computed_regression() -> None:
    # Lengths 5, 3, 2, 9, 3 with budget 8. Stable ascending sort gives
    # lengths [2, 3, 3, 5, 9] = indices [2, 1, 4, 0, 3]. Greedy fill:
    #   [2]        cost 1*2=2, then +idx1: 2*3=6 <= 8 -> [2, 1]
    #   +idx4:     3*3=9  > 8 -> flush [2, 1], start [4]
    #   +idx0:     2*5=10 > 8 -> flush [4], start [0]
    #   +idx3:     2*9=18 > 8 -> flush [0], start [3] (9 > 8: oversized alone)
    texts = ["e" * 5, "c" * 3, "a" * 2, "z" * 9, "d" * 3]
    batches = list(make_batches(_indexed(texts), 8))
    indices = [[index for index, _ in batch] for batch in batches]
    assert indices == [[2, 1], [4], [0], [3]]


def test_property_random_inputs_hold_invariants() -> None:
    rng = random.Random(42)
    for _ in range(200):
        n_texts = rng.randint(0, 40)
        texts = ["x" * rng.randint(0, 30) for _ in range(n_texts)]
        budget = rng.randint(1, 60)
        _assert_invariants(texts, budget)


def test_empty_input_yields_nothing() -> None:
    assert list(make_batches([], 10)) == []


def test_custom_length_fn() -> None:
    # With a constant length_fn of 1, the budget becomes a plain item count.
    texts = ["short", "a much longer text", "x"] * 2
    batches = list(make_batches(_indexed(texts), 2, length_fn=lambda _: 1))
    assert [len(batch) for batch in batches] == [2, 2, 2]


@pytest.mark.parametrize("budget", [0, -1])
def test_invalid_budget_raises_eagerly(budget: int) -> None:
    with pytest.raises(ValueError, match="max_batch_tokens"):
        make_batches([(0, "a")], budget)
