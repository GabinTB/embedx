"""Tests for length-sorted token-budget batching (task 03)."""

from __future__ import annotations

import random

import pytest

from embedx.engine import make_batches, pack_count


def _indexed(texts: list[str]) -> list[tuple[int, str]]:
    return list(enumerate(texts))


def _padded_cost(batch: list[tuple[int, str]]) -> int:
    return len(batch) * max(len(text) for _, text in batch)


def _assert_invariants(texts: list[str], budget: int, max_items: int | None = None) -> None:
    batches = list(make_batches(_indexed(texts), budget, max_batch_items=max_items))

    for batch in batches:
        assert batch, "empty batch emitted"
        if max_items is not None:
            assert len(batch) <= max_items
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
        max_items = rng.randint(1, 8) if rng.random() < 0.5 else None
        _assert_invariants(texts, budget, max_items)


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


# --------------------------------------------------------------------------- #
# max_batch_items
# --------------------------------------------------------------------------- #


def test_max_batch_items_bounds_empty_strings() -> None:
    # Zero-length items have zero padded cost, so without the cap they all
    # land in one batch; the cap must bound the count.
    texts = [""] * 100
    assert len(list(make_batches(_indexed(texts), 10))) == 1
    batches = list(make_batches(_indexed(texts), 10, max_batch_items=8))
    assert all(len(batch) <= 8 for batch in batches)
    assert sum(len(batch) for batch in batches) == 100


def test_max_batch_items_binds_before_budget() -> None:
    texts = ["ab"] * 6
    batches = list(make_batches(_indexed(texts), 100, max_batch_items=2))
    assert [len(batch) for batch in batches] == [2, 2, 2]


@pytest.mark.parametrize("cap", [0, -1])
def test_invalid_max_batch_items_raises_eagerly(cap: int) -> None:
    with pytest.raises(ValueError, match="max_batch_items"):
        make_batches([(0, "a")], 5, max_batch_items=cap)


# --------------------------------------------------------------------------- #
# pack_count
# --------------------------------------------------------------------------- #


def test_pack_count_ascending() -> None:
    lengths = [2, 3, 3, 5, 9]
    # 1*2=2, 2*3=6 <= 8, then 3*3=9 > 8 -> 2 items.
    assert pack_count(lengths, 0, 5, 1, 8) == 2
    assert pack_count(lengths, 2, 5, 1, 10) == 2  # 2*5=10 ok, 3*9=27 no


def test_pack_count_descending() -> None:
    lengths = [2, 3, 3, 5, 9]
    # From the top the first item is the running max: 18 // 9 = 2 items.
    assert pack_count(lengths, 4, -1, -1, 18) == 2
    assert pack_count(lengths, 4, -1, -1, 45) == 5  # whole span
    # All-zero span: cost-free, take everything.
    assert pack_count([0, 0, 0], 2, -1, -1, 5) == 3


def test_pack_count_at_least_one_when_span_nonempty() -> None:
    assert pack_count([50], 0, 1, 1, 10) == 1
    assert pack_count([50], 0, -1, -1, 10) == 1


def test_pack_count_empty_span_returns_zero() -> None:
    assert pack_count([], 0, 0, 1, 5) == 0
    assert pack_count([1, 2, 3], 1, 1, 1, 5) == 0
    assert pack_count([1, 2, 3], 2, 2, -1, 5) == 0


def test_pack_count_max_items_binds_before_budget() -> None:
    assert pack_count([1] * 10, 0, 10, 1, 100, max_items=3) == 3
    assert pack_count([1] * 10, 9, -1, -1, 100, max_items=3) == 3


def test_pack_count_validates_step_and_budget() -> None:
    with pytest.raises(ValueError, match="step"):
        pack_count([1], 0, 1, 2, 5)
    with pytest.raises(ValueError, match="budget"):
        pack_count([1], 0, 1, 1, 0)


def test_pack_count_matches_make_batches_boundaries() -> None:
    rng = random.Random(7)
    for _ in range(100):
        lengths = sorted(rng.randint(0, 20) for _ in range(rng.randint(1, 30)))
        budget = rng.randint(1, 40)
        max_items = rng.randint(1, 6) if rng.random() < 0.5 else None
        texts = ["x" * n for n in lengths]  # already ascending: sort is identity
        batch_sizes = [
            len(batch) for batch in make_batches(_indexed(texts), budget, max_batch_items=max_items)
        ]
        expected = []
        pos = 0
        while pos < len(lengths):
            count = pack_count(lengths, pos, len(lengths), 1, budget, max_items)
            expected.append(count)
            pos += count
        assert batch_sizes == expected
