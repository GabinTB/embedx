"""Tests for the converging work-stealing scheduler (task 04)."""

from __future__ import annotations

import random
import threading

import numpy as np
import pytest

from embedx.backend import FakeBackend
from embedx.engine import Scheduler, Side, assign_sides, make_batches


def _run_workers(
    scheduler: Scheduler,
    sides: list[Side],
    claim_args: random.Random | tuple[int, int | None],
) -> list[list[list[tuple[int, str]]]]:
    """Run one real thread per side; return each worker's claimed batches."""
    claimed: list[list[list[tuple[int, str]]]] = [[] for _ in sides]

    def worker(w: int, side: Side) -> None:
        rng = random.Random(w) if isinstance(claim_args, random.Random) else None
        while True:
            if rng is not None:
                budget = rng.randint(1, 50)
                max_items = rng.randint(1, 5) if rng.random() < 0.5 else None
            else:
                budget, max_items = claim_args  # type: ignore[misc]
            batch = scheduler.claim(side, budget, max_items)
            if batch is None:
                return
            claimed[w].append(batch)

    threads = [threading.Thread(target=worker, args=(w, side)) for w, side in enumerate(sides)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return claimed


def _flat_indices(claimed: list[list[list[tuple[int, str]]]]) -> list[int]:
    return [index for batches in claimed for batch in batches for index, _ in batch]


def test_exactly_once_under_threads_randomized() -> None:
    rng = random.Random(11)
    for _ in range(25):
        n = rng.randint(0, 200)
        texts = ["x" * rng.randint(0, 40) for _ in range(n)]
        scheduler = Scheduler(enumerate(texts))
        sides = assign_sides(rng.randint(1, 8))
        claimed = _run_workers(scheduler, sides, rng)
        assert sorted(_flat_indices(claimed)) == list(range(n))
        assert scheduler.done
        assert scheduler.remaining == 0


def test_meeting_point_no_gap_no_overlap() -> None:
    # Budget 4 over four length-2 items: each side's claim takes exactly 2,
    # and the frontiers meet in the middle with nothing lost or doubled.
    scheduler = Scheduler(enumerate(["aa", "bb", "cc", "dd"]))
    assert scheduler.claim(Side.SHORT, 4) == [(0, "aa"), (1, "bb")]
    assert scheduler.claim(Side.LONG, 4) == [(2, "cc"), (3, "dd")]
    assert scheduler.done
    assert scheduler.claim(Side.SHORT, 4) is None
    assert scheduler.claim(Side.LONG, 4) is None

    # Odd remainder: one item left in the middle after both sides claim.
    scheduler = Scheduler(enumerate(["a", "b", "c", "d", "e"]))
    assert scheduler.claim(Side.SHORT, 2) == [(0, "a"), (1, "b")]
    assert scheduler.claim(Side.LONG, 2) == [(3, "d"), (4, "e")]
    assert scheduler.remaining == 1
    assert scheduler.claim(Side.LONG, 2) == [(2, "c")]
    assert scheduler.done

    # One side racing ahead: SHORT drains everything, LONG must see None.
    scheduler = Scheduler(enumerate(["a", "b", "c"]))
    assert scheduler.claim(Side.SHORT, 100) == [(0, "a"), (1, "b"), (2, "c")]
    assert scheduler.claim(Side.LONG, 100) is None


def test_long_claims_come_in_ascending_order() -> None:
    scheduler = Scheduler(enumerate(["a", "bb", "ccc", "dddd"]))
    batch = scheduler.claim(Side.LONG, 8)
    assert batch == [(2, "ccc"), (3, "dddd")]  # natural ascending order


def test_reassembly_matches_single_threaded_reference() -> None:
    rng = random.Random(5)
    backend = FakeBackend(dim=8)
    for _ in range(10):
        n = rng.randint(1, 120)
        texts = ["".join(rng.choices("abcdef ", k=rng.randint(0, 30))) for _ in range(n)]
        reference = backend.embed(texts)

        scheduler = Scheduler(enumerate(texts))
        out = np.zeros((n, 8), dtype=np.float32)

        def worker(side: Side, budget: int, scheduler: Scheduler, out: np.ndarray) -> None:
            while (batch := scheduler.claim(side, budget)) is not None:
                vectors = backend.embed([text for _, text in batch])
                for (index, _), vector in zip(batch, vectors, strict=True):
                    out[index] = vector

        threads = [
            threading.Thread(target=worker, args=(side, budget, scheduler, out))
            for side, budget in [(Side.LONG, 64), (Side.SHORT, 24)]
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        np.testing.assert_array_equal(out, reference)


def test_stress_many_workers_tiny_budget() -> None:
    for _ in range(5):
        n = 300
        scheduler = Scheduler(enumerate(["x"] * n))
        sides = assign_sides(12)
        claimed = _run_workers(scheduler, sides, (1, None))
        assert sorted(_flat_indices(claimed)) == list(range(n))
        assert all(len(batch) == 1 for batches in claimed for batch in batches)
        assert scheduler.done


def test_assign_sides() -> None:
    assert assign_sides(1) == [Side.SHORT]
    assert assign_sides(2) == [Side.LONG, Side.SHORT]
    assert assign_sides(3) == [Side.LONG, Side.LONG, Side.SHORT]
    assert assign_sides(4) == [Side.LONG, Side.LONG, Side.SHORT, Side.SHORT]
    with pytest.raises(ValueError, match="n_workers"):
        assign_sides(0)


def test_exhausted_scheduler_returns_none_repeatedly() -> None:
    empty = Scheduler([])
    assert empty.done
    assert empty.remaining == 0
    drained = Scheduler([(0, "abc")])
    assert drained.claim(Side.LONG, 10) == [(0, "abc")]
    for scheduler in (empty, drained):
        for side in (Side.SHORT, Side.LONG):
            for _ in range(3):
                assert scheduler.claim(side, 10) is None


# --------------------------------------------------------------------------- #
# Simulated makespan: the test that justifies the design. Deterministic,
# thread-free — worker clocks advance analytically using the FakeBackend
# cost model, so it is stable in CI.
# --------------------------------------------------------------------------- #


def _batch_cost(backend: FakeBackend, batch: list[tuple[int, str]]) -> float:
    return sum(backend.latency_s + backend.latency_per_token * len(text) for _, text in batch)


def test_converging_scheduler_beats_static_split_makespan() -> None:
    rng = random.Random(123)
    # Skewed workload: many short texts, a heavy tail of long ones.
    lengths = [rng.randint(1, 10) for _ in range(180)] + [rng.randint(200, 300) for _ in range(30)]
    rng.shuffle(lengths)
    items = list(enumerate("x" * n for n in lengths))

    fast = FakeBackend(dim=4, latency_s=0.01, latency_per_token=0.001)
    slow = FakeBackend(dim=4, latency_s=0.02, latency_per_token=0.002)  # 2x slower
    fast_budget, slow_budget = 400, 120

    # Converging: advance whichever worker's clock is lowest; it claims and
    # its clock grows by the batch's simulated cost.
    scheduler = Scheduler(items)
    workers = [(Side.LONG, fast_budget, fast), (Side.SHORT, slow_budget, slow)]
    clocks = [0.0, 0.0]
    active = [True, True]
    converging_indices: list[int] = []
    while any(active):
        w = min((i for i in range(2) if active[i]), key=lambda i: clocks[i])
        side, budget, backend = workers[w]
        batch = scheduler.claim(side, budget)
        if batch is None:
            active[w] = False
            continue
        clocks[w] += _batch_cost(backend, batch)
        converging_indices.extend(index for index, _ in batch)
    converging_makespan = max(clocks)

    # Static baseline: split the same sorted list into two contiguous halves
    # up front — fast worker gets the long half (the favorable assignment) —
    # each half batched with make_batches.
    sorted_items = sorted(items, key=lambda item: len(item[1]))
    mid = len(sorted_items) // 2
    static_indices: list[int] = []
    static_clocks = []
    for half, budget, backend in [
        (sorted_items[mid:], fast_budget, fast),
        (sorted_items[:mid], slow_budget, slow),
    ]:
        clock = 0.0
        for batch in make_batches(half, budget):
            clock += _batch_cost(backend, batch)
            static_indices.extend(index for index, _ in batch)
        static_clocks.append(clock)
    static_makespan = max(static_clocks)

    # Same work either way, but the converging schedule finishes materially
    # sooner because neither worker ever idles while work remains.
    assert sorted(converging_indices) == sorted(static_indices) == list(range(len(items)))
    assert converging_makespan < 0.85 * static_makespan
