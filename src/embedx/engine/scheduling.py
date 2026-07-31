"""Converging work-stealing scheduler over a single length-sorted queue.

All items live in one ascending-sorted list with a shared cursor pair
`(lo, hi)`. SHORT workers claim from the cheap end and advance `lo`; LONG
workers claim from the expensive end and retreat `hi`. The two frontiers
converge until they meet, so load balances itself: a fast worker simply
comes back for more, and nothing is partitioned up front. One lock makes
each claim atomic — read cursors, size the batch with `pack_count`, commit
the new cursor — which is what guarantees every item is claimed exactly
once with no gap or overlap at the meeting point.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from enum import Enum
from math import ceil

from embedx.engine.batching import pack_count


class Side(Enum):
    """Which end of the sorted queue a worker consumes from."""

    SHORT = "short"
    LONG = "long"


class Scheduler:
    """Shared claim queue for N heterogeneous workers.

    Items are stably sorted ascending by `length_fn(text)` once at
    construction. `claim` never blocks and never waits: `None` is the one
    and only exhaustion signal.
    """

    def __init__(
        self,
        items_with_index: Iterable[tuple[int, str]],
        length_fn: Callable[[str], int] = len,
    ) -> None:
        self._items = sorted(items_with_index, key=lambda item: length_fn(item[1]))
        self._lengths = [length_fn(text) for _, text in self._items]
        self._lock = threading.Lock()
        self._lo = 0
        self._hi = len(self._items)

    def claim(
        self, side: Side, budget: int, max_items: int | None = None
    ) -> list[tuple[int, str]] | None:
        """Atomically claim the next batch from `side`, or None if exhausted.

        The `pack_count` call and the cursor commit share one critical
        section: sizing the batch against stale cursors would let two
        converging workers overlap on the same items. `pack_count` is
        O(count), cheap enough to hold the lock through.
        """
        with self._lock:
            if self._lo >= self._hi:
                return None
            if side is Side.SHORT:
                count = pack_count(self._lengths, self._lo, self._hi, 1, budget, max_items)
                batch = self._items[self._lo : self._lo + count]
                self._lo += count
            else:
                count = pack_count(self._lengths, self._hi - 1, self._lo - 1, -1, budget, max_items)
                batch = self._items[self._hi - count : self._hi]
                self._hi -= count
            return batch

    @property
    def remaining(self) -> int:
        """Count of unclaimed items."""
        with self._lock:
            return self._hi - self._lo

    @property
    def done(self) -> bool:
        """True once every item has been claimed."""
        with self._lock:
            return self._lo >= self._hi


def assign_sides(n_workers: int) -> list[Side]:
    """Map workers (fastest first) to queue sides.

    A lone worker gets SHORT — plain ascending length-sorted batching. With
    more, the faster `ceil(n / 2)` take LONG so the expensive tail lands on
    fast devices, and the rest sweep the cheap end.
    """
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")
    if n_workers == 1:
        return [Side.SHORT]
    n_long = ceil(n_workers / 2)
    return [Side.LONG] * n_long + [Side.SHORT] * (n_workers - n_long)
