"""Length-sorted token-budget batching.

Batches are filled to a token budget rather than a fixed item count. The
cost of a batch is its *padded* cost, `count * max_len`, because the
backend pads every item in a batch to the longest one. Sorting by length
first packs similar lengths together and minimizes that padding waste.

`pack_count` is the single home of the padded-cost model: `make_batches`
walks it ascending here, and the scheduler (task 04) reuses it descending
to carve work from the expensive end of a shared queue.

Items travel as `(original_index, text)` pairs so downstream code can
restore input order after batches complete in any order (the ordering
invariant lives there, not here).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence


def pack_count(
    lengths: Sequence[int],
    start: int,
    stop: int,
    step: int,
    budget: int,
    max_items: int | None = None,
) -> int:
    """How many consecutive items fit in one padded-cost-bounded batch.

    Walks `lengths` from `start` toward `stop` (exclusive) with `step` of
    +1 or -1, taking items while the padded cost `count * max(taken)` stays
    within `budget`, and never more than `max_items`. Returns 0 on an empty
    span, otherwise at least 1 — an oversized item forms a batch of one and
    nothing is ever dropped.

    `lengths` is assumed ascending-sorted, which the two directions exploit
    differently: ascending (+1), the running max is the last item taken;
    descending (-1), it is the first, so the whole count is a single
    division `budget // lengths[start]` clamped to the span and `max_items`.
    """
    if step not in (1, -1):
        raise ValueError(f"step must be +1 or -1, got {step}")
    if budget <= 0:
        raise ValueError(f"budget must be > 0, got {budget}")
    if max_items is not None and max_items <= 0:
        raise ValueError(f"max_items must be > 0 when set, got {max_items}")
    span = (stop - start) * step
    if span <= 0:
        return 0
    limit = span if max_items is None else min(span, max_items)

    if step == -1:
        head = lengths[start]
        if head <= 0:
            return limit
        return max(1, min(limit, budget // head))

    count = 0
    running_max = 0
    for taken in range(limit):
        running_max = max(running_max, lengths[start + taken])
        if count and (count + 1) * running_max > budget:
            break
        count += 1
    return count


def make_batches(
    items_with_index: Iterable[tuple[int, str]],
    max_batch_tokens: int,
    length_fn: Callable[[str], int] = len,
    max_batch_items: int | None = None,
) -> Iterator[list[tuple[int, str]]]:
    """Greedily pack `(index, text)` items into padded-cost-bounded batches.

    Items are stably sorted by `length_fn(text)`, then packed greedily via
    `pack_count`: a batch flushes when adding the next item would push its
    padded cost `(count + 1) * max_len` over `max_batch_tokens`. A single
    item longer than the whole budget forms a batch of one — texts are
    never dropped or split.

    `max_batch_items` additionally caps the batch size by count. The budget
    alone cannot: with zero- or tiny-length items the padded cost stays
    near zero and a batch could otherwise grow to thousands of items.

    `length_fn` defaults to `len` (character proxy) and is responsible for
    clamping to the model's maximum sequence length — a text longer than
    that is truncated by the backend, so its true padded cost is the model
    maximum, not its raw length. No clamping happens here.
    """
    if max_batch_tokens <= 0:
        raise ValueError(f"max_batch_tokens must be > 0, got {max_batch_tokens}")
    if max_batch_items is not None and max_batch_items <= 0:
        raise ValueError(f"max_batch_items must be > 0 when set, got {max_batch_items}")
    return _generate(items_with_index, max_batch_tokens, length_fn, max_batch_items)


def _generate(
    items_with_index: Iterable[tuple[int, str]],
    max_batch_tokens: int,
    length_fn: Callable[[str], int],
    max_batch_items: int | None,
) -> Iterator[list[tuple[int, str]]]:
    # Ascending, stable sort: equal lengths keep input order.
    items = sorted(items_with_index, key=lambda item: length_fn(item[1]))
    lengths = [length_fn(text) for _, text in items]

    pos = 0
    while pos < len(items):
        count = pack_count(lengths, pos, len(items), 1, max_batch_tokens, max_batch_items)
        yield items[pos : pos + count]
        pos += count
