"""Length-sorted token-budget batching.

Batches are filled to a token budget rather than a fixed item count. The
cost of a batch is its *padded* cost, `count * max_len`, because the
backend pads every item in a batch to the longest one. Sorting by length
first packs similar lengths together and minimizes that padding waste.

Items travel as `(original_index, text)` pairs so downstream code can
restore input order after batches complete in any order (the ordering
invariant lives there, not here).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator


def make_batches(
    items_with_index: Iterable[tuple[int, str]],
    max_batch_tokens: int,
    length_fn: Callable[[str], int] = len,
) -> Iterator[list[tuple[int, str]]]:
    """Greedily pack `(index, text)` items into padded-cost-bounded batches.

    Items are stably sorted by `length_fn(text)`, then packed greedily:
    a batch flushes when adding the next item would push its padded cost
    `(count + 1) * max_len` over `max_batch_tokens`. A single item longer
    than the whole budget forms a batch of one — texts are never dropped
    or split.

    `length_fn` defaults to `len` (character proxy); inject a real
    tokenizer length later without touching this algorithm.
    """
    if max_batch_tokens <= 0:
        raise ValueError(f"max_batch_tokens must be > 0, got {max_batch_tokens}")
    return _generate(items_with_index, max_batch_tokens, length_fn)


def _generate(
    items_with_index: Iterable[tuple[int, str]],
    max_batch_tokens: int,
    length_fn: Callable[[str], int],
) -> Iterator[list[tuple[int, str]]]:
    # SPEC-GAP: batching.md (the authoritative algorithm doc) is absent from
    # the repo; "length-sorted" leaves the direction unspecified. Ascending is
    # the simplest correct choice: each added item is the new max, so the
    # projected cost is just (count + 1) * length of the incoming item.
    items = sorted(items_with_index, key=lambda item: length_fn(item[1]))

    batch: list[tuple[int, str]] = []
    max_len = 0
    for index, text in items:
        item_len = length_fn(text)
        new_max = max(max_len, item_len)
        if batch and (len(batch) + 1) * new_max > max_batch_tokens:
            yield batch
            batch = []
            new_max = item_len
        batch.append((index, text))
        max_len = new_max
    if batch:
        yield batch
