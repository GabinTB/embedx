# Task 03 — Length-sorted token-budget batching

## Scope

Implement the per-worker batching described in `batching.md`. Pure Python, fully
CPU-testable.

## Do

- `src/embedx/engine/batching.py`:
  - `make_batches(items_with_index, max_batch_tokens, length_fn) -> Iterator[list[tuple[int, str]]]`
    following the exact algorithm in `batching.md` (greedy token-budget fill,
    single-oversized-item forms its own batch, budget accounting
    `count * current_max_len`).
  - `length_fn` defaults to `len`; pluggable for true token lengths later.
  - Validate `max_batch_tokens > 0`.

## Files

- `src/embedx/engine/__init__.py`
- `src/embedx/engine/batching.py`
- `tests/test_batching.py`

## Tests to add (see batching.md + testing.md)

- No batch exceeds budget except the unavoidable single-oversized item.
- Every input index appears exactly once; order preserved.
- Hand-computed regression example with known boundaries.
- Property test: random strings/budgets → invariants always hold.

## Gate

Full gate green. Commit:
`feat(engine): length-sorted token-budget batching`.
