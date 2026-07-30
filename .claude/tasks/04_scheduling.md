# Task 04 — Converging work-stealing queue

## Scope

Implement the scheduler in `scheduling.md`. This is the core novelty and the
most concurrency-sensitive code. Pure Python + threads, CPU-testable with the
fake backend.

## Do

- `src/embedx/engine/scheduling.py`:
  - Shared `(lo, hi)` cursors over a length-sorted list, guarded by a
    `threading.Lock`.
  - `Side` enum (`SHORT`, `LONG`).
  - `claim(side, budget, lengths) -> list[tuple[int, str]] | None` implementing
    the exact atomic claim protocol (single shared frontier; SHORT grows `lo`
    up, LONG shrinks `hi` down; at-least-one-item guarantee; `None` when
    `lo >= hi`).
  - A `Scheduler` object holding sorted items + a precomputed `lengths` array,
    exposing `claim` and a `done` check.
  - Rank→side assignment helper for N workers (faster half LONG, slower half
    SHORT; N==1 single consumer; N==2 fast=LONG slow=SHORT).

## Files

- `src/embedx/engine/scheduling.py`
- `tests/test_scheduling.py`

## Tests to add (see scheduling.md invariants)

- Exactly-once across random N, sizes, budgets (multiset of indices == range(n)).
- No gaps/no duplicates at the meeting point.
- Order-independence vs single-threaded reference (with FakeBackend).
- Concurrency stress: many workers, tiny budgets, many repetitions, real Lock.
- Side-assignment helper correctness for N=1,2,3,4.

## Gate

Full gate green, scheduling tests run many randomized iterations. Commit:
`feat(engine): converging work-stealing scheduler across N heterogeneous workers`.
