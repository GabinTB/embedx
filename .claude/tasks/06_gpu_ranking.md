# Task 06 — GPU discovery, ranking, budgets (no torch at import)

## Scope

Implement `gpu/discovery.py` and `gpu/budgets.py` per `gpu.md`. Must import
without torch.

## Do

- `src/embedx/gpu/discovery.py`:
  - `DeviceInfo` dataclass (no torch types).
  - `discover_devices()` — torch imported lazily inside the function, guarded by
    availability; returns `[]`/synthetic when torch absent.
  - `rank_devices(infos, weights_override) -> list[DeviceInfo]` — static score
    formula (documented), applies config weight overrides, returns
    fastest→slowest with `score`/`weight` filled.
- `src/embedx/gpu/budgets.py`:
  - derive per-device `max_batch_tokens` from memory + global default + overrides.

## Files

- `src/embedx/gpu/__init__.py`
- `src/embedx/gpu/discovery.py`
- `src/embedx/gpu/budgets.py`
- `tests/test_gpu.py`

## Tests to add

- Ranking order from synthetic DeviceInfo.
- Weight/budget override application.
- No-torch-at-import (import module in env without gpu extra; assert no raise).

## Gate

Full gate green. Commit: `feat(gpu): static device ranking and per-device budgets`.
