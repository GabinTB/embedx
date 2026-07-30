# Task 11 — Dev notebooks and validation

## Scope

Exploratory notebooks under `dev/` validating the design claims. Not shipped, not
imported by the package or tests.

## Do

- `dev/01_scheduling_visualization.ipynb`: simulate fake fast/slow workers over a
  synthetic length distribution; plot cursor convergence over time and per-worker
  item/token counts. Show the faster worker consumes more, slower gets shorter
  texts.
- `dev/02_padding_savings.ipynb`: on a sample corpus, compare total padding tokens
  for naive fixed-batch vs length-sorted token-budget batching; quantify savings.
- `dev/03_real_gpu_throughput.ipynb` (GPU): measure tokens/sec per real device and
  wall-clock with vs without the converging queue; show both GPUs finish close to
  the same time (balanced).
- Add `dev/README.md` explaining these are exploratory and how to run them
  (`uv run --extra gpu jupyter lab`).

## Files

- `dev/*.ipynb`, `dev/README.md`

## Tests to add

- None (notebooks are excluded from tests). Ensure `.gitignore` excludes
  checkpoint dirs; ensure ruff/pytest do not scan `dev/`.

## Gate

Full gate green (notebooks not linted/tested). Commit:
`docs(dev): validation notebooks for scheduling, padding savings, throughput`.
