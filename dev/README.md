# dev/

Exploratory notebooks that validate embedx's design claims. **Not part of the
package**: not imported by `embedx`, not covered by tests, excluded from ruff
(see `[tool.ruff] exclude` in pyproject.toml).

Jupyter and matplotlib are not project dependencies; run them ephemerally:

```
# 01 and 02 (CPU only; 02 uses a synthetic fallback without `datasets`):
uv run --with jupyterlab --with matplotlib jupyter lab

# 02 with the real corpus, and 03 (CUDA host only):
uv sync --extra dev --extra gpu
uv run --with jupyterlab --with matplotlib --with datasets jupyter lab
```

- `01_scheduling_visualization.ipynb` — pure simulation over the real
  `Scheduler`: cursor convergence, per-worker item/token balance, and a
  1x–10x worker-speed sweep. Committed without outputs; deterministic
  (seeded), and the measured numbers are stated in its markdown cells.
- `02_padding_savings.ipynb` — padding waste of fixed item-count batching vs
  `make_batches`, budget sweep, and the one-long-document degenerate case.
  Committed without outputs; numbers stated in markdown.
- `03_real_gpu_throughput.ipynb` — the evidence notebook: four
  configurations (fast GPU alone, static 50/50, static weighted, converging
  queue) on real hardware, with correctness asserted before timing, warmup
  discarded, per-device idle time, and an H2D bandwidth measurement per
  device. Run it on the GPU host and **commit it executed, with outputs**.
