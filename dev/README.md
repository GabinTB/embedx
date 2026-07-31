# dev/

Exploratory notebooks that validate embedx's design claims. **Not part of the
package**: not imported by `embedx`, not covered by tests, excluded from ruff
(see `[tool.ruff] exclude` in pyproject.toml).

The notebook stack (jupyterlab, ipykernel, matplotlib, datasets) ships in the
`dev` extra, so no ad-hoc `--with` flags are needed:

```
# 01 and 02 (CPU only):
uv sync --extra dev
uv run jupyter lab

# 03 additionally needs a CUDA host and the gpu extra:
uv sync --extra dev --extra gpu
uv run jupyter lab
```

To execute one headlessly instead of in the browser:

```
uv run jupyter nbconvert --to notebook --execute --inplace dev/03_real_gpu_throughput.ipynb
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
