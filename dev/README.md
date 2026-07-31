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
  1x–10x worker-speed sweep. Deterministic (seeded), and the measured
  numbers are stated in its markdown cells.
- `02_padding_savings.ipynb` — padding waste of fixed item-count batching vs
  `make_batches`, budget sweep, and the one-long-document degenerate case.
  Measured on `fancyzhx/ag_news`; numbers stated in markdown.
- `03_real_gpu_throughput.ipynb` — the evidence notebook: four
  configurations (fast GPU alone, static 50/50, static weighted, converging
  queue) on real hardware, with correctness asserted before timing, warmup
  discarded, per-device idle time, and an H2D bandwidth measurement per
  device. Run it on the GPU host and **commit it executed, with outputs**.

## Two rules these notebooks enforce

**No synthetic corpus by accident.** 02 and 03 load `fancyzhx/ag_news` and
*raise* if it is unavailable. Both used to fall back to a log-normal guess,
and both silently did so once the bare `ag_news` id stopped resolving under
`datasets>=5` — 03 spent a full benchmark on repeated `"lorem ipsum "`, and
02 reported a length distribution with a much heavier tail than ag_news
really has, overstating the padding win. Set `ALLOW_SYNTHETIC = True` in the
corpus cell only to test plumbing; it labels every number `DO NOT PUBLISH`.

**No retyped numbers.** 03 writes `dev/output/results.json` — makespans,
per-device items/tokens/idle, H2D bandwidth, driver, torch build, and device
capabilities. README benchmark figures are transcribed from that file. If a
number is not in it, it was not measured.
