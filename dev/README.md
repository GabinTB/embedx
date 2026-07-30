# dev/

Exploratory notebooks that validate embedx's design claims. **Not part of the
package**, not imported by `embedx`, not covered by tests, not linted.

Run with the GPU extra for the throughput notebook:

```
uv sync --extra dev --extra gpu
uv run jupyter lab
```

Planned notebooks (see .claude/tasks/11_notebooks_and_validation.md):

- `01_scheduling_visualization.ipynb` — cursor convergence + per-worker balance.
- `02_padding_savings.ipynb` — padding-token savings of length-sorted batching.
- `03_real_gpu_throughput.ipynb` — real per-GPU tokens/sec and wall-clock balance.
