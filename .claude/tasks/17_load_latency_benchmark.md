# Task 17 — Cold/warm model-load latency benchmark

## Scope

Measure, rather than assert, what a cold model load actually costs and why —
this directly settles a question raised during design: is Ollama's perceived
load speed for large models mostly physics (PCIe transfer, unavoidable and
identical for any runtime) or mostly software overhead (something embedx
could actually improve)? Answer it with a number on this hardware, for a
model in the size class actually planned for production (Qwen3-Embedding-4B
or whatever ends up largest), and use the result to calibrate
`default_keep_alive_s` (task 15) instead of guessing.

## Do

New notebook `dev/04_model_load_latency.ipynb`, same measurement rigor as
`dev/03_real_gpu_throughput.ipynb`: real hardware, medians over several
runs, methodology stated plainly. Warmup discipline here is deliberately
*different* from notebook 03 — the entire point is separating cold-cache
from warm-cache behavior, so don't copy notebook 03's "discard warmups"
pattern uncritically; describe explicitly which stages are measured cold
and which warm, and why.

Isolate four stages, timed separately, never lumped into one "load" number:

1. **Disk → OS page cache.** The first-ever read of the checkpoint files. If
   you have permission to drop caches on this host, measure true cold;
   otherwise say plainly that this stage wasn't isolated cleanly and why,
   rather than reporting a number that's actually partially warm.
2. **OS page cache → host RAM materialization.** `from_pretrained` with the
   file already page-cached (confirm `disable_mmap` is not set, i.e. mmap
   loading is active, since that's the mechanism actually being tested).
3. **Host RAM → device VRAM.** The PCIe-bound copy. Cross-check this against
   notebook 03's measured H2D bandwidth per device: does
   `model_size_bytes / measured_h2d_gib_s` roughly match what's observed
   here? A mismatch is itself a finding worth reporting, not something to
   paper over.
4. **First-inference kernel warmup.** Compare a first `embed()` call on a
   freshly loaded model against the second call on the same, now-warm
   model — this isolates cuBLAS/cuDNN kernel autotuning, a one-time
   per-process cost, from the load itself.

Run this for the largest production-relevant model on both devices
independently (some stages, like stage 3, are expected to differ
significantly given the measured PCIe gap between the two cards), and once
more for a small model (e5-small-v2 or MiniLM) as a size-scaling
comparison point, so the write-up can say whether the dominant cost scales
with model size or is roughly fixed per load.

State the conclusion plainly, whichever way it lands: if stage 3 (PCIe
transfer) dominates, that cost is identical for any runtime including
Ollama's, and there's nothing to build. If stage 2 (materialization) or
stage 4 (kernel warmup) dominates, say so and note it as a genuine
optimization opportunity for a later task rather than acting on it here —
this is a measurement task, not an optimization task.

From the result, propose a `default_keep_alive_s` value with the reasoning
spelled out: too short re-pays this cost on every brief lull between
requests; too long leaves an idle model's weights resident on the GPU
consuming power for no work, which is the exact concern that motivated TTL
eviction in the first place.

## Files

`dev/04_model_load_latency.ipynb`,
`dev/output/model_load_results.json` — same transcription discipline as
task 12: any number that ends up in `DEPLOY.md` or `README.md` is
transcribed from this file, never retyped from a notebook cell.

## Tests to add

None required — this is a `dev/` notebook, same as 01-03, outside
`pytest`'s scope. If a reusable stage-timing helper gets factored out and is
genuinely useful at runtime (e.g. a future `/info` field reporting how long
the last load took), it may move under `src/embedx`; if so it needs tests
and the full gate applies to it. Otherwise keep it notebook-local and say
why in a comment.

## Gate

Does not block the four-command gate unless a reusable helper is promoted
to `src/embedx` as described above, in which case the full gate applies to
that helper: `uv run ruff check .`, `uv run ruff format --check .`,
`uv run mypy src/embedx`, `uv run pytest -m "not gpu"`.

Commit: `docs(dev): cold/warm model-load latency benchmark`
