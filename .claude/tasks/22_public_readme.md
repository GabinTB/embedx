# Task 22 — Public README

## Scope

Rewrite `README.md` as the front page of a project a stranger might adopt.
This is a rewrite for someone who has never seen the repo, not a polish pass.

Last of three sequential commits (20 audit, 21 split, 22 README). Both must
land first — the README documents the install paths task 21 creates, and must
not contain anything task 20 removed.

## Sections

Every one earns its place. Do not pad.

- **What it is**, one sentence, then the specific reason it exists rather
  than TEI or vLLM: one embedding model across GPUs of **different** speed
  and memory, all kept busy, with no configured load ratio. TEI is
  one-model-one-GPU; vLLM's tensor parallelism assumes homogeneous devices.
  Accurate and without disparagement — both are good tools solving different
  problems.
- **Badges**, only if real and passing. Nothing aspirational.
- **Install**, two paths, clearly separated, in this order:
  1. Docker, for the server. The primary, verified path.
  2. pip, for the library.
  Show both working end to end: a curl and the OpenAI-SDK equivalent for the
  server, and a short Python snippet for the library. Note that `pooling`
  goes in `extra_body` for the SDK — it is the single most likely thing to
  trip someone up. Explain in one line why the pip name differs from the
  import name, so a reader who notices finds the answer immediately instead
  of suspecting the docs are wrong.
- **How it works**: length-sorted batching, token-budget batches, converging
  work-stealing queue consumed from both ends. Concrete and brief.
- **Measured results**, transcribed from `dev/output/*.json`, never retyped,
  and only numbers that survive task 20. Report the modest margins honestly:
  1.14x over a single GPU and 1.09x over a tuned weighted static split. Then
  make the real argument, which is not raw speed: the static split needs a
  ratio someone has to find and re-find whenever the model, corpus or
  co-tenants change, the converging queue needs none, and it drove both
  devices' idle time under 8 ms where the weighted split left the small card
  idle for 64% of the run.
- **Model support**: any sentence-transformers or `AutoModel` checkpoint,
  from the hub or a local path, including private and custom fine-tunes —
  a first-class use case worth advertising. Pooling is explicit and required
  on first load, and say why: a wrong pooling produces plausible garbage with
  no error anywhere. Weights must be safetensors.
- **API reference**: the endpoints, the non-standard request fields with
  their first-load-only semantics, and the status codes that matter (409
  pooling conflict, 503 placement or capacity).
- **Configuration**, a table generated from `Settings` so it cannot drift.
- **Deployment**: link `DEPLOY.md`, do not duplicate it.
- **Limitations**, up front and unhedged. This section earns more trust than
  the feature list: CUDA only (there is a vendor seam but no ROCm or XPU
  implementation); Linux + NVIDIA first class, Windows via WSL2 with
  friction, macOS not at all; multi-GPU balancing applies only to models
  small enough to replicate on **every** device, so on a 16GB + 4GB pair
  anything above roughly 3.5GB at bf16 runs single-GPU and the balancing
  contributes nothing; one model per request; no pre-tokenized input;
  RoPE-family models JIT-compile at first inference and need a C toolchain
  with libc headers, which is what the Docker image solves.
- **Contributing**: the four-command gate, one task per commit.
- **License**.

## Tone

Direct and technical. No marketing voice, no "blazing fast", no emoji. A
reader should be able to tell within thirty seconds whether this solves their
problem, and the limitations section should do as much of that work as the
feature list.

## Gate

`uv run ruff check .`, `uv run ruff format --check .`,
`uv run mypy src/embedx`, `uv run pytest -m "not gpu"`.

Commit: `docs: rewrite README for public release`
