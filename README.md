# embedx

**Heterogeneous multi-GPU text embedding server.**

`embedx` serves text-embedding models across an arbitrary set of GPUs that may
differ in speed and memory, and keeps all of them busy by combining three ideas:

1. **Length-sorted batching** — inputs are sorted by length so each batch is
   length-homogeneous, minimizing padding waste.
2. **Token-budget batches** — each GPU fills batches up to a per-device token
   budget rather than a fixed item count, so short-text batches are large and
   long-text batches are small, and every batch loads the GPU evenly.
3. **Converging work-stealing queue** — the sorted work list is consumed from
   both ends inward: slower GPUs pull short texts from one end, faster GPUs pull
   long texts from the other, and they meet in the middle. Faster GPUs naturally
   consume more work. No fixed ratio, no tuning.

It exposes an **OpenAI-compatible `/v1/embeddings`** endpoint (plus a native
`/embed` and `/health`), runs as a **systemd service**, and is designed to be
locked to a private network (Tailscale / direct link) with an optional API key,
exactly like a self-hosted Ollama.

Generalizes from 1 GPU to N GPUs. Model-agnostic: any Hugging Face /
sentence-transformers embedding model.

## Status

Early scaffold. See [`.claude/skills/embedx-project-coding/`](.claude/skills/embedx-project-coding/)
for the full design specification and [`tasks/`](tasks/) for the commit-oriented
build plan.

## Quickstart (target UX, not all implemented yet)

```bash
# install (editable) with the GPU backend
uv pip install -e ".[gpu,dev]"

# serve a model across all detected GPUs
embedx serve --model-id BAAI/bge-m3

# call it, OpenAI style
curl http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "bge-m3", "input": ["hello world", "a longer piece of text ..."]}'
```

## License

Apache-2.0.
