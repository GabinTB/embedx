# Task 15 — Config and CLI migration (breaking change)

## Scope

Remove the single-model config surface from 0.1.0. Call this what it is: a
breaking change to `Settings`, not an additive one. Existing deployments
setting `EMBEDX_MODEL_ID` etc. will need to migrate to request-time `model`.

## Do

- Remove from `Settings`: `model_id`, `pooling`, `dtype`, `max_seq_len` as
  top-level required/singular fields. They move to being per-request
  concerns (task 14) resolved once per model id (task 13).
- Add: `default_keep_alive_s: int` — pick a default and justify it in a
  comment citing Ollama's documented sane keep-alive range (roughly
  10-15 minutes) as prior art, not an arbitrary number; > 0 validated.
  `max_loaded_models: int | None = None` — optional cap. When set and a new
  load would exceed it, evict the least-recently-used resident model to
  make room rather than refusing the request; document this LRU behavior
  in the field's docstring.
- Keep as-is, unchanged: `devices`, `max_batch_tokens`, `max_batch_items`,
  `device_weights`, `device_batch_tokens`. These describe the hardware, not
  any one checkpoint, and stay server-wide.
- `embedx serve` starts with zero models loaded. The eager `build_engine`
  call at startup goes away entirely — the registry (task 13) replaces it,
  constructed empty and populated on first request per model.
- `embedx check` no longer preflights a specific model, since there isn't
  one at startup anymore. It still validates: config parses, every
  requested device index exists, at least one device is available. Consider
  an optional `--warm <model_id>` flag that performs a real load-and-evict
  cycle as an end-to-end preflight, without leaving the model resident
  afterward — useful for systemd's `ExecStartPre`, but not required if it
  adds meaningfully more scope than the rest of this task.
- `embedx info` reports server-wide config plus whatever's currently
  resident (normally empty right after `check`, since nothing has loaded a
  model yet).
- Update `DEPLOY.md`: the env-file section no longer lists
  `EMBEDX_MODEL_ID`/`EMBEDX_POOLING` as required. Add a section summarizing
  the new safety floor since it replaces what config-time validation used to
  guarantee: pooling is required on first load and conflict-checked after
  that (never silently reapplied), and dynamically loaded weights must be
  safetensors.
- `CHANGELOG.md`: new section (`[Unreleased]` or `0.2.0`, match whatever the
  project's versioning convention already implies) stating plainly that this
  is a breaking change, naming the removed fields and their replacement.
- `README.md`: update the quickstart curl/client example to include a
  `model` field if it didn't reference one explicitly before.

## Files

`src/embedx/config.py`, `src/embedx/cli.py`, `DEPLOY.md`, `CHANGELOG.md`,
`README.md`

## Tests to add

- `Settings()` on defaults (or with only network/device fields set) now
  constructs successfully — replaces task 05's tests asserting `model_id`
  and `pooling` are required, which are no longer true.
- `default_keep_alive_s` and `max_loaded_models` validated (positive;
  `max_loaded_models` genuinely optional).
- `check --warm <id>` (if built): succeeds, and the model is not resident
  afterward (assert via the registry/stub).
- Re-run task 09's CLI precedence regression test against the new field set;
  it must still pass unchanged in spirit.

## Gate

`uv run ruff check .`, `uv run ruff format --check .`,
`uv run mypy src/embedx`, `uv run pytest -m "not gpu"`.

Commit: `feat(config): drop single-model settings for the multi-model registry`

Include a `BREAKING CHANGE:` paragraph in the commit body naming the removed
fields and pointing at the replacement (request-time `model`, task 14).
