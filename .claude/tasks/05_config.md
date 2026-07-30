# Task 05 — Configuration

## Scope

Implement `config.py` per `config.md` with pydantic-settings.

## Do

- `src/embedx/config.py`: `Settings(BaseSettings)` with env prefix `EMBEDX_`,
  all fields from `config.md`, validators (required model_id, budget clamps,
  device-index validation), and parsers for override strings
  (`device_weights`, `device_batch_tokens`).
- Precedence: CLI > env > file (`EMBEDX_CONFIG`) > defaults.
- Loopback default host; warning surfaced (as a returned flag/log) when bound
  beyond loopback without an API key.

## Files

- `src/embedx/config.py`
- `tests/test_config.py`

## Tests to add

- Env/file/CLI precedence.
- Validation: missing model_id fails clearly; bad device index errors.
- Override-string parsing incl. malformed input.
- Exposure warning logic (bound non-loopback + no key → warn flag true).

## Gate

Full gate green. Commit: `feat(config): pydantic-settings config with overrides and validation`.
