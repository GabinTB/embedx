# Task 12 — Release readiness

## Scope

Converge everything into a shippable, PyPI-publishable 0.1.0.

## Do

- Finalize `README.md`: real quickstart, feature summary, config table link,
  deployment link, benchmarks from the notebooks (padding savings, balance).
- `CHANGELOG.md` with 0.1.0 notes.
- Verify `uv build` produces a valid wheel+sdist; `twine check dist/*` clean.
- Ensure `mypy src/embedx`, full ruff, and `pytest -m "not gpu"` all green; run
  `pytest -m gpu` on a GPU host and record results in the PR.
- Tag `v0.1.0`.
- (Optional) TestPyPI dry-run upload.

## Files

- `README.md`, `CHANGELOG.md`

## Tests to add

- A packaging smoke test (build in CI, import from the built wheel in a clean env).

## Gate

Full gate green + `uv build` + `twine check` clean. Commit:
`release: v0.1.0`.
