# Task 20 — Pre-publication audit and third-party model scrub

## Scope

The repo goes public and gets an accompanying post. This task finds anything
that should not be public and removes references to proprietary models that
embedx has no relationship with. Audit first, then act; nothing here is a
feature change.

This is the first of three sequential commits (20 audit, 21 package split,
22 README). Do not start the split or the README rewrite here.

## Part A — audit, report before acting

Walk the working tree **and the git history**. A secret deleted in a later
commit is still public once the repo is.

- Any API key, token, password or credential in tracked files. Check
  `git log -p` and `git log --all --diff-filter=A --name-only`.
- Confirm `.env`, `docker-compose.override.yml`, `cache/` and any private dev
  output are gitignored **and were never committed**.
- Personal paths and identifiers: `/home/luxor`, hostnames, the Tailscale
  address, the direct-link address, email addresses. RFC1918 and CGNAT
  addresses are not secrets and are reasonable as concrete examples in
  `DEPLOY.md` — flag each with a recommendation rather than blanket-removing.
- `LICENSE` present and Apache-2.0, matching what `pyproject.toml` declares.

Report findings as a list. If anything requires rewriting history rather than
a normal commit, stop and say so — that decision is mine, not the agent's.

## Part B — proprietary model references must not appear

RavenBERT and RavenPack are proprietary licensed models. embedx has no
relationship to them, must not name them, must not reference their paths, and
must not imply it ships or depends on them. They were used as concrete
examples during development and appear in at least `.claude/tasks/17` and
`.claude/tasks/19`, and possibly in notebooks, `dev/output/*.json`, comments
or docs.

Grep the repo **and the history** for: `ravenbert`, `ravenpack`, `raven`.
Report every hit with file and context, then replace with a publicly
available equivalent of similar size (MiniLM or e5-small) wherever a concrete
example is genuinely needed.

**A measured number cannot simply have its label swapped.** If any benchmark
value in `dev/output/` or the README was produced against one of those
models, say so explicitly. It either gets re-measured against a public model
or it gets removed — silently relabelling a measurement is the one outcome
that is not acceptable here.

## Part C — documentation drift

The CI packaging smoke command referenced CLI flags removed in task 15 and
stayed broken for several commits without anyone noticing, so assume other
drift exists.

- Every command in `DEPLOY.md` and any other doc: does it still work against
  the current CLI and config surface?
- Every documented env var exists in `Settings`, and every `Settings` field a
  user needs is documented.
- No task file, notebook or comment contradicts current behaviour.

## Note for the README (task 22, not here)

Support for local paths, private checkpoints and custom fine-tunes is a
first-class use case and should be advertised — stated generically, never
with the proprietary names attached.

## Gate

`uv run ruff check .`, `uv run ruff format --check .`,
`uv run mypy src/embedx`, `uv run pytest -m "not gpu"`.

Commit: `chore: pre-publication audit and third-party reference scrub`
