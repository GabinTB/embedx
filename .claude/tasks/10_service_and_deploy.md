# Task 10 — Service and deployment

## Scope

Ship the systemd unit and deployment docs; enforce safe network defaults.

## Do

- `src/embedx/service/embedx.service` per `deployment.md` (with the NVIDIA device
  access note and a known-good hardening variant).
- `docs/deployment.md` (or top-level `DEPLOY.md`): install, env file, enable on
  boot, Tailscale/UFW binding, API key, running alongside Ollama, CPU fallback.
- Startup exposure warning wired (from config task) into `serve` logs.

## Files

- `src/embedx/service/embedx.service`
- `DEPLOY.md`

## Tests to add

- Unit test that the exposure-warning fires for non-loopback bind + no key
  (logic-level, already partly in config tests; assert integration in serve
  preflight without binding a port).

## Gate

Full gate green. Commit: `feat: systemd service + deployment docs + safe defaults`.
