# Security and privacy

MotionSense processes live camera video of people in their homes, so privacy and safe defaults
matter more than features. This page explains what the software stores, what it exposes on the
network, and how to report a problem.

## Reporting a vulnerability

Please **do not open a public issue** for security problems. Use GitHub's
[private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
("Security" tab → "Report a vulnerability") and include steps to reproduce. You should get a reply
within a week.

## What stays on your computer

| Data | Where | Retention |
|---|---|---|
| Live video | Memory only, never written to disk | — |
| Incident snapshots (a few seconds before/after a fall alert) | `%LOCALAPPDATA%\motionsense\events\` (Windows), `~/.local/share/motionsense/events/` (Linux/macOS) | Newest 100 incidents (`sinks.snapshot_max_events`); disable with `sinks.snapshots: false` |
| Event log (activity changes, falls; no images) | `…/motionsense/events.jsonl` | Until you delete it; disable with `sinks.jsonl: false` |
| Recordings you make with `motionsense record` | Path you choose (default `recordings/`, which git ignores) | Until you delete them |
| AI models | `…/motionsense/models/` | — |

Nothing is uploaded anywhere unless **you** configure a webhook. Webhooks receive only the event
JSON (type, time, confidence, message), never images. Model weights are downloaded from the
Ultralytics GitHub releases on first run.

## Network exposure

- By default the server listens on **127.0.0.1 only**, so it is reachable from this computer alone.
- Without a token, requests must be addressed to `localhost` (this blocks DNS-rebinding attacks),
  and WebSocket connections from other websites are refused. Ordinary web pages you visit therefore
  cannot read the camera, events or snapshots.
- To reach it from a phone or another PC, you must use `--host 0.0.0.0 --token <secret>`. The
  server **refuses to start** on a network address without a token (12+ characters). Generate one
  with `python -c "import secrets; print(secrets.token_urlsafe(24))"`.
- With a token, every endpoint except a minimal `/api/v1/health` (version and status) requires
  `Authorization: Bearer <token>` or `?token=<token>`. Prefer the header for scripts; the query
  form exists for the browser dashboard. The dashboard sends `Referrer-Policy: no-referrer` so the
  token never leaks to other sites, and it cannot be embedded in other pages
  (`X-Frame-Options: DENY`, CSP `frame-ancestors 'none'`).
- The server speaks plain HTTP. **Do not expose it to the internet.** For remote access use a VPN
  (for example Tailscale or WireGuard) or a reverse proxy with HTTPS and proper authentication.
- Camera passwords in stream URLs (`rtsp://user:pass@…`) are never shown in the API or the
  dashboard. Webhook URLs (which often contain a secret) are logged by host name only.

## For contributors

- Never commit recordings, snapshots, datasets, model weights, `.env` files or local config.
  `.gitignore` covers the default locations; check `git status` before committing.
- CI runs ruff, mypy and the full test suite; Dependabot keeps dependencies and GitHub Actions
  up to date. Run `uvx bandit -r src` and `uvx pip-audit` before a release.

## Safety disclaimer

MotionSense is a prototype, **not a medical device or a certified alarm system**. It can miss
falls and raise false alarms. Do not rely on it as the only protection for a vulnerable person.
