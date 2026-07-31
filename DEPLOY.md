# Deploying embedx

This walks through a single-host systemd deployment. The shipped unit file
lives inside the package at `embedx/service/embedx.service`; print its path
with:

```bash
python -c "from importlib import resources; print(resources.files('embedx') / 'service' / 'embedx.service')"
```

## Install

One assumption up front: the host has a working NVIDIA driver (`nvidia-smi`
lists your cards). The torch wheels pulled in by the `gpu` extra bundle their
own CUDA runtime, so nothing else CUDA-related needs to be installed
system-wide — but without the driver there is no GPU, and embedx does not
serve on CPU.

With uv, into a dedicated venv (the path the unit file expects):

```bash
uv venv /opt/embedx/.venv
VIRTUAL_ENV=/opt/embedx/.venv uv pip install "embedx[gpu]"
```

Or with pipx:

```bash
pipx install "embedx[gpu]"
```

If you use pipx or another venv location, adjust `ExecStartPre` and
`ExecStart` in the unit file accordingly.

Create the service user and config directory:

```bash
sudo useradd --system --home-dir /var/lib/embedx --shell /usr/sbin/nologin embedx
sudo mkdir -p /etc/embedx
```

## The env file

The unit reads `/etc/embedx/embedx.env`. Two variables are **required** —
there are no defaults for them, and pooling is deliberately never inferred
(a wrong pooling produces plausible garbage vectors with no error):

```bash
# /etc/embedx/embedx.env
EMBEDX_MODEL_ID=sentence-transformers/all-MiniLM-L6-v2
EMBEDX_POOLING=mean                 # cls | mean | last_token

# Common optional settings:
#EMBEDX_HOST=127.0.0.1              # default; see Binding below
#EMBEDX_PORT=8477
#EMBEDX_API_KEY=...                 # unset disables auth entirely
#EMBEDX_DEVICES=0,1                 # default: all visible devices
#EMBEDX_MAX_BATCH_TOKENS=16384
#EMBEDX_DEVICE_WEIGHTS=0=1.0,1=0.35
#EMBEDX_DEVICE_BATCH_TOKENS=0=16384,1=4096
#EMBEDX_LOG_LEVEL=INFO
```

Permissions: the file can carry the API key, so keep it `0600` owned by
root — systemd (running as root) reads `EnvironmentFile=` before dropping to
the service user, so the service user does not need read access:

```bash
sudo chown root:root /etc/embedx/embedx.env
sudo chmod 0600 /etc/embedx/embedx.env
```

If other tooling running as the service group must read it, use
`root:embedx` with `0640` instead; never world-readable.

## CUDA_VISIBLE_DEVICES and EMBEDX_DEVICES interact

`EMBEDX_DEVICES` indices are relative to **what CUDA already exposes**, not
to physical slots. Setting both is how people end up serving the wrong card.

Concrete example: the host has three GPUs, physical 0/1/2.

```
CUDA_VISIBLE_DEVICES=1,2   # CUDA renumbers: physical 1 -> index 0, physical 2 -> index 1
EMBEDX_DEVICES=1           # selects CUDA index 1 = PHYSICAL GPU 2
```

If you expected "physical GPU 1", you are now serving from the wrong card,
silently. Pick **one** mechanism: either restrict with
`CUDA_VISIBLE_DEVICES` and leave `EMBEDX_DEVICES` unset, or expose
everything and select with `EMBEDX_DEVICES` alone.

## Enable on boot

```bash
sudo cp "$(python -c "from importlib import resources; print(resources.files('embedx') / 'service' / 'embedx.service')")" \
    /etc/systemd/system/embedx.service
# edit paths (venv location, DeviceAllow lines for your card count), then:
sudo systemctl daemon-reload
sudo systemctl enable --now embedx
```

On first start, check the log for the preflight and startup lines:

```bash
journalctl -u embedx -f
```

You want to see, in order:

- `check ok: config valid, N device(s) available` (the `ExecStartPre`
  preflight; if this fails the unit stops before crash-looping serve),
- `binding http://<host>:<port>`,
- `api key: set` (or `not set (open access)` — see Binding),
- `pooling=... dtype=... normalize=...` — verify the pooling is the one your
  model needs,
- one `device N: <name> (weight ..., max_batch_tokens ...)` line per card.

The first start may sit in `activating` for minutes while model weights
download (this is why the unit sets `TimeoutStartSec=900`).

## Binding

The default bind is `127.0.0.1`: nothing off-host can reach the server.

For remote access prefer a non-routable interface over opening the port:

- **Tailscale**: set `EMBEDX_HOST` to the machine's tailnet address
  (`tailscale ip -4`), and uncomment the `After=tailscaled.service` lines in
  the unit so the bind address exists before the service starts.
- **Direct-link subnet**: bind the specific interface address
  (e.g. `10.0.0.1` on a point-to-point link), not `0.0.0.0`.

Keep the firewall closed to WAN regardless:

```bash
sudo ufw allow in on tailscale0 to any port 8477 proto tcp
sudo ufw deny 8477/tcp
```

Stated plainly: **binding `0.0.0.0` without `EMBEDX_API_KEY` exposes the
server to everything that can route to it** — every pod on the LAN, every
container on the bridge, the whole tailnet, or the internet if the box has a
public address. embedx logs a warning in this state (`embedx check` shows it
too) but it will serve.

## The API key

```bash
EMBEDX_API_KEY=$(openssl rand -hex 32)
```

Put it in the env file (see permissions above). Clients send
`Authorization: Bearer <key>`. This is bearer auth over **plain HTTP**: the
key crosses the wire unencrypted unless something terminates TLS in front
(Caddy, nginx, a Tailscale tunnel — Tailscale traffic is encrypted at the
WireGuard layer, which is one reason it is the recommended remote path).

## Running alongside Ollama

Both can share a host; two things to know:

- **Port**: Ollama defaults to 11434, embedx to 8477 — no clash by default,
  but if you moved either, they must differ.
- **VRAM**: both allocate on the same devices. A large Ollama model resident
  on a card shrinks the free memory embedx can use for activations, which
  effectively shrinks its batches (and can OOM a batch sized for an empty
  card). Ollama's keep-alive is the knob: `OLLAMA_KEEP_ALIVE=5m` (or `0` to
  release immediately) controls how long Ollama keeps models resident after
  the last request. Alternatively pin the two services to different cards
  with `CUDA_VISIBLE_DEVICES`.

## CPU fallback

There is none. `build_engine` requires at least one CUDA device and raises
with a clear error otherwise; `embedx check` exits non-zero on a GPU-less
host. If you need CPU embedding, use a different tool — do not deploy this
expecting a silent CPU mode.

## Troubleshooting

**`check failed: no CUDA device available`** — `nvidia-smi` works for root
but the service sees nothing: usually the hardening block. `DevicePolicy=
closed` only allows the listed `DeviceAllow` nodes; confirm there is a
`DeviceAllow=/dev/nvidiaN rw` line for every card index on this host. Also
check the driver actually loaded (`lsmod | grep nvidia`) and that
`CUDA_VISIBLE_DEVICES` in the env file is not empty or filtering everything
out.

**Model download timeout on first start** — the unit is killed while still
`activating`: the weights did not download within `TimeoutStartSec`.
Pre-download as the service user instead of raising it blindly:

```bash
sudo -u embedx HF_HOME=/var/lib/embedx/hf \
    /opt/embedx/.venv/bin/python -c \
    "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')"
```

**Permission errors on the HF cache** — `ProtectSystem=strict` makes the
entire filesystem read-only for the service except `ReadWritePaths` (and
`StateDirectory`). If you changed `HF_HOME`, or the cache was first created
by another user (root, after a manual run), the download fails with
`Permission denied` / `Read-only file system`. Keep `HF_HOME` inside
`/var/lib/embedx` and fix ownership:

```bash
sudo chown -R embedx:embedx /var/lib/embedx
```
