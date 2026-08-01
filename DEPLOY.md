# Deploying embedx

Two supported paths, both single-host. They are alternatives, not layers.

| | Use it when |
|---|---|
| **[Docker](#docker)** | You want the Python / CUDA-wheel / Triton dependency problem solved for you. Recommended if you are not already running a matching Python and a C toolchain. |
| **[systemd](#install)** | The host already has the right Python, a C compiler, and the driver, and you would rather not run a 12 GB image. Lower overhead, more assembly. |

Neither is faster than the other in any way you will measure. GPU compute
overhead under the NVIDIA Container Toolkit is roughly 1–2%. The one place
containerization can genuinely cost you time is **disk I/O, and only if the
model cache is misconfigured** — see [the cache volumes](#the-cache-volumes-are-the-part-that-matters).

The systemd path below uses the unit file shipped inside the package at
`embedx/service/embedx.service`; print its path with:

```bash
python -c "from importlib import resources; print(resources.files('embedx') / 'service' / 'embedx.service')"
```

---

## Docker

### Prerequisites

1. An NVIDIA driver on the host. **525 or newer** — that is the floor implied
   by the CUDA 12.8 base image, through CUDA minor-version compatibility.
   Check with `nvidia-smi`.
2. The **NVIDIA Container Toolkit**, which is what lets a container see the
   GPU. Docker alone will not:

   ```bash
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
     | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
     | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
     | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

   Verify before going further — if this fails, nothing below will work:

   ```bash
   docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
   ```

**Linux + NVIDIA is the first-class target.** Windows + NVIDIA works through
WSL2 with friction. **macOS is not supported at all** — Apple dropped NVIDIA
driver support after 10.13 and Docker Desktop on macOS has no GPU
passthrough. This is not "cross-platform".

### Build and run

```bash
git clone https://github.com/<you>/embedx && cd embedx
cp .env.example .env          # set EMBEDX_UID/GID to `id -u` / `id -g`
mkdir -p cache/hf cache/triton
docker compose up -d --build
```

Expect **12 GB and ~5 minutes**, almost all of it torch and the CUDA
libraries. That size is normal for a torch + CUDA image and is why CI does
not build it.

```bash
curl http://127.0.0.1:8477/health
curl -X POST http://127.0.0.1:8477/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"sentence-transformers/all-MiniLM-L6-v2","pooling":"mean","input":["hello"]}'
```

No weights are baked into the image. The first request naming a model
downloads it into the mounted cache; subsequent starts reuse it.

### The cache volumes are the part that matters

Both are **bind mounts to host storage**, deliberately not named volumes.

`/cache/hf` (`HF_HOME`) holds model weights. Task 17 measured the disk read
as the dominant stage of a cold load — **6.38 s of a 9.36 s** Qwen3-4B load,
68% of it, at 1.17 GiB/s. Losing this cache on every container recreate
re-pays the single largest cost there is. Put it on **fast local storage**;
a network filesystem here is the one way containerization will cost you
measurable throughput.

`/cache/triton` holds JIT-compiled kernels (see below). Persisting it means
the compile happens once rather than on every container start.

**Both must be owned by the uid the container runs as.** This is the most
common way to break the setup: Docker creates a missing bind-mount source as
`root`, the unprivileged container user cannot write to it, and you get

```
PermissionError: [Errno 13] Permission denied: '/cache/triton/...'
```

at first inference — after the model has loaded and registered as resident.
Creating the directories yourself and setting `EMBEDX_UID`/`EMBEDX_GID` to
your own ids, as in the quick start, is what avoids it. If you prefer a
dedicated service account, `chown` both directories to match instead.

### Why the image ships a C compiler

`gcc` and `libc6-dev` are in the runtime stage on purpose. **Do not strip
them as bloat.**

Since the load-time smoke test landed, a host missing the toolchain fails
the **load** with a 503 naming the model, instead of registering a resident
model that raises on every request. That turns a silent, permanent fault
into a loud one — but it does not remove the requirement, and on the
systemd path (where you supply the toolchain yourself) it is still the
thing to check first when loads start failing.

torch routes RoPE-family models (Qwen3) through a Triton kernel that
JIT-compiles a C helper at *first inference*, not at load. Without a
compiler the model loads cleanly, registers as resident, and then every
request raises — which is precisely how this project's bare-metal host
failed once. Python headers are *not* needed separately: the image's
standalone CPython ships its own.

To verify the toolchain in a running container:

```bash
docker compose exec embedx python -c "
import torch, triton, triton.language as tl
@triton.jit
def k(x, y, o, n, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B); m = i < n
    tl.store(o + i, tl.load(x + i, mask=m) + tl.load(y + i, mask=m), mask=m)
x = torch.rand(1024, device='cuda'); y = torch.rand(1024, device='cuda'); o = torch.empty_like(x)
k[(1,)](x, y, o, 1024, B=1024); torch.cuda.synchronize()
assert torch.allclose(o, x + y); print('triton JIT OK')"
```

### Confirming Blackwell kernels

The build **fails** if the torch wheel lacks `sm_120`, rather than letting
you discover it as `no kernel image is available for execution on the
device` at first inference. To check a running container:

```bash
docker compose exec embedx python -c "import torch; print(torch.cuda.get_arch_list())"
# ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
```

### Changing CUDA or torch

The two vendor-specific decisions are build arguments, not hardcoded:

```bash
docker compose build \
  --build-arg CUDA_IMAGE=nvidia/cuda:12.8.1-base-ubuntu24.04 \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
```

CUDA 12.8 is the **floor** for `sm_120` and deliberately the *oldest* CUDA
that clears it. Raising the base to 13.x would push the minimum host driver
from 525+ to 580+ and exclude most users for no benefit. Note that cu128
currently tops out at torch 2.11 — 2.12 and later ship cu130 only.

### Sharing a GPU with Ollama

If Ollama runs on the same host it allocates VRAM on the same devices, and a
resident Ollama model directly shrinks what embedx can batch. embedx sizes
its token budgets from **total** device memory, not free memory, so it will
not notice and back off. Either cap Ollama's residency
(`OLLAMA_MAX_LOADED_MODELS`, `OLLAMA_KEEP_ALIVE`) or pin the two to
different cards.

---

## Install

One assumption up front: the host has a working NVIDIA driver (`nvidia-smi`
lists your cards). The torch wheels pulled in by the `gpu` extra bundle their
own CUDA runtime, so nothing else *CUDA-related* needs to be installed
system-wide — but without the driver there is no GPU, and embedx does not
serve on CPU.

That is not the same as "nothing else needs to be installed". RoPE-family
models need a **C toolchain at run time**; see the next section before you
serve one. The Docker image exists partly because it gets this right for
you.

### RoPE-family models need a C toolchain at run time

**Symptom:** the model loads cleanly, appears in `/info` as resident, and
then *every* request raises. Nothing looks wrong until you send traffic.
Since the load-time smoke test landed, embedx catches this and fails the
load with a 503 naming the model instead — but the requirement is
unchanged, and this is still the first thing to check when loads start
failing on a model like Qwen3.

**Cause:** torch routes RoPE-family models through a Triton kernel that
**JIT-compiles a C helper at first inference, not at load**. So the compiler
is needed on the serving host, at request time, long after `pip install`
finished.

Triton's compile needs three things. Which one is missing depends on how the
host is put together, and this project has hit two different ones:

| Needed | Provided by | How it fails when absent |
|---|---|---|
| A C compiler | `gcc` | `RuntimeError: Failed to find C compiler` |
| libc headers | `libc6-dev` | `fatal error: stdlib.h: No such file or directory` |
| Python headers for the venv's interpreter | see below | `fatal error: Python.h: No such file or directory` |

```bash
sudo apt-get install gcc libc6-dev
```

`gcc` alone is **not** enough — with `--no-install-recommends` it pulls no
libc headers, and the compile dies on `stdlib.h`. That is the failure the
container image hit.

**Python headers depend on where your interpreter came from**, and this is
the part that catches people:

- **A uv-managed Python** (`uv python install 3.14`, then
  `uv venv --python 3.14 /opt/embedx/.venv`) **ships its own headers.**
  Nothing to install. This is what the Docker image does, and it is the
  path of least resistance here too.
- **The distro Python** needs the matching `pythonX.Y-dev` package. Check
  what your venv actually uses before assuming — `home = /usr/bin` in
  `pyvenv.cfg` means the distro interpreter:

  ```bash
  grep -E '^(home|version)' /opt/embedx/.venv/pyvenv.cfg
  sudo apt-get install python3.14-dev      # match YOUR minor version
  ```

  That package does not exist for every minor version on every release —
  Ubuntu 24.04 ships `python3.12-dev` and has no `python3.14-dev` at all.
  If yours is missing, use a uv-managed Python instead of hunting for it.

Verify the whole chain rather than trusting the package list, using the venv
that will actually serve:

```bash
/opt/embedx/.venv/bin/python -c "
import torch, triton, triton.language as tl
@triton.jit
def k(x, y, o, n, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B); m = i < n
    tl.store(o + i, tl.load(x + i, mask=m) + tl.load(y + i, mask=m), mask=m)
x = torch.rand(1024, device='cuda'); y = torch.rand(1024, device='cuda'); o = torch.empty_like(x)
k[(1,)](x, y, o, 1024, B=1024); torch.cuda.synchronize()
assert torch.allclose(o, x + y); print('triton JIT OK')"
```

**This can break on a torch upgrade rather than a config change.** Whether
Triton is invoked *at all* depends on the torch version: torch 2.13 on this
project's bare-metal host routes Qwen3 through Triton, while the 2.11.0
pinned in the container image does not take that path for the same model. A
host with no C toolchain can therefore serve Qwen3 happily for months and
then fail on every request after a routine `pip install -U torch`, with
nothing in your own configuration having changed. If you upgrade torch,
re-run the check above.

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

The unit reads `/etc/embedx/embedx.env`. **Nothing is required.** A server is
configured without naming a checkpoint: models are named per request and
loaded on demand. `EMBEDX_MODEL_ID` and `EMBEDX_POOLING` no longer exist. Leaving
them here is a startup error rather than a no-op — deliberately, because the
alternative is a server that starts fine while its operator believes it is
still pinned to that model. Delete them, along with `EMBEDX_DTYPE` and
`EMBEDX_MAX_SEQ_LEN`.

```bash
# /etc/embedx/embedx.env — every line optional

#EMBEDX_HOST=127.0.0.1              # default; see Binding below
#EMBEDX_PORT=8477
#EMBEDX_API_KEY=...                 # unset disables auth entirely
#EMBEDX_DEVICES=0,1                 # default: all visible devices
#EMBEDX_MAX_BATCH_TOKENS=16384
#EMBEDX_DEVICE_WEIGHTS=0=1.0,1=0.35
#EMBEDX_DEVICE_BATCH_TOKENS=0=16384,1=4096
#EMBEDX_LOG_LEVEL=INFO

# Model residency:
#EMBEDX_DEFAULT_KEEP_ALIVE_S=600    # idle seconds before a model is unloaded
#EMBEDX_MAX_LOADED_MODELS=3         # unset = no cap

# Concurrency:
#EMBEDX_MAX_CONCURRENT_REQUESTS=8   # in-flight embedding requests
#EMBEDX_REQUEST_QUEUE_TIMEOUT_S=30  # queued longer than this -> 503
#EMBEDX_MAX_CONCURRENT_LOADS=2      # simultaneous cold model loads
```

`EMBEDX_DEFAULT_KEEP_ALIVE_S` is how long an idle model stays on the GPU
before its memory is released. A request may override it per model with
`keep_alive`, including `keep_alive: 0` to unload as soon as that request
finishes. `EMBEDX_MAX_LOADED_MODELS` caps how many models are resident at
once: at the cap, a new load evicts the least-recently-used model that no
request is using. If every resident model is busy the new load fails with
503 rather than pulling a model out from under a request in flight.

## Concurrency and backpressure

Two caps, and they are separate on purpose.

`EMBEDX_MAX_CONCURRENT_REQUESTS` bounds embedding requests in flight. Beyond
it, requests queue; a request that waits longer than
`EMBEDX_REQUEST_QUEUE_TIMEOUT_S` gets **503** rather than queueing forever,
so a client always gets a definite answer. The default of 8 is not a
throughput target: `Engine` holds a per-device lock across each embed, so
actual GPU parallelism is capped at your device count regardless. Admitting
more only overlaps tokenizing and output assembly with GPU work, and bounds
thread count and the memory held by in-flight inputs.

`EMBEDX_MAX_CONCURRENT_LOADS` bounds *cold model loads* — the expensive case,
which contends for VRAM and PCIe bandwidth. It has its own semaphore, not a
share of the request cap. If the two were merged, a handful of slow cold
loads would fill the request cap and requests to already-resident models
would queue behind them, which is the opposite of what a cap is for. A
request to a warm model never waits on the load cap.

A load cap larger than the request cap is rejected at startup rather than
accepted and ignored: every load runs inside a request and holds a request
slot while it does, so the larger number could never bind.

Both are visible in `GET /info` under `concurrency`, alongside live
`in_flight_requests` and `in_flight_loads`. Check there first when clients
report unexplained 503s — `/info` itself never consumes a request slot, so
it still answers when every slot is busy.

## What a cold load costs

Measured on the reference two-GPU host (RTX PRO 2000 Blackwell + RTX A400);
raw values in `dev/output/model_load_results.json`, method in
`dev/04_model_load_latency.ipynb`. Medians, each run in a fresh process with
the checkpoint's page cache verifiably evicted.

| stage | Qwen3-Embedding-4B (7.49 GiB) | MiniLM-L6-v2 (0.085 GiB) |
|---|---|---|
| 1 — disk → OS page cache | **6.382 s** | 0.097 s |
| 2 — page cache → host RAM | 0.205 s | 0.143 s |
| 2b — deferred mmap fault-in | 0.268 s | 0.044 s |
| 3 — host RAM → VRAM (PCIe) | 1.316 s | 0.013 s |
| 4 — first-inference kernel warmup | 1.193 s | **0.251 s** |
| total, cold page cache | 9.363 s | 0.548 s |
| total, warm page cache | 2.981 s | 0.451 s |

**Disk dominates a large load: 68% of it.** The PCIe copy people assume is
the bottleneck is 14%, and it is genuinely bus-bound (5.69 GiB/s against a
measured 6.36 GiB/s ceiling), so there is nothing to reclaim there.

**Practically, this means where you put the HF cache matters more than
anything else you can change.** Stage 1 read at 1.174 GiB/s here. On a
network filesystem, a spinning disk, or a container layer without a
persistent volume, that stage — already the largest — gets worse, and every
cold load pays it. Point `HF_HOME` at fast local storage and give it a
persistent volume, so the cache survives restarts:

```bash
#EMBEDX_... (embedx has no setting for this; it is HF's own)
HF_HOME=/var/lib/embedx/hf
```

A model reloaded while its files are still in the OS page cache skips stage 1
entirely — 2.981 s instead of 9.363 s for the 4B model. That is why
`EMBEDX_DEFAULT_KEEP_ALIVE_S` is 600 s rather than something aggressive: a
short TTL re-pays this on every quiet period.

Small models are dominated by stage 4, first-inference kernel autotune,
which is roughly fixed per model rather than proportional to size. It is
paid by whichever request arrives first after a load, so the first request
to a freshly loaded model is always slower than the ones behind it.

### A model that fits only the larger card runs single-GPU

Qwen3-Embedding-4B **does not fit the RTX A400**: 7.49 GiB of bf16 weights
against a 3.68 GiB card, which fails during the transfer with
`CUDA out of memory. Tried to allocate 48.00 MiB`.

This is handled, not fatal — embedx tries each device in turn and keeps
whichever succeed, so the model is served from the Blackwell card alone. But
be clear about what that means: **for such a model the multi-GPU balancing
contributes nothing.** The converging scheduler needs the model resident on
more than one device to have anything to balance. On this pair, a model
larger than the A400's 3.68 GiB is effectively a single-GPU deployment, and
the second card only helps for models small enough to be replicated on it.

## The safety floor

Config-time validation used to guarantee that a running server had a
deliberate pooling and a checkpoint someone had chosen by hand. Neither is
true when any request can name any model, so the guarantees moved into the
load path instead of disappearing:

- **Pooling is required on a model's first load** and is never inferred. The
  request that loads a model must carry `pooling`; without it the load is
  refused with 400. The resolution is logged at WARNING, so which pooling a
  model is serving under is in the journal.
- **Pooling is conflict-checked afterwards.** A later request naming the same
  model with a *different* pooling gets 409. It is never silently reapplied
  (wrong vectors for that caller), never silently ignored, and never triggers
  a reload (wrong vectors for everyone else mid-flight).
- **Weights must be safetensors.** A server that loads whatever a request
  names cannot also run pickle deserialization, which executes code. A
  checkpoint whose only weights are `.bin`/`.pt` is refused with 400, before
  anything is handed to torch.
- **The format check fails closed.** If the file listing cannot be obtained
  at all — hub unreachable and nothing cached — the load is refused with 503
  rather than proceeding unchecked. A model already in the local HF cache is
  verified against the cache, so an air-gapped host still serves what it has.

No allowlist: any hub id or local path a request names may be loaded. If that
is not acceptable for your deployment, keep the server behind an API key and
a trusted network — see Binding and The API key below.

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

### Under Docker it is a chain of three, not two

Compose adds a third filter *upstream* of both, and they compose in order:

```
compose device_ids  ->  CUDA_VISIBLE_DEVICES  ->  EMBEDX_DEVICES
   (what the             (what CUDA exposes        (which of those
    container sees)       inside that)              embedx uses)
```

Each stage renumbers from zero for the stage after it. So with three
physical GPUs:

```yaml
devices:
  - driver: nvidia
    device_ids: ["1", "2"]   # container sees physical 1,2 as 0,1
    capabilities: [gpu]
```

```
EMBEDX_DEVICES=1             # CUDA index 1 = PHYSICAL GPU 2
```

Same trap as above, one level deeper — and harder to see, because
`nvidia-smi` on the host still shows all three. Run `nvidia-smi` *inside the
container* to see what embedx actually has:

```bash
docker compose exec embedx nvidia-smi -L
```

The advice is unchanged, and stronger here: pick one mechanism. Selecting
with compose's `device_ids` and leaving both environment variables unset is
the clearest, because `docker compose config` then shows the whole story.

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
- `normalize=... wrapping=... default_keep_alive_s=... max_loaded_models=...`
  and the two concurrency caps,
- one `device N: <name> (weight ..., max_batch_tokens ...)` line per card,
- `no models loaded; each is loaded on the first request that names it`.

Startup is now fast: nothing is downloaded or loaded until a request names a
model. The wait moved to that first request, which blocks while the model
downloads and loads — `TimeoutStartSec=900` in the unit no longer covers it,
so give your first client a generous timeout instead.

Pooling is *not* in the startup log any more, because no model is loaded yet.
It appears at WARNING when each model is first loaded:

```
pooling for <model> resolved to mean on first load (dtype=..., device(s) [0, 1]);
later requests naming a different pooling will be refused
```

That line is the one to check when a model returns vectors that seem wrong.

### Preflighting a specific model

`ExecStartPre` runs `embedx check`, which validates config and devices but
loads nothing. To also prove a specific model loads on this box — weights
reachable, format accepted, fits in VRAM — add `--warm`:

```ini
ExecStartPre=/usr/local/bin/embedx check --warm sentence-transformers/all-MiniLM-L6-v2 --warm-pooling mean
```

It loads the model, embeds one string, unloads it, and leaves nothing
resident, so the server still starts empty. `--warm-pooling` is required with
`--warm` for the same reason it is required on a first load: pooling is never
inferred. Note this makes startup pay the download/load cost, which is
exactly what `TimeoutStartSec=900` is for — use it if you would rather find
out at boot than on the first request.

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

There is none. `embedx serve` requires at least one CUDA device and fails
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
