# Notebook API + Domain + RunComfy

This is the **current target setup**:

| Piece | Role |
|--------|------|
| **Blender add-on** | Calls `https://api.richardandrys.com` |
| **Your notebook** | Runs FastAPI (`uvicorn`); orchestration only |
| **RunComfy** | GPU / ComfyUI workflow execution |
| **Domain** | `api.richardandrys.com` points at your notebook via a **tunnel** (HTTPS) |

You do **not** host models on the notebook when using RunComfy. The notebook only runs Python, SQLite, and HTTP to RunComfy.

## 0. Setup order (do this sequence)

Follow these steps in order so nothing depends on a missing piece.

| Step | What | Why |
|------|------|-----|
| **A** | **RunComfy** | In the RunComfy dashboard: create or select a **deployment**, note `RUNCOMFY_DEPLOYMENT_ID` and create an **API token** (`RUNCOMFY_API_TOKEN`). Export your workflow as API JSON if needed (`RUNCOMFY_WORKFLOW_JSON_PATH`). Map node IDs in `.env` (`RUNCOMFY_IMAGE_NODE_IDS`, prompts, etc.). | Without this, panorama jobs cannot run on GPU. |
| **B** | **Notebook: Python + `.env`** | Install Python 3.12+, create venv in `hdri_api_server`, `pip install -r requirements.txt`. Copy `.env.example` to `.env` and set at least: `HDRI_PUBLIC_BASE_URL=https://api.richardandrys.com`, `HDRI_SIGNING_SECRET`, `HDRI_ADMIN_TOKEN`, `HDRI_REQUIRE_API_KEY=1`, `HDRI_REMOTE_PROVIDER=runcomfy`, RunComfy vars, and node mappings. | Server must match the public URL you will use in Blender. |
| **C** | **Local test (no tunnel yet)** | `uvicorn app:app --host 127.0.0.1 --port 8000` then open `http://127.0.0.1:8000/v1/config` and `/docs`. | Confirms the app runs before networking. |
| **D** | **Tunnel on the notebook** | Install **Cloudflare Tunnel** (`cloudflared`) on Windows. Create a tunnel and add a **public hostname**: `api.richardandrys.com` -> `http://127.0.0.1:8000`. Cloudflare will show the **CNAME target** for DNS (e.g. `xxxxx.cfargotunnel.com`). | HTTPS + reach your PC without opening router ports. |
| **E** | **DNS at WEDOS** | In WEDOS DNS for `richardandrys.com`, add **CNAME**: name `api`, target = exact hostname from step D. Save/apply. | Makes `api.richardandrys.com` resolve to the tunnel. |
| **F** | **Listen on all interfaces** | Stop local-only bind. Run: `uvicorn app:app --host 0.0.0.0 --port 8000` (tunnel connects to localhost anyway). | Ensures the tunnel can reach uvicorn reliably. |
| **G** | **Verify HTTPS** | Wait for DNS (often 30 min - few hours). Test `https://api.richardandrys.com/v1/config` in a browser. | Proves domain + tunnel + API. |
| **H** | **API keys + Blender** | Create a user key (`POST /v1/accounts` with `X-Admin-Token`, or bootstrap key in `.env` for dev). In the add-on: Base URL `https://api.richardandrys.com`, paste key. | End-to-end from Blender. |

**Note:** If your domainís **nameservers** point to **Cloudflare** (not WEDOS), add the `api` **CNAME in Cloudflare DNS** instead of WEDOS. Only one DNS provider is authoritative.

Official Cloudflare Tunnel docs: [Cloudflare Zero Trust - Tunnels](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/).

## 1. Environment (copy into `hdri_api_server/.env`)

Minimum for this architecture:

```env
# Public URL must match what Blender uses (no trailing slash)
HDRI_PUBLIC_BASE_URL=https://api.richardandrys.com

# Strong random secrets (generate new values for production)
HDRI_SIGNING_SECRET=your-long-random-secret
HDRI_ADMIN_TOKEN=your-admin-token-for-provisioning

# Auth for real tests
HDRI_REQUIRE_API_KEY=1

# Optional dev bootstrap (remove in production)
# HDRI_BOOTSTRAP_API_KEY=...
# HDRI_BOOTSTRAP_ACCOUNT_ID=dev-account
# HDRI_BOOTSTRAP_TOKENS=25

# Hosted panorama + GPU work on RunComfy
HDRI_REMOTE_PROVIDER=runcomfy
RUNCOMFY_API_TOKEN=your-runcomfy-token
RUNCOMFY_DEPLOYMENT_ID=your-deployment-id
RUNCOMFY_BASE_URL=https://api.runcomfy.net

# Map your ComfyUI workflow nodes (required for image + prompts to reach the graph)
# RUNCOMFY_IMAGE_NODE_IDS=...
# RUNCOMFY_PROMPT_NODE_IDS=...
# ... see hdri_api_server/.env.example

# Panorama mode is ignored for panorama generation when HDRI_REMOTE_PROVIDER=runcomfy
# (keep defaults or set explicitly; see README)
PANORAMA_MODE=http_json
```

**Critical:** `HDRI_PUBLIC_BASE_URL` must be exactly `https://api.richardandrys.com` so signed `.hdr` download links in job responses work when Blender fetches them.

## 2. Run the API on the notebook

From PowerShell:

```powershell
cd c:\Users\richa\hdri_plugin\hdri_api_server
.\.venv\Scripts\activate
uvicorn app:app --host 0.0.0.0 --port 8000
```

- `0.0.0.0` allows the tunnel process on the same machine to connect to `localhost:8000`.
- Keep the PC awake while testing; sleep/hibernate stops the API.

## 3. Expose `api.richardandrys.com` to the notebook (HTTPS)

The notebook is not a static public IP suitable for a plain `A` record in most home setups. Use a **tunnel** so:

- TLS is terminated at the edge
- traffic reaches `127.0.0.1:8000` on your notebook

### Option A: Cloudflare Tunnel (recommended)

1. Create a Cloudflare account; add zone `richardandrys.com` (or use a partial DNS setup as Cloudflare documents).
2. Install `cloudflared` on Windows.
3. Create a tunnel and route **hostname** `api.richardandrys.com` to `http://127.0.0.1:8000`.
4. In DNS (Cloudflare or WEDOS, depending on where `richardandrys.com` nameservers point), add the **CNAME** Cloudflare shows for `api` (often `something.cfargotunnel.com`).

After DNS propagates, `https://api.richardandrys.com/v1/config` should load.

### Option B: ngrok (quick test)

- Use ngrok with a **fixed domain** or reserved domain if you need a stable `api.richardandrys.com` (often paid).
- For free tests, you may get a random `*.ngrok-free.app` URL; then set `HDRI_PUBLIC_BASE_URL` to that URL **and** use the same URL in Blender until you switch to the real subdomain.

## 4. WEDOS DNS: `api.richardandrys.com`

You need a DNS record so the name **`api.richardandrys.com`** resolves to your tunnel (or later to a VPS IP).

### Where you edit DNS

- Log in to **WEDOS customer administration** (https://client.wedos.com or the current WEDOS client URL).
- Open **Domains** and select **`richardandrys.com`**.
- Open **DNS records** / **DNS management** for that domain (wording may be "Sprùva DNS", "DNS zùznamy", or similar).

### What to add (notebook + tunnel)

Your tunnel provider (e.g. Cloudflare Tunnel) will give you **exact** values. Typical case:

| Field | What to enter |
|--------|----------------|
| **Type** | `CNAME` |
| **Name / Subdomain** | `api` only (not `api.richardandrys.com` - WEDOS adds the domain) |
| **Target / Data** | The hostname the tunnel shows (e.g. `xxxx.cfargotunnel.com`) |
| **TTL** | `300` or default |

WEDOS knowledge base notes: subdomains are entered **without** the full domain name (e.g. only `api`, not `api.richardandrys.com`). See [WEDOS: DNS subdomains](https://kb.wedos.com/dns-subdomeny/).

Save the record and **apply** changes if the UI has a separate "apply" step. Propagation often takes **30 minutes to a few hours** (sometimes up to 24-48 hours).

### Rules that often confuse people

- **CNAME and other records on the same name:** You cannot have both a `CNAME` and an `A` record for `api` at the same time. Use only what your tunnel needs.
- **Nameservers:** If you moved DNS to **Cloudflare** (nameservers point to Cloudflare), you edit DNS **in Cloudflare**, not in WEDOS. Only one place is authoritative.

### If you use a VPS later with a fixed IP

Instead of `CNAME`, you may use:

| Type | Name | Data |
|------|------|------|
| `A` | `api` | Your server IPv4 |

## 5. Blender add-on

In add-on preferences:

- **API Base URL:** `https://api.richardandrys.com`  
  (no path, no trailing slash)

Paste the API key you created (bootstrap or `POST /v1/accounts` with `X-Admin-Token`).

## 6. Verify

```text
https://api.richardandrys.com/v1/config
https://api.richardandrys.com/docs
```

Then submit a job from Blender and confirm job polling + HDR download.

## 7. Later: Oracle / Hetzner VPS

Same code and `.env`; only change:

- run `uvicorn` (or Docker) on the VPS 24/7
- point `api.richardandrys.com` to the VPS public IP or tunnel target
- keep `HDRI_PUBLIC_BASE_URL=https://api.richardandrys.com`

See also [HOSTED_DEPLOYMENT.md](HOSTED_DEPLOYMENT.md) and [SCALING_MIGRATION.md](SCALING_MIGRATION.md).
