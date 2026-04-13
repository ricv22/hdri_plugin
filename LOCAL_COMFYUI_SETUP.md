# Local ComfyUI Setup (V1)

This project supports a local-first generation path:

- Blender addon -> `hdri_api_server`
- `hdri_api_server` -> local panorama worker (`examples/comfyui_worker.py`)
- worker -> local ComfyUI

## 1) Start ComfyUI

Install and run ComfyUI on your machine, then confirm:

- `http://127.0.0.1:8188`

Install required models/workflow assets:

- base model checkpoint (non-distilled, Flux-compatible)
- Flux.2 Klein 4B 360 ERP outpaint LoRA
- any custom nodes required by your workflow

## 2) Configure API server

In `hdri_api_server/.env`:

```env
PANORAMA_MODE=http_json
PANORAMA_HTTP_URL=http://127.0.0.1:8001/v1/panorama
HDR_RECONSTRUCTION_MODE_DEFAULT=comfyui_hdr
AI_HDR_FAILOVER_MODE=heuristic
AI_HDR_MODEL_NAME=embedded
HDR_HTTP_URL=http://127.0.0.1:8001/v1/hdr_restore
HDRI_PUBLIC_BASE_URL=http://127.0.0.1:8000
HDRI_SIGNING_SECRET=change-me
```

Run:

```powershell
cd hdri_api_server
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8000
```

## 2b) Optional: GMNet gain-map HDR (recommended over Flux img2img for reconstruction-shaped HDR)

This repo includes a ComfyUI custom node package: `comfyui_gmnet_itm/` (copy or symlink it into ComfyUI’s `custom_nodes/` folder).

1. Clone [GMNet](https://github.com/qtlark/GMNet) and download a checkpoint (e.g. `checkpoints/G_synthetic.pth` from the repo / release assets).
2. Install node deps inside ComfyUI’s Python (OpenCV): `pip install -r comfyui_gmnet_itm/requirements.txt` (path relative to this repo).
3. Set environment variables **before starting ComfyUI** (system env, or use this repo’s launcher):

- **Launcher (Windows):** `scripts/run_comfyui_with_gmnet.bat` — sets `GMNET_*` and starts ComfyUI with `D:\ComfyUI\.venv\Scripts\python.exe` and `D:\ComfyUI\resources\ComfyUI\main.py`. Edit paths inside the `.bat` if your layout differs.

Or set manually:

```env
GMNET_CODES_ROOT=D:/gmnet/GMNet/codes
GMNET_CHECKPOINT=D:/gmnet/GMNet/checkpoints/G_synthetic.pth
```

Optional: `GMNET_REPO_ROOT=D:/gmnet/GMNet` if you omit `GMNET_CHECKPOINT` and keep weights at `GMNET_REPO_ROOT/checkpoints/G_synthetic.pth`.

4. **HDR workflow JSON — set on the panorama worker (port 8001), not on ComfyUI.** The worker reads `COMFYUI_HDR_WORKFLOW_TEMPLATE` when it handles `POST /v1/hdr_restore`. Put it in the **same place you set `COMFYUI_SERVER_URL`** (worker’s shell, or a `.env` loaded before `uvicorn examples.comfyui_worker`):

```env
COMFYUI_HDR_WORKFLOW_TEMPLATE=examples/comfyui_gmnet_hdr_restore_api.json
```

If you start the worker with `cd hdri_api_server`, that path is relative to **`hdri_api_server`** and resolves to `hdri_api_server/examples/comfyui_gmnet_hdr_restore_api.json`. For a fixed path, use an absolute path to that file instead.

Workflow file in repo: `hdri_api_server/examples/comfyui_gmnet_hdr_restore_api.json` — **LoadImage → GMNetHDRITM → SaveImage**.

## 3) Configure and run local worker

Worker env options (set in shell or `.env` for your worker process):

```env
COMFYUI_SERVER_URL=http://127.0.0.1:8188
COMFYUI_WORKFLOW_TEMPLATE=examples/comfyui_flux2_klein_template.json
COMFYUI_HDR_WORKFLOW_TEMPLATE=examples/comfyui_flux2_klein_4b_hdr_restore_api.json
# Optional overrides (prefer workflow defaults if using exported API JSON)
# COMFYUI_BASE_MODEL=flux-2-klein-base-4b.safetensors
# COMFYUI_KLEIN_LORA=flux-2-klein-4B-360-erp-outpaint-lora\\flux-2-klein-4B-360-erp-outpaint-lora_V1.safetensors
COMFYUI_BALANCED_STEPS=28
COMFYUI_DEFAULT_STRENGTH=1.0
COMFYUI_HDR_BALANCED_STEPS=28
COMFYUI_HDR_DEFAULT_STRENGTH=0.35
```

Run:

```powershell
cd hdri_api_server
python -m uvicorn examples.comfyui_worker:app --host 127.0.0.1 --port 8001
```

Check:

- `http://127.0.0.1:8001/health`

## 4) Point Blender addon to local API

Addon preferences:

- API base URL: `http://127.0.0.1:8000`
- Timeout: start with `180` or `300`
- HDR reconstruction: `ComfyUI HDR` (recommended) or `AI Fast` as fallback

Panel defaults for V1:

- Output resolution: `2048x1024`
- ERP layout: `single_front`
- Reference coverage: `0.60`
- Seam fix: enabled

## 5) Benchmark quick check

With worker running:

```powershell
cd hdri_api_server
$env:PANORAMA_HTTP_URL="http://127.0.0.1:8001/v1/panorama"
python benchmarks/run_local_worker_benchmark.py
```

Review:

- `hdri_api_server/benchmarks/local_worker_report.md`
- generated outputs under `hdri_api_server/benchmarks/local_worker_panos/`
