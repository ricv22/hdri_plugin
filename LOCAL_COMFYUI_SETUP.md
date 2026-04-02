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

## 3) Configure and run local worker

Worker env options (set in shell or `.env` for your worker process):

```env
COMFYUI_SERVER_URL=http://127.0.0.1:8188
COMFYUI_WORKFLOW_TEMPLATE=examples/comfyui_flux2_klein_template.json
COMFYUI_BASE_MODEL=flux1-dev.safetensors
COMFYUI_KLEIN_LORA=flux-2-klein-4B-360-erp-outpaint-lora.safetensors
COMFYUI_BALANCED_STEPS=28
COMFYUI_DEFAULT_STRENGTH=0.58
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

Panel defaults for V1:

- Output resolution: `2048x1024`
- ERP layout: `single_front`
- Reference coverage: `0.25`
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
