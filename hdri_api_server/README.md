# HDRI API Server (MVP)

Returns a signed download URL (`hdri_url` / `exr_url`) for a generated HDR environment map.

## Why `.hdr` instead of OpenEXR?

On Windows, `pip install OpenEXR` often requires a native C++ toolchain.  
This server writes Radiance RGBE `.hdr` using pure Python + NumPy.

## Default sizes

The API now supports these 2:1 output sizes:

- `1024x512`
- `2048x1024` (default)
- `4096x2048`

## Run API server

```powershell
cd hdri_api_server
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
$env:HDRI_PUBLIC_BASE_URL="http://127.0.0.1:8000"
$env:HDRI_SIGNING_SECRET="change-me"
uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs`.

## API

### `POST /v1/hdri`

Example body:

```json
{
  "provider": "D",
  "image_b64": "<base64>",
  "scene_mode": "auto",
  "quality_mode": "balanced",
  "preset": "none",
  "output_width": 2048,
  "output_height": 1024,
  "assume_upright": true,
  "panorama_prompt": "optional",
  "panorama_negative_prompt": null,
  "panorama_seed": null,
  "panorama_strength": null,
  "erp_layout_mode": "single_front",
  "reference_coverage": 0.4,
  "seam_fix": true,
  "erp_canvas_width": null,
  "erp_canvas_height": null,
  "panorama_extra": null,
  "hdr_reconstruction_mode": "ai_fast",
  "hdr_exposure_bias": 0.0,
  "hue_shift": 0.0,
  "sat_scale": 1.0,
  "blur_sigma": 0.0,
  "color_gain": 1.0
}
```

Response:

```json
{
  "hdri_url": "http://127.0.0.1:8000/v1/files/<uuid>.hdr?exp=...&sig=...",
  "exr_url": "http://127.0.0.1:8000/v1/files/<uuid>.hdr?exp=...&sig=...",
  "width": 2048,
  "height": 1024,
  "format": "hdr_rgbe",
  "panorama_mode": "http_json",
  "hdr_reconstruction_mode": "ai_fast"
}
```

### `POST /v1/jobs/hdri`

Creates an async generation job and returns:

```json
{
  "job_id": "uuid",
  "status": "queued"
}
```

### `GET /v1/jobs/{job_id}`

Poll until `status` is `succeeded` or `failed`.

### `GET /v1/files/{id}.hdr?exp=...&sig=...`

Downloads the HDR file.

### `GET /v1/config`

Shows active panorama backend (`panorama_mode`).

## Local ComfyUI worker (recommended for V1)

Set API server to forward panorama generation to your local worker:

```powershell
$env:PANORAMA_MODE="http_json"
$env:PANORAMA_HTTP_URL="http://127.0.0.1:8001/v1/panorama"
```

Run worker:

```powershell
cd hdri_api_server
python -m uvicorn examples.comfyui_worker:app --host 127.0.0.1 --port 8001
```

Worker health check:

- `http://127.0.0.1:8001/health`

### Worker request contract

Base fields:

- `image_b64`
- `width`, `height` (2:1)
- `scene_mode`
- `quality_mode`
- `prompt`
- `negative_prompt`
- `seed`
- `strength`

ERP placement fields (V1):

- `erp_layout_mode` (`single_front`)
- `reference_coverage` (`0.15..0.85`, default `0.60`)
- `seam_fix` (optional bool, quality-based default if omitted)
- `erp_canvas_width` / `erp_canvas_height` (optional 2:1 control canvas)

Response must include one of:

- `image_b64`
- `image_url`
- `output_url`

## HDR reconstruction modes

The API applies HDR reconstruction after panorama generation and before `.hdr` export.

Request field:

- `hdr_reconstruction_mode`: `ai_fast` | `heuristic` | `off`
- `hdr_exposure_bias`: EV offset applied after HDR reconstruction

Mode behavior:

- `ai_fast` (recommended): lightweight AI HDR reconstruction (`ai_hdr.py`)
- `heuristic`: legacy `_fake_hdr_lift()` path
- `off`: flat linear export (`rgb_lin * 2.5`)

Server defaults / failover env:

- `HDR_RECONSTRUCTION_MODE_DEFAULT=ai_fast`
- `AI_HDR_FAILOVER_MODE=heuristic`
- `AI_HDR_MODEL_NAME=embedded|torchscript`
- `AI_HDR_MODEL_PATH=...` (required when using `torchscript`)

## ComfyUI configuration for the worker

Environment variables used by `examples/comfyui_worker.py`:

- `COMFYUI_SERVER_URL` (default `http://127.0.0.1:8188`)
- `COMFYUI_WORKFLOW_TEMPLATE` (default `examples/comfyui_flux2_klein_template.json`)
- `COMFYUI_BASE_MODEL` (base model checkpoint filename)
- `COMFYUI_KLEIN_LORA` (LoRA filename)
- `COMFYUI_FAST_STEPS` / `COMFYUI_BALANCED_STEPS` / `COMFYUI_HIGH_STEPS`
- `COMFYUI_CFG`
- `COMFYUI_DEFAULT_STRENGTH`
- `COMFYUI_DEFAULT_PROMPT`
- `COMFYUI_DEFAULT_NEGATIVE_PROMPT`
- `COMFYUI_POLL_TIMEOUT_S`
- `COMFYUI_POLL_INTERVAL_S`

The included workflow template is a starter baseline.  
You will usually adapt node IDs/inputs to your actual ComfyUI graph.

## Panorama backend modes

Set `PANORAMA_MODE`:

- `resize`: stretch input to 2:1
- `replicate`: call Replicate predictions API
- `http_json`: call your worker endpoint
- `hf_dit360`: call Hugging Face inference API

## Benchmarks

Replicate benchmark:

```powershell
python benchmarks/run_replicate_benchmark.py
```

Local worker benchmark:

```powershell
$env:PANORAMA_HTTP_URL="http://127.0.0.1:8001/v1/panorama"
python benchmarks/run_local_worker_benchmark.py
```

Outputs:

- `benchmarks/local_worker_metrics.csv`
- `benchmarks/local_worker_report.md`
