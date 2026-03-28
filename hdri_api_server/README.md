# HDRI API Server (MVP)

Returns a **signed download URL** (`hdri_url` / `exr_url`) for a **1024×512** environment map.

## Why `.hdr` instead of OpenEXR?

On **Windows**, `pip install OpenEXR` often tries to **compile C++** (CMake + MSVC). That fails if you do not have the Visual Studio build tools.

This server writes **Radiance RGBE `.hdr`** using pure Python + NumPy — **no compiler**. Blender’s Environment Texture loads `.hdr` fine for HDRI lighting.

## Python version

Use **Python 3.11 or 3.12** for the venv if possible. **Python 3.14** is very new; many packages may not ship wheels yet.

## Run

```powershell
cd hdri_api_server
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
$env:HDRI_PUBLIC_BASE_URL="http://127.0.0.1:8000"
$env:HDRI_SIGNING_SECRET="change-me"
uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` to verify the server is up.

## API

### `POST /v1/hdri`

Body (JSON):

```json
{
  "provider": "D",
  "image_b64": "<base64>",
  "scene_mode": "auto",
  "quality_mode": "balanced",
  "preset": "none",
  "output_width": 1024,
  "output_height": 512,
  "assume_upright": true,
  "panorama_prompt": "optional — forwarded to http_json worker",
  "panorama_negative_prompt": null,
  "panorama_seed": null,
  "panorama_strength": null,
  "panorama_extra": null
}
```

Response:

```json
{
  "hdri_url": "http://127.0.0.1:8000/v1/files/<uuid>.hdr?exp=...&sig=...",
  "exr_url": "http://127.0.0.1:8000/v1/files/<uuid>.hdr?exp=...&sig=...",
  "width": 1024,
  "height": 512,
  "format": "hdr_rgbe",
  "panorama_mode": "resize"
}
```

(`exr_url` is kept as an alias of `hdri_url` for older clients; the file is still `.hdr`.)

### `GET /v1/files/{id}.hdr?exp=...&sig=...`

Downloads the HDR file.

### `GET /v1/config`

Returns which **panorama backend** is active (`panorama_mode`).

---

## Image-conditioned panorama (`http_json` / img2img)

For **true image → equirectangular** (inpainting, outpainting, DiT360-style ERP control images, ComfyUI, etc.), use:

```powershell
$env:PANORAMA_MODE="http_json"
$env:PANORAMA_HTTP_URL="http://127.0.0.1:8001/v1/panorama"
```

Your worker receives a **JSON POST** with at least:

| Field | Meaning |
|--------|--------|
| `image_b64` | User photo (same as Blender/API client sent) |
| `width` | Target width (e.g. 1024) |
| `height` | Target height (e.g. 512) |
| `scene_mode` | `auto` / `outdoor` / `indoor` / `studio` |
| `quality_mode` | `fast` / `balanced` / `high` |

Optional fields (sent when set on `POST /v1/hdri`):

| Field | Meaning |
|--------|--------|
| `prompt` | Main prompt for img2img / outpainting |
| `negative_prompt` | Negative prompt |
| `seed` | Integer seed |
| `strength` | Img2img strength 0–1 (if your worker supports it) |

Plus any keys from **`panorama_extra`** on the request, and static keys from env **`PANORAMA_HTTP_BODY_JSON`** (merged before; request fields override).

**Response** must include one of: **`image_b64`**, **`image_url`**, **`output_url`** (PNG/JPEG equirectangular or any image the HDR step can interpret).

### Stub worker (test without GPU)

From the `hdri_api_server` folder:

```powershell
cd hdri_api_server
pip install fastapi uvicorn pillow
python -m uvicorn examples.img2pano_worker_stub:app --host 127.0.0.1 --port 8001
```

Point `PANORAMA_HTTP_URL` at `http://127.0.0.1:8001/v1/panorama`. The stub only **resizes** the input to 2:1 so you can verify wiring; replace `panorama()` with your DiT360 / ComfyUI pipeline.

---

## Panorama diffusion (before HDR lift)

The pipeline is: **photo → 2:1 equirectangular image → HDR lift → `.hdr`**.

Set **`PANORAMA_MODE`**:

| Mode | What it does |
|------|----------------|
| `resize` | **Default.** Stretch input to 1024×512 (no external API). |
| `replicate` | Calls **[Replicate](https://replicate.com)** predictions API, then resizes to 1024×512. |
| `http_json` | `POST` JSON to **`PANORAMA_HTTP_URL`**; expects `image_b64` or `image_url` / `output_url` back. |
| `hf_dit360` | **[Hugging Face Inference API](https://huggingface.co/docs/api-inference/tasks/text-to-image)** for **`Insta360-Research/DiT360-Panorama-Image-Generation`**. |

### Hugging Face DiT360 (`hf_dit360`)

Insta360 documents using DiT360 as a component with Hub access. This server calls the **text-to-image** inference endpoint:

- **Endpoint (default):** `https://api-inference.huggingface.co/models/Insta360-Research/DiT360-Panorama-Image-Generation`  
  Override with **`HF_INFERENCE_URL`** if you use a [Dedicated Inference Endpoint](https://huggingface.co/docs/inference-endpoints/index) or a future router URL.

**Important caveats**

1. **Deployment:** On the [model card](https://huggingface.co/Insta360-Research/DiT360-Panorama-Image-Generation), Hugging Face may show *“This model isn't deployed by any Inference Provider”*. In that case **serverless API calls can fail** until a provider hosts the model, or you deploy it yourself on **Inference Endpoints** / run the [GitHub](https://github.com/Insta360-Research-Team/DiT360) code locally.
2. **Image conditioning:** The official `inference.py` example is **text-to-panorama** only. Your **uploaded photo is not sent** to DiT360 in this mode (unless you later add e.g. captioning + prompt). For **image → panorama** with conditioning, use **`replicate`**, **`http_json`** to your own worker, or **local** DiT360 with their full pipeline.

**Environment (Windows-friendly)**

```powershell
$env:PANORAMA_MODE="hf_dit360"
$env:HF_API_TOKEN="hf_..."   # token with permission to call Inference API / providers
# Optional:
# $env:HF_DIT360_PROMPT="Your full prompt overriding scene defaults"
# $env:HF_GUIDANCE_SCALE="2.8"
# $env:HF_NUM_INFERENCE_STEPS="28"
# $env:HF_INFERENCE_RETRIES="6"
# $env:HF_INFERENCE_RETRY_DELAY_S="3"
```

### Replicate

1. Pick a model on Replicate that outputs a **360° / equirectangular** image (or image URL). Copy the **version id** (long hash).
2. Set:

```powershell
$env:PANORAMA_MODE="replicate"
$env:REPLICATE_API_TOKEN="r8_..."
$env:REPLICATE_MODEL_VERSION="<version-hash>"
```

Optional:

- **`REPLICATE_PROMPT`** — overrides the default text prompt (if you do not use `REPLICATE_INPUT_JSON`).
- **`REPLICATE_INPUT_JSON`** — full JSON object merged into Replicate `input` (for model-specific keys). If you omit the image field, the server injects a JPEG **data URI** into **`REPLICATE_IMAGE_FIELD`** (default `image`).
- **`REPLICATE_IMAGE_FIELD`** — e.g. `image` or `input_image` depending on the model.
- **`REPLICATE_POLL_TIMEOUT_S`** — default `300`.
- **`REPLICATE_POLL_INTERVAL_S`** — default `1.0`.

### Generic HTTP JSON

Your service receives:

```json
{
  "image_b64": "<same as client sent>",
  "width": 1024,
  "height": 512,
  "scene_mode": "auto|outdoor|indoor|studio",
  "quality_mode": "fast|balanced|high"
}
```

Merged with optional **`PANORAMA_HTTP_BODY_JSON`** (object).

Set:

```powershell
$env:PANORAMA_MODE="http_json"
$env:PANORAMA_HTTP_URL="https://your-api.example.com/v1/panorama"
$env:PANORAMA_HTTP_API_KEY="..."   # optional
$env:PANORAMA_HTTP_HEADERS_JSON='{"X-Custom":"1"}'   # optional
```

Response must include one of: **`image_b64`**, **`image_url`**, **`output_url`**.

The `POST /v1/hdri` response includes **`panorama_mode`** so you can confirm which path ran.
