"""
Minimal **local panorama worker** for PANORAMA_MODE=http_json (img2img contract).

This does NOT run DiT360 — it only resizes the input to 2:1 and returns PNG base64
so you can verify the HDRI server → worker → HDR lift pipeline end-to-end.

Replace the body of `panorama()` with:
  - ComfyUI API call
  - Insta360 DiT360 pipeline (inpaint/outpaint, green ERP canvas, etc.)
  - Or any service that accepts image_b64 and returns image_b64 / image_url

Run (separate terminal), from the `hdri_api_server` directory:

  pip install fastapi uvicorn pillow
  python -m uvicorn examples.img2pano_worker_stub:app --host 127.0.0.1 --port 8001

Then:

  set PANORAMA_MODE=http_json
  set PANORAMA_HTTP_URL=http://127.0.0.1:8001/v1/panorama
"""

from __future__ import annotations

import base64
import io
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field
from PIL import Image

app = FastAPI(title="Img2Pano worker (stub)", version="0.1.0")


class PanoramaRequest(BaseModel):
    """Subset of fields sent by hdri_api_server/panorama.py — accept extras."""

    image_b64: str
    width: int = 1024
    height: int = 512
    scene_mode: str = "auto"
    quality_mode: str = "balanced"
    prompt: str | None = None
    negative_prompt: str | None = None
    strength: float | None = Field(None, ge=0.0, le=1.0)
    seed: int | None = None

    model_config = {"extra": "allow"}


@app.get("/health")
def health():
    return {"ok": True, "note": "stub worker — replace with real img2img / DiT360 / ComfyUI"}


@app.post("/v1/panorama")
def panorama(body: PanoramaRequest) -> dict[str, Any]:
    # Decode input (accept raw base64 or data URL)
    s = body.image_b64.strip()
    if "," in s:
        s = s.split(",", 1)[1]
    raw = base64.b64decode(s)
    im = Image.open(io.BytesIO(raw)).convert("RGB")

    # --- Replace this block with your real pipeline ---
    # Example: im = run_dit360_outpaint(im, prompt=body.prompt, ...)
    out = im.resize((body.width, body.height), resample=Image.BICUBIC)
    # ---------------------------------------------------

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return {
        "image_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
        "note": "stub: no diffusion; input was only resized",
    }
