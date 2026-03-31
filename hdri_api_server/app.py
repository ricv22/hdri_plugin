import base64
import hashlib
import hmac
import os
import threading
import time
import uuid
from typing import Any, Literal


def _load_local_env() -> None:
    """Load optional ``.env`` next to this file (KEY=value). Does not override existing OS env."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.isfile(path):
        return
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
    except OSError:
        pass


_load_local_env()

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from PIL import Image

from panorama import build_equirectangular, get_mode
from rgbe_hdr import write_rgbe_hdr

APP_NAME = "HDRI API Server (MVP)"

# Storage
DATA_DIR = os.environ.get("HDRI_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
os.makedirs(DATA_DIR, exist_ok=True)

# Signed URL (HMAC)
SIGNING_SECRET = os.environ.get("HDRI_SIGNING_SECRET", "dev-secret-change-me").encode("utf-8")
SIGNED_URL_TTL_S = int(os.environ.get("HDRI_SIGNED_URL_TTL_S", "3600"))

# Public base URL (used to build download URL)
PUBLIC_BASE_URL = os.environ.get("HDRI_PUBLIC_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


class HdriRequest(BaseModel):
    provider: Literal["D"] = "D"
    image_b64: str = Field(..., description="Base64-encoded input image bytes.")

    scene_mode: Literal["auto", "outdoor", "indoor", "studio"] = "auto"
    quality_mode: Literal["fast", "balanced", "high"] = "balanced"
    preset: Literal["none", "sunset", "overcast", "dramatic", "studio_soft", "cyberpunk"] = "none"

    output_width: int = 2048
    output_height: int = 1024
    assume_upright: bool = True

    # Only used when PANORAMA_MODE=http_json — forwarded to your img2img / panorama worker
    panorama_prompt: str | None = Field(
        None,
        description="Prompt for image-conditioned panorama (img2img / outpainting).",
    )
    panorama_negative_prompt: str | None = Field(None, description="Negative prompt for the worker.")
    panorama_seed: int | None = Field(None, description="Optional RNG seed for the worker.")
    panorama_strength: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Optional img2img strength (0–1) if the worker supports it.",
    )
    panorama_extra: dict[str, Any] | None = Field(
        None,
        description="Arbitrary extra fields merged into the POST JSON to PANORAMA_HTTP_URL (after env body).",
    )
    erp_layout_mode: str | None = Field(
        None,
        description="Worker ERP placement mode (e.g. single_front).",
    )
    reference_coverage: float | None = Field(
        None,
        ge=0.15,
        le=0.85,
        description="Relative width coverage of source image on ERP control canvas.",
    )
    seam_fix: bool | None = Field(
        None,
        description="If set, overrides worker seam-fix default behavior.",
    )
    erp_canvas_width: int | None = Field(
        None,
        ge=512,
        description="Optional ERP control canvas width; must be 2x erp_canvas_height.",
    )
    erp_canvas_height: int | None = Field(
        None,
        ge=256,
        description="Optional ERP control canvas height; must be 1/2 erp_canvas_width.",
    )

    heuristic_hdr_lift: bool = Field(
        True,
        description="If True, apply server-side HDR boost (_fake_hdr_lift). If False, mild linear scale only (flatter).",
    )
    # Optional baked controls if the client wants the generated file itself adjusted.
    hue_shift: float = Field(0.0, ge=-1.0, le=1.0, description="Hue shift in normalized turns (-1..1).")
    sat_scale: float = Field(1.0, ge=0.0, le=2.0, description="Saturation multiplier for baked output.")
    blur_sigma: float = Field(0.0, ge=0.0, le=16.0, description="Gaussian blur sigma for baked output.")
    color_gain: float = Field(1.0, ge=0.0, le=8.0, description="Post-color gain multiplier for baked output.")


class HdriResponse(BaseModel):
    """Signed download URL. Uses Radiance .hdr (RGBE) — no OpenEXR build on Windows."""

    hdri_url: str
    # Back-compat: same URL as hdri_url (Option B originally said exr_url)
    exr_url: str
    width: int
    height: int
    format: str = "hdr_rgbe"
    # How the 2:1 panorama was produced before HDR lift (see PANORAMA_MODE)
    panorama_mode: str = "resize"


class HdriJobCreateResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running"]


class HdriJobStatusResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    hdri_url: str | None = None
    exr_url: str | None = None
    width: int | None = None
    height: int | None = None
    format: str | None = None
    panorama_mode: str | None = None
    error: str | None = None


def _b64_to_bytes(s: str) -> bytes:
    try:
        return base64.b64decode(s, validate=True)
    except Exception:
        if "," in s:
            try:
                return base64.b64decode(s.split(",", 1)[1], validate=True)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid image_b64: {e}")
        raise HTTPException(status_code=400, detail="Invalid image_b64")


def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1.0 + a)) ** 2.4)


def _apply_preset(rgb_lin: np.ndarray, preset: str) -> np.ndarray:
    if preset == "none":
        return rgb_lin

    out = rgb_lin.copy()

    def lift_gamma_gain(img, lift, gamma, gain):
        img = img * gain + lift
        img = np.clip(img, 0.0, None)
        img = img ** (1.0 / max(gamma, 1e-6))
        return img

    if preset == "sunset":
        out[..., 0] *= 1.15
        out[..., 2] *= 0.95
        out = lift_gamma_gain(out, lift=0.0, gamma=1.05, gain=1.1)
    elif preset == "overcast":
        out = out ** 0.9
        out *= 0.85
    elif preset == "dramatic":
        out = out ** 0.85
        out *= 1.25
    elif preset == "studio_soft":
        gray = out.mean(axis=-1, keepdims=True)
        out = out * 0.75 + gray * 0.25
        out *= 1.05
    elif preset == "cyberpunk":
        out[..., 0] *= 1.10
        out[..., 1] *= 0.95
        out[..., 2] *= 1.15
        out = out ** 0.9
        out *= 1.15

    return np.clip(out, 0.0, None)


def _fake_hdr_lift(rgb_lin: np.ndarray, quality_mode: str) -> np.ndarray:
    if quality_mode == "fast":
        boost, knee = 6.0, 0.6
    elif quality_mode == "balanced":
        boost, knee = 12.0, 0.5
    else:
        boost, knee = 18.0, 0.45

    lum = 0.2126 * rgb_lin[..., 0] + 0.7152 * rgb_lin[..., 1] + 0.0722 * rgb_lin[..., 2]
    lum = lum[..., None]
    t = np.clip((lum - knee) / max(1e-6, (1.0 - knee)), 0.0, 1.0)
    gain = 1.0 + t * boost
    out = rgb_lin * gain

    h = out.shape[0]
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    sky = 1.0 + 0.35 * (1.0 - y)
    ground = 1.0 - 0.15 * y
    out = out * (sky * ground)

    return np.clip(out, 0.0, None).astype(np.float32)


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    cmax = np.max(rgb, axis=-1)
    cmin = np.min(rgb, axis=-1)
    delta = cmax - cmin

    h = np.zeros_like(cmax)
    mask = delta > 1e-8
    rmask = mask & (cmax == r)
    gmask = mask & (cmax == g)
    bmask = mask & (cmax == b)
    h[rmask] = ((g[rmask] - b[rmask]) / delta[rmask]) % 6.0
    h[gmask] = ((b[gmask] - r[gmask]) / delta[gmask]) + 2.0
    h[bmask] = ((r[bmask] - g[bmask]) / delta[bmask]) + 4.0
    h = (h / 6.0) % 1.0

    s = np.zeros_like(cmax)
    nz = cmax > 1e-8
    s[nz] = delta[nz] / cmax[nz]
    v = cmax
    return np.stack([h, s, v], axis=-1)


def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    h = (hsv[..., 0] % 1.0) * 6.0
    s = np.clip(hsv[..., 1], 0.0, 1.0)
    v = np.clip(hsv[..., 2], 0.0, None)
    i = np.floor(h).astype(np.int32)
    f = h - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    i_mod = i % 6
    out = np.zeros_like(hsv)
    out[i_mod == 0] = np.stack([v, t, p], axis=-1)[i_mod == 0]
    out[i_mod == 1] = np.stack([q, v, p], axis=-1)[i_mod == 1]
    out[i_mod == 2] = np.stack([p, v, t], axis=-1)[i_mod == 2]
    out[i_mod == 3] = np.stack([p, q, v], axis=-1)[i_mod == 3]
    out[i_mod == 4] = np.stack([t, p, v], axis=-1)[i_mod == 4]
    out[i_mod == 5] = np.stack([v, p, q], axis=-1)[i_mod == 5]
    return out


def _apply_baked_adjustments(rgb_lin: np.ndarray, req: HdriRequest) -> np.ndarray:
    out = rgb_lin
    if req.blur_sigma > 0:
        tmp = np.clip(out, 0.0, 1.0)
        pil = Image.fromarray((tmp * 255.0).astype(np.uint8), mode="RGB")
        # Pillow ImageFilter import kept local to avoid startup overhead.
        from PIL import ImageFilter

        pil = pil.filter(ImageFilter.GaussianBlur(radius=req.blur_sigma))
        out = np.asarray(pil).astype(np.float32) / 255.0

    if abs(req.hue_shift) > 1e-6 or abs(req.sat_scale - 1.0) > 1e-6:
        hsv = _rgb_to_hsv(np.clip(out, 0.0, None))
        hsv[..., 0] = (hsv[..., 0] + req.hue_shift) % 1.0
        hsv[..., 1] = np.clip(hsv[..., 1] * req.sat_scale, 0.0, 1.0)
        out = _hsv_to_rgb(hsv)

    if abs(req.color_gain - 1.0) > 1e-6:
        out = out * req.color_gain
    return np.clip(out, 0.0, None).astype(np.float32)


def _validate_output_size(width: int, height: int) -> None:
    allowed = {(1024, 512), (2048, 1024), (4096, 2048)}
    if (width, height) not in allowed:
        raise HTTPException(
            status_code=400,
            detail="Output size must be one of 1024x512, 2048x1024, or 4096x2048.",
        )
    if width != 2 * height:
        raise HTTPException(status_code=400, detail="Output must use 2:1 equirectangular ratio.")


def _sign(file_id: str, exp: int) -> str:
    msg = f"{file_id}:{exp}".encode("utf-8")
    return hmac.new(SIGNING_SECRET, msg, hashlib.sha256).hexdigest()


def _verify(file_id: str, exp: int, sig: str) -> bool:
    if exp < int(time.time()):
        return False
    expected = _sign(file_id, exp)
    return hmac.compare_digest(expected, sig)


app = FastAPI(title=APP_NAME)
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def _build_panorama_overrides(req: HdriRequest) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if req.panorama_prompt is not None:
        overrides["prompt"] = req.panorama_prompt
    if req.panorama_negative_prompt is not None:
        overrides["negative_prompt"] = req.panorama_negative_prompt
    if req.panorama_seed is not None:
        overrides["seed"] = req.panorama_seed
    if req.panorama_strength is not None:
        overrides["strength"] = req.panorama_strength
    if req.erp_layout_mode is not None:
        overrides["erp_layout_mode"] = req.erp_layout_mode
    if req.reference_coverage is not None:
        overrides["reference_coverage"] = req.reference_coverage
    if req.seam_fix is not None:
        overrides["seam_fix"] = req.seam_fix
    if req.erp_canvas_width is not None:
        overrides["erp_canvas_width"] = req.erp_canvas_width
    if req.erp_canvas_height is not None:
        overrides["erp_canvas_height"] = req.erp_canvas_height
    if req.panorama_extra:
        overrides.update(req.panorama_extra)
    return overrides


def _generate_hdri(req: HdriRequest) -> HdriResponse:
    _validate_output_size(req.output_width, req.output_height)

    panorama_overrides = _build_panorama_overrides(req)

    try:
        im, pano_mode = build_equirectangular(
            req.image_b64,
            req.output_width,
            req.output_height,
            req.scene_mode,
            req.quality_mode,
            http_json_overrides=panorama_overrides or None,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Panorama step failed: {e}") from e

    im = im.convert("RGB")

    rgb = np.asarray(im).astype(np.float32) / 255.0
    rgb_lin = _srgb_to_linear(rgb)
    rgb_lin = _apply_preset(rgb_lin, req.preset)
    rgb_lin = _apply_baked_adjustments(rgb_lin, req)
    if req.heuristic_hdr_lift:
        rgb_hdr = _fake_hdr_lift(rgb_lin, req.quality_mode)
    else:
        # Flatter: linear radiance ~ display linear, small headroom (user can raise Exposure in Blender)
        rgb_hdr = np.clip(rgb_lin.astype(np.float32) * 2.5, 0.0, None)

    file_id = str(uuid.uuid4())
    hdr_path = os.path.join(DATA_DIR, f"{file_id}.hdr")
    write_rgbe_hdr(hdr_path, rgb_hdr)

    exp = int(time.time()) + SIGNED_URL_TTL_S
    sig = _sign(file_id, exp)
    url = f"{PUBLIC_BASE_URL}/v1/files/{file_id}.hdr?exp={exp}&sig={sig}"

    return HdriResponse(
        hdri_url=url,
        exr_url=url,
        width=req.output_width,
        height=req.output_height,
        format="hdr_rgbe",
        panorama_mode=pano_mode,
    )


@app.get("/v1/config")
def config():
    """Non-secret hints for debugging (which panorama backend is active)."""
    return {
        "panorama_mode": get_mode(),
        "note": "Set PANORAMA_MODE=replicate | http_json | hf_dit360; see README.",
    }


@app.post("/v1/hdri", response_model=HdriResponse)
def create_hdri(req: HdriRequest):
    return _generate_hdri(req)


def _run_job(job_id: str, req: HdriRequest) -> None:
    with _jobs_lock:
        _jobs[job_id]["status"] = "running"
        _jobs[job_id]["updated_at"] = int(time.time())
    try:
        result = _generate_hdri(req)
        with _jobs_lock:
            _jobs[job_id]["status"] = "succeeded"
            _jobs[job_id]["result"] = result.model_dump()
            _jobs[job_id]["updated_at"] = int(time.time())
    except Exception as e:
        with _jobs_lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(e)
            _jobs[job_id]["updated_at"] = int(time.time())


@app.post("/v1/jobs/hdri", response_model=HdriJobCreateResponse)
def create_hdri_job(req: HdriRequest):
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued",
            "created_at": int(time.time()),
            "updated_at": int(time.time()),
            "result": None,
            "error": None,
        }
    t = threading.Thread(target=_run_job, args=(job_id, req), daemon=True)
    t.start()
    return HdriJobCreateResponse(job_id=job_id, status="queued")


@app.get("/v1/jobs/{job_id}", response_model=HdriJobStatusResponse)
def get_hdri_job(job_id: str):
    with _jobs_lock:
        row = _jobs.get(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="Job not found.")
    if row["status"] == "succeeded" and row["result"]:
        return HdriJobStatusResponse(job_id=job_id, status="succeeded", **row["result"])
    return HdriJobStatusResponse(
        job_id=job_id,
        status=row["status"],
        error=row.get("error"),
    )


@app.get("/v1/files/{file_name}")
def get_file(file_name: str, exp: int, sig: str):
    if not (file_name.endswith(".hdr") or file_name.endswith(".exr")):
        raise HTTPException(status_code=400, detail="Only .hdr or .exr is supported.")

    ext = os.path.splitext(file_name)[1]
    file_id = os.path.splitext(file_name)[0]

    if not _verify(file_id, exp, sig):
        raise HTTPException(status_code=403, detail="Invalid or expired URL.")

    disk_path = os.path.join(DATA_DIR, f"{file_id}{ext}")
    if not os.path.exists(disk_path):
        raise HTTPException(status_code=404, detail="Not found.")

    if file_name.endswith(".hdr"):
        media = "image/vnd.radiance"
    else:
        media = "image/x-exr"

    return FileResponse(disk_path, media_type=media, filename=file_name)
