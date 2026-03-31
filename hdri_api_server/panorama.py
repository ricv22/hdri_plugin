"""
Pluggable equirectangular (2:1) panorama generation before HDR lift.

Modes (env PANORAMA_MODE):
  - resize   — stretch/crop input to target size (no external API; default)
  - replicate — Replicate predictions API (set REPLICATE_API_TOKEN + REPLICATE_MODEL_VERSION)
  - http_json — POST JSON to your URL; response image_b64 or image_url
  - hf_dit360 — Hugging Face Inference API for Insta360 DiT360 (text-to-panorama; see README)

All modes must yield a PIL Image RGB at the requested width × height.
"""
from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.parse
import urllib.error
import urllib.request
from typing import Any

from PIL import Image


def get_mode() -> str:
    return os.environ.get("PANORAMA_MODE", "resize").strip().lower()


def image_to_jpeg_data_uri(im: Image.Image, quality: int = 92) -> str:
    buf = io.BytesIO()
    im_rgb = im.convert("RGB")
    im_rgb.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _download_url(url: str, timeout_s: int = 120) -> bytes:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return resp.read()


def _http_json_request(url: str, payload: dict, headers: dict[str, str], timeout_s: int = 120) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in headers.items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _log_provider_event(provider: str, event: str, **fields: Any) -> None:
    """Simple structured logs for provider comparisons and failure triage."""
    safe_fields = {k: v for k, v in fields.items() if v is not None}
    print(json.dumps({"provider": provider, "event": event, **safe_fields}))


def panorama_resize(im: Image.Image, width: int, height: int) -> Image.Image:
    return im.convert("RGB").resize((width, height), resample=Image.BICUBIC)


def panorama_replicate(
    im: Image.Image,
    width: int,
    height: int,
    scene_mode: str,
    quality_mode: str,
    request_overrides: dict[str, Any] | None = None,
) -> Image.Image:
    token = os.environ.get("REPLICATE_API_TOKEN", "").strip()
    version = os.environ.get("REPLICATE_MODEL_VERSION", "").strip()
    if not token or not version:
        raise RuntimeError(
            "PANORAMA_MODE=replicate requires REPLICATE_API_TOKEN and REPLICATE_MODEL_VERSION "
            "(model version hash from replicate.com)"
        )

    # Build model-specific input; override with REPLICATE_INPUT_JSON if set.
    # Suggested baseline model/version for MVP:
    # pearsonkyle/360-panorama:5f638558adec78589e947290cb7083b0730dcfc8e8c9c9db6eebc5261edc09ca
    data_uri = image_to_jpeg_data_uri(im)
    input_obj: dict[str, Any] = {}

    extra = os.environ.get("REPLICATE_INPUT_JSON", "").strip()
    if extra:
        input_obj = json.loads(extra)
        field = os.environ.get("REPLICATE_IMAGE_FIELD", "image").strip() or "image"
        if field not in input_obj and "image" not in input_obj and "input_image" not in input_obj:
            input_obj[field] = data_uri
    else:
        # Sensible defaults — many 360 / img2pano models use "image" + optional prompt
        field = os.environ.get("REPLICATE_IMAGE_FIELD", "image").strip() or "image"
        input_obj[field] = data_uri
        prompt = os.environ.get("REPLICATE_PROMPT", "").strip()
        if not prompt:
            prompt = (
                "seamless 360 degree equirectangular panorama, photorealistic, "
                "full spherical projection, no seams at the horizon"
            )
        if scene_mode == "outdoor":
            prompt += ", outdoor, natural sky"
        elif scene_mode == "indoor":
            prompt += ", indoor architecture"
        elif scene_mode == "studio":
            prompt += ", studio lighting, clean backdrop"
        if quality_mode == "high":
            prompt += ", highly detailed lighting"
        elif quality_mode == "fast":
            prompt += ", simple lighting"
        input_obj["prompt"] = prompt

    # Per-request prompt overrides (shared with API panorama_* fields).
    if request_overrides:
        if isinstance(request_overrides.get("prompt"), str):
            input_obj["prompt"] = request_overrides["prompt"]
        if isinstance(request_overrides.get("negative_prompt"), str):
            input_obj["negative_prompt"] = request_overrides["negative_prompt"]
        if request_overrides.get("seed") is not None:
            input_obj["seed"] = request_overrides["seed"]
        if request_overrides.get("strength") is not None:
            input_obj["strength"] = request_overrides["strength"]
        # Preserve any extra model-specific knobs.
        for k, v in request_overrides.items():
            if k not in ("prompt", "negative_prompt", "seed", "strength"):
                input_obj[k] = v

    # Allow forcing output size if the model supports width/height
    for key_w, key_h in (("width", "height"), ("output_width", "output_height")):
        if key_w not in input_obj and key_h not in input_obj:
            input_obj[key_w] = width
            input_obj[key_h] = height
            break

    create_url = "https://api.replicate.com/v1/predictions"
    body = json.dumps({"version": version, "input": input_obj}).encode("utf-8")
    req = urllib.request.Request(create_url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Token {token}")

    start_s = time.time()
    _log_provider_event(
        "replicate",
        "create_prediction",
        model_version=version,
        create_url=create_url,
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            pred = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")[:2000]
        _log_provider_event("replicate", "create_failed", status_code=e.code, error=err_body[:600])
        raise RuntimeError(f"Replicate create failed: {e.code} {err_body}") from e

    pred_id = pred.get("id")
    if not pred_id:
        raise RuntimeError(f"Replicate: no prediction id in response: {pred}")

    poll_url = f"https://api.replicate.com/v1/predictions/{pred_id}"
    deadline = time.time() + float(os.environ.get("REPLICATE_POLL_TIMEOUT_S", "300"))
    poll_interval = float(os.environ.get("REPLICATE_POLL_INTERVAL_S", "1.0"))

    while time.time() < deadline:
        req2 = urllib.request.Request(poll_url, method="GET")
        req2.add_header("Authorization", f"Token {token}")
        with urllib.request.urlopen(req2, timeout=60) as resp2:
            pred = json.loads(resp2.read().decode("utf-8"))
        status = pred.get("status")
        if status == "succeeded":
            out = pred.get("output")
            break
        if status in ("failed", "canceled"):
            _log_provider_event(
                "replicate",
                "prediction_failed",
                prediction_id=pred_id,
                status=status,
                error=pred.get("error"),
                latency_ms=int((time.time() - start_s) * 1000),
            )
            raise RuntimeError(f"Replicate prediction {status}: {pred.get('error') or pred}")
        time.sleep(poll_interval)
    else:
        _log_provider_event(
            "replicate",
            "prediction_timeout",
            prediction_id=pred_id,
            timeout_s=os.environ.get("REPLICATE_POLL_TIMEOUT_S", "300"),
            latency_ms=int((time.time() - start_s) * 1000),
        )
        raise RuntimeError("Replicate: polling timed out")

    # output: URL string, list of URLs, or nested
    url: str | None = None
    if isinstance(out, str) and out.startswith("http"):
        url = out
    elif isinstance(out, list) and out:
        url = out[0] if isinstance(out[0], str) else None
    if not url and isinstance(out, dict):
        for k in ("url", "image", "output", "output_url"):
            v = out.get(k)
            if isinstance(v, str) and v.startswith("http"):
                url = v
                break
    if not url:
        raise RuntimeError(f"Replicate: unexpected output shape: {out!r}")

    _log_provider_event(
        "replicate",
        "prediction_succeeded",
        prediction_id=pred_id,
        output_url_host=urllib.parse.urlparse(url).netloc,
        latency_ms=int((time.time() - start_s) * 1000),
    )
    raw = _download_url(url, timeout_s=120)
    pano = Image.open(io.BytesIO(raw))
    return pano.convert("RGB").resize((width, height), resample=Image.BICUBIC)


def _dit360_prompt(scene_mode: str) -> str:
    custom = os.environ.get("HF_DIT360_PROMPT", "").strip()
    if custom:
        return custom
    base = (
        "This is a panorama. The image shows a photorealistic seamless 360-degree equirectangular "
        "environment in 2:1 aspect ratio, level horizon, continuous left-right wrap, no visible seam."
    )
    if scene_mode == "outdoor":
        return base + " Outdoor scene, natural sky and landscape."
    if scene_mode == "indoor":
        return base + " Indoor architectural interior."
    if scene_mode == "studio":
        return base + " Professional studio environment, soft even lighting."
    return base + " Detailed immersive environment."


def panorama_hf_dit360(
    _im: Image.Image,
    width: int,
    height: int,
    scene_mode: str,
    quality_mode: str,
) -> Image.Image:
    """
    Call Hugging Face Inference API (text-to-image) for DiT360 model weights.

    Note: The public `inference.py` for DiT360 is **text-to-panorama** only.
    The uploaded photo is **not** sent to HF unless you add a captioning step yourself.
    HF may return 503 while a cold model loads — we retry a few times.

    Model card: https://huggingface.co/Insta360-Research/DiT360-Panorama-Image-Generation
    If the hub says "not deployed by any Inference Provider", requests fail until HF/providers enable it,
    or you use a dedicated Inference Endpoint / self-host.
    """
    token = os.environ.get("HF_API_TOKEN", os.environ.get("HUGGING_FACE_HUB_TOKEN", "")).strip()
    if not token:
        raise RuntimeError(
            "PANORAMA_MODE=hf_dit360 requires HF_API_TOKEN (Hugging Face access token with inference permissions)."
        )

    model_id = os.environ.get("HF_MODEL_ID", "Insta360-Research/DiT360-Panorama-Image-Generation").strip()
    base_url = os.environ.get(
        "HF_INFERENCE_URL",
        f"https://api-inference.huggingface.co/models/{model_id}",
    ).strip()

    prompt = _dit360_prompt(scene_mode)
    guidance = float(os.environ.get("HF_GUIDANCE_SCALE", "2.8"))
    steps = int(os.environ.get("HF_NUM_INFERENCE_STEPS", "28"))
    if quality_mode == "fast":
        steps = min(steps, 16)
    elif quality_mode == "high":
        steps = max(steps, 28)

    payload = {
        "inputs": prompt,
        "parameters": {
            "width": width,
            "height": height,
            "guidance_scale": guidance,
            "num_inference_steps": steps,
        },
    }

    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    max_retries = int(os.environ.get("HF_INFERENCE_RETRIES", "6"))
    delay_s = float(os.environ.get("HF_INFERENCE_RETRY_DELAY_S", "3.0"))

    last_err: str | None = None
    for attempt in range(max_retries):
        req = urllib.request.Request(base_url, data=body, method="POST")
        for k, v in headers.items():
            req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                raw = resp.read()
            # Success path: image bytes (PNG/WEBP/JPEG)
            try:
                pano = Image.open(io.BytesIO(raw))
                return pano.convert("RGB").resize((width, height), resample=Image.LANCZOS)
            except Exception:
                # Maybe JSON error body
                try:
                    err = json.loads(raw.decode("utf-8"))
                    last_err = str(err.get("error") or err)
                except Exception:
                    last_err = raw[:500].decode("utf-8", errors="replace")
                raise RuntimeError(f"Hugging Face returned non-image body: {last_err}")
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")[:4000]
            last_err = f"HTTP {e.code}: {err_body}"
            if e.code == 503 and attempt < max_retries - 1:
                time.sleep(delay_s * (attempt + 1))
                continue
            raise RuntimeError(f"Hugging Face inference failed: {last_err}") from e

    raise RuntimeError(f"Hugging Face inference failed after retries: {last_err}")


def panorama_http_json(
    image_b64: str,
    width: int,
    height: int,
    scene_mode: str,
    quality_mode: str,
    request_overrides: dict[str, Any] | None = None,
) -> Image.Image:
    url = os.environ.get("PANORAMA_HTTP_URL", "").strip()
    if not url:
        raise RuntimeError("PANORAMA_MODE=http_json requires PANORAMA_HTTP_URL")

    headers: dict[str, str] = {}
    hk = os.environ.get("PANORAMA_HTTP_HEADERS_JSON", "").strip()
    if hk:
        headers = {str(k): str(v) for k, v in json.loads(hk).items()}
    api_key = os.environ.get("PANORAMA_HTTP_API_KEY", "").strip()
    if api_key:
        headers.setdefault("Authorization", f"Bearer {api_key}")

    # Merge order: base → env PANORAMA_HTTP_BODY_JSON → per-request overrides (wins)
    payload: dict[str, Any] = {
        "image_b64": image_b64,
        "width": width,
        "height": height,
        "scene_mode": scene_mode,
        "quality_mode": quality_mode,
    }
    extra = os.environ.get("PANORAMA_HTTP_BODY_JSON", "").strip()
    if extra:
        payload.update(json.loads(extra))
    if request_overrides:
        payload.update(request_overrides)

    data = _http_json_request(url, payload, headers, timeout_s=int(os.environ.get("PANORAMA_HTTP_TIMEOUT_S", "300")))

    if "image_b64" in data:
        raw = base64.b64decode(data["image_b64"])
        pano = Image.open(io.BytesIO(raw))
    elif "image_url" in data:
        raw = _download_url(str(data["image_url"]))
        pano = Image.open(io.BytesIO(raw))
    elif "output_url" in data:
        raw = _download_url(str(data["output_url"]))
        pano = Image.open(io.BytesIO(raw))
    else:
        raise RuntimeError(f"http_json: expected image_b64, image_url, or output_url in response, got keys: {list(data)}")

    return pano.convert("RGB").resize((width, height), resample=Image.BICUBIC)


def _decode_image_b64(image_b64: str) -> bytes:
    try:
        return base64.b64decode(image_b64, validate=True)
    except Exception:
        if "," in image_b64:
            return base64.b64decode(image_b64.split(",", 1)[1])
        raise


def build_equirectangular(
    image_b64: str,
    width: int,
    height: int,
    scene_mode: str,
    quality_mode: str,
    http_json_overrides: dict[str, Any] | None = None,
) -> tuple[Image.Image, str]:
    """
    Returns (PIL RGB image at width×height, mode name used).
    """
    mode = get_mode()
    raw_bytes = _decode_image_b64(image_b64)
    im = Image.open(io.BytesIO(raw_bytes))

    if mode == "resize":
        return panorama_resize(im, width, height), mode
    if mode == "replicate":
        return (
            panorama_replicate(
                im,
                width,
                height,
                scene_mode,
                quality_mode,
                request_overrides=http_json_overrides,
            ),
            mode,
        )
    if mode == "http_json":
        return (
            panorama_http_json(
                image_b64,
                width,
                height,
                scene_mode,
                quality_mode,
                request_overrides=http_json_overrides,
            ),
            mode,
        )
    if mode in ("hf_dit360", "huggingface_dit360", "dit360"):
        return panorama_hf_dit360(im, width, height, scene_mode, quality_mode), "hf_dit360"

    raise RuntimeError(
        f"Unknown PANORAMA_MODE={mode!r}. Use resize | replicate | http_json | hf_dit360"
    )
