from __future__ import annotations

import base64
import copy
import io
import json
import os
import time
import uuid
import urllib.parse
import urllib.request
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image

try:
    from .erp_layout import build_single_front_erp_layout
except Exception:
    from erp_layout import build_single_front_erp_layout


def _load_local_env() -> None:
    """Load optional .env from hdri_api_server root without overriding OS env."""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root_dir, ".env")
    if not os.path.isfile(path):
        return
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
    except OSError:
        pass


_load_local_env()

app = FastAPI(title="ComfyUI Panorama Worker", version="0.1.0")


class PanoramaRequest(BaseModel):
    image_b64: str
    width: int = 2048
    height: int = 1024
    scene_mode: str = "auto"
    quality_mode: str = "balanced"
    prompt: str | None = None
    negative_prompt: str | None = None
    strength: float | None = Field(None, ge=0.0, le=1.0)
    seed: int | None = None

    # ERP placement controls
    erp_layout_mode: str = "single_front"
    reference_coverage: float = Field(0.60, ge=0.15, le=0.85)
    erp_canvas_width: int | None = None
    erp_canvas_height: int | None = None
    seam_fix: bool | None = None

    model_config = {"extra": "allow"}


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _decode_image_b64(s: str) -> bytes:
    t = s.strip()
    if "," in t:
        t = t.split(",", 1)[1]
    return base64.b64decode(t)


def _encode_png_b64(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _quality_steps(mode: str) -> int:
    if mode == "fast":
        return int(_env("COMFYUI_FAST_STEPS", "18"))
    if mode == "high":
        return int(_env("COMFYUI_HIGH_STEPS", "36"))
    return int(_env("COMFYUI_BALANCED_STEPS", "28"))


def _default_seam_fix(mode: str) -> bool:
    if mode == "fast":
        return False
    return True


def _workflow_has_node_type(workflow: dict[str, Any], class_type: str) -> bool:
    for node in workflow.values():
        if isinstance(node, dict) and str(node.get("class_type", "")) == class_type:
            return True
    return False


def _seam_blend_wrap(pano: Image.Image, band_px: int) -> Image.Image:
    """
    Seam smoothing fallback for ERP wraps. This is not full inpaint,
    but reduces visible left/right mismatch for many generations.
    """
    img = pano.convert("RGB")
    w, h = img.size
    if band_px <= 0 or band_px * 2 >= w:
        return img
    px = img.load()
    for x in range(band_px):
        t = (x + 1) / float(band_px + 1)
        lx = x
        rx = w - band_px + x
        for y in range(h):
            lr, lg, lb = px[lx, y]
            rr, rg, rb = px[rx, y]
            nr = int((1.0 - t) * lr + t * rr)
            ng = int((1.0 - t) * lg + t * rg)
            nb = int((1.0 - t) * lb + t * rb)
            px[lx, y] = (nr, ng, nb)
            px[rx, y] = (nr, ng, nb)
    return img


def _json_request(url: str, payload: dict[str, Any], timeout_s: int = 120) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _multipart_upload_image(base_url: str, image_bytes: bytes, filename: str, overwrite: bool = True) -> dict[str, Any]:
    boundary = f"----CursorBoundary{uuid.uuid4().hex}"
    parts = []
    parts.append(f"--{boundary}\r\n".encode("utf-8"))
    parts.append(
        (
            f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
            "Content-Type: image/png\r\n\r\n"
        ).encode("utf-8")
    )
    parts.append(image_bytes)
    parts.append(b"\r\n")
    parts.append(f"--{boundary}\r\n".encode("utf-8"))
    parts.append(
        (
            f'Content-Disposition: form-data; name="overwrite"\r\n\r\n'
            f'{"true" if overwrite else "false"}\r\n'
        ).encode("utf-8")
    )
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(parts)

    upload_url = urllib.parse.urljoin(base_url.rstrip("/") + "/", "upload/image")
    req = urllib.request.Request(upload_url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _load_workflow_template(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _deep_replace(obj: Any, replacements: dict[str, Any]) -> Any:
    if isinstance(obj, dict):
        return {k: _deep_replace(v, replacements) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_replace(v, replacements) for v in obj]
    if isinstance(obj, str):
        if obj in replacements:
            return replacements[obj]
        out = obj
        for k, v in replacements.items():
            if isinstance(v, (str, int, float)):
                out = out.replace(str(k), str(v))
        return out
    return obj


def _coverage_to_fov_deg(reference_coverage: float) -> float:
    # Tuned so 0.40 coverage stays near the previous 85 degree default.
    fov = reference_coverage * 212.5
    return max(35.0, min(140.0, fov))


def _build_panorama_stickers_state_json(
    control_name: str,
    control_subfolder: str,
    width: int,
    reference_coverage: float,
) -> str:
    asset_id = "asset_uploaded"
    sticker_id = "st_uploaded"
    fov_deg = _coverage_to_fov_deg(reference_coverage)
    state = {
        "version": 1,
        "projection_model": "pinhole_rectilinear",
        "alpha_mode": "straight",
        "bg_color": "#00ff00",
        "output_preset": int(width),
        "assets": {
            asset_id: {
                "type": "comfy_image",
                "filename": control_name,
                "subfolder": control_subfolder,
                "storage": "input",
                "name": control_name,
            }
        },
        "stickers": [
            {
                "id": sticker_id,
                "asset_id": asset_id,
                "yaw_deg": 0.0,
                "pitch_deg": 0.0,
                "hFOV_deg": fov_deg,
                "vFOV_deg": fov_deg,
                "rot_deg": 0.0,
                "z_index": 1,
            }
        ],
        "shots": [],
        "ui_settings": {
            "invert_view_x": False,
            "invert_view_y": False,
            "preview_quality": "balanced",
        },
        "active": {"selected_sticker_id": sticker_id, "selected_shot_id": None},
    }
    return json.dumps(state, separators=(",", ":"))


def _adapt_api_workflow_for_worker(
    workflow: dict[str, Any],
    *,
    control_name: str,
    control_subfolder: str,
    request_prompt: str,
    request_neg: str,
    seed: int,
    strength: float,
    steps: int,
    cfg: float,
    body_width: int,
    body_height: int,
    reference_coverage: float,
    lora_name: str,
    base_model: str,
    clip_name1: str,
    clip_name2: str,
    vae_name: str,
) -> tuple[dict[str, Any], list[str]]:
    vae_decode_id: str | None = None
    panorama_cutout_ids: set[str] = set()

    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        class_type = str(node.get("class_type", ""))
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue

        if class_type == "VAEDecode" and vae_decode_id is None:
            vae_decode_id = node_id
        if class_type == "PanoramaCutout":
            panorama_cutout_ids.add(node_id)

        if class_type == "CLIPTextEncode":
            title = str(node.get("_meta", {}).get("title", "")).lower()
            if "negative" in title:
                inputs["text"] = request_neg
            else:
                inputs["text"] = request_prompt
        elif class_type == "KSampler":
            inputs["seed"] = seed
            inputs["steps"] = steps
            inputs["cfg"] = cfg
            inputs["denoise"] = strength
        elif class_type == "CLIPTextEncodeFlux":
            inputs["clip_l"] = request_neg if "negative" in str(node.get("_meta", {}).get("title", "")).lower() else request_prompt
            inputs["t5xxl"] = request_neg if "negative" in str(node.get("_meta", {}).get("title", "")).lower() else request_prompt
            inputs["guidance"] = cfg
        elif class_type == "LoraLoaderModelOnly":
            if lora_name:
                inputs["lora_name"] = lora_name
            inputs["strength_model"] = 1.0
        elif class_type == "UNETLoader":
            if base_model:
                inputs["unet_name"] = base_model
        elif class_type == "CLIPLoader":
            if clip_name1:
                inputs["clip_name"] = clip_name1
        elif class_type == "DualCLIPLoader":
            if clip_name1:
                inputs["clip_name1"] = clip_name1
            if clip_name2:
                inputs["clip_name2"] = clip_name2
            inputs["type"] = "flux"
        elif class_type == "VAELoader":
            if vae_name:
                inputs["vae_name"] = vae_name
        elif class_type == "ModelSamplingFlux":
            inputs["width"] = body_width
            inputs["height"] = body_height
        elif class_type == "PanoramaStickers":
            inputs["output_preset"] = f"{body_width} x {body_height}"
            inputs["bg_color"] = "#00ff00"
            inputs["state_json"] = _build_panorama_stickers_state_json(
                control_name=control_name,
                control_subfolder=control_subfolder,
                width=body_width,
                reference_coverage=reference_coverage,
            )

    preferred_save_ids: list[str] = []
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        if str(node.get("class_type", "")) != "SaveImage":
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        link = inputs.get("images")
        if (
            isinstance(link, list)
            and len(link) >= 1
            and isinstance(link[0], str)
            and link[0] in panorama_cutout_ids
            and vae_decode_id
        ):
            # Force full panorama output instead of PanoramaCutout output.
            inputs["images"] = [vae_decode_id, 0]
        preferred_save_ids.append(node_id)

    return workflow, preferred_save_ids


def _extract_output_image(base_url: str, history_row: dict[str, Any], preferred_node_ids: list[str] | None = None) -> bytes:
    outputs = history_row.get("outputs", {})
    ordered_ids: list[str] = []
    if preferred_node_ids:
        ordered_ids.extend([nid for nid in preferred_node_ids if nid in outputs])
    ordered_ids.extend([nid for nid in outputs.keys() if nid not in ordered_ids])

    for node_id in ordered_ids:
        node_data = outputs.get(node_id) or {}
        imgs = node_data.get("images") or []
        if imgs and isinstance(imgs, list):
            item = imgs[-1]
            filename = item.get("filename")
            subfolder = item.get("subfolder", "")
            ftype = item.get("type", "output")
            if filename:
                q = urllib.parse.urlencode(
                    {
                        "filename": filename,
                        "subfolder": subfolder,
                        "type": ftype,
                    }
                )
                view_url = urllib.parse.urljoin(base_url.rstrip("/") + "/", f"view?{q}")
                with urllib.request.urlopen(view_url, timeout=120) as resp:
                    return resp.read()
    raise RuntimeError("ComfyUI history had no usable output images.")


def run_comfyui_generation(
    body: PanoramaRequest,
    control_png: bytes,
    mask_png: bytes,
) -> Image.Image:
    base_url = _env("COMFYUI_SERVER_URL", "http://127.0.0.1:8188")
    template_path = _env(
        "COMFYUI_WORKFLOW_TEMPLATE",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "comfyui_flux2_klein_template.json"),
    )
    if not os.path.isfile(template_path):
        raise RuntimeError(
            "ComfyUI workflow template not found. Set COMFYUI_WORKFLOW_TEMPLATE to your workflow JSON."
        )

    uploaded_control = _multipart_upload_image(base_url, control_png, f"erp_control_{uuid.uuid4().hex}.png")
    uploaded_mask = _multipart_upload_image(base_url, mask_png, f"erp_mask_{uuid.uuid4().hex}.png")
    control_name = uploaded_control.get("name") or uploaded_control.get("filename")
    control_subfolder = str(uploaded_control.get("subfolder", ""))
    mask_name = uploaded_mask.get("name") or uploaded_mask.get("filename")
    if not control_name or not mask_name:
        raise RuntimeError("ComfyUI upload did not return image names.")

    workflow = _load_workflow_template(template_path)
    request_prompt = (body.prompt or _env("COMFYUI_DEFAULT_PROMPT")).strip()
    if not request_prompt:
        request_prompt = (
            "photorealistic seamless 360 degree equirectangular panorama, "
            "coherent lighting, continuous horizon, no visible seam"
        )
    request_neg = (body.negative_prompt or _env("COMFYUI_DEFAULT_NEGATIVE_PROMPT")).strip()
    seed = body.seed if body.seed is not None else int(time.time()) % 2_147_483_647
    strength = body.strength if body.strength is not None else float(_env("COMFYUI_DEFAULT_STRENGTH", "1.0"))
    steps = _quality_steps(body.quality_mode)
    cfg = float(_env("COMFYUI_CFG", "3.0"))
    lora_name = _env("COMFYUI_KLEIN_LORA", "")
    base_model = _env("COMFYUI_BASE_MODEL", "")
    clip_name1 = _env("COMFYUI_CLIP_NAME1", "")
    clip_name2 = _env("COMFYUI_CLIP_NAME2", clip_name1)
    vae_name = _env("COMFYUI_VAE_NAME", "")

    # Panorama outpainting generally needs high denoise; lower values often preserve
    # the green control canvas too aggressively.
    if body.strength is None and _workflow_has_node_type(workflow, "PanoramaStickers"):
        strength = max(0.95, strength)

    replacements: dict[str, Any] = {
        "__CONTROL_IMAGE_NAME__": control_name,
        "__MASK_IMAGE_NAME__": mask_name,
        "__PROMPT__": request_prompt,
        "__NEGATIVE_PROMPT__": request_neg,
        "__SEED__": seed,
        "__STRENGTH__": strength,
        "__WIDTH__": body.width,
        "__HEIGHT__": body.height,
        "__STEPS__": steps,
        "__CFG__": cfg,
        "__LORA_NAME__": lora_name,
        "__BASE_MODEL__": base_model,
        "__CLIP_NAME1__": clip_name1,
        "__CLIP_NAME2__": clip_name2,
        "__VAE_NAME__": vae_name,
    }
    workflow = _deep_replace(copy.deepcopy(workflow), replacements)
    workflow, preferred_output_node_ids = _adapt_api_workflow_for_worker(
        workflow,
        control_name=control_name,
        control_subfolder=control_subfolder,
        request_prompt=request_prompt,
        request_neg=request_neg,
        seed=seed,
        strength=strength,
        steps=steps,
        cfg=cfg,
        body_width=body.width,
        body_height=body.height,
        reference_coverage=body.reference_coverage,
        lora_name=lora_name,
        base_model=base_model,
        clip_name1=clip_name1,
        clip_name2=clip_name2,
        vae_name=vae_name,
    )

    client_id = str(uuid.uuid4())
    submit_url = urllib.parse.urljoin(base_url.rstrip("/") + "/", "prompt")
    submitted = _json_request(submit_url, {"prompt": workflow, "client_id": client_id}, timeout_s=120)
    prompt_id = submitted.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"ComfyUI prompt submission failed: {submitted}")

    history_url = urllib.parse.urljoin(base_url.rstrip("/") + "/", f"history/{prompt_id}")
    timeout_s = int(_env("COMFYUI_POLL_TIMEOUT_S", "900"))
    interval_s = float(_env("COMFYUI_POLL_INTERVAL_S", "1.5"))
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with urllib.request.urlopen(history_url, timeout=120) as resp:
            history = json.loads(resp.read().decode("utf-8"))
        row = history.get(prompt_id)
        if row and isinstance(row, dict):
            status = row.get("status", {})
            if status.get("status_str") in ("success", "completed") or row.get("outputs"):
                raw = _extract_output_image(base_url, row, preferred_node_ids=preferred_output_node_ids)
                out = Image.open(io.BytesIO(raw)).convert("RGB")
                return out.resize((body.width, body.height), resample=Image.LANCZOS)
            if status.get("status_str") in ("error", "failed"):
                raise RuntimeError(f"ComfyUI generation failed: {status}")
        time.sleep(interval_s)
    raise RuntimeError("ComfyUI generation timed out.")


@app.get("/health")
def health():
    return {
        "ok": True,
        "mode": "comfyui_local",
        "comfyui_server_url": _env("COMFYUI_SERVER_URL", "http://127.0.0.1:8188"),
        "workflow_template": _env("COMFYUI_WORKFLOW_TEMPLATE", "examples/comfyui_flux2_klein_template.json"),
    }


@app.post("/v1/panorama")
def panorama(body: PanoramaRequest) -> dict[str, Any]:
    if body.width != 2 * body.height:
        raise HTTPException(status_code=400, detail="Worker expects 2:1 equirectangular target.")

    try:
        src = Image.open(io.BytesIO(_decode_image_b64(body.image_b64))).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image_b64: {e}") from e

    canvas_w = int(body.erp_canvas_width or body.width)
    canvas_h = int(body.erp_canvas_height or body.height)
    if canvas_w != 2 * canvas_h:
        raise HTTPException(status_code=400, detail="erp_canvas_width/height must be 2:1.")
    if body.erp_layout_mode != "single_front":
        raise HTTPException(status_code=400, detail="V1 supports erp_layout_mode='single_front' only.")

    layout = build_single_front_erp_layout(
        source_rgb=src,
        canvas_width=canvas_w,
        canvas_height=canvas_h,
        scene_mode=body.scene_mode,
        reference_coverage=body.reference_coverage,
    )
    control_png = io.BytesIO()
    layout.control_rgb.save(control_png, format="PNG")
    mask_png = io.BytesIO()
    layout.mask_l.save(mask_png, format="PNG")

    try:
        out = run_comfyui_generation(body, control_png.getvalue(), mask_png.getvalue())
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"ComfyUI generation failed: {e}") from e

    use_seam_fix = body.seam_fix if body.seam_fix is not None else _default_seam_fix(body.quality_mode)
    if use_seam_fix:
        band_px = max(8, int(out.width * 0.01))
        out = _seam_blend_wrap(out, band_px=band_px)

    return {
        "image_b64": _encode_png_b64(out),
        "meta": {
            "erp_layout_mode": body.erp_layout_mode,
            "bbox_xywh": list(layout.bbox_xywh),
            "seam_fix_applied": bool(use_seam_fix),
            "worker": "comfyui_local_v1",
            "width": out.width,
            "height": out.height,
        },
    }

