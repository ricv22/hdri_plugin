from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
import math
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any
import urllib.error
import urllib.request

from PIL import Image
from panorama import build_equirectangular


class RunComfyHTTPError(RuntimeError):
    def __init__(self, status_code: int, url: str, body: str):
        self.status_code = status_code
        self.url = url
        self.body = body
        super().__init__(f"HTTP {status_code} from {url}: {body[:2000]}")


@dataclass
class ProviderSubmitResult:
    provider_job_id: str


@dataclass
class ProviderStatusResult:
    status: str
    image_bytes: bytes | None = None
    image_url: str | None = None
    error: str | None = None


class RemoteProvider:
    """
    Provider adapter contract for hosted workflow execution.

    Current implementation intentionally keeps behavior backwards-compatible by
    using existing `panorama.build_equirectangular()` modes under the hood.
    This lets us wire async job lifecycle and accounting now, then swap this
    implementation to a third-party workflow API without changing addon/API contracts.
    """

    @staticmethod
    def _provider_mode() -> str:
        return os.environ.get("HDRI_REMOTE_PROVIDER", "legacy").strip().lower()

    @staticmethod
    def _runcomfy_http_timeout_s() -> float:
        """Socket timeout per HTTP call to RunComfy (inference POST, poll, result, image download)."""
        try:
            return max(30.0, float(os.environ.get("RUNCOMFY_HTTP_TIMEOUT_S", "120")))
        except ValueError:
            return 120.0

    @staticmethod
    def _env_truthy(name: str, default: str = "0") -> bool:
        return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _write_runcomfy_debug_payload(payload: dict[str, Any]) -> None:
        """Optionally persist the exact inference payload for placement debugging."""
        if not RemoteProvider._env_truthy("RUNCOMFY_DEBUG_OVERRIDES", "0"):
            return
        data_dir = os.environ.get("HDRI_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
        os.makedirs(data_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        debug_path = os.environ.get(
            "RUNCOMFY_DEBUG_OVERRIDES_PATH",
            os.path.join(data_dir, f"runcomfy_payload_{ts}.json"),
        ).strip()
        try:
            with open(debug_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=True)
            latest_path = os.path.join(data_dir, "runcomfy_payload_latest.json")
            with open(latest_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=True)
            print(f"[runcomfy-debug] wrote exact payload to: {debug_path}")
            print(f"[runcomfy-debug] updated latest payload at: {latest_path}")
        except Exception as e:
            print(f"[runcomfy-debug] failed to write payload dump: {e}")

    @staticmethod
    def _http_json(url: str, method: str, payload: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, method=method)
        if body is not None:
            req.add_header("Content-Type", "application/json")
        for k, v in (headers or {}).items():
            if v:
                req.add_header(k, v)
        timeout = RemoteProvider._runcomfy_http_timeout_s()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RunComfyHTTPError(e.code, url, body) from e

    @staticmethod
    def _download_headers() -> dict[str, str]:
        """Headers for fetching binary files from RunComfy temp storage (not JSON API)."""
        return {
            "Accept": "image/*,*/*;q=0.8",
            "User-Agent": "curl/8.5.0",
        }

    @staticmethod
    def _http_download_bytes(url: str, headers: dict[str, str] | None = None) -> bytes:
        req = urllib.request.Request(url, method="GET")
        merged = RemoteProvider._download_headers()
        merged.update(headers or {})
        for k, v in merged.items():
            if v:
                req.add_header(k, v)
        timeout = RemoteProvider._runcomfy_http_timeout_s()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {e.code} downloading {url}: {body[:500]}") from e

    @staticmethod
    def _collect_https_urls(obj: Any) -> list[str]:
        found: list[str] = []

        def walk(x: Any) -> None:
            if isinstance(x, dict):
                for v in x.values():
                    walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)
            elif isinstance(x, str) and x.startswith("https://"):
                found.append(x)

        walk(obj)
        return found

    @staticmethod
    def _pick_runcomfy_output_image_url(urls: list[str]) -> str | None:
        if not urls:
            return None
        uniq = list(dict.fromkeys(urls))

        prefs_raw = os.environ.get(
            "RUNCOMFY_OUTPUT_URL_CONTAINS",
            "serverless-api-storage.runcomfy.net,deployment_requests,temp/,ComfyUI_temp",
        )
        prefs = [p.strip() for p in prefs_raw.split(",") if p.strip()]

        def score(u: str) -> tuple[int, int]:
            s = sum(1 for p in prefs if p in u)
            ext = u.lower().rpartition(".")[2]
            ext_bonus = 2 if ext in {"png", "webp", "jpg", "jpeg"} else 0
            return (s + ext_bonus, len(u))

        ranked = sorted(uniq, key=score, reverse=True)
        return ranked[0]

    @staticmethod
    def _select_runcomfy_image_url(
        result_data: dict[str, Any],
        *,
        prefer_output_node_ids: list[str] | None = None,
    ) -> tuple[str | None, str | None]:
        """Select the panorama output image URL from a RunComfy /result payload.

        Prefers nodes listed in ``prefer_output_node_ids`` (4k SaveImage, etc.), then
        ``RUNCOMFY_OUTPUT_NODE_IDS`` (comma-separated). If unset, prefers any image with
        ``type=="output"`` (SaveImage), then ``type=="temp"`` (PreviewImage), and finally
        falls back to a recursive scan. Skips nodes listed in ``RUNCOMFY_OUTPUT_SKIP_NODE_IDS``
        (typically the LoadImage we inject).
        Returns ``(url, source_node_id)``.
        """
        outputs = result_data.get("outputs") if isinstance(result_data, dict) else None
        if not isinstance(outputs, dict):
            return None, None

        base_whitelist = [
            x.strip()
            for x in os.environ.get("RUNCOMFY_OUTPUT_NODE_IDS", "").split(",")
            if x.strip()
        ]
        if prefer_output_node_ids:
            whitelist = RemoteProvider._dedupe_node_ids([*prefer_output_node_ids, *base_whitelist])
        else:
            whitelist = base_whitelist
        skip = {
            x.strip()
            for x in os.environ.get("RUNCOMFY_OUTPUT_SKIP_NODE_IDS", "").split(",")
            if x.strip()
        }
        # Also implicitly skip the LoadImage node we injected (its preview is the input).
        for node_id in os.environ.get("RUNCOMFY_IMAGE_NODE_IDS", "").split(","):
            node_id = node_id.strip()
            if node_id:
                skip.add(node_id)
        # PanoramaStickers nodes often expose preview/temp images (green canvas + sticker).
        # Those are control intermediates, not the final outpainted result.
        for node_id in os.environ.get("RUNCOMFY_PANORAMA_STICKERS_NODE_IDS", "").split(","):
            node_id = node_id.strip()
            if node_id:
                skip.add(node_id)

        def _image_entries_from_node(node_val: dict[str, Any]) -> list[dict[str, Any]]:
            entries: list[dict[str, Any]] = []
            for key in ("images", "pano_input_images", "pano_output_images"):
                vals = node_val.get(key)
                if isinstance(vals, list):
                    entries.extend([img for img in vals if isinstance(img, dict)])
            return entries

        def _images_for(node_id: str) -> list[dict[str, Any]]:
            node_val = outputs.get(node_id)
            if not isinstance(node_val, dict):
                return []
            return _image_entries_from_node(node_val)

        def _first_url(images: list[dict[str, Any]]) -> str | None:
            for img in images:
                url = img.get("url") or img.get("URL")
                if isinstance(url, str) and url.startswith(("http://", "https://")):
                    return url
            return None

        # 1. Whitelist takes priority, in order.
        for node_id in whitelist:
            if node_id in skip:
                continue
            url = _first_url(_images_for(node_id))
            if url:
                return url, node_id

        # 2. Otherwise prefer SaveImage outputs, then preview/temp.
        for desired_type in ("output", "temp"):
            for node_id, node_val in outputs.items():
                if node_id in skip or not isinstance(node_val, dict):
                    continue
                for img in _image_entries_from_node(node_val):
                    if str(img.get("type", "")).strip().lower() != desired_type:
                        continue
                    url = img.get("url") or img.get("URL")
                    if isinstance(url, str) and url.startswith(("http://", "https://")):
                        return url, str(node_id)

        # 3. Last resort: any URL on any non-skipped node.
        for node_id, node_val in outputs.items():
            if node_id in skip or not isinstance(node_val, dict):
                continue
            url = _first_url(_image_entries_from_node(node_val))
            if url:
                return url, str(node_id)

        return None, None

    def _runcomfy_headers(self) -> dict[str, str]:
        token = os.environ.get("RUNCOMFY_API_TOKEN", "").strip()
        if not token:
            raise RuntimeError("RUNCOMFY_API_TOKEN is required when HDRI_REMOTE_PROVIDER=runcomfy")
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "User-Agent": "curl/8.5.0",
        }

    def _runcomfy_base(self) -> str:
        return os.environ.get("RUNCOMFY_BASE_URL", "https://api.runcomfy.net").rstrip("/")

    def _runcomfy_deployment_id(self) -> str:
        deployment_id = os.environ.get("RUNCOMFY_DEPLOYMENT_ID", "").strip()
        if not deployment_id:
            raise RuntimeError("RUNCOMFY_DEPLOYMENT_ID is required when HDRI_REMOTE_PROVIDER=runcomfy")
        return deployment_id

    @staticmethod
    def _normalise_image_bytes(image_b64: str) -> bytes:
        raw = image_b64.strip()
        if raw.startswith("data:image/"):
            _, _, raw = raw.partition(",")
        image_bytes = base64.b64decode(raw, validate=False)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image.thumbnail((1536, 1536), resample=Image.LANCZOS)
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85, optimize=True)
        return buf.getvalue()

    @staticmethod
    def _image_data_uri(image_b64: str) -> str:
        try:
            encoded = base64.b64encode(RemoteProvider._normalise_image_bytes(image_b64)).decode("ascii")
            return f"data:image/jpeg;base64,{encoded}"
        except Exception:
            # Fall back to the original data if Pillow cannot decode it.
            raw = image_b64.strip()
            if raw.startswith("data:image/"):
                return raw
            return f"data:image/jpeg;base64,{raw}"

    @staticmethod
    def _signed_input_image_url(image_b64: str) -> str | None:
        if os.environ.get("RUNCOMFY_INPUT_IMAGE_TRANSPORT", "url").strip().lower() != "url":
            return None
        public_base = os.environ.get("HDRI_PUBLIC_BASE_URL", "").strip().rstrip("/")
        if not public_base:
            return None
        try:
            image_bytes = RemoteProvider._normalise_image_bytes(image_b64)
        except Exception:
            return None
        data_dir = os.environ.get("HDRI_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
        os.makedirs(data_dir, exist_ok=True)
        file_id = f"runcomfy_input_{uuid.uuid4()}"
        disk_path = os.path.join(data_dir, f"{file_id}.jpg")
        with open(disk_path, "wb") as f:
            f.write(image_bytes)
        ttl = int(os.environ.get("RUNCOMFY_INPUT_URL_TTL_S", os.environ.get("HDRI_SIGNED_URL_TTL_S", "3600")))
        exp = int(time.time()) + ttl
        secret = os.environ.get("HDRI_SIGNING_SECRET", "dev-secret-change-me").encode("utf-8")
        msg = f"{file_id}:{exp}".encode("utf-8")
        sig = hmac.new(secret, msg, hashlib.sha256).hexdigest()
        return f"{public_base}/v1/input-files/{file_id}.jpg?exp={exp}&sig={sig}"

    @staticmethod
    def _build_erp_control_png(
        image_b64: str,
        *,
        width: int,
        height: int,
        scene_mode: str,
        reference_coverage: float,
    ) -> bytes | None:
        try:
            try:
                from erp_projection import coverage_to_fov_deg, project_pinhole_to_erp
            except Exception:
                from .erp_projection import coverage_to_fov_deg, project_pinhole_to_erp  # type: ignore
            raw = image_b64.strip()
            if raw.startswith("data:image/"):
                _, _, raw = raw.partition(",")
            image_bytes = base64.b64decode(raw, validate=False)
            src = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            fov_deg = coverage_to_fov_deg(reference_coverage)
            mode = (scene_mode or "auto").strip().lower()
            pitch_deg = 5.0 if mode == "outdoor" else 0.0

            try:
                custom_fov = float(os.environ.get("RUNCOMFY_ERP_HFOV_DEG", "").strip())
            except ValueError:
                custom_fov = 0.0
            if custom_fov > 0.0:
                fov_deg = custom_fov

            try:
                pitch_override = os.environ.get("RUNCOMFY_ERP_PITCH_DEG", "").strip()
                if pitch_override != "":
                    pitch_deg = float(pitch_override)
            except ValueError:
                pass

            bg = os.environ.get("RUNCOMFY_PANORAMA_BG_COLOR", "#00ff00").strip() or "#00ff00"

            # Match ComfyUI-Panorama-Stickers: derive vFOV from hFOV and the *source* image
            # aspect, not the ERP canvas. Using v_fov_deg=fov_deg squashes non-square inputs
            # vertically on the sphere, which reads as a flat poster in Blender world textures.
            src_w, src_h = src.size
            if src_w > 0 and src_h > 0:
                v_fov_deg = math.degrees(
                    2.0 * math.atan(math.tan(math.radians(fov_deg) * 0.5) * (src_h / src_w))
                )
                v_fov_deg = max(0.1, min(179.0, v_fov_deg))
            else:
                v_fov_deg = fov_deg

            erp = project_pinhole_to_erp(
                src,
                canvas_width=int(width),
                canvas_height=int(height),
                yaw_deg=0.0,
                pitch_deg=pitch_deg,
                h_fov_deg=fov_deg,
                v_fov_deg=v_fov_deg,
                rot_deg=0.0,
                bg_color=bg,
            )

            buf = io.BytesIO()
            erp.save(buf, format="PNG")
            return buf.getvalue()
        except Exception:
            return None

    @staticmethod
    def _signed_erp_control_url(
        image_b64: str,
        *,
        width: int,
        height: int,
        scene_mode: str,
        reference_coverage: float,
    ) -> str | None:
        if os.environ.get("RUNCOMFY_INPUT_IMAGE_TRANSPORT", "url").strip().lower() != "url":
            return None
        public_base = os.environ.get("HDRI_PUBLIC_BASE_URL", "").strip().rstrip("/")
        if not public_base:
            return None
        png_bytes = RemoteProvider._build_erp_control_png(
            image_b64,
            width=width,
            height=height,
            scene_mode=scene_mode,
            reference_coverage=reference_coverage,
        )
        if png_bytes is None:
            return None
        data_dir = os.environ.get("HDRI_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
        os.makedirs(data_dir, exist_ok=True)
        file_id = f"runcomfy_erp_{uuid.uuid4()}"
        disk_path = os.path.join(data_dir, f"{file_id}.png")
        with open(disk_path, "wb") as f:
            f.write(png_bytes)
        ttl = int(os.environ.get("RUNCOMFY_INPUT_URL_TTL_S", os.environ.get("HDRI_SIGNED_URL_TTL_S", "3600")))
        exp = int(time.time()) + ttl
        secret = os.environ.get("HDRI_SIGNING_SECRET", "dev-secret-change-me").encode("utf-8")
        msg = f"{file_id}:{exp}".encode("utf-8")
        sig = hmac.new(secret, msg, hashlib.sha256).hexdigest()
        return f"{public_base}/v1/input-files/{file_id}.png?exp={exp}&sig={sig}"

    @staticmethod
    def _parse_node_ids(env_name: str) -> list[str]:
        raw = os.environ.get(env_name, "").strip()
        if not raw:
            return []
        return [x.strip() for x in raw.split(",") if x.strip()]

    @staticmethod
    def _workflow_nodes_with_class(workflow_api_json: dict[str, Any] | None, class_type: str) -> list[str]:
        if not isinstance(workflow_api_json, dict):
            return []
        out: list[str] = []
        for node_id, node_val in workflow_api_json.items():
            if not isinstance(node_val, dict):
                continue
            if str(node_val.get("class_type", "")).strip() == class_type:
                out.append(str(node_id))
        return out

    @staticmethod
    def _dedupe_node_ids(node_ids: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for node_id in node_ids:
            node_id = str(node_id).strip()
            if not node_id or node_id in seen:
                continue
            seen.add(node_id)
            out.append(node_id)
        return out

    @staticmethod
    def _set_override_value(dst: dict[str, Any], node_id: str, input_name: str, value: Any) -> None:
        node = dst.setdefault(str(node_id), {})
        inputs = node.setdefault("inputs", {})
        inputs[input_name] = value

    @staticmethod
    def _quality_steps(quality_mode: str) -> int:
        if quality_mode == "fast":
            return int(os.environ.get("RUNCOMFY_FAST_STEPS", "16"))
        if quality_mode == "high":
            return int(os.environ.get("RUNCOMFY_HIGH_STEPS", "32"))
        return int(os.environ.get("RUNCOMFY_BALANCED_STEPS", "24"))

    @staticmethod
    def _runcomfy_coverage_to_fov_deg(reference_coverage: float) -> float:
        """Map placement coverage to a camera-like rectilinear sticker hFOV."""
        cov = max(0.15, min(0.85, float(reference_coverage)))
        t = (cov - 0.15) / 0.70
        return 35.0 + t * 60.0

    @staticmethod
    def _runcomfy_output_preset(width: int, height: int) -> str:
        """Return the deployed PanoramaStickers preset value."""
        _ = height
        return str(int(width))

    @staticmethod
    def _decoded_image_size(image_b64: str) -> tuple[int, int] | None:
        try:
            raw = image_b64.strip()
            if raw.startswith("data:image/"):
                _, _, raw = raw.partition(",")
            image_bytes = base64.b64decode(raw, validate=False)
            with Image.open(io.BytesIO(image_bytes)) as src:
                w, h = src.size
            if int(w) > 0 and int(h) > 0:
                return int(w), int(h)
        except Exception:
            return None
        return None

    @staticmethod
    def _build_runcomfy_sticker_state_json(
        *,
        reference_coverage: float,
        scene_mode: str,
        source_width: int | None,
        source_height: int | None,
        yaw_deg: float | None = None,
        pitch_deg: float | None = None,
        rot_deg: float | None = None,
        h_fov_deg: float | None = None,
    ) -> str:
        mode = (scene_mode or "auto").strip().lower()
        default_pitch = 5.0 if mode == "outdoor" else 0.0
        source_aspect = 1.0
        if source_width and source_height and source_width > 0 and source_height > 0:
            source_aspect = float(source_width) / float(source_height)

        resolved_hfov = (
            float(h_fov_deg)
            if h_fov_deg is not None
            else RemoteProvider._runcomfy_coverage_to_fov_deg(reference_coverage)
        )
        resolved_hfov = max(1.0, min(179.0, resolved_hfov))
        resolved_vfov = math.degrees(
            2.0 * math.atan(math.tan(math.radians(resolved_hfov) * 0.5) / max(source_aspect, 1e-6))
        )
        resolved_vfov = max(1.0, min(179.0, resolved_vfov))
        resolved_rot = float(0.0 if rot_deg is None else rot_deg)

        payload: dict[str, Any] = {
            "kind": "pano_sticker_state",
            "version": 1,
            "pose": {
                "yaw_deg": float(0.0 if yaw_deg is None else yaw_deg),
                "pitch_deg": float(default_pitch if pitch_deg is None else pitch_deg),
                # PanoramaStickers naming uses rot_deg; keep roll_deg alias for compatibility.
                "rot_deg": resolved_rot,
                "roll_deg": resolved_rot,
                "hFOV_deg": float(resolved_hfov),
                "vFOV_deg": float(resolved_vfov),
            },
            "source_aspect": source_aspect,
        }
        return json.dumps(payload, separators=(",", ":"))

    @staticmethod
    def _build_runcomfy_panorama_stickers_state_json(
        *,
        image_data_uri: str,
        width: int,
        reference_coverage: float,
        bg_color: str,
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
        rot_deg: float = 0.0,
        h_fov_deg: float | None = None,
        source_width: int | None = None,
        source_height: int | None = None,
    ) -> str:
        """
        Build PanoramaStickers `state_json` for RunComfy. RunComfy accepts media as
        HTTPS URLs or data URIs in overrides; we pass the same data URI in `filename`
        so the hosted ComfyUI graph can load the control image (see RunComfy quickstart).
        """
        asset_id = "asset_uploaded"
        sticker_id = "st_uploaded"
        fov_deg = float(h_fov_deg) if h_fov_deg is not None else RemoteProvider._runcomfy_coverage_to_fov_deg(reference_coverage)
        source_aspect = 1.0
        if source_width and source_height and source_width > 0 and source_height > 0:
            source_aspect = float(source_width) / float(source_height)
        v_fov_deg = math.degrees(2.0 * math.atan(math.tan(math.radians(fov_deg) * 0.5) / max(source_aspect, 1e-6)))
        v_fov_deg = max(1.0, min(179.0, v_fov_deg))
        state: dict[str, Any] = {
            "version": 1,
            "projection_model": "pinhole_rectilinear",
            "alpha_mode": "straight",
            "bg_color": bg_color,
            "output_preset": int(width),
            "assets": {
                asset_id: {
                    "type": "comfy_image",
                    "filename": image_data_uri,
                    "subfolder": "",
                    "storage": "input",
                    "name": "upload",
                }
            },
            "stickers": [
                {
                    "id": sticker_id,
                    "asset_id": asset_id,
                    "yaw_deg": float(yaw_deg),
                    "pitch_deg": float(pitch_deg),
                    "hFOV_deg": fov_deg,
                    "vFOV_deg": v_fov_deg,
                    "rot_deg": float(rot_deg),
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

    def _build_runcomfy_overrides(
        self,
        *,
        image_b64: str,
        width: int,
        height: int,
        scene_mode: str,
        quality_mode: str,
        overrides: dict[str, Any] | None,
        workflow_api_json: dict[str, Any] | None = None,
        delivery_width: int | None = None,
        delivery_height: int | None = None,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}

        # Optional static baseline overrides, useful for deployment-specific defaults.
        static_overrides = os.environ.get("RUNCOMFY_OVERRIDES_JSON", "").strip()
        if static_overrides:
            parsed = json.loads(static_overrides)
            if isinstance(parsed, dict):
                out.update(parsed)

        # If caller already provided RunComfy-style overrides, preserve them.
        if isinstance(overrides, dict):
            runcomfy_like = True
            for k, v in overrides.items():
                if not isinstance(k, str) or not isinstance(v, dict) or "inputs" not in v:
                    runcomfy_like = False
                    break
            if runcomfy_like:
                out.update(overrides)
                return out

        generic = overrides or {}
        ref_cov_val = generic.get("placement_coverage", generic.get("reference_coverage"))
        if ref_cov_val is None:
            try:
                ref_cov = float(os.environ.get("RUNCOMFY_DEFAULT_REFERENCE_COVERAGE", "0.4"))
            except ValueError:
                ref_cov = 0.4
        else:
            ref_cov = float(ref_cov_val)

        placement_yaw = generic.get("placement_yaw_deg")
        placement_pitch = generic.get("placement_pitch_deg")
        placement_rot = generic.get("placement_rotation_deg")
        placement_hfov = generic.get("placement_hfov_deg")
        placement_requested = any(v is not None for v in (placement_yaw, placement_pitch, placement_rot, placement_hfov))

        ps_ids = self._parse_node_ids("RUNCOMFY_PANORAMA_STICKERS_NODE_IDS")
        if not ps_ids:
            ps_ids = self._workflow_nodes_with_class(workflow_api_json, "PanoramaStickers")
        ps_ids = self._dedupe_node_ids(ps_ids)

        sticker_image_node_ids: list[str] = []
        if isinstance(workflow_api_json, dict):
            for ps_id in ps_ids:
                node_val = workflow_api_json.get(ps_id)
                if not isinstance(node_val, dict):
                    continue
                inputs = node_val.get("inputs")
                if not isinstance(inputs, dict):
                    continue
                sticker_image_ref = inputs.get("sticker_image")
                if isinstance(sticker_image_ref, list) and sticker_image_ref:
                    sticker_image_node_ids.append(str(sticker_image_ref[0]))
        sticker_image_node_ids = self._dedupe_node_ids(sticker_image_node_ids)

        load_image_node_ids = self._parse_node_ids("RUNCOMFY_IMAGE_NODE_IDS")
        if not load_image_node_ids and sticker_image_node_ids:
            load_image_node_ids = list(sticker_image_node_ids)
        load_image_node_ids = self._dedupe_node_ids(load_image_node_ids)
        target_image_node_ids = (
            list(sticker_image_node_ids)
            if (placement_requested and sticker_image_node_ids)
            else list(load_image_node_ids)
        )
        use_native_sticker_inputs = (
            bool(ps_ids)
            and (
                placement_requested
                or os.environ.get("RUNCOMFY_PANORAMA_STICKERS_USE_EXTERNAL_IMAGE", "1").strip().lower()
                in {"1", "true", "yes", "on"}
            )
        )

        # For the panorama outpainting workflow, the LoadImage node receives a
        # pre-composited 2:1 ERP control PNG (green outpaint area + source
        # placed front-center). This avoids depending on PanoramaStickers asset
        # resolution when using state_json/assets.
        #
        # When RUNCOMFY_PANORAMA_STICKERS_USE_EXTERNAL_IMAGE=1 and a workflow
        # wires LoadImage->PanoramaStickers.sticker_image, prefer native
        # PanoramaStickers projection driven by sticker_state.
        compose_erp = (
            not placement_requested
            and
            not use_native_sticker_inputs
            and os.environ.get("RUNCOMFY_LOAD_IMAGE_COMPOSE_ERP", "1").strip().lower() in {"1", "true", "yes", "on"}
        )
        image_ref: str | None = None
        if target_image_node_ids and compose_erp:
            image_ref = self._signed_erp_control_url(
                image_b64,
                width=width,
                height=height,
                scene_mode=scene_mode,
                reference_coverage=ref_cov,
            )
        if image_ref is None:
            image_ref = self._signed_input_image_url(image_b64) or self._image_data_uri(image_b64)

        for node_id in target_image_node_ids:
            self._set_override_value(out, node_id, os.environ.get("RUNCOMFY_IMAGE_INPUT_NAME", "image"), image_ref)

        # Prompt nodes
        prompt = generic.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            for node_id in self._parse_node_ids("RUNCOMFY_PROMPT_NODE_IDS"):
                self._set_override_value(out, node_id, os.environ.get("RUNCOMFY_PROMPT_INPUT_NAME", "text"), prompt)

        negative = generic.get("negative_prompt")
        if isinstance(negative, str) and negative.strip():
            for node_id in self._parse_node_ids("RUNCOMFY_NEGATIVE_PROMPT_NODE_IDS"):
                self._set_override_value(out, node_id, os.environ.get("RUNCOMFY_NEGATIVE_PROMPT_INPUT_NAME", "text"), negative)

        if generic.get("seed") is not None:
            for node_id in self._parse_node_ids("RUNCOMFY_SEED_NODE_IDS"):
                self._set_override_value(out, node_id, os.environ.get("RUNCOMFY_SEED_INPUT_NAME", "seed"), generic["seed"])

        if generic.get("strength") is not None:
            for node_id in self._parse_node_ids("RUNCOMFY_STRENGTH_NODE_IDS"):
                self._set_override_value(out, node_id, os.environ.get("RUNCOMFY_STRENGTH_INPUT_NAME", "denoise"), generic["strength"])

        coverage_value = generic.get("placement_coverage")
        if coverage_value is None:
            coverage_value = generic.get("reference_coverage")
        if coverage_value is not None:
            for node_id in self._parse_node_ids("RUNCOMFY_REFERENCE_COVERAGE_NODE_IDS"):
                self._set_override_value(
                    out,
                    node_id,
                    os.environ.get("RUNCOMFY_REFERENCE_COVERAGE_INPUT_NAME", "reference_coverage"),
                    coverage_value,
                )

        # Resolution and quality controls.
        for node_id in self._parse_node_ids("RUNCOMFY_DIMENSION_NODE_IDS"):
            self._set_override_value(out, node_id, os.environ.get("RUNCOMFY_WIDTH_INPUT_NAME", "width"), width)
            self._set_override_value(out, node_id, os.environ.get("RUNCOMFY_HEIGHT_INPUT_NAME", "height"), height)

        for node_id in self._parse_node_ids("RUNCOMFY_STEPS_NODE_IDS"):
            self._set_override_value(out, node_id, os.environ.get("RUNCOMFY_STEPS_INPUT_NAME", "steps"), self._quality_steps(quality_mode))

        # PanoramaStickers (e.g. examples/comfyui_flux2_klein_4b_api.json node 56).
        # Preferred path for hosted RunComfy: wire LoadImage->PanoramaStickers.sticker_image
        # in the workflow and provide pose via sticker_state. This avoids CPU-side ERP
        # pre-composition and uses PanoramaStickers' canonical projection in-graph.
        #
        # Legacy path (state_json/assets) is retained for backward compatibility.
        if ps_ids:
            bg = os.environ.get("RUNCOMFY_PANORAMA_BG_COLOR", "#00ff00").strip() or "#00ff00"
            preset = self._runcomfy_output_preset(width, height)
            source_size = self._decoded_image_size(image_b64)
            src_w = source_size[0] if source_size is not None else None
            src_h = source_size[1] if source_size is not None else None
            for node_id in ps_ids:
                self._set_override_value(out, node_id, "output_preset", preset)
                self._set_override_value(out, node_id, "bg_color", bg)
                if use_native_sticker_inputs:
                    sticker_state = self._build_runcomfy_sticker_state_json(
                        reference_coverage=ref_cov,
                        scene_mode=scene_mode,
                        source_width=src_w,
                        source_height=src_h,
                        yaw_deg=float(placement_yaw) if placement_yaw is not None else None,
                        pitch_deg=float(placement_pitch) if placement_pitch is not None else None,
                        rot_deg=float(placement_rot) if placement_rot is not None else None,
                        h_fov_deg=float(placement_hfov) if placement_hfov is not None else None,
                    )
                    self._set_override_value(
                        out,
                        node_id,
                        os.environ.get("RUNCOMFY_PANORAMA_STICKER_STATE_INPUT_NAME", "sticker_state"),
                        sticker_state,
                    )
                    # Compatibility fallback: some deployments still read state_json pose.
                    # Keep sticker_state as primary, but also mirror pose into state_json.
                    native_fallback = os.environ.get(
                        "RUNCOMFY_PANORAMA_STICKERS_NATIVE_STATEJSON_FALLBACK",
                        "1" if placement_requested else "0",
                    ).strip().lower() in {"1", "true", "yes", "on"}
                    if native_fallback:
                        state_str = self._build_runcomfy_panorama_stickers_state_json(
                            image_data_uri=image_ref,
                            width=width,
                            reference_coverage=ref_cov,
                            bg_color=bg,
                            yaw_deg=float(placement_yaw) if placement_yaw is not None else 0.0,
                            pitch_deg=float(placement_pitch) if placement_pitch is not None else 0.0,
                            rot_deg=float(placement_rot) if placement_rot is not None else 0.0,
                            h_fov_deg=float(placement_hfov) if placement_hfov is not None else None,
                            source_width=src_w,
                            source_height=src_h,
                        )
                        self._set_override_value(out, node_id, "state_json", state_str)
                    else:
                        # Ensure stale asset-based state doesn't override external sticker image path.
                        self._set_override_value(out, node_id, "state_json", "")
                else:
                    state_str = self._build_runcomfy_panorama_stickers_state_json(
                        image_data_uri=image_ref,
                        width=width,
                        reference_coverage=ref_cov,
                        bg_color=bg,
                        yaw_deg=float(placement_yaw) if placement_yaw is not None else 0.0,
                        pitch_deg=float(placement_pitch) if placement_pitch is not None else 0.0,
                        rot_deg=float(placement_rot) if placement_rot is not None else 0.0,
                        h_fov_deg=float(placement_hfov) if placement_hfov is not None else None,
                        source_width=src_w,
                        source_height=src_h,
                    )
                    self._set_override_value(out, node_id, "state_json", state_str)

        if (
            delivery_width == 4096
            and delivery_height == 2048
            and os.environ.get("RUNCOMFY_4K_WORKFLOW_JSON_PATH", "").strip()
        ):
            dit_model = os.environ.get("RUNCOMFY_4K_SEEDVR2_DIT_MODEL", "").strip()
            if dit_model:
                for node_id in self._parse_node_ids("RUNCOMFY_4K_SEEDVR2_DIT_MODEL_NODE_IDS") or ["69"]:
                    self._set_override_value(out, node_id, "model", dit_model)
            vae_model = os.environ.get("RUNCOMFY_4K_SEEDVR2_VAE_MODEL", "").strip()
            if vae_model:
                for node_id in self._parse_node_ids("RUNCOMFY_4K_SEEDVR2_VAE_MODEL_NODE_IDS") or ["70"]:
                    self._set_override_value(out, node_id, "model", vae_model)
            resolution_raw = os.environ.get("RUNCOMFY_4K_SEEDVR2_RESOLUTION", "").strip()
            if resolution_raw:
                try:
                    resolution = int(resolution_raw)
                except ValueError:
                    resolution = 0
                if resolution > 0:
                    for node_id in self._parse_node_ids("RUNCOMFY_4K_SEEDVR2_UPSCALER_NODE_IDS") or ["71"]:
                        self._set_override_value(out, node_id, "resolution", resolution)
            seedvr2_seed = os.environ.get("RUNCOMFY_4K_SEEDVR2_SEED", "").strip()
            if seedvr2_seed:
                try:
                    seed_val = int(seedvr2_seed)
                except ValueError:
                    seed_val = None
                if seed_val is not None:
                    for node_id in self._parse_node_ids("RUNCOMFY_4K_SEEDVR2_UPSCALER_NODE_IDS") or ["71"]:
                        self._set_override_value(out, node_id, "seed", seed_val)

        return out

    @staticmethod
    def _runcomfy_workflow_json_path_for_request(width: int, height: int) -> str:
        """Workflow file to send for this output size.

        4096x2048 requires ``RUNCOMFY_4K_WORKFLOW_JSON_PATH`` (2k generate + SeedVR2 upscale).
        """
        if int(width) == 4096 and int(height) == 2048:
            p = os.environ.get("RUNCOMFY_4K_WORKFLOW_JSON_PATH", "").strip()
            if not p:
                raise RuntimeError(
                    "4096x2048 output requires RUNCOMFY_4K_WORKFLOW_JSON_PATH (see "
                    "examples/comfyui_flux2_klein_4b_api_4k_upscale.json). "
                    "Generation uses 2048x1024 control plus SeedVR2 upscale; set this on the API host."
                )
            return p
        return os.environ.get("RUNCOMFY_WORKFLOW_JSON_PATH", "").strip()

    @staticmethod
    def _runcomfy_generation_dimensions(width: int, height: int) -> tuple[int, int]:
        """PanoramaStickers / VAE encode pixel size sent to RunComfy overrides.

        For 4k delivery, overrides stay at 2048x1024 so the graph matches a working 2k outpaint;
        the workflow refines to 4096x2048 in Comfy.
        """
        if int(width) == 4096 and int(height) == 2048 and os.environ.get("RUNCOMFY_4K_WORKFLOW_JSON_PATH", "").strip():
            return 2048, 1024
        return int(width), int(height)

    def _runcomfy_payload(
        self,
        *,
        image_b64: str,
        width: int,
        height: int,
        scene_mode: str,
        quality_mode: str,
        overrides: dict[str, Any] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        workflow_path = RemoteProvider._runcomfy_workflow_json_path_for_request(width, height)
        gen_w, gen_h = RemoteProvider._runcomfy_generation_dimensions(width, height)
        if workflow_path:
            with open(workflow_path, encoding="utf-8") as f:
                payload["workflow_api_json"] = json.load(f)
        runcomfy_overrides = self._build_runcomfy_overrides(
            image_b64=image_b64,
            width=gen_w,
            height=gen_h,
            scene_mode=scene_mode,
            quality_mode=quality_mode,
            overrides=overrides,
            workflow_api_json=payload.get("workflow_api_json") if isinstance(payload.get("workflow_api_json"), dict) else None,
            delivery_width=int(width),
            delivery_height=int(height),
        )
        if runcomfy_overrides:
            payload["overrides"] = runcomfy_overrides
        webhook = os.environ.get("RUNCOMFY_WEBHOOK_URL", "").strip()
        if webhook:
            payload["webhook"] = webhook
        return payload

    def submit_job(
        self,
        *,
        image_b64: str,
        width: int,
        height: int,
        scene_mode: str,
        quality_mode: str,
        overrides: dict[str, Any] | None = None,
    ) -> ProviderSubmitResult:
        mode = self._provider_mode()
        if mode == "runcomfy":
            base = self._runcomfy_base()
            deployment_id = self._runcomfy_deployment_id()
            url = f"{base}/prod/v1/deployments/{deployment_id}/inference"
            payload = self._runcomfy_payload(
                image_b64=image_b64,
                width=width,
                height=height,
                scene_mode=scene_mode,
                quality_mode=quality_mode,
                overrides=overrides,
            )
            self._write_runcomfy_debug_payload(payload)
            data = self._http_json(url, "POST", payload=payload, headers=self._runcomfy_headers())
            request_id = str(data.get("request_id", "")).strip()
            if not request_id:
                raise RuntimeError(f"RunComfy submit missing request_id: {data}")
            return ProviderSubmitResult(provider_job_id=request_id)
        return ProviderSubmitResult(provider_job_id=f"local-{uuid.uuid4()}")

    def wait_for_result(
        self,
        *,
        provider_job_id: str | None = None,
        image_b64: str,
        width: int,
        height: int,
        scene_mode: str,
        quality_mode: str,
        overrides: dict[str, Any] | None = None,
        poll_interval_s: float = 0.2,
    ) -> tuple[Any, str]:
        mode = self._provider_mode()
        if mode == "runcomfy":
            base = self._runcomfy_base()
            deployment_id = self._runcomfy_deployment_id()
            request_id = provider_job_id
            if not request_id:
                request_id = self.submit_job(
                    image_b64=image_b64,
                    width=width,
                    height=height,
                    scene_mode=scene_mode,
                    quality_mode=quality_mode,
                    overrides=overrides,
                ).provider_job_id
            status_url = f"{base}/prod/v1/deployments/{deployment_id}/requests/{request_id}/status"
            result_url = f"{base}/prod/v1/deployments/{deployment_id}/requests/{request_id}/result"
            deadline = time.time() + float(os.environ.get("RUNCOMFY_POLL_TIMEOUT_S", "900"))
            transient_statuses = {429, 502, 503, 504}
            while time.time() < deadline:
                try:
                    status_data = self._http_json(status_url, "GET", headers=self._runcomfy_headers())
                except RunComfyHTTPError as e:
                    if e.status_code not in transient_statuses:
                        raise
                    time.sleep(max(1.0, float(poll_interval_s)))
                    continue
                status = str(status_data.get("status", "")).strip().lower()
                if status in {"in_queue", "queued", "in_progress", "running", "processing"}:
                    time.sleep(max(0.2, float(poll_interval_s)))
                    continue
                if status in {"cancelled", "failed", "error"}:
                    raise RuntimeError(f"RunComfy job {request_id} failed: {status_data}")
                if status in {"completed", "succeeded", "success"}:
                    break
                # Unknown but non-empty status: keep polling briefly.
                time.sleep(max(0.2, float(poll_interval_s)))
            else:
                raise RuntimeError(f"RunComfy polling timed out for request_id={request_id}")

            while time.time() < deadline:
                try:
                    result_data = self._http_json(result_url, "GET", headers=self._runcomfy_headers())
                    break
                except RunComfyHTTPError as e:
                    if e.status_code not in transient_statuses:
                        raise
                    time.sleep(max(1.0, float(poll_interval_s)))
            else:
                raise RuntimeError(f"RunComfy result fetch timed out for request_id={request_id}")
            if str(result_data.get("status", "")).strip().lower() in {"failed", "error"}:
                raise RuntimeError(f"RunComfy result failed: {result_data}")

            prefer_save = None
            if int(width) == 4096 and int(height) == 2048:
                prefer_save = RemoteProvider._parse_node_ids("RUNCOMFY_4K_OUTPUT_NODE_IDS")
                if not prefer_save:
                    prefer_save = None
            image_url, source_node = self._select_runcomfy_image_url(
                result_data, prefer_output_node_ids=prefer_save
            )
            if not image_url:
                # If RunComfy returned a normal outputs map but no acceptable non-skipped
                # output, do not fall back to arbitrary preview/control URLs. That can
                # accidentally apply PanoramaStickers' green conditioning image.
                if isinstance(result_data.get("outputs"), dict):
                    raise RuntimeError(
                        "RunComfy result did not include a final output image from a non-skipped node. "
                        "Check RUNCOMFY_OUTPUT_NODE_IDS points to the final SaveImage node and that the "
                        f"workflow supports {width}x{height}."
                    )
                candidates = RemoteProvider._collect_https_urls(result_data)
                image_url = RemoteProvider._pick_runcomfy_output_image_url(candidates)
            if not image_url:
                raise RuntimeError(f"RunComfy result missing output image URL: {result_data}")
            print(
                f"[runcomfy] downloading panorama from node={source_node or '?'} url={image_url}"
            )
            raw = self._http_download_bytes(image_url)
            pano = Image.open(io.BytesIO(raw))
            return pano.convert("RGB").resize((width, height), resample=Image.BICUBIC), "runcomfy"

        _ = provider_job_id
        # Placeholder for real hosted polling loop:
        # submit -> poll provider status -> download result.
        # For now this delegates to existing panorama backend selection.
        time.sleep(0.01)
        return build_equirectangular(
            image_b64,
            width,
            height,
            scene_mode,
            quality_mode,
            http_json_overrides=overrides,
        )
