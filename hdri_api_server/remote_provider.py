from __future__ import annotations

import io
import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any
import urllib.request

from PIL import Image
from panorama import build_equirectangular


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
    def _http_json(url: str, method: str, payload: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, method=method)
        if body is not None:
            req.add_header("Content-Type", "application/json")
        for k, v in (headers or {}).items():
            if v:
                req.add_header(k, v)
        timeout = RemoteProvider._runcomfy_http_timeout_s()
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    @staticmethod
    def _http_download_bytes(url: str, headers: dict[str, str] | None = None) -> bytes:
        req = urllib.request.Request(url, method="GET")
        for k, v in (headers or {}).items():
            if v:
                req.add_header(k, v)
        timeout = RemoteProvider._runcomfy_http_timeout_s()
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()

    def _runcomfy_headers(self) -> dict[str, str]:
        token = os.environ.get("RUNCOMFY_API_TOKEN", "").strip()
        if not token:
            raise RuntimeError("RUNCOMFY_API_TOKEN is required when HDRI_REMOTE_PROVIDER=runcomfy")
        return {"Authorization": f"Bearer {token}"}

    def _runcomfy_base(self) -> str:
        return os.environ.get("RUNCOMFY_BASE_URL", "https://api.runcomfy.net").rstrip("/")

    def _runcomfy_deployment_id(self) -> str:
        deployment_id = os.environ.get("RUNCOMFY_DEPLOYMENT_ID", "").strip()
        if not deployment_id:
            raise RuntimeError("RUNCOMFY_DEPLOYMENT_ID is required when HDRI_REMOTE_PROVIDER=runcomfy")
        return deployment_id

    @staticmethod
    def _image_data_uri(image_b64: str) -> str:
        raw = image_b64.strip()
        if raw.startswith("data:image/"):
            return raw
        return f"data:image/jpeg;base64,{raw}"

    @staticmethod
    def _parse_node_ids(env_name: str) -> list[str]:
        raw = os.environ.get(env_name, "").strip()
        if not raw:
            return []
        return [x.strip() for x in raw.split(",") if x.strip()]

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
        """Match examples/comfyui_worker.py tuning (coverage → sticker FOV)."""
        fov = float(reference_coverage) * 212.5
        return max(35.0, min(140.0, fov))

    @staticmethod
    def _build_runcomfy_panorama_stickers_state_json(
        *,
        image_data_uri: str,
        width: int,
        reference_coverage: float,
        bg_color: str,
    ) -> str:
        """
        Build PanoramaStickers `state_json` for RunComfy. RunComfy accepts media as
        HTTPS URLs or data URIs in overrides; we pass the same data URI in `filename`
        so the hosted ComfyUI graph can load the control image (see RunComfy quickstart).
        """
        asset_id = "asset_uploaded"
        sticker_id = "st_uploaded"
        fov_deg = RemoteProvider._runcomfy_coverage_to_fov_deg(reference_coverage)
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

    def _build_runcomfy_overrides(
        self,
        *,
        image_b64: str,
        width: int,
        height: int,
        quality_mode: str,
        overrides: dict[str, Any] | None,
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
        data_uri = self._image_data_uri(image_b64)

        # Image nodes (LoadImage.image)
        for node_id in self._parse_node_ids("RUNCOMFY_IMAGE_NODE_IDS"):
            self._set_override_value(out, node_id, os.environ.get("RUNCOMFY_IMAGE_INPUT_NAME", "image"), data_uri)

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

        if generic.get("reference_coverage") is not None:
            for node_id in self._parse_node_ids("RUNCOMFY_REFERENCE_COVERAGE_NODE_IDS"):
                self._set_override_value(
                    out,
                    node_id,
                    os.environ.get("RUNCOMFY_REFERENCE_COVERAGE_INPUT_NAME", "reference_coverage"),
                    generic["reference_coverage"],
                )

        # Resolution and quality controls.
        for node_id in self._parse_node_ids("RUNCOMFY_DIMENSION_NODE_IDS"):
            self._set_override_value(out, node_id, os.environ.get("RUNCOMFY_WIDTH_INPUT_NAME", "width"), width)
            self._set_override_value(out, node_id, os.environ.get("RUNCOMFY_HEIGHT_INPUT_NAME", "height"), height)

        for node_id in self._parse_node_ids("RUNCOMFY_STEPS_NODE_IDS"):
            self._set_override_value(out, node_id, os.environ.get("RUNCOMFY_STEPS_INPUT_NAME", "steps"), self._quality_steps(quality_mode))

        # PanoramaStickers (e.g. examples/comfyui_flux2_klein_4b_api.json node 56): image lives in
        # `state_json`, not LoadImage. RunComfy: media as data URI in asset filename (see docs).
        ps_ids = self._parse_node_ids("RUNCOMFY_PANORAMA_STICKERS_NODE_IDS")
        if ps_ids:
            ref = generic.get("reference_coverage")
            if ref is None:
                try:
                    ref_cov = float(os.environ.get("RUNCOMFY_DEFAULT_REFERENCE_COVERAGE", "0.4"))
                except ValueError:
                    ref_cov = 0.4
            else:
                ref_cov = float(ref)
            bg = os.environ.get("RUNCOMFY_PANORAMA_BG_COLOR", "#00ff00").strip() or "#00ff00"
            state_str = self._build_runcomfy_panorama_stickers_state_json(
                image_data_uri=data_uri,
                width=width,
                reference_coverage=ref_cov,
                bg_color=bg,
            )
            preset = f"{width} x {height}"
            for node_id in ps_ids:
                self._set_override_value(out, node_id, "output_preset", preset)
                self._set_override_value(out, node_id, "bg_color", bg)
                self._set_override_value(out, node_id, "state_json", state_str)

        return out

    def _runcomfy_payload(
        self,
        *,
        image_b64: str,
        width: int,
        height: int,
        quality_mode: str,
        overrides: dict[str, Any] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        workflow_path = os.environ.get("RUNCOMFY_WORKFLOW_JSON_PATH", "").strip()
        if workflow_path:
            with open(workflow_path, encoding="utf-8") as f:
                payload["workflow_api_json"] = json.load(f)
        runcomfy_overrides = self._build_runcomfy_overrides(
            image_b64=image_b64,
            width=width,
            height=height,
            quality_mode=quality_mode,
            overrides=overrides,
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
        _ = scene_mode
        mode = self._provider_mode()
        if mode == "runcomfy":
            base = self._runcomfy_base()
            deployment_id = self._runcomfy_deployment_id()
            url = f"{base}/prod/v1/deployments/{deployment_id}/inference"
            payload = self._runcomfy_payload(
                image_b64=image_b64,
                width=width,
                height=height,
                quality_mode=quality_mode,
                overrides=overrides,
            )
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
            while time.time() < deadline:
                status_data = self._http_json(status_url, "GET", headers=self._runcomfy_headers())
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

            result_data = self._http_json(result_url, "GET", headers=self._runcomfy_headers())
            if str(result_data.get("status", "")).strip().lower() in {"failed", "error"}:
                raise RuntimeError(f"RunComfy result failed: {result_data}")

            image_url: str | None = None
            outputs = result_data.get("outputs")
            if isinstance(outputs, dict):
                for node_val in outputs.values():
                    if not isinstance(node_val, dict):
                        continue
                    images = node_val.get("images")
                    if isinstance(images, list):
                        for img in images:
                            if isinstance(img, dict):
                                url = img.get("url")
                                if isinstance(url, str) and url.startswith("http"):
                                    image_url = url
                                    break
                        if image_url:
                            break
            if not image_url:
                raise RuntimeError(f"RunComfy result missing output image URL: {result_data}")
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
