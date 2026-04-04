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
    def _http_json(url: str, method: str, payload: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, method=method)
        if body is not None:
            req.add_header("Content-Type", "application/json")
        for k, v in (headers or {}).items():
            if v:
                req.add_header(k, v)
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))

    @staticmethod
    def _http_download_bytes(url: str, headers: dict[str, str] | None = None) -> bytes:
        req = urllib.request.Request(url, method="GET")
        for k, v in (headers or {}).items():
            if v:
                req.add_header(k, v)
        with urllib.request.urlopen(req, timeout=120) as resp:
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

    def _runcomfy_payload(self, overrides: dict[str, Any] | None) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        workflow_path = os.environ.get("RUNCOMFY_WORKFLOW_JSON_PATH", "").strip()
        if workflow_path:
            with open(workflow_path, encoding="utf-8") as f:
                payload["workflow_api_json"] = json.load(f)
        if overrides:
            payload["overrides"] = overrides
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
        _ = (image_b64, width, height, scene_mode, quality_mode)
        mode = self._provider_mode()
        if mode == "runcomfy":
            base = self._runcomfy_base()
            deployment_id = self._runcomfy_deployment_id()
            url = f"{base}/prod/v1/deployments/{deployment_id}/inference"
            payload = self._runcomfy_payload(overrides)
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
