from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _quality_scale(quality_mode: str) -> float:
    if quality_mode == "fast":
        return 1.05
    if quality_mode == "high":
        return 1.45
    return 1.25


def _embedded_neural_hdr(rgb_lin: np.ndarray, quality_mode: str) -> np.ndarray:
    """
    Lightweight neural-style HDR expansion.
    This path is dependency-free and deterministic for environments without Torch.
    """
    x = np.clip(rgb_lin.astype(np.float32), 0.0, None)
    lum = (0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2])[..., None]
    log_lum = np.log1p(lum)
    feats = np.concatenate([x, lum, log_lum, np.ones_like(lum)], axis=-1)

    # Tiny MLP weights (frozen). Keeps behavior stable across machines.
    w1 = np.array(
        [
            [0.85, -0.15, 0.12, 0.50, 0.08, -0.35, 0.10, 0.20],
            [-0.12, 0.92, 0.07, 0.55, 0.10, 0.05, -0.18, 0.22],
            [0.04, -0.08, 0.88, 0.48, 0.12, 0.16, 0.24, -0.14],
            [0.62, 0.58, 0.54, 0.95, 0.30, 0.22, 0.12, 0.15],
            [0.38, 0.35, 0.32, 0.72, 0.24, 0.18, 0.10, 0.12],
            [0.05, 0.05, 0.05, 0.10, 0.02, 0.02, 0.02, 0.02],
        ],
        dtype=np.float32,
    )
    b1 = np.array([0.03, 0.03, 0.03, 0.06, 0.02, 0.01, 0.01, 0.01], dtype=np.float32)
    h1 = np.maximum(0.0, np.tensordot(feats, w1, axes=([2], [0])) + b1)

    w2 = np.array(
        [
            [0.52, 0.08, 0.04, 0.45, 0.20],
            [0.08, 0.50, 0.05, 0.44, 0.20],
            [0.06, 0.07, 0.50, 0.43, 0.20],
            [0.40, 0.40, 0.40, 0.72, 0.25],
            [0.18, 0.18, 0.18, 0.36, 0.12],
            [0.10, 0.08, 0.07, 0.12, 0.08],
            [0.05, 0.06, 0.08, 0.10, 0.07],
            [0.07, 0.05, 0.06, 0.11, 0.07],
        ],
        dtype=np.float32,
    )
    b2 = np.array([0.06, 0.06, 0.06, 0.08, 0.03], dtype=np.float32)
    h2 = np.maximum(0.0, np.tensordot(h1, w2, axes=([2], [0])) + b2)

    scale = _quality_scale(quality_mode)
    chroma_gain = 1.0 + _softplus(h2[..., :3]) * (0.22 * scale)
    spec_gain = _softplus(h2[..., 3:4]) * (0.95 * scale)
    base = x * chroma_gain
    # AI-like highlight reconstruction from learned features.
    hdr = base + lum * spec_gain
    return np.clip(hdr, 0.0, None).astype(np.float32)


@lru_cache(maxsize=1)
def _load_torchscript_model(model_path: str):
    if torch is None:
        raise RuntimeError("torch is not installed")
    device_name = os.environ.get("AI_HDR_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model, device


def _torchscript_hdr(rgb_lin: np.ndarray, quality_mode: str, model_path: str) -> np.ndarray:
    model, device = _load_torchscript_model(model_path)
    x = np.clip(rgb_lin.astype(np.float32), 0.0, None)
    t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(t)
    if not isinstance(out, torch.Tensor):
        raise RuntimeError("TorchScript model must return a tensor")
    if out.ndim != 4 or out.shape[1] not in (1, 3):
        raise RuntimeError(f"Unexpected model output shape: {tuple(out.shape)}")
    if out.shape[1] == 1:
        out = out.repeat(1, 3, 1, 1)
    y = out.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
    # Keep a quality-dependent amplification to map model output into practical Blender range.
    return np.clip(y * _quality_scale(quality_mode), 0.0, None)


def reconstruct_ai_hdr(
    rgb_lin: np.ndarray,
    *,
    quality_mode: str,
    exposure_bias: float = 0.0,
    model_name: str | None = None,
) -> np.ndarray:
    """
    AI HDR reconstruction entrypoint.

    Backends:
    - embedded (default): tiny neural-style model with frozen weights.
    - torchscript: load model from AI_HDR_MODEL_PATH and run inference via Torch.
    """
    backend = (model_name or os.environ.get("AI_HDR_MODEL_NAME", "embedded")).strip().lower()
    if backend == "torchscript":
        model_path = os.environ.get("AI_HDR_MODEL_PATH", "").strip()
        if not model_path:
            raise RuntimeError("AI_HDR_MODEL_PATH is required for torchscript backend")
        hdr = _torchscript_hdr(rgb_lin, quality_mode, model_path)
    else:
        hdr = _embedded_neural_hdr(rgb_lin, quality_mode)

    if abs(exposure_bias) > 1e-6:
        hdr = hdr * (2.0 ** float(exposure_bias))
    return np.clip(hdr, 0.0, None).astype(np.float32)
