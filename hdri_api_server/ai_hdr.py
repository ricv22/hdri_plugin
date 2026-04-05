from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    if edge1 <= edge0:
        return np.zeros_like(x, dtype=np.float32)
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def _quality_scale(quality_mode: str) -> float:
    if quality_mode == "fast":
        return 1.0
    if quality_mode == "high":
        return 1.18
    return 1.08


def _saturation(rgb: np.ndarray) -> np.ndarray:
    cmax = np.max(rgb, axis=-1, keepdims=True)
    cmin = np.min(rgb, axis=-1, keepdims=True)
    return np.where(cmax > 1e-5, (cmax - cmin) / cmax, 0.0).astype(np.float32)


def _luminance_expand(
    rgb_lin: np.ndarray,
    *,
    base_gain: float,
    mid_gain: float,
    source_gain: float,
    source_desat: float,
    learned_bias: np.ndarray | None = None,
) -> np.ndarray:
    x = np.clip(rgb_lin.astype(np.float32), 0.0, None)
    lum = (0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2])[..., None]
    p50 = float(np.percentile(lum, 50))
    p90 = float(np.percentile(lum, 90))
    p97 = float(np.percentile(lum, 97))

    mid_ref = max(p50, 0.10)
    hot_ref = max(p97, p90 + 1e-3, mid_ref * 1.75)
    mid_mask = _smoothstep(mid_ref * 0.5, mid_ref * 1.4, lum)
    source_mask = _smoothstep(hot_ref * 0.82, hot_ref * 1.18, lum)

    if learned_bias is not None:
        base_gain = base_gain + learned_bias[..., 0:1] * 0.06
        mid_gain = mid_gain * (0.9 + learned_bias[..., 1:2] * 0.25)
        source_gain = source_gain * (0.75 + learned_bias[..., 2:3] * 0.45)

    target_lum = lum * np.maximum(1.0, base_gain + mid_gain * mid_mask + source_gain * source_mask)
    chroma = x / np.maximum(lum, 1e-4)
    hdr = chroma * target_lum

    # Slightly neutralize the hottest reconstructed emitters to avoid neon clipping artifacts.
    neutral = np.repeat(target_lum, 3, axis=-1)
    hdr = hdr * (1.0 - source_desat * source_mask) + neutral * (source_desat * source_mask)
    return np.clip(hdr, 0.0, None).astype(np.float32)


def reconstruct_heuristic_hdr(rgb_lin: np.ndarray, *, quality_mode: str) -> np.ndarray:
    if quality_mode == "fast":
        params = (1.08, 0.12, 0.70)
    elif quality_mode == "high":
        params = (1.18, 0.28, 1.80)
    else:
        params = (1.12, 0.20, 1.20)
    return _luminance_expand(
        rgb_lin,
        base_gain=params[0],
        mid_gain=params[1],
        source_gain=params[2],
        source_desat=0.10,
    )


def _embedded_itm_hdr(rgb_lin: np.ndarray, quality_mode: str) -> np.ndarray:
    """
    Lightweight inverse-tone-mapping style reconstruction.
    It keeps the pano close to its LDR structure, then adds most extra headroom
    only around clipped or near-clipped light sources.
    """
    x = np.clip(rgb_lin.astype(np.float32), 0.0, None)
    lum = (0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2])[..., None]
    sat = _saturation(x)
    log_lum = np.log1p(lum)
    feats = np.concatenate([x, lum, log_lum, sat, np.ones_like(lum)], axis=-1)

    w1 = np.array(
        [
            [0.74, -0.10, 0.08, 0.42, 0.10, 0.08, 0.18, -0.12],
            [-0.09, 0.76, 0.08, 0.44, 0.10, 0.10, -0.14, 0.16],
            [0.03, -0.04, 0.80, 0.40, 0.08, 0.16, 0.14, 0.12],
            [0.48, 0.48, 0.48, 0.88, 0.26, 0.16, 0.10, 0.12],
            [0.22, 0.22, 0.22, 0.34, 0.30, 0.14, 0.08, 0.10],
            [0.12, 0.12, 0.12, 0.18, 0.55, 0.10, 0.06, 0.10],
            [0.06, 0.05, 0.05, 0.12, 0.08, 0.06, 0.05, 0.05],
        ],
        dtype=np.float32,
    )
    b1 = np.array([0.03, 0.03, 0.03, 0.05, 0.02, 0.01, 0.01, 0.01], dtype=np.float32)
    h1 = np.maximum(0.0, np.tensordot(feats, w1, axes=([2], [0])) + b1)

    w2 = np.array(
        [
            [0.42, 0.22, 0.38, 0.20, 0.18],
            [0.22, 0.42, 0.38, 0.20, 0.18],
            [0.20, 0.20, 0.40, 0.22, 0.18],
            [0.52, 0.52, 0.62, 0.28, 0.26],
            [0.12, 0.12, 0.22, 0.36, 0.22],
            [0.08, 0.08, 0.16, 0.44, 0.14],
            [0.06, 0.05, 0.08, 0.10, 0.12],
            [0.05, 0.06, 0.08, 0.08, 0.10],
        ],
        dtype=np.float32,
    )
    b2 = np.array([0.04, 0.04, 0.05, 0.05, 0.04], dtype=np.float32)
    h2 = np.maximum(0.0, np.tensordot(h1, w2, axes=([2], [0])) + b2)
    gates = _sigmoid(h2)

    if quality_mode == "fast":
        base_gain, mid_gain, source_gain = 1.10, 0.12, 1.35
        emitter_scale = 0.95
        max_shoulder = 4.0
    elif quality_mode == "high":
        base_gain, mid_gain, source_gain = 1.20, 0.28, 2.10
        emitter_scale = 1.80
        max_shoulder = 6.5
    else:
        base_gain, mid_gain, source_gain = 1.15, 0.20, 1.70
        emitter_scale = 1.30
        max_shoulder = 5.2

    base = _luminance_expand(
        x,
        base_gain=base_gain,
        mid_gain=mid_gain,
        source_gain=source_gain,
        source_desat=0.10,
        learned_bias=gates[..., 0:3],
    )
    base_lum = (0.2126 * base[..., 0] + 0.7152 * base[..., 1] + 0.0722 * base[..., 2])[..., None]

    p90 = float(np.percentile(lum, 90))
    p98 = float(np.percentile(lum, 98))
    hot_ref = max(p98, p90 + 1e-3, 0.18)
    source_mask = _smoothstep(hot_ref * 0.72, hot_ref * 1.10, lum)
    clipped_mask = _smoothstep(0.82, 0.98, np.max(x, axis=-1, keepdims=True))
    emitter_gain = emitter_scale * (0.7 + gates[..., 3:4] * 0.9)
    emitter_extra = hot_ref * emitter_gain * source_mask * (0.30 + 1.85 * clipped_mask)

    target_lum = base_lum + emitter_extra
    shoulder = 2.8 + gates[..., 4:5] * (max_shoulder - 2.8)
    target_lum = np.minimum(target_lum, np.maximum(base_lum * shoulder, hot_ref * (1.8 + emitter_gain * 1.6)))

    chroma = base / np.maximum(base_lum, 1e-4)
    hdr = chroma * target_lum
    neutral = np.repeat(target_lum, 3, axis=-1)
    emitter_mix = np.clip(0.16 + 0.22 * clipped_mask, 0.0, 0.38)
    hdr = hdr * (1.0 - emitter_mix) + neutral * emitter_mix
    return np.clip(hdr * (1.02 + 0.04 * _quality_scale(quality_mode)), 0.0, None).astype(np.float32)


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

    if quality_mode == "fast":
        base_gain, mid_gain, source_gain = 1.10, 0.14, 0.90
    elif quality_mode == "high":
        base_gain, mid_gain, source_gain = 1.22, 0.32, 2.30
    else:
        base_gain, mid_gain, source_gain = 1.15, 0.24, 1.55

    learned_bias = _sigmoid(h2[..., :3])
    hdr = _luminance_expand(
        x,
        base_gain=base_gain,
        mid_gain=mid_gain,
        source_gain=source_gain,
        source_desat=0.12,
        learned_bias=learned_bias,
    )
    # Keep a modest quality-dependent amplification so higher quality modes remain meaningfully punchier.
    return np.clip(hdr * _quality_scale(quality_mode), 0.0, None).astype(np.float32)


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


def _torchscript_itm_hdr(rgb_lin: np.ndarray, quality_mode: str, model_path: str) -> np.ndarray:
    hdr = _torchscript_hdr(rgb_lin, quality_mode, model_path)
    base_lum = (0.2126 * hdr[..., 0] + 0.7152 * hdr[..., 1] + 0.0722 * hdr[..., 2])[..., None]
    clipped_mask = _smoothstep(0.82, 0.98, np.max(np.clip(rgb_lin, 0.0, None), axis=-1, keepdims=True))
    target_lum = base_lum * (1.0 + clipped_mask * (1.0 if quality_mode == "fast" else 1.4 if quality_mode == "balanced" else 1.8))
    chroma = hdr / np.maximum(base_lum, 1e-4)
    neutral = np.repeat(target_lum, 3, axis=-1)
    out = chroma * target_lum
    out = out * (1.0 - 0.18 * clipped_mask) + neutral * (0.18 * clipped_mask)
    return np.clip(out, 0.0, None).astype(np.float32)


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


def reconstruct_itm_hdr(
    rgb_lin: np.ndarray,
    *,
    quality_mode: str,
    exposure_bias: float = 0.0,
    model_name: str | None = None,
) -> np.ndarray:
    """
    Inverse-tone-mapping flavored HDR reconstruction.

    Backends:
    - embedded (default): deterministic emitter-aware reconstruction
    - torchscript: load model from AI_HDR_ITM_MODEL_PATH (or AI_HDR_MODEL_PATH) and run inference
    """
    backend = (model_name or os.environ.get("AI_HDR_ITM_MODEL_NAME", "embedded")).strip().lower()
    if backend == "torchscript":
        model_path = os.environ.get("AI_HDR_ITM_MODEL_PATH", "").strip() or os.environ.get("AI_HDR_MODEL_PATH", "").strip()
        if not model_path:
            raise RuntimeError("AI_HDR_ITM_MODEL_PATH or AI_HDR_MODEL_PATH is required for torchscript backend")
        hdr = _torchscript_itm_hdr(rgb_lin, quality_mode, model_path)
    else:
        hdr = _embedded_itm_hdr(rgb_lin, quality_mode)

    if abs(exposure_bias) > 1e-6:
        hdr = hdr * (2.0 ** float(exposure_bias))
    return np.clip(hdr, 0.0, None).astype(np.float32)
