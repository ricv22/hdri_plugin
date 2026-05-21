from __future__ import annotations

import numpy as np


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    if edge1 <= edge0:
        return np.zeros_like(x, dtype=np.float32)
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _luminance_expand(
    rgb_lin: np.ndarray,
    *,
    base_gain: float,
    mid_gain: float,
    source_gain: float,
    source_desat: float,
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

    target_lum = lum * np.maximum(1.0, base_gain + mid_gain * mid_mask + source_gain * source_mask)
    chroma = x / np.maximum(lum, 1e-4)
    hdr = chroma * target_lum

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
