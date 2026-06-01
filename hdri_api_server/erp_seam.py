"""Equirectangular left/right seam repair for 2:1 panoramas."""

from __future__ import annotations

import numpy as np


def _gaussian_kernel1d(sigma: float) -> np.ndarray:
    if sigma <= 1e-6:
        return np.array([1.0], dtype=np.float32)
    radius = max(1, int(round(float(sigma) * 3.0)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / float(sigma)) ** 2)
    return (k / np.sum(k)).astype(np.float32)


def _convolve_horizontal_wrap(rgb: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    pad = len(kernel) // 2
    w = rgb.shape[1]
    out = np.empty_like(rgb)
    for j in range(w):
        cols = (np.arange(j - pad, j + pad + 1) % w).astype(np.int32)
        out[:, j, :] = np.tensordot(kernel, rgb[:, cols, :], axes=([0], [1]))
    return out


def _seam_band_px(width: int, band_frac: float) -> int:
    """Narrow edge band only — avoids smearing detail across the panorama."""
    w = int(width)
    if w < 8:
        return 0
    target = int(round(w * float(band_frac)))
    return max(6, min(w // 24, target))


def seam_fix_erp_wrap_blur(
    rgb: np.ndarray,
    *,
    band_frac: float = 0.012,
    blur_sigma: float = 4.0,
    blend_strength: float = 0.35,
    mask_power: float = 2.0,
) -> np.ndarray:
    """
    Subtle ERP wrap repair: lightly nudge left/right edges toward each other, then
    apply a wrap-aware horizontal blur only in a narrow edge feather (center untouched).
    """
    out = np.asarray(rgb, dtype=np.float32).copy()
    if out.ndim != 3 or out.shape[2] < 3:
        return out

    h, w, _ = out.shape
    if w < 8 or h < 4:
        return out

    band = _seam_band_px(w, band_frac)
    if band < 2:
        return out

    strength = float(np.clip(blend_strength, 0.0, 1.0))
    if strength > 1e-6:
        t = np.linspace(0.0, 1.0, band, dtype=np.float32)
        alpha = (0.5 - 0.5 * np.cos(np.pi * t)) * strength
        for i in range(band):
            a = float(alpha[i])
            left = i
            right = w - band + i
            left_px = out[:, left, :]
            right_px = out[:, right, :]
            out[:, left, :] = (1.0 - a) * left_px + a * right_px
            out[:, right, :] = (1.0 - a) * right_px + a * left_px

    if blur_sigma <= 0.1:
        return out

    kernel = _gaussian_kernel1d(float(blur_sigma))
    blurred = _convolve_horizontal_wrap(out, kernel)

    col_idx = np.arange(w, dtype=np.float32)
    dist_edge = np.minimum(col_idx, (w - 1) - col_idx)
    mask = np.clip(1.0 - dist_edge / float(band), 0.0, 1.0) ** float(mask_power)
    mask = mask[None, :, None]
    return out * (1.0 - mask) + blurred * mask
