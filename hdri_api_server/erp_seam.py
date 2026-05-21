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


def seam_fix_erp_wrap_blur(
    rgb: np.ndarray,
    *,
    band_frac: float = 0.04,
    blur_sigma: float = 10.0,
) -> np.ndarray:
    """
    Reduce visible ERP wrap seams by cosine-blending the left/right edges, then
    applying a horizontal wrap-aware Gaussian blur feathered near the seam.
    """
    out = np.asarray(rgb, dtype=np.float32).copy()
    if out.ndim != 3 or out.shape[2] < 3:
        return out

    h, w, _ = out.shape
    if w < 8 or h < 4:
        return out

    band = max(32, min(w // 6, int(round(w * float(band_frac)))))
    if band < 2:
        return out

    t = np.linspace(0.0, 1.0, band, dtype=np.float32)
    alpha = 0.5 - 0.5 * np.cos(np.pi * t)
    for i in range(band):
        a = float(alpha[i])
        left = i
        right = w - band + i
        merged = (1.0 - a) * out[:, left, :] + a * out[:, right, :]
        out[:, left, :] = merged
        out[:, right, :] = merged

    if blur_sigma <= 0.1:
        return out

    kernel = _gaussian_kernel1d(float(blur_sigma))
    blurred = _convolve_horizontal_wrap(out, kernel)

    col_idx = np.arange(w, dtype=np.float32)
    dist_edge = np.minimum(col_idx, (w - 1) - col_idx)
    mask = np.clip(1.0 - dist_edge / float(band), 0.0, 1.0) ** 0.65
    mask = mask[None, :, None]
    return out * (1.0 - mask) + blurred * mask
