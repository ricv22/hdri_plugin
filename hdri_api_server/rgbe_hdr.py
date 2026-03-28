"""
Radiance RGBE .hdr writer (no OpenEXR / no C++ build).
Matches Bruce Walter / OpenCV rgbe.c behavior: header uses FORMAT=32-bit_rle_rgbe,
then writes flat RGBE pixels (same as RGBE_WritePixels — readers auto-detect non-RLE).
"""
from __future__ import annotations

import numpy as np


def float2rgbe(rgb: np.ndarray) -> np.ndarray:
    """
    Vectorized float RGB (..., 3) -> uint8 RGBE (..., 4).
    Same formula as rgbe.c float2rgbe (Greg Ward).
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    rgbe = np.zeros(rgb.shape[:-1] + (4,), dtype=np.uint8)

    valid = mx >= 1e-32
    if not np.any(valid):
        return rgbe

    mant, exp = np.frexp(mx)
    scale = np.zeros_like(mx)
    scale[valid] = mant[valid] * 256.0 / mx[valid]

    rgbe[..., 0] = np.clip((r * scale), 0, 255).astype(np.uint8)
    rgbe[..., 1] = np.clip((g * scale), 0, 255).astype(np.uint8)
    rgbe[..., 2] = np.clip((b * scale), 0, 255).astype(np.uint8)
    rgbe[..., 3] = np.clip((exp + 128), 0, 255).astype(np.uint8)
    return rgbe


def write_rgbe_hdr(path: str, rgb_linear: np.ndarray) -> None:
    """
    Write linear float RGB (H, W, 3) as Radiance .hdr (flat RGBE pixels).
    """
    if rgb_linear.dtype != np.float32:
        rgb_linear = rgb_linear.astype(np.float32)
    h, w, _ = rgb_linear.shape
    rgbe = float2rgbe(rgb_linear)

    # Same header pattern as RGBE_WriteHeader in OpenCV rgbe.cpp
    header = (
        "#?RADIANCE\n"
        "# hdri_api_server (flat RGBE, OpenCV-compatible)\n"
        "FORMAT=32-bit_rle_rgbe\n"
        "\n"
        f"-Y {h} +X {w}\n"
    )

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        # Row-major: top scanline first (matches -Y height +X width)
        f.write(rgbe.tobytes())
