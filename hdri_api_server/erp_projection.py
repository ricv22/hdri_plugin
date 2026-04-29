from __future__ import annotations

import math

import numpy as np
from PIL import Image


_DEG2RAD = math.pi / 180.0


def _hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
    s = (hex_color or "#00ff00").strip().lstrip("#")
    if len(s) != 6:
        s = "00ff00"
    r = int(s[0:2], 16) / 255.0
    g = int(s[2:4], 16) / 255.0
    b = int(s[4:6], 16) / 255.0
    return r, g, b


def _orthonormal_basis_from_forward(forward: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = forward.astype(np.float32)
    f = f / (np.linalg.norm(f) + 1e-8)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(f, world_up))) > 0.999:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(world_up, f)
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(f, right)
    up = up / (np.linalg.norm(up) + 1e-8)
    return right, up, f


def _yaw_pitch_to_dir(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = yaw_deg * _DEG2RAD
    pitch = pitch_deg * _DEG2RAD
    cp = math.cos(pitch)
    return np.array(
        [cp * math.sin(yaw), math.sin(pitch), cp * math.cos(yaw)],
        dtype=np.float32,
    )


def _sample_rgb_bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w, _ = img.shape
    x = np.clip(x, 0.0, w - 1.0)
    y = np.clip(y, 0.0, h - 1.0)

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)

    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]

    c00 = img[y0, x0]
    c10 = img[y0, x1]
    c01 = img[y1, x0]
    c11 = img[y1, x1]

    c0 = c00 * (1.0 - fx) + c10 * fx
    c1 = c01 * (1.0 - fx) + c11 * fx
    return c0 * (1.0 - fy) + c1 * fy


def project_pinhole_to_erp(
    source_rgb: Image.Image,
    *,
    canvas_width: int,
    canvas_height: int,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    h_fov_deg: float = 127.5,
    v_fov_deg: float | None = None,
    rot_deg: float = 0.0,
    bg_color: str = "#00ff00",
) -> Image.Image:
    """Project a rectilinear (pinhole) image onto a 2:1 equirectangular canvas.

    Mirrors the math used by ComfyUI-Panorama-Stickers `compose_stickers_to_erp`
    for a single front-facing sticker so the API can produce the same control
    image that the deployment workflow was designed for. Returns a PIL RGB image.
    """
    if canvas_width <= 0 or canvas_height <= 0:
        raise ValueError("Canvas size must be positive.")
    if canvas_width != 2 * canvas_height:
        raise ValueError("ERP canvas must be 2:1.")

    if v_fov_deg is None:
        v_fov_deg = h_fov_deg
    h_fov_deg = max(0.1, float(h_fov_deg))
    v_fov_deg = max(0.1, float(v_fov_deg))

    src_arr = np.asarray(source_rgb.convert("RGB"), dtype=np.float32) / 255.0
    if src_arr.ndim != 3 or src_arr.shape[2] != 3:
        raise ValueError("Source image must be RGB.")

    bg = _hex_to_rgb01(bg_color)
    canvas = np.empty((canvas_height, canvas_width, 3), dtype=np.float32)
    canvas[..., 0] = bg[0]
    canvas[..., 1] = bg[1]
    canvas[..., 2] = bg[2]

    fwd = _yaw_pitch_to_dir(yaw_deg, pitch_deg)
    right, up, fwd = _orthonormal_basis_from_forward(fwd)

    max_fov = max(h_fov_deg, v_fov_deg)
    half_u = int(math.ceil(canvas_width * (max_fov / 360.0) * 1.2))
    half_v = int(math.ceil(canvas_height * (max_fov / 180.0) * 1.2))

    center_u = ((yaw_deg / 360.0) + 0.5) * canvas_width
    center_v = (0.5 - (pitch_deg / 180.0)) * canvas_height

    y_min = max(0, int(center_v - half_v))
    y_max = min(canvas_height, int(center_v + half_v))
    if y_max <= y_min:
        return Image.fromarray((canvas * 255.0).astype(np.uint8), mode="RGB")

    def _u_ranges(center: float, half: int, w: int):
        start = int(math.floor(center - half))
        end = int(math.ceil(center + half))
        if start < 0:
            yield (start + w, w)
            yield (0, end)
        elif end >= w:
            yield (start, w)
            yield (0, end - w)
        else:
            yield (start, end)

    xs_lin = np.arange(canvas_width, dtype=np.float32) + 0.5
    ys_lin = np.arange(y_min, y_max, dtype=np.float32) + 0.5

    rr = -float(rot_deg) * _DEG2RAD
    cr = math.cos(rr)
    sr = math.sin(rr)

    tan_half_h = math.tan(h_fov_deg * 0.5 * _DEG2RAD)
    tan_half_v = math.tan(v_fov_deg * 0.5 * _DEG2RAD)

    for ux0, ux1 in _u_ranges(center_u, half_u, canvas_width):
        ux0 = max(0, ux0)
        ux1 = min(canvas_width, ux1)
        if ux1 <= ux0:
            continue

        xs = xs_lin[ux0:ux1]
        xg, yg = np.meshgrid(xs, ys_lin)

        lon = (xg / canvas_width - 0.5) * (2.0 * math.pi)
        lat = (0.5 - yg / canvas_height) * math.pi
        dirs = np.stack(
            [
                np.cos(lat) * np.sin(lon),
                np.sin(lat),
                np.cos(lat) * np.cos(lon),
            ],
            axis=-1,
        ).astype(np.float32)

        z = np.sum(dirs * fwd[None, None, :], axis=-1)
        front = z > 1e-6
        if not np.any(front):
            continue

        z_safe = np.maximum(z, 1e-6)
        local_x = np.sum(dirs * right[None, None, :], axis=-1) / z_safe
        local_y = np.sum(dirs * up[None, None, :], axis=-1) / z_safe

        xr = local_x * cr - local_y * sr
        yr = local_x * sr + local_y * cr

        xn = xr / tan_half_h
        yn = yr / tan_half_v

        inside = front & (np.abs(xn) <= 1.0) & (np.abs(yn) <= 1.0)
        if not np.any(inside):
            continue

        su = xn * 0.5 + 0.5
        sv = 0.5 - yn * 0.5

        ih, iw, _ = src_arr.shape
        px = su * (iw - 1)
        py = sv * (ih - 1)
        rgb = _sample_rgb_bilinear(src_arr, px, py)

        patch = canvas[y_min:y_max, ux0:ux1, :]
        patch[inside] = rgb[inside]
        canvas[y_min:y_max, ux0:ux1, :] = patch

    canvas_u8 = np.clip(canvas, 0.0, 1.0)
    canvas_u8 = (canvas_u8 * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(canvas_u8, mode="RGB")


def coverage_to_fov_deg(reference_coverage: float) -> float:
    fov = float(reference_coverage) * 212.5
    return max(35.0, min(140.0, fov))
