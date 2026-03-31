from __future__ import annotations

from dataclasses import dataclass

from PIL import Image


@dataclass
class LayoutResult:
    control_rgb: Image.Image
    mask_l: Image.Image
    bbox_xywh: tuple[int, int, int, int]


def _vertical_anchor(scene_mode: str) -> float:
    mode = (scene_mode or "auto").strip().lower()
    if mode == "outdoor":
        return 0.45
    return 0.50


def build_single_front_erp_layout(
    source_rgb: Image.Image,
    canvas_width: int,
    canvas_height: int,
    scene_mode: str,
    reference_coverage: float = 0.40,
) -> LayoutResult:
    """
    Build a fixed ERP control image:
      - 2:1 panorama canvas with green outpaint area
      - source image centered horizontally
      - vertical anchor from scene_mode
      - white mask marks regions to outpaint
    """
    if canvas_width <= 0 or canvas_height <= 0:
        raise ValueError("Canvas size must be positive.")
    if canvas_width != 2 * canvas_height:
        raise ValueError("ERP canvas must be 2:1.")

    reference_coverage = max(0.15, min(0.85, float(reference_coverage)))
    src = source_rgb.convert("RGB")

    target_w = max(16, int(canvas_width * reference_coverage))
    scale = target_w / max(1, src.width)
    target_h = max(16, int(src.height * scale))
    if target_h > canvas_height:
        scale2 = canvas_height / max(1, target_h)
        target_w = max(16, int(target_w * scale2))
        target_h = max(16, int(target_h * scale2))

    resized = src.resize((target_w, target_h), resample=Image.LANCZOS)
    cx = canvas_width // 2
    cy = int(canvas_height * _vertical_anchor(scene_mode))
    x0 = max(0, min(canvas_width - target_w, cx - target_w // 2))
    y0 = max(0, min(canvas_height - target_h, cy - target_h // 2))

    # Green screen for outpaint area.
    control = Image.new("RGB", (canvas_width, canvas_height), (0, 255, 0))
    control.paste(resized, (x0, y0))

    # White means "to fill"/outpaint, black means "keep source region".
    mask = Image.new("L", (canvas_width, canvas_height), 255)
    keep = Image.new("L", (target_w, target_h), 0)
    mask.paste(keep, (x0, y0))

    return LayoutResult(
        control_rgb=control,
        mask_l=mask,
        bbox_xywh=(x0, y0, target_w, target_h),
    )

