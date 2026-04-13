"""
GMNet (gain-map) inverse tone mapping for ComfyUI.

Inference matches the GMNet ``test.py`` reconstruction (peak / QGM), using the same
input preprocessing idea as ``LQGT_base_dataset`` (full-res LQ + 256x256 thumbnail).
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

_NET: torch.nn.Module | None = None
_NET_KEY: str | None = None


def _gmnet_codes_root() -> str:
    root = os.environ.get("GMNET_CODES_ROOT", "").strip()
    if not root:
        raise RuntimeError(
            "Set GMNET_CODES_ROOT to the GMNet repository's ``codes`` directory "
            "(the folder containing ``models/`` and ``utils/``). "
            "Example: C:/src/GMNet/codes"
        )
    if not os.path.isdir(root):
        raise RuntimeError(f"GMNET_CODES_ROOT is not a directory: {root}")
    return root


def _default_checkpoint() -> str:
    p = os.environ.get("GMNET_CHECKPOINT", "").strip()
    if p and os.path.isfile(p):
        return p
    # Common layout: clone GMNet next to ComfyUI and put checkpoints under GMNet/checkpoints/
    root = os.environ.get("GMNET_REPO_ROOT", "").strip()
    if root:
        cand = os.path.join(root, "checkpoints", "G_synthetic.pth")
        if os.path.isfile(cand):
            return cand
    raise RuntimeError(
        "Set GMNET_CHECKPOINT to G_synthetic.pth (or G_real.pth), or set "
        "GMNET_REPO_ROOT to the GMNet repo root containing checkpoints/G_synthetic.pth"
    )


def _ensure_gmnet_import_path() -> str:
    codes = _gmnet_codes_root()
    if codes not in sys.path:
        sys.path.insert(0, codes)
    return codes


def _load_net(ckpt: str, device: torch.device) -> torch.nn.Module:
    global _NET, _NET_KEY
    key = f"{ckpt}|{device}"
    if _NET is not None and _NET_KEY == key:
        return _NET

    _ensure_gmnet_import_path()
    import models.networks as networks  # type: ignore  # noqa: E402

    opt: dict[str, Any] = {
        "network_G": {
            "which_model_G": "GMNet",
            "in_nc": 3,
            "out_nc": 1,
            "nf": 64,
            "nb": 16,
            "act_type": "relu",
        },
    }
    net = networks.define_G(opt).to(device)
    try:
        state = torch.load(ckpt, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    clean: dict[str, Any] = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        clean[nk] = v
    net.load_state_dict(clean, strict=False)
    net.eval()
    _NET = net
    _NET_KEY = key
    return net


def _tensor_qgm_to_numpy(qgm_chw: torch.Tensor) -> np.ndarray:
    """QGM single-channel CHW -> HW1 float32 (same idea as GMNet util.tensor2numpy 1ch)."""
    x = qgm_chw.detach().float().cpu().numpy()
    x = np.clip(x, 0.0, None)
    return np.transpose(x, (1, 2, 0)).astype(np.float32)


def _tensor_org_to_lq_bgr_hwc(org_rgb_chw: torch.Tensor) -> np.ndarray:
    """RGB CHW (org, gamma 2.2) -> BGR HWC float (matches GMNet test util.tensor2numpy)."""
    x = org_rgb_chw.detach().float().cpu().numpy()
    x[x < 0] = 0
    x = x[[2, 1, 0], :, :]
    return np.transpose(x, (1, 2, 0)).astype(np.float32)


def _preprocess_from_comfy_rgb(
    rgb_hwc: np.ndarray, *, scale: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    rgb_hwc: float32 HxWx3 in [0,1] (Comfy IMAGE convention).
    LQ at 1/scale of full res; MN is 256x256 from full-res SDR (GMNet dataset convention).
    org is full-res gamma 2.2 for the HDR merge in test.py.
    """
    import cv2

    if rgb_hwc.dtype != np.float32:
        rgb_hwc = rgb_hwc.astype(np.float32)
    rgb_hwc = np.clip(rgb_hwc, 0.0, 1.0)
    bgr_full = rgb_hwc[..., ::-1].copy()
    h, w = bgr_full.shape[:2]

    if abs(scale - 1.0) > 1e-6:
        bgr_lq = cv2.resize(
            bgr_full,
            (max(1, int(w / scale)), max(1, int(h / scale))),
            interpolation=cv2.INTER_CUBIC,
        )
    else:
        bgr_lq = bgr_full
    bgr_lq = np.clip(bgr_lq, 0.0, 1.0)

    mn_bgr = cv2.resize(bgr_full, (256, 256), interpolation=cv2.INTER_CUBIC)
    mn_bgr = np.clip(mn_bgr, 0.0, 1.0)

    lq_rgb = bgr_lq[:, :, ::-1]
    mn_rgb = mn_bgr[:, :, ::-1]

    lq = torch.from_numpy(np.ascontiguousarray(np.transpose(lq_rgb, (2, 0, 1)))).float()
    mn_t = torch.from_numpy(np.ascontiguousarray(np.transpose(mn_rgb, (2, 0, 1)))).float()

    org_rgb = np.power(rgb_hwc, 2.2)
    org = torch.from_numpy(np.ascontiguousarray(np.transpose(org_rgb, (2, 0, 1)))).float()

    return lq, mn_t, org


def _hdr_bgr_from_qgm(lq_bgr_hwc: np.ndarray, qgm_hw1: np.ndarray, *, peak: float, scale: float) -> np.ndarray:
    """GMNet ``test.py`` merge: upscale QGM by ``scale`` to match full-res ``org``."""
    import cv2

    sr = qgm_hw1
    if sr.ndim == 2:
        sr = sr[..., np.newaxis]
    h0, w0 = lq_bgr_hwc.shape[:2]
    sr = cv2.resize(sr, None, None, fx=float(scale), fy=float(scale), interpolation=cv2.INTER_LINEAR)
    if sr.shape[0] != h0 or sr.shape[1] != w0:
        sr = cv2.resize(sr, (w0, h0), interpolation=cv2.INTER_LINEAR)
    if sr.ndim == 2:
        sr = sr[..., np.newaxis]
    exp = np.power(2.0, np.clip(sr, 0.0, 1.0) * np.log2(peak)) / peak
    return lq_bgr_hwc * exp


def _linear_rgb_to_display_srgb(rgb_lin: np.ndarray) -> np.ndarray:
    """Map linear RGB (unbounded) to [0,1] display sRGB for PNG / Comfy IMAGE."""
    x = np.clip(rgb_lin, 0.0, None)
    med = float(np.percentile(x, 99.5)) if x.size else 1.0
    scale = max(med, 1e-4)
    x = x / (1.0 + x / scale)
    x = np.clip(x, 0.0, 1.0)
    # sRGB OETF
    a = 0.055
    mask = x <= 0.0031308
    out = np.empty_like(x, dtype=np.float32)
    out[mask] = 12.92 * x[mask]
    out[~mask] = (1.0 + a) * np.power(x[~mask], 1.0 / 2.4) - a
    return np.clip(out, 0.0, 1.0)


class GMNetHDRITM:
    """
    Gain-map inverse tone mapping (GMNet). Outputs display-referred sRGB in Comfy ``IMAGE``
    form [0,1] for Save Image and for the HDRI API worker (which re-linearizes PNG).
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "images": ("IMAGE",),
                "peak": (
                    "FLOAT",
                    {"default": 8.0, "min": 2.0, "max": 32.0, "step": 0.5, "tooltip": "HDR peak (same role as GMNet test config ``peak``)"},
                ),
                "scale": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.25, "max": 1.0, "step": 0.25, "tooltip": "Downsample factor for LQ branch (1 = full resolution)"},
                ),
            },
            "optional": {
                "checkpoint_path": (
                    "STRING",
                    {"default": "", "tooltip": "Path to G_synthetic.pth; empty = GMNET_CHECKPOINT / GMNET_REPO_ROOT"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "image/HDR"

    def apply(
        self,
        images: torch.Tensor,
        peak: float,
        scale: float,
        checkpoint_path: str = "",
    ) -> tuple[torch.Tensor]:
        ckpt = checkpoint_path.strip() if checkpoint_path else ""
        if not ckpt:
            ckpt = _default_checkpoint()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = _load_net(ckpt, device)

        out_frames: list[np.ndarray] = []
        batch = images.shape[0]
        for bi in range(batch):
            rgb = images[bi].cpu().numpy()
            lq, mn_t, org = _preprocess_from_comfy_rgb(rgb, scale=float(scale))
            lq_b = lq.unsqueeze(0).to(device)
            mn_b = mn_t.unsqueeze(0).to(device)

            with torch.no_grad():
                _, qgm = net((lq_b, mn_b))
            qgm = qgm[0]
            org_t = org  # full-res; merge uses same space as GMNet test ``org``
            lq_bgr = _tensor_org_to_lq_bgr_hwc(org_t.cpu())
            qgm_np = _tensor_qgm_to_numpy(qgm.cpu())
            hdr_bgr = _hdr_bgr_from_qgm(lq_bgr, qgm_np, peak=float(peak), scale=float(scale))
            hdr_rgb = hdr_bgr[:, :, ::-1]
            disp = _linear_rgb_to_display_srgb(hdr_rgb)
            out_frames.append(disp)

        stacked = np.stack(out_frames, axis=0)
        return (torch.from_numpy(stacked.astype(np.float32)),)


NODE_CLASS_MAPPINGS = {
    "GMNetHDRITM": GMNetHDRITM,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GMNetHDRITM": "GMNet HDR (gain map iTM)",
}
