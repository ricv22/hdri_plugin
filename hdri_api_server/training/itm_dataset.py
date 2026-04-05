from __future__ import annotations

from pathlib import Path
import random

import numpy as np

from rgbe_hdr import read_rgbe_hdr

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - optional dependency for training only
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment]


def _require_torch() -> None:
    if torch is None or F is None:
        raise RuntimeError("torch is required for HDR ITM training")


def list_hdr_training_files(root: str | Path) -> list[Path]:
    root_path = Path(root)
    files: list[Path] = []
    for pattern in ("*.hdr", "*.npy", "*.npz"):
        files.extend(root_path.rglob(pattern))
    return sorted(p for p in files if p.is_file())


def _load_npz(path: Path) -> np.ndarray:
    with np.load(path) as data:
        if "hdr" in data:
            arr = data["hdr"]
        else:
            first_key = next(iter(data.keys()))
            arr = data[first_key]
    return np.asarray(arr, dtype=np.float32)


def load_hdr_array(path: str | Path) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".hdr":
        arr = read_rgbe_hdr(str(path))
    elif suffix == ".npy":
        arr = np.load(path).astype(np.float32)
    elif suffix == ".npz":
        arr = _load_npz(path)
    else:
        raise ValueError(f"Unsupported HDR training file: {path}")

    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"HDR training image must have shape (H, W, 3): {path}")
    return np.clip(arr.astype(np.float32), 0.0, None)


def _srgb_to_linear_torch(x):
    return torch.where(x <= 0.04045, x / 12.92, torch.pow((x + 0.055) / 1.055, 2.4))


def _linear_to_srgb_torch(x):
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * torch.pow(torch.clamp(x, min=0.0), 1.0 / 2.4) - 0.055)


def synthesize_ldr_from_hdr(hdr, *, max_hdr: float = 32.0):
    """
    Generate a plausible display-linear LDR pano from a linear HDR target.
    This is the synthetic supervision bridge for inverse tone mapping.
    """
    _require_torch()
    x = torch.clamp(hdr, min=0.0, max=max_hdr)

    exposure_ev = random.uniform(-1.5, 1.5)
    exposure_scale = 2.0 ** exposure_ev
    wb = torch.tensor(
        [random.uniform(0.92, 1.08), random.uniform(0.94, 1.06), random.uniform(0.92, 1.08)],
        device=x.device,
        dtype=x.dtype,
    ).view(3, 1, 1)
    work = x * exposure_scale * wb

    curve_choice = random.randint(0, 2)
    if curve_choice == 0:
        ldr_lin = work / (1.0 + work)
    elif curve_choice == 1:
        ldr_lin = 1.0 - torch.exp(-work * random.uniform(0.8, 1.6))
    else:
        ldr_lin = torch.pow(work / (1.0 + work), random.uniform(0.85, 1.10))

    contrast = random.uniform(0.95, 1.08)
    ldr_lin = torch.pow(torch.clamp(ldr_lin, 0.0, 1.0), contrast)

    srgb = torch.clamp(_linear_to_srgb_torch(ldr_lin), 0.0, 1.0)
    noise = torch.randn_like(srgb) * random.uniform(0.0, 0.01)
    srgb = torch.clamp(srgb + noise, 0.0, 1.0)
    srgb = torch.round(srgb * 255.0) / 255.0
    return torch.clamp(_srgb_to_linear_torch(srgb), 0.0, 1.0)


class SyntheticITMDataset(Dataset):  # type: ignore[misc]
    def __init__(
        self,
        files: list[str | Path],
        *,
        image_size: tuple[int, int] = (256, 512),
        max_hdr: float = 32.0,
    ):
        _require_torch()
        self.files = [Path(p) for p in files]
        self.image_size = image_size
        self.max_hdr = float(max_hdr)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        arr = load_hdr_array(self.files[idx])
        hdr = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        hdr = F.interpolate(hdr.unsqueeze(0), size=self.image_size, mode="bilinear", align_corners=False).squeeze(0)
        hdr = torch.clamp(hdr, min=0.0, max=self.max_hdr)
        ldr = synthesize_ldr_from_hdr(hdr, max_hdr=self.max_hdr)
        return {
            "ldr": ldr,
            "hdr": hdr,
            "path": str(self.files[idx]),
        }

