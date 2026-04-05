from __future__ import annotations

import argparse
from pathlib import Path

from training.itm_model import ITMNet

try:
    import torch
except Exception as exc:  # pragma: no cover - optional dependency for training only
    raise RuntimeError("torch is required for HDR ITM export") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Export HDR ITM checkpoint to TorchScript")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint saved by training/train_itm.py")
    parser.add_argument("--output", required=True, help="Path to save TorchScript model")
    parser.add_argument("--base-channels", type=int, default=None, help="Override base channels if absent in checkpoint")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if "state_dict" not in checkpoint:
        raise SystemExit("Checkpoint must contain a 'state_dict' entry.")

    base_channels = args.base_channels or int(checkpoint.get("base_channels", 24))
    model = ITMNet(base_channels=base_channels)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    scripted = torch.jit.script(model)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(output_path))
    print(f"Saved TorchScript HDR ITM model to {output_path}")


if __name__ == "__main__":
    main()
