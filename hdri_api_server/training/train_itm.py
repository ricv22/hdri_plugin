from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from training.itm_dataset import SyntheticITMDataset, list_hdr_training_files
from training.itm_model import ITMNet

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader
except Exception as exc:  # pragma: no cover - optional dependency for training only
    raise RuntimeError("torch is required for HDR ITM training") from exc


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_files(files: list[Path], val_fraction: float, seed: int) -> tuple[list[Path], list[Path]]:
    items = files[:]
    rng = random.Random(seed)
    rng.shuffle(items)
    if len(items) < 2 or val_fraction <= 0.0:
        return items, items[:0]
    val_count = max(1, int(len(items) * val_fraction))
    return items[val_count:], items[:val_count]


def _highlight_mask(hdr: torch.Tensor) -> torch.Tensor:
    lum = 0.2126 * hdr[:, 0:1] + 0.7152 * hdr[:, 1:2] + 0.0722 * hdr[:, 2:3]
    flat = lum.flatten(1)
    q = torch.quantile(flat, 0.98, dim=1, keepdim=True).view(-1, 1, 1, 1)
    return (lum >= q).to(hdr.dtype)


def _hdr_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = torch.clamp(pred, min=0.0)
    target = torch.clamp(target, min=0.0)

    log_loss = F.l1_loss(torch.log1p(pred), torch.log1p(target))
    pred_lum = 0.2126 * pred[:, 0:1] + 0.7152 * pred[:, 1:2] + 0.0722 * pred[:, 2:3]
    tgt_lum = 0.2126 * target[:, 0:1] + 0.7152 * target[:, 1:2] + 0.0722 * target[:, 2:3]
    lum_loss = F.l1_loss(torch.log1p(pred_lum), torch.log1p(tgt_lum))
    hi_mask = _highlight_mask(target)
    hi_loss = F.l1_loss(torch.log1p(pred * hi_mask), torch.log1p(target * hi_mask))
    return log_loss + 0.35 * lum_loss + 0.45 * hi_loss


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    optimizer=None,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_count = 0

    for batch in loader:
        ldr = batch["ldr"].to(device)
        hdr = batch["hdr"].to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                pred = model(ldr)
                loss = _hdr_loss(pred, hdr)
            if is_train:
                assert optimizer is not None
                if scaler is not None and device.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        batch_size = int(ldr.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HDR inverse tone mapping model")
    parser.add_argument("--dataset-root", required=True, help="Folder containing .hdr/.npy/.npz HDR panoramas")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoints and metadata")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--max-hdr", type=float, default=32.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    _set_seed(args.seed)
    files = list_hdr_training_files(args.dataset_root)
    if not files:
        raise SystemExit("No HDR training files found. Expected .hdr, .npy, or .npz panoramas.")

    train_files, val_files = _split_files(files, args.val_fraction, args.seed)
    if not train_files:
        raise SystemExit("Training split is empty; add more HDR files or reduce val_fraction.")

    train_ds = SyntheticITMDataset(
        train_files,
        image_size=(args.image_height, args.image_width),
        max_hdr=args.max_hdr,
    )
    val_ds = SyntheticITMDataset(
        val_files or train_files[: min(len(train_files), 4)],
        image_size=(args.image_height, args.image_width),
        max_hdr=args.max_hdr,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device(args.device)
    model = ITMNet(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss = _run_epoch(model, train_loader, optimizer=optimizer, device=device, scaler=scaler)
        val_loss = _run_epoch(model, val_loader, optimizer=None, device=device, scaler=None)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        checkpoint = {
            "state_dict": model.state_dict(),
            "base_channels": args.base_channels,
            "image_height": args.image_height,
            "image_width": args.image_width,
            "max_hdr": args.max_hdr,
            "history": history,
        }
        torch.save(checkpoint, out_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, out_dir / "best.pt")

    summary = {
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "train_count": len(train_files),
        "val_count": len(val_ds),
        "best_val_loss": best_val,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": [args.image_height, args.image_width],
        "base_channels": args.base_channels,
        "max_hdr": args.max_hdr,
        "history": history,
    }
    with open(out_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
