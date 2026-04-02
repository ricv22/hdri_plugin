import base64
import csv
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

THIS_DIR = Path(__file__).resolve().parent
SERVER_DIR = THIS_DIR.parent
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from panorama import build_equirectangular


OUT_DIR = THIS_DIR
INPUT_DIR = OUT_DIR / "generated_inputs"
PANOS_DIR = OUT_DIR / "generated_panos"
METRICS_CSV = OUT_DIR / "replicate_metrics.csv"
REPORT_MD = OUT_DIR / "replicate_report.md"


def _to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _gen_scene(kind: str, idx: int, w: int = 1024, h: int = 768) -> Image.Image:
    img = Image.new("RGB", (w, h), (20, 20, 20))
    d = ImageDraw.Draw(img)
    if kind == "outdoor":
        for y in range(h):
            t = y / max(1, h - 1)
            c = int(50 + (190 * (1.0 - t)))
            d.line([(0, y), (w, y)], fill=(c, c + 20, min(255, c + 40)))
        d.rectangle([0, int(h * 0.6), w, h], fill=(70, 120, 70))
        d.ellipse([int(w * 0.7), int(h * 0.1), int(w * 0.85), int(h * 0.3)], fill=(255, 230, 120))
    elif kind == "indoor":
        d.rectangle([0, 0, w, h], fill=(130, 125, 120))
        d.rectangle([int(w * 0.15), int(h * 0.2), int(w * 0.85), int(h * 0.85)], outline=(220, 220, 220), width=8)
        d.rectangle([int(w * 0.35), int(h * 0.05), int(w * 0.65), int(h * 0.2)], fill=(255, 245, 220))
    else:
        d.rectangle([0, 0, w, h], fill=(80, 80, 85))
        d.rectangle([int(w * 0.1), int(h * 0.2), int(w * 0.9), int(h * 0.8)], fill=(120, 120, 130))
        d.ellipse([int(w * 0.2), int(h * 0.35), int(w * 0.45), int(h * 0.65)], fill=(170, 170, 180))
        d.ellipse([int(w * 0.55), int(h * 0.35), int(w * 0.8), int(h * 0.65)], fill=(170, 170, 180))
    d.text((20, 20), f"{kind}_{idx:02d}", fill=(255, 255, 255))
    return img


def _seam_score(rgb: np.ndarray) -> float:
    left = rgb[:, 0, :].astype(np.float32)
    right = rgb[:, -1, :].astype(np.float32)
    return float(np.mean(np.abs(left - right)) / 255.0)


def _horizon_score(rgb: np.ndarray) -> float:
    mid = rgb[rgb.shape[0] // 2, :, :].astype(np.float32)
    lum = 0.2126 * mid[:, 0] + 0.7152 * mid[:, 1] + 0.0722 * mid[:, 2]
    return float(np.std(lum) / 255.0)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    PANOS_DIR.mkdir(parents=True, exist_ok=True)

    cases: list[tuple[str, int]] = []
    for k in ("outdoor", "indoor", "studio"):
        for i in range(1, 6):
            cases.append((k, i))

    rows: list[dict] = []
    start = time.time()
    for kind, idx in cases:
        src = _gen_scene(kind, idx)
        src_path = INPUT_DIR / f"{kind}_{idx:02d}.png"
        src.save(src_path)

        row = {"case": f"{kind}_{idx:02d}", "scene_mode": kind if kind != "studio" else "studio"}
        try:
            b64 = _to_b64(src)
            pano, mode = build_equirectangular(
                image_b64=b64,
                width=1024,
                height=512,
                scene_mode=row["scene_mode"],
                quality_mode="balanced",
                http_json_overrides=None,
            )
            pano_path = PANOS_DIR / f"{kind}_{idx:02d}.png"
            pano.save(pano_path)
            rgb = np.asarray(pano.convert("RGB"))
            row["panorama_mode"] = mode
            row["width"] = pano.width
            row["height"] = pano.height
            row["seam_score"] = round(_seam_score(rgb), 6)
            row["horizon_score"] = round(_horizon_score(rgb), 6)
            row["status"] = "ok"
            row["error"] = ""
        except Exception as e:
            row["panorama_mode"] = os.environ.get("PANORAMA_MODE", "resize")
            row["width"] = 0
            row["height"] = 0
            row["seam_score"] = ""
            row["horizon_score"] = ""
            row["status"] = "error"
            row["error"] = str(e)
        rows.append(row)

    with METRICS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "scene_mode",
                "panorama_mode",
                "width",
                "height",
                "seam_score",
                "horizon_score",
                "status",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    ok_rows = [r for r in rows if r["status"] == "ok"]
    err_rows = [r for r in rows if r["status"] != "ok"]
    avg_seam = sum(float(r["seam_score"]) for r in ok_rows) / max(1, len(ok_rows))
    avg_horizon = sum(float(r["horizon_score"]) for r in ok_rows) / max(1, len(ok_rows))

    with REPORT_MD.open("w", encoding="utf-8") as f:
        f.write("# Replicate Benchmark Report\n\n")
        f.write(f"- Cases: {len(rows)}\n")
        f.write(f"- Success: {len(ok_rows)}\n")
        f.write(f"- Errors: {len(err_rows)}\n")
        f.write(f"- Average seam score: {avg_seam:.6f}\n")
        f.write(f"- Average horizon score: {avg_horizon:.6f}\n")
        f.write(f"- Runtime seconds: {time.time() - start:.2f}\n\n")
        f.write("## Blender lighting review checklist\n\n")
        f.write("For at least 5 generated panoramas, check in Blender:\n\n")
        f.write("- specular highlight believability on preview sphere\n")
        f.write("- dominant light direction consistency\n")
        f.write("- seam visibility at left/right wrap\n")
        f.write("- horizon plausibility when rotating yaw\n")
        f.write("- effect of blur/hue/sat/post exposure controls\n\n")
        if err_rows:
            f.write("## Errors\n\n")
            for r in err_rows:
                f.write(f"- `{r['case']}`: {r['error']}\n")

    print(json.dumps({"metrics_csv": str(METRICS_CSV), "report_md": str(REPORT_MD), "cases": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
