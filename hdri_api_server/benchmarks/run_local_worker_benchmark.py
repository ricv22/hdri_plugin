import base64
import csv
import io
import json
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

OUT_DIR = Path(__file__).resolve().parent
INPUT_DIR = OUT_DIR / "local_worker_inputs"
PANOS_DIR = OUT_DIR / "local_worker_panos"
METRICS_CSV = OUT_DIR / "local_worker_metrics.csv"
REPORT_MD = OUT_DIR / "local_worker_report.md"


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


def _post_json(url: str, payload: dict, timeout_s: int = 900) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    worker_url = os.environ.get("PANORAMA_HTTP_URL", "http://127.0.0.1:8001/v1/panorama").strip()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    PANOS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    cases = [(kind, i) for kind in ("outdoor", "indoor", "studio") for i in range(1, 6)]
    start = time.time()

    for kind, idx in cases:
        src = _gen_scene(kind, idx)
        src_path = INPUT_DIR / f"{kind}_{idx:02d}.png"
        src.save(src_path)
        payload = {
            "image_b64": _to_b64(src),
            "width": 2048,
            "height": 1024,
            "scene_mode": kind,
            "quality_mode": "balanced",
            "erp_layout_mode": "single_front",
            "reference_coverage": 0.40,
            "seam_fix": False,
        }
        row = {"case": f"{kind}_{idx:02d}", "scene_mode": kind}
        try:
            data = _post_json(worker_url, payload)
            img_b64 = data.get("image_b64", "")
            if not img_b64:
                raise RuntimeError(f"No image_b64 in response keys={list(data)}")
            raw = base64.b64decode(img_b64)
            pano = Image.open(io.BytesIO(raw)).convert("RGB")
            pano_path = PANOS_DIR / f"{kind}_{idx:02d}.png"
            pano.save(pano_path)
            rgb = np.asarray(pano)
            row["width"] = pano.width
            row["height"] = pano.height
            row["seam_score"] = round(_seam_score(rgb), 6)
            row["horizon_score"] = round(_horizon_score(rgb), 6)
            row["status"] = "ok"
            row["error"] = ""
        except Exception as e:
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
            fieldnames=["case", "scene_mode", "width", "height", "seam_score", "horizon_score", "status", "error"],
        )
        writer.writeheader()
        writer.writerows(rows)

    ok_rows = [r for r in rows if r["status"] == "ok"]
    err_rows = [r for r in rows if r["status"] != "ok"]
    avg_seam = sum(float(r["seam_score"]) for r in ok_rows) / max(1, len(ok_rows))
    avg_horizon = sum(float(r["horizon_score"]) for r in ok_rows) / max(1, len(ok_rows))
    with REPORT_MD.open("w", encoding="utf-8") as f:
        f.write("# Local ComfyUI Worker Benchmark Report\n\n")
        f.write(f"- Worker URL: `{worker_url}`\n")
        f.write(f"- Cases: {len(rows)}\n")
        f.write(f"- Success: {len(ok_rows)}\n")
        f.write(f"- Errors: {len(err_rows)}\n")
        f.write(f"- Average seam score: {avg_seam:.6f}\n")
        f.write(f"- Average horizon score: {avg_horizon:.6f}\n")
        f.write(f"- Runtime seconds: {time.time() - start:.2f}\n\n")
        f.write("## Blender review checklist\n\n")
        f.write("- seam continuity while rotating yaw\n")
        f.write("- horizon plausibility\n")
        f.write("- source image identity preservation\n")
        f.write("- lighting/reflection usefulness on preview sphere\n")
        f.write("- effect of hue/sat/exposure controls\n\n")
        if err_rows:
            f.write("## Errors\n\n")
            for r in err_rows:
                f.write(f"- `{r['case']}`: {r['error']}\n")

    print(json.dumps({"metrics_csv": str(METRICS_CSV), "report_md": str(REPORT_MD), "cases": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
