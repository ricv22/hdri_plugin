# 4K delivery workflow — 2k outpaint + SeedVR2 upscale

> **Deferred for v1.** Not used unless `HDRI_ENABLE_4K_UPSCALE=1` on the API host and the addon exposes 4096×2048 again.

Canonical API JSON: `comfyui_flux2_klein_4b_api_4k_upscale.json`

## Strategy

| Stage | Resolution | Compute |
|-------|------------|---------|
| Outpaint (Flux + PanoramaStickers) | **2048×1024** | Same as today (main cost) |
| **SeedVR2** (`SeedVR2VideoUpscaler`) | **4096×2048** | **Medium–high** (diffusion upscale; better detail than GAN 2×) |

We **do not** run a second full Flux pass at 4k (no extra VAE encode + KSampler + decode at latent 4k).

## Graph (4k branch only)

```text
... main outpaint ...
31 KSampler -> 8 VAEDecode (full 2k ERP)
  -> 69 SeedVR2LoadDiTModel
  -> 70 SeedVR2LoadVAEModel
  -> 71 SeedVR2VideoUpscaler (resolution=2048, batch_size=1)
  -> 66 SaveImage (4k)
```

| Node | Class | Connect / notes |
|------|--------|-----------------|
| **8** | `VAEDecode` | Full 2k outpaint (not PanoramaCutout **59**) |
| **69** | `SeedVR2LoadDiTModel` | Default `seedvr2_ema_3b_fp8_e4m3fn.safetensors`, BlockSwap for VRAM |
| **70** | `SeedVR2LoadVAEModel` | Default `ema_vae_fp16.safetensors`, tiled encode/decode at 4k |
| **71** | `SeedVR2VideoUpscaler` | **image** ← **8**, **dit** ← **69**, **vae** ← **70**, **resolution** = **2048** (shortest edge → 4096×2048) |
| **66** | `SaveImage` | **images** ← **71** |

## Compute vs other options

| Method | Extra cost vs 2k-only | Typical ERP quality |
|--------|------------------------|---------------------|
| `ImageScale` / lanczos | Very low | Soft, no real detail |
| Latent 2× + KSampler refine | Very high | Often smeared / unstable |
| RealESRGAN 2× | Low–medium | Sharp; can look filtered on sky |
| **SeedVR2 (this workflow)** | **Medium–high** | Natural detail; good for HDRI |
| Tiled SD / Ultimate upscale | High | Optional future |

## RunComfy requirements

1. Install **ComfyUI-SeedVR2_VideoUpscaler** on the worker ([numz pack](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)).
2. DiT/VAE weights download on first run (defaults in node **69** / **70**).
3. For tight VRAM: raise `blocks_to_swap`, keep `offload_device=cpu`, keep VAE `encode_tiled` / `decode_tiled` enabled.
4. Re-export or sync this JSON after ComfyUI edits (node IDs must match overrides).

## API env

```env
RUNCOMFY_4K_WORKFLOW_JSON_PATH=examples/comfyui_flux2_klein_4b_api_4k_upscale.json
RUNCOMFY_4K_OUTPUT_NODE_IDS=66
# Optional overrides (defaults match workflow JSON):
# RUNCOMFY_4K_SEEDVR2_DIT_MODEL=seedvr2_ema_3b_fp8_e4m3fn.safetensors
# RUNCOMFY_4K_SEEDVR2_DIT_MODEL_NODE_IDS=69
# RUNCOMFY_4K_SEEDVR2_VAE_MODEL=ema_vae_fp16.safetensors
# RUNCOMFY_4K_SEEDVR2_VAE_MODEL_NODE_IDS=70
# RUNCOMFY_4K_SEEDVR2_RESOLUTION=2048
# RUNCOMFY_4K_SEEDVR2_SEED=42
# RUNCOMFY_4K_SEEDVR2_UPSCALER_NODE_IDS=71
# Sync API request seed to upscale: RUNCOMFY_SEED_NODE_IDS=31,71
```

If seams are visible at 4k, enable **Seam fix** in the Blender addon or add a light ERP seam blend after download (API `seam_fix`).

## Manual test at 2048 output size

To preview SeedVR2 alone in ComfyUI, run only **8 → 71** with `resolution=1024` on a 512-wide ERP slice, or full `resolution=2048` on a completed 2k outpaint.

## Troubleshooting: `KeyError: 'flash_attn'` on node import

ComfyUI fails loading **ComfyUI-SeedVR2_VideoUpscaler** with `RuntimeError: ... 'flash_attn'` when `transformers` ≥ 5.5 probes Flash Attention before `flash_attn` is in `PACKAGE_DISTRIBUTION_MAPPING` (common on Windows / embedded Python). See [SeedVR2 #566](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler/issues/566).

**Fix A (recommended, one-line patch)** — at the **very top** of the custom node `__init__.py` (before any other imports), add:

```python
try:
    from transformers.utils.import_utils import PACKAGE_DISTRIBUTION_MAPPING
    if "flash_attn" not in PACKAGE_DISTRIBUTION_MAPPING:
        PACKAGE_DISTRIBUTION_MAPPING["flash_attn"] = ["flash_attn", "flash-attn"]
except Exception:
    pass
```

Path on your machine is typically:

`D:\ComfyUI\resources\ComfyUI\custom_nodes\seedvr2_videoupscaler\__init__.py`

Restart ComfyUI after saving.

**Fix B** — In ComfyUI Manager, set **ComfyUI-SeedVR2_VideoUpscaler** to release **v2.5.18** (some users report newer tags break on certain ComfyUI builds).

**Fix C** — In the ComfyUI venv (`D:\ComfyUI\.venv`), pin or upgrade `transformers` until the upstream fix is installed, e.g.:

```powershell
D:\ComfyUI\.venv\Scripts\pip.exe install "transformers>=4.46,<5.5"
```

Only if A/B are not enough; may conflict with other nodes that need `transformers` 5.x.

SeedVR2 does **not** require Flash Attention for basic use — DiT node `attention_mode` defaults to **sdpa**.
