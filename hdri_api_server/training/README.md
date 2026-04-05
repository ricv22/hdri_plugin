# HDR ITM Training

This folder contains the first real training path for the `ai_itm` HDR reconstruction mode.

## Expected data

Training expects true HDR panoramas in one of these formats:

- `.hdr` (Radiance RGBE)
- `.npy` containing `(H, W, 3)` float RGB
- `.npz` containing a `hdr` array or a single array entry

All inputs should be linear RGB panorama images.

## Install training dependencies

```bash
cd hdri_api_server
python3 -m pip install --user -r requirements-train.txt
```

## Train

```bash
cd hdri_api_server
python3 -m training.train_itm \
  --dataset-root /path/to/hdr_panos \
  --output-dir training_runs/itm_v1 \
  --epochs 20 \
  --batch-size 2 \
  --image-height 256 \
  --image-width 512
```

The trainer:

- loads HDR panoramas
- synthesizes plausible LDR inputs with exposure / white-balance / tone-mapping degradation
- trains a small U-Net on log-HDR and highlight-aware loss
- writes `best.pt`, `last.pt`, and `training_summary.json`

## Export TorchScript

```bash
cd hdri_api_server
python3 -m training.export_itm_torchscript \
  --checkpoint training_runs/itm_v1/best.pt \
  --output training_runs/itm_v1/itm_model.ts
```

Then point the server at it:

```bash
export AI_HDR_ITM_MODEL_NAME=torchscript
export AI_HDR_ITM_MODEL_PATH=/absolute/path/to/itm_model.ts
```
