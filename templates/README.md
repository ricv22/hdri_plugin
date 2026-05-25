# Ground projection template

The addon loads **`ground_projection.blend`** from this folder on install (fresh enable).

- Default: created automatically on first register if the file is missing.
- **Your custom setup:** run **Save Ground Template** in the addon, then copy or commit
  `ground_projection.blend` here so every fresh install uses your cleaned node group.

The node group must be named **`HDRI Ground Projection`** with inputs:
`Vector`, `Size`, `Horizon`, `Rotation` and output `Vector`.
