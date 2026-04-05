from __future__ import annotations

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - optional dependency for training only
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


def _require_torch() -> None:
    if torch is None or nn is None or F is None:
        raise RuntimeError("torch is required for HDR ITM training")


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):  # type: ignore[misc]
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        _require_torch()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)

    def forward(self, x):
        x = F.silu(self.norm1(self.conv1(x)))
        x = F.silu(self.norm2(self.conv2(x)))
        return x


class UpBlock(nn.Module):  # type: ignore[misc]
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        _require_torch()
        self.block = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class ITMNet(nn.Module):  # type: ignore[misc]
    """
    Small U-Net for inverse tone mapping on equirectangular panoramas.
    Input is 3-channel display-linear LDR RGB in [0, +inf).
    Output is 3-channel HDR RGB in linear radiance space.
    """

    def __init__(self, base_channels: int = 24):
        super().__init__()
        _require_torch()
        self.base_channels = int(base_channels)
        c = self.base_channels

        self.enc1 = ConvBlock(7, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.enc4 = ConvBlock(c * 4, c * 8)
        self.bottleneck = ConvBlock(c * 8, c * 8)

        self.down = nn.AvgPool2d(kernel_size=2, stride=2)
        self.up3 = UpBlock(c * 8, c * 8, c * 4)
        self.up2 = UpBlock(c * 4, c * 4, c * 2)
        self.up1 = UpBlock(c * 2, c * 2, c)
        self.head = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(c, 3, kernel_size=1),
        )

    def _build_features(self, x):
        x = torch.clamp(x, min=0.0)
        lum = 0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]
        cmax = torch.amax(x, dim=1, keepdim=True)
        cmin = torch.amin(x, dim=1, keepdim=True)
        sat = torch.where(cmax > 1e-5, (cmax - cmin) / torch.clamp(cmax, min=1e-5), torch.zeros_like(cmax))
        clipped = torch.sigmoid((cmax - 0.92) * 18.0)
        return torch.cat([x, lum, torch.log1p(lum), sat, clipped], dim=1)

    def forward(self, x):
        feats = self._build_features(x)
        e1 = self.enc1(feats)
        e2 = self.enc2(self.down(e1))
        e3 = self.enc3(self.down(e2))
        e4 = self.enc4(self.down(e3))
        b = self.bottleneck(self.down(e4))
        d3 = self.up3(b, e4)
        d2 = self.up2(d3, e3)
        d1 = self.up1(d2, e2)
        out = F.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        out = out + e1
        return F.softplus(self.head(out))

