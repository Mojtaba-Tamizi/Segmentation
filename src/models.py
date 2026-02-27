# src/models.py
"""
Simple UNet baseline for binary segmentation.

- Input:  [B, C, H, W]  (C=1 for teacher; can be >1 for student later)
- Output: logits [B, 1, H, W]

Designed to be stable + easy to debug first.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str = "bn", dropout: float = 0.0):
        super().__init__()
        norm = norm.lower()
        if norm not in ("bn", "in", "gn", "none"):
            raise ValueError("norm must be one of: bn, in, gn, none")

        def _norm_layer(c: int) -> nn.Module:
            if norm == "bn":
                return nn.BatchNorm2d(c)
            if norm == "in":
                return nn.InstanceNorm2d(c, affine=True)
            if norm == "gn":
                # 8 groups is a common safe default
                g = 8 if c >= 8 else 1
                return nn.GroupNorm(g, c)
            return nn.Identity()

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.n1 = _norm_layer(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.n2 = _norm_layer(out_ch)
        self.drop = nn.Dropout2d(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.n1(self.conv1(x)), inplace=True)
        x = self.drop(x)
        x = F.relu(self.n2(self.conv2(x)), inplace=True)
        return x


class UNet(nn.Module):
    """
    Classic UNet with transpose-conv upsampling.
    Defaults are modest (base_channels=32) to fit Colab GPUs.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 4,
        norm: str = "bn",
        dropout: float = 0.0,
    ):
        super().__init__()
        assert depth >= 2, "depth should be >= 2"

        feats: List[int] = [base_channels * (2**i) for i in range(depth)]
        self.depth = depth

        self.enc_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()

        prev = in_channels
        for f in feats:
            self.enc_blocks.append(ConvBlock(prev, f, norm=norm, dropout=dropout))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev = f

        self.bottleneck = ConvBlock(feats[-1], feats[-1] * 2, norm=norm, dropout=dropout)

        self.upconvs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        rev_feats = list(reversed(feats))
        prev = feats[-1] * 2
        for f in rev_feats:
            self.upconvs.append(nn.ConvTranspose2d(prev, f, kernel_size=2, stride=2))
            self.dec_blocks.append(ConvBlock(f + f, f, norm=norm, dropout=dropout))
            prev = f

        self.head = nn.Conv2d(base_channels, out_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        # Kaiming init for convs
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d) or isinstance(m, nn.GroupNorm):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _center_crop(enc_feat: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        """Crop enc_feat to target spatial size (H,W) if mismatch due to odd sizes."""
        _, _, H, W = enc_feat.shape
        th, tw = target_hw
        if H == th and W == tw:
            return enc_feat
        y0 = max(0, (H - th) // 2)
        x0 = max(0, (W - tw) // 2)
        return enc_feat[:, :, y0:y0 + th, x0:x0 + tw]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: List[torch.Tensor] = []

        for i in range(self.depth):
            x = self.enc_blocks[i](x)
            skips.append(x)
            x = self.pools[i](x)

        x = self.bottleneck(x)

        skips = list(reversed(skips))
        for i in range(self.depth):
            x = self.upconvs[i](x)
            skip = self._center_crop(skips[i], (x.shape[-2], x.shape[-1]))
            x = torch.cat([skip, x], dim=1)
            x = self.dec_blocks[i](x)

        logits = self.head(x)
        return logits


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)