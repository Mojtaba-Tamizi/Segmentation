# src/losses.py
"""
Losses for binary segmentation (logits-based).
Includes optional pixel-wise weights for Stage 2 (uncertainty weighting).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    logits:  [B,1,H,W] (or [B,H,W])
    targets: [B,1,H,W] float in {0,1} (or [B,H,W])
    weight:  optional [B,1,H,W] or [B,H,W] (>=0), used as per-pixel weights
    """
    if logits.ndim == 4:
        logits = logits[:, 0]
    if targets.ndim == 4:
        targets = targets[:, 0]
    probs = torch.sigmoid(logits)

    probs = probs.contiguous().view(probs.shape[0], -1)
    targets = targets.contiguous().view(targets.shape[0], -1)

    if weight is not None:
        if weight.ndim == 4:
            weight = weight[:, 0]
        weight = weight.contiguous().view(weight.shape[0], -1).clamp_min(0.0)
        probs = probs * weight
        targets = targets * weight

    inter = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def bce_with_logits_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    pos_weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Pixel-wise weighted BCE with logits.
    weight: per-pixel weights (same shape as targets) or None
    pos_weight: torch.Tensor scalar or shape [1] for positive class weighting
    """
    if logits.ndim == 4 and targets.ndim == 4:
        # BCE expects matching shapes
        pass
    elif logits.ndim == 3 and targets.ndim == 3:
        pass
    else:
        # allow [B,1,H,W] vs [B,H,W]
        if logits.ndim == 4:
            logits = logits[:, 0]
        if targets.ndim == 4:
            targets = targets[:, 0]
        if weight is not None and weight.ndim == 4:
            weight = weight[:, 0]

    loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        weight=weight,
        pos_weight=pos_weight,
        reduction=reduction,
    )
    return loss


def focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: Optional[float] = None,
    weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Binary focal loss from logits.

    alpha: if set, balances positives (alpha) vs negatives (1-alpha).
    weight: optional per-pixel weight.
    """
    if logits.ndim == 4:
        logits_ = logits[:, 0]
    else:
        logits_ = logits
    if targets.ndim == 4:
        targets_ = targets[:, 0]
    else:
        targets_ = targets

    bce = F.binary_cross_entropy_with_logits(logits_, targets_, reduction="none")
    p = torch.sigmoid(logits_)
    pt = p * targets_ + (1 - p) * (1 - targets_)
    mod = (1 - pt).pow(gamma)

    if alpha is not None:
        a_t = alpha * targets_ + (1 - alpha) * (1 - targets_)
        loss = a_t * mod * bce
    else:
        loss = mod * bce

    if weight is not None:
        if weight.ndim == 4:
            weight = weight[:, 0]
        loss = loss * weight.clamp_min(0.0)

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


@dataclass
class LossConfig:
    dice_weight: float = 1.0
    bce_weight: float = 1.0
    use_focal: bool = False
    focal_gamma: float = 2.0
    focal_alpha: Optional[float] = None
    eps: float = 1e-6


class BCEDiceLoss(nn.Module):
    def __init__(self, cfg: LossConfig = LossConfig(), pos_weight: Optional[float] = None):
        super().__init__()
        self.cfg = cfg
        self.pos_weight = None if pos_weight is None else torch.tensor([float(pos_weight)], dtype=torch.float32)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        dice = soft_dice_loss_with_logits(logits, targets, weight=weight, eps=self.cfg.eps)

        if self.cfg.use_focal:
            bce = focal_loss_with_logits(
                logits, targets,
                gamma=self.cfg.focal_gamma,
                alpha=self.cfg.focal_alpha,
                weight=weight,
                reduction="mean",
            )
        else:
            pos_w = None
            if self.pos_weight is not None:
                pos_w = self.pos_weight.to(device=logits.device, dtype=logits.dtype)
            bce = bce_with_logits_loss(logits, targets, weight=weight, pos_weight=pos_w, reduction="mean")

        return self.cfg.dice_weight * dice + self.cfg.bce_weight * bce