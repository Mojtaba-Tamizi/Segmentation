# src/metrics.py
"""
Metrics for binary segmentation with mm-aware distance metrics.

Implements:
- Dice, IoU (pixel metrics)
- HD95 and ASSD (surface distance metrics in mm if spacing provided)

Spacing:
- expects (row_spacing_mm, col_spacing_mm)
- if spacing is invalid (<=0), falls back to 1.0mm and flags via return dict.

Empty-mask policy (consistent):
- GT empty & Pred empty -> Dice=1, IoU=1, HD95=0, ASSD=0
- GT empty & Pred nonempty -> Dice=0, IoU=0, HD95=inf, ASSD=inf
- GT nonempty & Pred empty -> Dice=0, IoU=0, HD95=inf, ASSD=inf
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

try:
    from scipy.ndimage import binary_erosion, distance_transform_edt
except Exception as e:
    binary_erosion = None
    distance_transform_edt = None


def _as_bool(x: np.ndarray) -> np.ndarray:
    return (x.astype(np.uint8) > 0)


def dice_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    p = _as_bool(pred)
    g = _as_bool(gt)
    inter = np.logical_and(p, g).sum(dtype=np.float64)
    denom = p.sum(dtype=np.float64) + g.sum(dtype=np.float64)
    if denom == 0:
        return 1.0
    return float((2.0 * inter + eps) / (denom + eps))


def iou_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    p = _as_bool(pred)
    g = _as_bool(gt)
    inter = np.logical_and(p, g).sum(dtype=np.float64)
    union = np.logical_or(p, g).sum(dtype=np.float64)
    if union == 0:
        return 1.0
    return float((inter + eps) / (union + eps))


def _surface_mask(mask: np.ndarray) -> np.ndarray:
    """
    Surface = mask XOR eroded(mask).
    """
    if binary_erosion is None:
        raise ImportError("scipy is required for HD95/ASSD. Install: pip install scipy")
    m = _as_bool(mask)
    if m.sum() == 0:
        return m
    er = binary_erosion(m, structure=np.ones((3, 3), dtype=bool), border_value=0)
    surf = np.logical_xor(m, er)
    return surf


def _sanitize_spacing(row_mm: float, col_mm: float) -> Tuple[Tuple[float, float], bool]:
    bad = False
    r = float(row_mm)
    c = float(col_mm)
    if not np.isfinite(r) or r <= 0:
        r = 1.0
        bad = True
    if not np.isfinite(c) or c <= 0:
        c = 1.0
        bad = True
    return (r, c), bad


def surface_distances_mm(pred: np.ndarray, gt: np.ndarray, spacing_rc: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (dist_pred_to_gt, dist_gt_to_pred) distances in mm for surface points.
    """
    if distance_transform_edt is None:
        raise ImportError("scipy is required for HD95/ASSD. Install: pip install scipy")

    p = _as_bool(pred)
    g = _as_bool(gt)
    sp = _surface_mask(p)
    sg = _surface_mask(g)

    # distance to nearest surface pixel: use dt on ~surface (True everywhere except surface)
    dt_to_sg = distance_transform_edt(~sg, sampling=spacing_rc)
    dt_to_sp = distance_transform_edt(~sp, sampling=spacing_rc)

    d_p_to_g = dt_to_sg[sp]  # distances for pred surface points to GT surface
    d_g_to_p = dt_to_sp[sg]  # distances for GT surface points to pred surface

    return d_p_to_g.astype(np.float64), d_g_to_p.astype(np.float64)


def hd95_assd_mm(
    pred: np.ndarray,
    gt: np.ndarray,
    row_spacing_mm: float,
    col_spacing_mm: float,
    cap_mm: Optional[float] = None,
) -> Tuple[float, float, bool]:
    """
    Returns (hd95, assd, used_fallback_spacing).
    cap_mm: if set, replaces inf with cap_mm (useful for averaging); still report failure rate separately.
    """
    p = _as_bool(pred)
    g = _as_bool(gt)

    spacing_rc, used_fallback = _sanitize_spacing(row_spacing_mm, col_spacing_mm)

    p_sum = int(p.sum())
    g_sum = int(g.sum())

    # empty handling
    if g_sum == 0 and p_sum == 0:
        return 0.0, 0.0, used_fallback
    if g_sum == 0 and p_sum > 0:
        hd, asd = np.inf, np.inf
        if cap_mm is not None:
            hd, asd = float(cap_mm), float(cap_mm)
        return float(hd), float(asd), used_fallback
    if g_sum > 0 and p_sum == 0:
        hd, asd = np.inf, np.inf
        if cap_mm is not None:
            hd, asd = float(cap_mm), float(cap_mm)
        return float(hd), float(asd), used_fallback

    d_p_to_g, d_g_to_p = surface_distances_mm(p, g, spacing_rc)
    all_d = np.concatenate([d_p_to_g, d_g_to_p], axis=0)

    # hd95: 95th percentile of symmetric surface distances
    hd95 = float(np.percentile(all_d, 95)) if all_d.size > 0 else 0.0
    # assd: mean of symmetric surface distances
    assd = float(all_d.mean()) if all_d.size > 0 else 0.0

    return hd95, assd, used_fallback


@dataclass
class MetricConfig:
    threshold: float = 0.5
    cap_mm: Optional[float] = None  # optional cap for inf (e.g., 1000.0)


def compute_all_metrics(
    prob_or_pred: np.ndarray,
    gt: np.ndarray,
    row_spacing_mm: float,
    col_spacing_mm: float,
    cfg: MetricConfig = MetricConfig(),
    input_is_prob: bool = False,
) -> Dict[str, float]:
    """
    prob_or_pred:
      - if input_is_prob=True: float prob map in [0,1]
      - else: binary pred mask (0/1)
    """
    if input_is_prob:
        pred = (prob_or_pred >= float(cfg.threshold)).astype(np.uint8)
    else:
        pred = (prob_or_pred.astype(np.uint8) > 0).astype(np.uint8)

    d = dice_score(pred, gt)
    j = iou_score(pred, gt)
    hd95, assd, used_fallback = hd95_assd_mm(
        pred, gt,
        row_spacing_mm=row_spacing_mm,
        col_spacing_mm=col_spacing_mm,
        cap_mm=cfg.cap_mm,
    )

    out = {
        "dice": float(d),
        "iou": float(j),
        "hd95_mm": float(hd95),
        "assd_mm": float(assd),
        "used_fallback_spacing": float(1.0 if used_fallback else 0.0),
    }
    return out