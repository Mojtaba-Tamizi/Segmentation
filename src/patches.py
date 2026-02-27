# src/patches.py
"""
Patch sampling + normalization utilities for mammography segmentation.

This module is designed to work with:
- train_teacher.py (patch-based training, full-image val via tiling)
- tiling.py (expects normalized full image input)

Expected .npz schema (minimum):
- "image": float/uint16 array [H, W]
- "mask_breast": uint8/bool array [H, W] (1 inside breast)
- target mask key: e.g. "mask_mass" or "mask_any" as uint8/bool [H, W]

Optional keys (passed through in meta when present):
- "row_spacing_mm", "col_spacing_mm"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


# -------------------------
# Configs
# -------------------------

@dataclass
class PatchConfig:
    patch_size: int = 512
    p_pos: float = 0.50
    p_hardneg: float = 0.25
    hardneg_exclusion_radius: int = 16  # pixels (dilate positives to exclude near-tumor negatives)
    min_breast_fraction: float = 0.60   # retry until patch has this breast area
    max_tries: int = 20
    pad_mode: str = "reflect"           # for image padding


@dataclass
class NormConfig:
    clip_p_lo: float = 1.0
    clip_p_hi: float = 99.0

    # Choose ONE normalization path:
    use_global_zscore: bool = True
    global_mean: float = 0.0
    global_std: float = 1.0

    use_minmax: bool = False  # if True, do minmax after clipping


# -------------------------
# Normalization helpers
# -------------------------

def compute_clip_values(
    image: np.ndarray,
    mask_breast: Optional[np.ndarray],
    p_lo: float,
    p_hi: float,
) -> Tuple[float, float]:
    """Compute robust clip values from breast ROI (fallback to full image)."""
    img = image.astype(np.float32, copy=False)
    if mask_breast is not None and mask_breast.shape == img.shape:
        roi = img[mask_breast > 0]
        if roi.size >= 32:
            lo = float(np.percentile(roi, p_lo))
            hi = float(np.percentile(roi, p_hi))
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                return lo, hi

    # Fallback
    lo = float(np.percentile(img, p_lo))
    hi = float(np.percentile(img, p_hi))
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        lo = float(img.min())
        hi = float(img.max() if img.max() > img.min() else img.min() + 1.0)
    return lo, hi


def apply_norm(
    image: np.ndarray,
    mask_breast: Optional[np.ndarray],
    norm_cfg: NormConfig,
    precomputed_clip: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Apply: clip using breast ROI percentiles -> (global z-score OR per-image minmax).

    Returns float32 [H,W].
    """
    img = image.astype(np.float32, copy=False)
    if precomputed_clip is None:
        lo, hi = compute_clip_values(img, mask_breast, norm_cfg.clip_p_lo, norm_cfg.clip_p_hi)
    else:
        lo, hi = precomputed_clip

    img_c = np.clip(img, lo, hi)

    if norm_cfg.use_minmax:
        denom = (hi - lo) if (hi > lo) else 1.0
        out = (img_c - lo) / denom
    else:
        std = float(norm_cfg.global_std) if float(norm_cfg.global_std) > 0 else 1.0
        out = (img_c - float(norm_cfg.global_mean)) / std

    out = out.astype(np.float32)

    # Optional: zero outside breast for stability
    if mask_breast is not None and mask_breast.shape == out.shape:
        out = out * (mask_breast > 0).astype(np.float32)

    return out


# -------------------------
# Morphology (dependency-free)
# -------------------------

def _dilate_binary(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation via max-pooling (CPU torch), returns uint8 {0,1}."""
    if radius <= 0:
        return (mask > 0).astype(np.uint8)
    m = torch.from_numpy((mask > 0).astype(np.float32))[None, None, ...]  # [1,1,H,W]
    k = int(2 * radius + 1)
    out = torch.nn.functional.max_pool2d(m, kernel_size=k, stride=1, padding=radius)
    return (out[0, 0].numpy() > 0.5).astype(np.uint8)


def _erode_binary(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary erosion via dilation on inverted mask."""
    if radius <= 0:
        return (mask > 0).astype(np.uint8)
    inv = (mask == 0).astype(np.uint8)
    inv_d = _dilate_binary(inv, radius)
    return (inv_d == 0).astype(np.uint8)


# -------------------------
# Patch extraction
# -------------------------

def _extract_patch_2d(
    arr: np.ndarray,
    y0: int,
    x0: int,
    size: int,
    pad_mode: str = "reflect",
    pad_value: float = 0.0,
) -> np.ndarray:
    """Extract [size,size] patch from [H,W], padding if needed."""
    H, W = arr.shape
    y1 = y0 + size
    x1 = x0 + size

    pad_top = max(0, -y0)
    pad_left = max(0, -x0)
    pad_bot = max(0, y1 - H)
    pad_right = max(0, x1 - W)

    yy0 = max(0, y0)
    xx0 = max(0, x0)
    yy1 = min(H, y1)
    xx1 = min(W, x1)

    patch = arr[yy0:yy1, xx0:xx1]

    if pad_top or pad_left or pad_bot or pad_right:
        if pad_mode == "constant":
            patch = np.pad(
                patch,
                ((pad_top, pad_bot), (pad_left, pad_right)),
                mode="constant",
                constant_values=pad_value,
            )
        else:
            patch = np.pad(patch, ((pad_top, pad_bot), (pad_left, pad_right)), mode=pad_mode)

    # Guarantee exact size
    if patch.shape != (size, size):
        patch = patch[:size, :size]
        if patch.shape != (size, size):
            patch2 = np.zeros((size, size), dtype=patch.dtype)
            patch2[: patch.shape[0], : patch.shape[1]] = patch
            patch = patch2

    return patch


# -------------------------
# Dataset
# -------------------------

class InbreastPatchDataset(Dataset):
    """
    Virtual-length patch dataset:
    - __len__ returns epoch_len (can be huge)
    - __getitem__(idx) uses RNG seeded with (seed + idx) for determinism

    Use an epoch-aware sampler (recommended) to avoid repeating patches each epoch.
    """

    def __init__(
        self,
        split_list: Union[str, Path],
        stage: str,
        patch_cfg: PatchConfig,
        norm_cfg: NormConfig,
        target_key: str = "mask_mass",
        epoch_len: int = 100_000,
        seed: int = 42,
        cache_items: int = 8,
        return_torch: bool = True,
    ):
        super().__init__()
        self.split_list = Path(split_list)
        assert self.split_list.exists(), f"Missing split list: {self.split_list}"
        self.paths: List[Path] = [Path(x.strip()) for x in self.split_list.read_text().splitlines() if x.strip()]
        assert len(self.paths) > 0, f"Empty split list: {self.split_list}"

        self.stage = str(stage).lower()
        if self.stage not in ("teacher", "student", "val"):
            # we only implement teacher sampling, but allow others to pass
            self.stage = "teacher"

        self.patch_cfg = patch_cfg
        self.norm_cfg = norm_cfg
        self.target_key = target_key
        self.epoch_len = int(epoch_len)
        self.seed = int(seed)
        self.cache_items = int(cache_items)
        self.return_torch = bool(return_torch)

        # simple LRU-ish cache: path -> dict(np arrays)
        self._cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._cache_order: List[str] = []

        # validate probabilities
        self.p_pos = float(self.patch_cfg.p_pos)
        self.p_hardneg = float(self.patch_cfg.p_hardneg)
        if self.p_pos < 0 or self.p_hardneg < 0:
            raise ValueError("p_pos and p_hardneg must be >= 0")
        if self.p_pos + self.p_hardneg > 1.0 + 1e-6:
            raise ValueError("p_pos + p_hardneg must be <= 1")

    def __len__(self) -> int:
        return self.epoch_len

    def _load_npz(self, p: Path) -> Dict[str, np.ndarray]:
        key = str(p)
        if key in self._cache:
            # bump to end
            if key in self._cache_order:
                self._cache_order.remove(key)
            self._cache_order.append(key)
            return self._cache[key]

        d = np.load(p, allow_pickle=True)
        dd: Dict[str, np.ndarray] = {k: d[k] for k in d.files}
        d.close()

        # insert into cache
        self._cache[key] = dd
        self._cache_order.append(key)
        if self.cache_items > 0 and len(self._cache_order) > self.cache_items:
            old = self._cache_order.pop(0)
            if old in self._cache:
                del self._cache[old]
        return dd

    def _pick_image(self, rng: np.random.Generator) -> Path:
        i = int(rng.integers(0, len(self.paths)))
        return self.paths[i]

    def _pick_center_pos(self, rng: np.random.Generator, tgt: np.ndarray) -> Optional[Tuple[int, int]]:
        ys, xs = np.where(tgt > 0)
        if ys.size == 0:
            return None
        j = int(rng.integers(0, ys.size))
        return int(ys[j]), int(xs[j])

    def _pick_center_from_mask(self, rng: np.random.Generator, mask: np.ndarray) -> Optional[Tuple[int, int]]:
        ys, xs = np.where(mask > 0)
        if ys.size == 0:
            return None
        j = int(rng.integers(0, ys.size))
        return int(ys[j]), int(xs[j])

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.seed + int(idx))

        size = int(self.patch_cfg.patch_size)
        rad = int(self.patch_cfg.hardneg_exclusion_radius)

        # retry loop to satisfy breast fraction
        last_pack = None
        tag = "na"

        for _ in range(int(self.patch_cfg.max_tries)):
            p = self._pick_image(rng)
            d = self._load_npz(p)

            img = d["image"].astype(np.float32, copy=False)
            breast = d.get("mask_breast", np.ones_like(img, dtype=np.uint8))
            breast = (breast > 0).astype(np.uint8)

            # target
            if self.target_key in d:
                tgt = d[self.target_key]
            else:
                # fallback common names
                tgt = d.get("mask_mass", d.get("mask_any", np.zeros_like(img, dtype=np.uint8)))
            tgt = (tgt > 0).astype(np.uint8)

            # normalize per image (clip from breast ROI)
            clip_vals = compute_clip_values(img, breast, self.norm_cfg.clip_p_lo, self.norm_cfg.clip_p_hi)
            img_n = apply_norm(img, breast, self.norm_cfg, precomputed_clip=clip_vals)

            # choose sampling mode
            u = float(rng.random())
            center = None

            if (tgt.sum() > 0) and (u < self.p_pos):
                center = self._pick_center_pos(rng, tgt)
                tag = "pos"
            elif u < (self.p_pos + self.p_hardneg):
                # hard negative: inside breast but away from positives
                dil = _dilate_binary(tgt, rad) if (rad > 0 and tgt.sum() > 0) else tgt
                cand = (breast > 0) & (dil == 0)
                center = self._pick_center_from_mask(rng, cand.astype(np.uint8))
                tag = "hardneg"
            else:
                # random within breast
                center = self._pick_center_from_mask(rng, breast)
                tag = "rand"

            if center is None:
                # fallback uniform anywhere
                y = int(rng.integers(0, img.shape[0]))
                x = int(rng.integers(0, img.shape[1]))
                center = (y, x)
                tag = "uniform"

            cy, cx = center
            y0 = int(cy - size // 2)
            x0 = int(cx - size // 2)

            x_patch = _extract_patch_2d(img_n, y0, x0, size, pad_mode=self.patch_cfg.pad_mode, pad_value=0.0)
            y_patch = _extract_patch_2d(tgt, y0, x0, size, pad_mode="constant", pad_value=0.0).astype(np.uint8)
            b_patch = _extract_patch_2d(breast, y0, x0, size, pad_mode="constant", pad_value=0.0).astype(np.uint8)

            breast_frac = float((b_patch > 0).mean())
            last_pack = (x_patch, y_patch, b_patch, p, clip_vals, center)

            if breast_frac >= float(self.patch_cfg.min_breast_fraction):
                break

        assert last_pack is not None
        x_patch, y_patch, b_patch, p, clip_vals, center = last_pack

        meta = {
            "npz_path": str(p),
            "sample_id": Path(p).stem,
            "tag": tag,
            "center_y": int(center[0]),
            "center_x": int(center[1]),
            "clip_lo": float(clip_vals[0]),
            "clip_hi": float(clip_vals[1]),
            "breast_fraction": float((b_patch > 0).mean()),
        }

        # spacing if available
        if "row_spacing_mm" in self._cache[str(p)]:
            try:
                meta["row_spacing_mm"] = float(self._cache[str(p)]["row_spacing_mm"])
            except Exception:
                pass
        if "col_spacing_mm" in self._cache[str(p)]:
            try:
                meta["col_spacing_mm"] = float(self._cache[str(p)]["col_spacing_mm"])
            except Exception:
                pass

        if self.return_torch:
            x_t = torch.from_numpy(x_patch).float().unsqueeze(0)  # [1,H,W]
            y_t = torch.from_numpy(y_patch.astype(np.float32)).float().unsqueeze(0)  # [1,H,W]
            return x_t, y_t, meta

        return x_patch.astype(np.float32), y_patch.astype(np.uint8), meta
