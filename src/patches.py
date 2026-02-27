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

# # src/patches.py
# """
# Patch sampling utilities for INbreast large mammograms stored as .npz.

# Supports:
# - teacher sampling (pos / hard-neg / random-breast mix)
# - student sampling (uncertainty/boundary/hardneg) using pseudo maps
# - per-image clip + normalization (ROI-aware using mask_breast)

# Expected keys in data npz:
# - image: float32 [H,W]
# - mask_breast: uint8 [H,W] (0/1)
# - mask_mass / mask_any: uint8 [H,W] (0/1)

# Pseudo maps (from pseudo_root):
# Preferred:
# - pseudo_root/all/<sample_id>.npz containing keys like:
#     prob, mi, unc_var, entropy_mean, boundary_band, edge_strength, fg_seed, bg_seed, etc.
# Fallback (legacy):
# - pseudo_root/prob_mean/<sample_id>.npz with key 'prob'
# - pseudo_root/uncertainty/<sample_id>.npz with key 'unc'

# Returned x:
# - teacher: [1,H,W]  (image only)
# - student: [1+len(pseudo_keys),H,W] where order = [image] + pseudo_keys

# Deps: numpy, torch
# Optional (for morphology): opencv-python or scikit-image
# """

# from __future__ import annotations
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Dict, List, Optional, Tuple, Union

# import numpy as np
# import torch
# from torch.utils.data import Dataset

# _HAS_CV2 = False
# _HAS_SKIMAGE = False
# try:
#     import cv2  # type: ignore
#     _HAS_CV2 = True
# except Exception:
#     _HAS_CV2 = False

# if not _HAS_CV2:
#     try:
#         from skimage.morphology import binary_dilation, binary_erosion, disk  # type: ignore
#         _HAS_SKIMAGE = True
#     except Exception:
#         _HAS_SKIMAGE = False


# # -------------------------
# # Normalization
# # -------------------------

# @dataclass
# class NormConfig:
#     clip_p_lo: float = 1.0
#     clip_p_hi: float = 99.0
#     use_global_zscore: bool = True
#     global_mean: float = 0.0
#     global_std: float = 1.0
#     # If True: per-image minmax after clipping
#     use_minmax: bool = False
#     eps: float = 1e-8


# def compute_clip_values(img: np.ndarray, breast_mask: Optional[np.ndarray], p_lo: float, p_hi: float) -> Tuple[float, float]:
#     if breast_mask is None:
#         roi = img.reshape(-1)
#     else:
#         roi = img[breast_mask > 0]
#         if roi.size < 100:
#             roi = img.reshape(-1)
#     lo = float(np.percentile(roi, p_lo))
#     hi = float(np.percentile(roi, p_hi))
#     if not np.isfinite(lo):
#         lo = float(np.min(img))
#     if not np.isfinite(hi):
#         hi = float(np.max(img))
#     if hi <= lo:
#         hi = lo + 1.0
#     return lo, hi


# def apply_norm(img: np.ndarray, breast_mask: Optional[np.ndarray], norm: NormConfig,
#                precomputed_clip: Optional[Tuple[float, float]] = None) -> np.ndarray:
#     """Return float32 normalized image [H,W]."""
#     if precomputed_clip is None:
#         lo, hi = compute_clip_values(img, breast_mask, norm.clip_p_lo, norm.clip_p_hi)
#     else:
#         lo, hi = precomputed_clip

#     x = np.clip(img, lo, hi).astype(np.float32)

#     if norm.use_minmax:
#         x = (x - lo) / max(norm.eps, (hi - lo))
#         return x.astype(np.float32)

#     if norm.use_global_zscore:
#         x = (x - float(norm.global_mean)) / float(max(norm.eps, norm.global_std))
#         return x.astype(np.float32)

#     # per-image z-score fallback
#     if breast_mask is not None and (breast_mask > 0).sum() > 100:
#         roi = x[breast_mask > 0]
#         mu = float(roi.mean())
#         sd = float(roi.std() + norm.eps)
#     else:
#         mu = float(x.mean())
#         sd = float(x.std() + norm.eps)
#     x = (x - mu) / sd
#     return x.astype(np.float32)


# # -------------------------
# # Morphology helpers (optional)
# # -------------------------

# def _binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
#     if radius <= 0:
#         return mask.astype(bool)
#     m = mask.astype(np.uint8)
#     if _HAS_CV2:
#         k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
#         return (cv2.dilate(m, k) > 0)
#     if _HAS_SKIMAGE:
#         return binary_dilation(mask.astype(bool), disk(radius))
#     # fallback: no dilation
#     return mask.astype(bool)


# def _binary_erode(mask: np.ndarray, radius: int) -> np.ndarray:
#     if radius <= 0:
#         return mask.astype(bool)
#     m = mask.astype(np.uint8)
#     if _HAS_CV2:
#         k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
#         return (cv2.erode(m, k) > 0)
#     if _HAS_SKIMAGE:
#         return binary_erosion(mask.astype(bool), disk(radius))
#     return mask.astype(bool)


# def boundary_band(mask: np.ndarray, radius: int = 3) -> np.ndarray:
#     """Approx boundary pixels as dilate(mask)-erode(mask)."""
#     m = mask.astype(bool)
#     dil = _binary_dilate(m, radius)
#     ero = _binary_erode(m, radius)
#     b = np.logical_and(dil, np.logical_not(ero))
#     return b.astype(np.uint8)


# # -------------------------
# # Patch extraction
# # -------------------------

# def extract_patch(arr: np.ndarray, cy: int, cx: int, ph: int, pw: int,
#                   pad_mode: str = "reflect", pad_value: float = 0.0) -> np.ndarray:
#     """Extract patch centered at (cy,cx), pads if needed."""
#     H, W = arr.shape[:2]
#     y0 = cy - ph // 2
#     x0 = cx - pw // 2
#     y1 = y0 + ph
#     x1 = x0 + pw

#     pad_top = max(0, -y0)
#     pad_left = max(0, -x0)
#     pad_bot = max(0, y1 - H)
#     pad_right = max(0, x1 - W)

#     y0c = max(0, y0)
#     x0c = max(0, x0)
#     y1c = min(H, y1)
#     x1c = min(W, x1)

#     patch = arr[y0c:y1c, x0c:x1c]

#     if pad_top or pad_left or pad_bot or pad_right:
#         if pad_mode == "constant":
#             patch = np.pad(
#                 patch,
#                 ((pad_top, pad_bot), (pad_left, pad_right)) + (() if patch.ndim == 2 else ((0, 0),)),
#                 mode=pad_mode,
#                 constant_values=pad_value,
#             )
#         else:
#             patch = np.pad(
#                 patch,
#                 ((pad_top, pad_bot), (pad_left, pad_right)) + (() if patch.ndim == 2 else ((0, 0),)),
#                 mode=pad_mode,
#             )
#     return patch


# def _random_true_index(mask: np.ndarray, rng: np.random.Generator) -> Optional[Tuple[int, int]]:
#     ys, xs = np.where(mask > 0)
#     if ys.size == 0:
#         return None
#     i = int(rng.integers(0, ys.size))
#     return int(ys[i]), int(xs[i])


# def _choose_center_teacher(
#     rng: np.random.Generator,
#     mask_pos: np.ndarray,
#     mask_breast: np.ndarray,
#     p_pos: float,
#     p_hardneg: float,
#     neg_exclusion_radius: int = 8,
# ) -> Tuple[int, int, str]:
#     r = float(rng.random())
#     H, W = mask_breast.shape

#     if r < p_pos:
#         idx = _random_true_index(mask_pos, rng)
#         if idx is not None:
#             return idx[0], idx[1], "pos"

#     if r < p_pos + p_hardneg:
#         if mask_pos.sum() > 0 and neg_exclusion_radius > 0:
#             avoid = _binary_dilate(mask_pos.astype(bool), neg_exclusion_radius)
#             cand = (mask_breast > 0) & (~avoid)
#         else:
#             cand = (mask_breast > 0) & (mask_pos == 0)
#         idx = _random_true_index(cand.astype(np.uint8), rng)
#         if idx is not None:
#             return idx[0], idx[1], "hardneg"

#     idx = _random_true_index((mask_breast > 0).astype(np.uint8), rng)
#     if idx is not None:
#         return idx[0], idx[1], "rand"

#     return int(rng.integers(0, H)), int(rng.integers(0, W)), "uniform"


# def _choose_center_student(
#     rng: np.random.Generator,
#     mask_breast: np.ndarray,
#     pseudo_uncertainty: Optional[np.ndarray],
#     pseudo_prob: Optional[np.ndarray],
#     mask_pos_hint: Optional[np.ndarray],
#     p_unc: float = 0.5,
#     p_boundary: float = 0.3,
#     p_hardneg: float = 0.2,
#     unc_topk: float = 0.10,
#     boundary_radius: int = 3,
#     hardneg_prob_lo: float = 0.15,
#     hardneg_prob_hi: float = 0.45,
# ) -> Tuple[int, int, str]:
#     H, W = mask_breast.shape
#     r = float(rng.random())

#     roi = (mask_breast > 0)
#     if roi.sum() == 0:
#         return int(rng.integers(0, H)), int(rng.integers(0, W)), "uniform"

#     # 1) high uncertainty
#     if pseudo_uncertainty is not None and r < p_unc:
#         u = pseudo_uncertainty.copy()
#         u[~roi] = -np.inf
#         valid = np.isfinite(u)
#         if valid.sum() > 0:
#             thr = np.quantile(u[valid], 1.0 - unc_topk)
#             cand = (u >= thr) & roi
#             idx = _random_true_index(cand.astype(np.uint8), rng)
#             if idx is not None:
#                 return idx[0], idx[1], "unc"

#     # 2) boundary
#     if r < p_unc + p_boundary:
#         if mask_pos_hint is not None and mask_pos_hint.sum() > 0:
#             band = boundary_band(mask_pos_hint, radius=boundary_radius).astype(bool)
#         elif pseudo_prob is not None:
#             band = (pseudo_prob > 0.4) & (pseudo_prob < 0.6)
#         else:
#             band = np.zeros((H, W), dtype=bool)
#         band = band & roi
#         idx = _random_true_index(band.astype(np.uint8), rng)
#         if idx is not None:
#             return idx[0], idx[1], "boundary"

#     # 3) hard negatives
#     if pseudo_prob is not None and r < p_unc + p_boundary + p_hardneg:
#         cand = roi & (pseudo_prob >= hardneg_prob_lo) & (pseudo_prob <= hardneg_prob_hi)
#         idx = _random_true_index(cand.astype(np.uint8), rng)
#         if idx is not None:
#             return idx[0], idx[1], "hardneg"

#     idx = _random_true_index(roi.astype(np.uint8), rng)
#     if idx is not None:
#         return idx[0], idx[1], "rand"

#     return int(rng.integers(0, H)), int(rng.integers(0, W)), "uniform"


# @dataclass
# class PatchConfig:
#     patch_size: int = 512
#     # Teacher sampling mix
#     p_pos: float = 0.50
#     p_hardneg: float = 0.25
#     # Student sampling mix
#     p_unc: float = 0.50
#     p_boundary: float = 0.30
#     p_hardneg_student: float = 0.20
#     # Exclude negatives close to positives
#     neg_exclusion_radius: int = 8
#     # Minimum breast coverage in patch (0..1)
#     min_breast_fraction: float = 0.60
#     # retries
#     max_tries: int = 20
#     pad_mode: str = "reflect"


# # -------------------------
# # NPZ cache
# # -------------------------

# class NpzCache:
#     """Tiny in-memory cache for loaded npz arrays."""
#     def __init__(self, max_items: int = 8):
#         self.max_items = int(max_items)
#         self._keys: List[str] = []
#         self._cache: Dict[str, Dict[str, np.ndarray]] = {}

#     def get(self, path: Union[str, Path]) -> Dict[str, np.ndarray]:
#         key = str(path)
#         if key in self._cache:
#             return self._cache[key]
#         d = np.load(key, allow_pickle=True)
#         out = {k: d[k] for k in d.files}
#         self._cache[key] = out
#         self._keys.append(key)
#         if len(self._keys) > self.max_items:
#             old = self._keys.pop(0)
#             self._cache.pop(old, None)
#         return out


# # -------------------------
# # Dataset
# # -------------------------

# class InbreastPatchDataset(Dataset):
#     """
#     Patch dataset that samples patches from full images on-the-fly.

#     __len__ is "epoch_len" (patches per epoch), not #images.
#     Deterministic sampling per idx via seed+idx.

#     For student:
#       x = [image] + pseudo maps in order of pseudo_keys
#     """
#     def __init__(
#         self,
#         split_list: Union[str, Path],
#         stage: str,  # "teacher" or "student"
#         patch_cfg: PatchConfig,
#         norm_cfg: NormConfig,
#         target_key: str = "mask_mass",
#         epoch_len: int = 4000,
#         seed: int = 123,
#         cache_items: int = 8,
#         pseudo_root: Optional[Union[str, Path]] = None,
#         pseudo_keys: Optional[List[str]] = None,   # <-- NEW
#         return_torch: bool = True,
#     ):
#         self.paths = [Path(p.strip()) for p in Path(split_list).read_text().splitlines() if p.strip()]
#         if len(self.paths) == 0:
#             raise ValueError(f"Empty split list: {split_list}")
#         if stage not in ("teacher", "student"):
#             raise ValueError("stage must be 'teacher' or 'student'")

#         self.stage = stage
#         self.patch_cfg = patch_cfg
#         self.norm_cfg = norm_cfg
#         self.target_key = target_key
#         self.epoch_len = int(epoch_len)
#         self.seed = int(seed)
#         self.cache = NpzCache(max_items=cache_items)
#         self.pseudo_root = Path(pseudo_root) if pseudo_root is not None else None

#         # NEW: which pseudo maps to load and stack as channels
#         if self.stage == "student":
#             self.pseudo_keys = pseudo_keys if pseudo_keys is not None else ["prob", "unc"]
#         else:
#             self.pseudo_keys = []  # teacher never loads pseudo maps

#         self.return_torch = bool(return_torch)

#     def __len__(self) -> int:
#         return self.epoch_len

#     # ---------- NEW: generic pseudo loader ----------
#     def _load_pseudo_dict(self, sample_id: str, keys: List[str]) -> Dict[str, np.ndarray]:
#         """
#         Preferred: pseudo_root/all/<sample_id>.npz
#         Fallback:
#           pseudo_root/prob_mean/<id>.npz ('prob')
#           pseudo_root/uncertainty/<id>.npz ('unc')
#         """
#         out: Dict[str, np.ndarray] = {}
#         if self.pseudo_root is None:
#             return out

#         all_path = self.pseudo_root / "all" / f"{sample_id}.npz"
#         if all_path.exists():
#             dd = np.load(all_path, allow_pickle=True)
#             for k in keys:
#                 if k in dd.files:
#                     out[k] = dd[k]
#             # allow also direct access to prob if present
#             if "prob" in dd.files and "prob" not in out and "prob" in keys:
#                 out["prob"] = dd["prob"]
#             return out

#         # legacy fallback
#         if "prob" in keys:
#             p_prob = self.pseudo_root / "prob_mean" / f"{sample_id}.npz"
#             if p_prob.exists():
#                 dd = np.load(p_prob, allow_pickle=True)
#                 out["prob"] = dd["prob"].astype(np.float32)

#         if "unc" in keys:
#             p_unc = self.pseudo_root / "uncertainty" / f"{sample_id}.npz"
#             if p_unc.exists():
#                 dd = np.load(p_unc, allow_pickle=True)
#                 out["unc"] = dd["unc"].astype(np.float32)

#         return out

#     def __getitem__(self, idx: int):
#         rng = np.random.default_rng(self.seed + idx)

#         # pick random image each patch
#         img_path = self.paths[int(rng.integers(0, len(self.paths)))]
#         d = self.cache.get(img_path)

#         img = d["image"].astype(np.float32)
#         breast = d.get("mask_breast", np.ones_like(img, dtype=np.uint8)).astype(np.uint8)
#         target = d.get(self.target_key, d.get("mask_any", np.zeros_like(img, dtype=np.uint8))).astype(np.uint8)

#         # per-image clip once
#         clip_vals = compute_clip_values(img, breast, self.norm_cfg.clip_p_lo, self.norm_cfg.clip_p_hi)
#         img_norm = apply_norm(img, breast, self.norm_cfg, precomputed_clip=clip_vals)

#         ph = pw = int(self.patch_cfg.patch_size)

#         # student: load pseudo maps dict
#         pseudo: Dict[str, np.ndarray] = {}
#         pseudo_prob_full: Optional[np.ndarray] = None
#         pseudo_unc_full: Optional[np.ndarray] = None

#         if self.stage == "student":
#             pseudo = self._load_pseudo_dict(img_path.stem, keys=self.pseudo_keys)

#             # sampling helpers (try best-effort)
#             pseudo_prob_full = pseudo.get("prob", None)
#             # choose an "uncertainty-like" map if not explicitly present
#             pseudo_unc_full = pseudo.get("unc", None)
#             if pseudo_unc_full is None:
#                 for cand in ("mi", "entropy_mean", "unc_var", "entropy_mean_of_samples"):
#                     if cand in pseudo:
#                         pseudo_unc_full = pseudo[cand].astype(np.float32)
#                         break

#         # sample center (with retries to satisfy breast coverage)
#         chosen_tag = "na"
#         for _ in range(self.patch_cfg.max_tries):
#             if self.stage == "teacher":
#                 cy, cx, chosen_tag = _choose_center_teacher(
#                     rng, target, breast,
#                     p_pos=self.patch_cfg.p_pos,
#                     p_hardneg=self.patch_cfg.p_hardneg,
#                     neg_exclusion_radius=self.patch_cfg.neg_exclusion_radius,
#                 )
#             else:
#                 cy, cx, chosen_tag = _choose_center_student(
#                     rng, breast,
#                     pseudo_uncertainty=pseudo_unc_full,
#                     pseudo_prob=pseudo_prob_full,
#                     mask_pos_hint=target,
#                     p_unc=self.patch_cfg.p_unc,
#                     p_boundary=self.patch_cfg.p_boundary,
#                     p_hardneg=self.patch_cfg.p_hardneg_student,
#                 )

#             p_breast = extract_patch(breast, cy, cx, ph, pw, pad_mode="constant", pad_value=0).astype(np.uint8)
#             frac = float((p_breast > 0).mean())
#             if frac >= self.patch_cfg.min_breast_fraction:
#                 break

#         # extract patches
#         p_img = extract_patch(img_norm, cy, cx, ph, pw, pad_mode=self.patch_cfg.pad_mode).astype(np.float32)
#         p_tgt = extract_patch(target, cy, cx, ph, pw, pad_mode="constant", pad_value=0).astype(np.uint8)

#         # ---------- NEW: stack pseudo channels in order of pseudo_keys ----------
#         chans: List[np.ndarray] = [p_img.astype(np.float32)]  # channel 0 = image

#         if self.stage == "student":
#             for k in self.pseudo_keys:
#                 if k not in pseudo:
#                     # missing => zeros (keeps shapes consistent; but you should ensure pseudo exists in practice)
#                     chans.append(np.zeros((ph, pw), dtype=np.float32))
#                 else:
#                     arr = pseudo[k]
#                     # cast to float32 (even if stored uint8 like boundary_band/seed)
#                     if arr.dtype != np.float32:
#                         arr = arr.astype(np.float32)
#                     p_arr = extract_patch(arr, cy, cx, ph, pw, pad_mode=self.patch_cfg.pad_mode).astype(np.float32)
#                     chans.append(p_arr)

#         x = np.stack(chans, axis=0).astype(np.float32)  # [C,H,W]
#         y = p_tgt[None, ...].astype(np.float32)         # [1,H,W]

#         meta = {
#             "path": str(img_path),
#             "sample_id": img_path.stem,
#             "tag": chosen_tag,
#             "clip_lo": float(clip_vals[0]),
#             "clip_hi": float(clip_vals[1]),
#             "stage": self.stage,
#             "pseudo_keys": ",".join(self.pseudo_keys),
#         }

#         if self.return_torch:
#             return torch.from_numpy(x), torch.from_numpy(y), meta
#         return x, y, meta