# src/gen_pseudo.py
"""
Generate rich pseudo maps for refinement:
- prob_mean
- unc_var (MC variance)
- entropy_mean (H(mean_prob))
- entropy_mean_of_samples (mean(H(p_k)))
- mi (mutual information proxy): H(mean) - mean(H(p_k))  (epistemic-ish)
- confidence, ambiguity
- edge_strength (sobel magnitude on normalized image)
- boundary_contour, boundary_band, dist_to_boundary, boundary_soft
- fg_seed, bg_seed (confidence + low-unc seeds)

Writes:
  pseudo_root/all/<sample_id>.npz  (all maps in one file)
Also writes compatibility:
  pseudo_root/prob_mean/<sample_id>.npz   key 'prob'
  pseudo_root/uncertainty/<sample_id>.npz key 'unc'  (selectable type)

Example:
python -m src.gen_pseudo \
  --data_dir data \
  --ckpt ckpt/teacher_best.pt \
  --split train \
  --pseudo_root pseudo/teacher_train \
  --mc_k 8 \
  --tta none \
  --unc_out mi \
  --tile_size 512 --overlap 0.5 --tile_batch 4 \
  --device cuda \
  --skip_existing
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.models import UNet
from src.patches import NormConfig, apply_norm, compute_clip_values
from src.tiling import TilingConfig, sliding_window_predict_proba, enable_mc_dropout

_HAS_CV2 = False
try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

_HAS_SCIPY = False
try:
    from scipy.ndimage import binary_erosion, binary_dilation, distance_transform_edt  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def load_ckpt(path: Path) -> dict:
    return torch.load(str(path), map_location="cpu")


def build_model_from_ckpt(ckpt: dict) -> UNet:
    args = ckpt.get("args", {}) or {}
    model = UNet(
        in_channels=int(args.get("in_channels", 1)),
        out_channels=1,
        base_channels=int(args.get("base_channels", 32)),
        depth=int(args.get("depth", 4)),
        norm=str(args.get("norm", "bn")),
        dropout=float(args.get("dropout", 0.0)),
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    return model


def has_dropout(model: torch.nn.Module) -> bool:
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            return True
    return False


def load_norm_cfg_from_ckpt_or_data(ckpt: dict, data_dir: Path, use_minmax: bool = False) -> NormConfig:
    if ckpt.get("norm_cfg", None) is not None:
        d = ckpt["norm_cfg"]
        return NormConfig(**{**d, "use_minmax": use_minmax, "use_global_zscore": (not use_minmax)})

    norm_path = data_dir / "normalization.json"
    if norm_path.exists():
        d = json.loads(norm_path.read_text())
        clip_lo, clip_hi = d.get("clip_percentiles", [1.0, 99.0])
        return NormConfig(
            clip_p_lo=float(clip_lo),
            clip_p_hi=float(clip_hi),
            use_global_zscore=not use_minmax,
            global_mean=float(d.get("global_mean_after_clip", 0.0)),
            global_std=float(d.get("global_std_after_clip", 1.0)),
            use_minmax=use_minmax,
        )
    return NormConfig(use_minmax=use_minmax, use_global_zscore=not use_minmax)


def binary_entropy(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)).astype(np.float32)


def tta_apply(img: np.ndarray, breast: Optional[np.ndarray], tta: str):
    tta = tta.lower()
    if tta == "none":
        return img, breast, (lambda x: x)
    if tta == "hflip":
        img_t = np.fliplr(img).copy()
        breast_t = None if breast is None else np.fliplr(breast).copy()
        def inv(x: np.ndarray) -> np.ndarray:
            return np.fliplr(x).copy()
        return img_t, breast_t, inv
    raise ValueError("tta must be one of: none, hflip")


def _erode(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return mask.astype(bool)
    m = (mask > 0).astype(np.uint8)
    if _HAS_CV2:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        return (cv2.erode(m, k) > 0)
    if _HAS_SCIPY:
        return binary_erosion(m.astype(bool), structure=np.ones((2 * r + 1, 2 * r + 1), bool))
    # fallback: no erosion
    return m.astype(bool)


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return mask.astype(bool)
    m = (mask > 0).astype(np.uint8)
    if _HAS_CV2:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        return (cv2.dilate(m, k) > 0)
    if _HAS_SCIPY:
        return binary_dilation(m.astype(bool), structure=np.ones((2 * r + 1, 2 * r + 1), bool))
    return m.astype(bool)


def contour_from_mask(mask: np.ndarray) -> np.ndarray:
    """1px-ish contour: mask XOR erode(mask)."""
    m = (mask > 0)
    e = _erode(m.astype(np.uint8), 1)
    c = np.logical_xor(m, e)
    return c.astype(np.uint8)


def dist_to_contour(contour: np.ndarray) -> Optional[np.ndarray]:
    """
    Distance (pixels) to nearest contour pixel.
    Returns float32 [H,W] or None if no backend.
    """
    c = (contour > 0).astype(np.uint8)
    if c.sum() == 0:
        return np.zeros_like(c, dtype=np.float32)

    if _HAS_CV2:
        # distanceTransform returns distance to nearest ZERO.
        inv = np.ones_like(c, dtype=np.uint8)
        inv[c > 0] = 0
        dt = cv2.distanceTransform(inv, distanceType=cv2.DIST_L2, maskSize=3)
        return dt.astype(np.float32)

    if _HAS_SCIPY:
        dt = distance_transform_edt(~(c.astype(bool)))
        return dt.astype(np.float32)

    return None


def edge_strength_sobel(img01: np.ndarray, breast: Optional[np.ndarray]) -> np.ndarray:
    """
    Sobel gradient magnitude on img01 in [0,1].
    Normalized by P99 inside breast (or full image).
    """
    x = img01.astype(np.float32)

    if _HAS_CV2:
        gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
        g = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    else:
        # simple conv
        kx = np.array([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]], dtype=np.float32)
        ky = np.array([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]], dtype=np.float32)
        g = _conv2d_absgrad(x, kx, ky)

    if breast is not None and (breast > 0).sum() > 100:
        roi = g[breast > 0]
    else:
        roi = g.reshape(-1)
    s = float(np.percentile(roi, 99)) if roi.size else float(np.percentile(g, 99))
    s = max(1e-6, s)
    g = np.clip(g / s, 0.0, 1.0).astype(np.float32)
    return g


def _conv2d_absgrad(x: np.ndarray, kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    # minimal convolution (reflect pad)
    xpad = np.pad(x, ((1, 1), (1, 1)), mode="reflect").astype(np.float32)
    gx = np.zeros_like(x, dtype=np.float32)
    gy = np.zeros_like(x, dtype=np.float32)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            patch = xpad[i:i + 3, j:j + 3]
            gx[i, j] = float((patch * kx).sum())
            gy[i, j] = float((patch * ky).sum())
    g = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    return g


@torch.no_grad()
def predict_stack(
    model: torch.nn.Module,
    img_norm: np.ndarray,
    breast: Optional[np.ndarray],
    tile_cfg: TilingConfig,
    mc_k: int,
    tta: str,
) -> np.ndarray:
    """
    Returns stack [K,H,W] of probability maps, including optional TTA.
    """
    preds: List[np.ndarray] = []

    # enable dropout for MC (BN stays eval)
    if mc_k > 1:
        enable_mc_dropout(model)
    else:
        model.eval()

    tta_list = [tta] if tta != "none" else ["none"]
    for t in tta_list:
        img_t, breast_t, inv = tta_apply(img_norm, breast, t)
        for _ in range(int(mc_k)):
            p = sliding_window_predict_proba(model, img_t, mask_breast=breast_t, preprocess=None, cfg=tile_cfg)
            preds.append(inv(p).astype(np.float32))

    return np.stack(preds, axis=0).astype(np.float32)  # [K,H,W]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    ap.add_argument("--pseudo_root", type=str, default="pseudo/teacher_train")

    ap.add_argument("--mc_k", type=int, default=8)
    ap.add_argument("--tta", type=str, default="none", choices=["none", "hflip"])
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--max_images", type=int, default=-1)

    ap.add_argument("--tile_size", type=int, default=512)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--tile_batch", type=int, default=4)
    ap.add_argument("--weight_mode", type=str, default="hann", choices=["hann", "gaussian", "uniform"])
    ap.add_argument("--device", type=str, default="cuda")

    # boundary/seed params
    ap.add_argument("--mask_thr", type=float, default=0.5, help="threshold to form provisional mask for boundary maps")
    ap.add_argument("--band_radius", type=int, default=4, help="boundary band dilation radius (pixels)")
    ap.add_argument("--boundary_soft_maxdist", type=float, default=20.0, help="max dist (pixels) for boundary_soft scaling")

    ap.add_argument("--t_hi", type=float, default=0.90, help="fg seed prob threshold")
    ap.add_argument("--t_lo", type=float, default=0.10, help="bg seed prob threshold")
    ap.add_argument("--seed_unc_source", type=str, default="mi", choices=["mi", "unc_var", "entropy_mean"],
                    help="which uncertainty map to use for seed selection")
    ap.add_argument("--seed_unc_pct", type=float, default=30.0, help="percentile (lower=more strict) for low-unc seeds")
    ap.add_argument("--seed_erode", type=int, default=1, help="erode seeds to keep only stable cores")

    # compatibility uncertainty output
    ap.add_argument("--unc_out", type=str, default="mi", choices=["mi", "unc_var", "entropy_mean"],
                    help="what to store as pseudo_root/uncertainty/* 'unc'")

    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    split_list = data_dir / "splits" / f"{args.split}.txt"
    if not split_list.exists():
        raise FileNotFoundError(f"Missing split list: {split_list}")

    pseudo_root = Path(args.pseudo_root)
    (pseudo_root / "all").mkdir(parents=True, exist_ok=True)
    (pseudo_root / "prob_mean").mkdir(parents=True, exist_ok=True)
    (pseudo_root / "uncertainty").mkdir(parents=True, exist_ok=True)

    ckpt = load_ckpt(Path(args.ckpt))
    model = build_model_from_ckpt(ckpt).to(args.device)
    model.eval()

    if args.mc_k > 1 and not has_dropout(model):
        print("[WARN] Teacher model has no Dropout layers. MC dropout variance/MI will be weak.")
        print("       Consider retraining teacher with dropout=0.1 (recommended).")

    norm_cfg = load_norm_cfg_from_ckpt_or_data(ckpt, data_dir, use_minmax=False)

    tile_cfg = TilingConfig(
        tile_size=args.tile_size,
        overlap=args.overlap,
        batch_size=args.tile_batch,
        weight_mode=args.weight_mode,
        device=args.device,
        amp=True,
    )

    paths = [Path(x.strip()) for x in split_list.read_text().splitlines() if x.strip()]
    if args.max_images > 0:
        paths = paths[: int(args.max_images)]

    for p in tqdm(paths, desc=f"Pseudo-rich ({args.split})"):
        sid = p.stem
        out_all = pseudo_root / "all" / f"{sid}.npz"
        out_prob = pseudo_root / "prob_mean" / f"{sid}.npz"
        out_unc = pseudo_root / "uncertainty" / f"{sid}.npz"
        if args.skip_existing and out_all.exists() and out_prob.exists() and out_unc.exists():
            continue

        d = np.load(p, allow_pickle=True)
        img = d["image"].astype(np.float32)
        breast = d.get("mask_breast", np.ones_like(img, dtype=np.uint8)).astype(np.uint8)

        # normalize for inference
        clip_vals = compute_clip_values(img, breast, norm_cfg.clip_p_lo, norm_cfg.clip_p_hi)
        img_norm = apply_norm(img, breast, norm_cfg, precomputed_clip=clip_vals)

        # also build [0,1] version for edge map using same clip
        lo, hi = clip_vals
        img01 = np.clip(img, lo, hi).astype(np.float32)
        img01 = (img01 - lo) / (max(1e-6, (hi - lo)))
        img01 = np.clip(img01, 0.0, 1.0)

        # predictions stack
        stack = predict_stack(model, img_norm, breast, tile_cfg, mc_k=max(1, args.mc_k), tta=args.tta)  # [K,H,W]
        mean = stack.mean(axis=0).astype(np.float32)
        var = stack.var(axis=0).astype(np.float32)

        # uncertainty / entropy family
        ent_mean = binary_entropy(mean)                      # H(mean)
        ent_samples = binary_entropy(stack)                  # [K,H,W]
        mean_ent = ent_samples.mean(axis=0).astype(np.float32)  # mean(H(p_k))
        mi = (ent_mean - mean_ent).astype(np.float32)        # MI proxy (>=0, ideally)
        mi = np.clip(mi, 0.0, None).astype(np.float32)

        # confidence/ambiguity
        conf = np.abs(mean - 0.5).astype(np.float32)         # 0..0.5
        amb = (1.0 - 2.0 * conf).astype(np.float32)          # 0..1 (high near 0.5)

        # edge map
        edge = edge_strength_sobel(img01, breast)

        # provisional mask + boundary maps
        m = (mean >= float(args.mask_thr)).astype(np.uint8)
        contour = contour_from_mask(m)                       # thin contour
        band = _dilate(contour, int(args.band_radius)).astype(np.uint8)

        dt = dist_to_contour(contour)
        if dt is None:
            dt = np.zeros_like(mean, dtype=np.float32)
        dt = dt.astype(np.float32)

        # boundary soft focus: 1 near boundary, 0 far away
        maxd = float(args.boundary_soft_maxdist)
        boundary_soft = np.clip(1.0 - (dt / max(1e-6, maxd)), 0.0, 1.0).astype(np.float32)

        # seeds
        # choose uncertainty map for seeds
        if args.seed_unc_source == "mi":
            u_seed = mi
        elif args.seed_unc_source == "entropy_mean":
            u_seed = ent_mean
        else:
            u_seed = var

        # compute low-unc threshold within breast
        roi = u_seed[breast > 0] if (breast > 0).sum() > 100 else u_seed.reshape(-1)
        u_thr = float(np.percentile(roi, float(args.seed_unc_pct))) if roi.size else float(np.percentile(u_seed, float(args.seed_unc_pct)))

        fg = (mean >= float(args.t_hi)) & (u_seed <= u_thr) & (breast > 0)
        bg = (mean <= float(args.t_lo)) & (u_seed <= u_thr) & (breast > 0)

        if args.seed_erode > 0:
            fg = _erode(fg.astype(np.uint8), int(args.seed_erode))
            bg = _erode(bg.astype(np.uint8), int(args.seed_erode))

        fg_seed = fg.astype(np.uint8)
        bg_seed = bg.astype(np.uint8)

        # select "unc" for compatibility output
        if args.unc_out == "mi":
            unc_out = mi
        elif args.unc_out == "entropy_mean":
            unc_out = ent_mean
        else:
            unc_out = var

        # save
        np.savez_compressed(out_prob, prob=mean.astype(np.float32))
        np.savez_compressed(out_unc, unc=unc_out.astype(np.float32))

        np.savez_compressed(
            out_all,
            prob=mean.astype(np.float32),
            unc_var=var.astype(np.float32),
            entropy_mean=ent_mean.astype(np.float32),
            entropy_mean_of_samples=mean_ent.astype(np.float32),
            mi=mi.astype(np.float32),
            confidence=conf.astype(np.float32),
            ambiguity=amb.astype(np.float32),
            edge_strength=edge.astype(np.float32),
            boundary_contour=contour.astype(np.uint8),
            boundary_band=band.astype(np.uint8),
            dist_to_boundary=dt.astype(np.float32),
            boundary_soft=boundary_soft.astype(np.float32),
            fg_seed=fg_seed.astype(np.uint8),
            bg_seed=bg_seed.astype(np.uint8),
        )

    print(f"Saved all maps: {pseudo_root/'all'}")
    print(f"Saved prob_mean: {pseudo_root/'prob_mean'}")
    print(f"Saved uncertainty: {pseudo_root/'uncertainty'}  (unc_out={args.unc_out})")


if __name__ == "__main__":
    main()