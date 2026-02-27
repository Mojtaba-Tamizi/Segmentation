# src/vis_pseudo.py
"""
Visual QC: compare GT mask, pseudo prob/uncertainty, teacher (stage-1) prediction,
and student/refiner (stage-2) prediction side-by-side.

It can:
- load a sample (.npz) from data/splits/{train|val|test}.txt OR a direct --npz path
- load pseudo maps from pseudo_root/{prob_mean,uncertainty}/{sample_id}.npz
- optionally run inference with teacher_ckpt and student_ckpt (stitched full image)

Outputs:
- saves PNGs to --out_dir (default: vis/qc)

Example:
python -m src.vis_pseudo \
  --data_dir data \
  --split train --n 8 \
  --teacher_ckpt ckpt/teacher_best.pt \
  --student_ckpt ckpt/student_best.pt \
  --pseudo_root pseudo/teacher_train \
  --out_dir vis/qc \
  --tile_size 512 --overlap 0.5 --tile_batch 4 --threshold 0.5

Notes:
- If pseudo maps are missing (val/test), they will show "N/A".
- If you trained an input-augmented student (image+prob+unc), this script supports it
  by tiling multi-channel input internally.

Deps:
  pip install numpy matplotlib torch tqdm
  pip install scipy   (only if you later want distance metrics here; not required for vis)
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models import UNet
from src.patches import NormConfig, apply_norm, compute_clip_values
from src.tiling import TilingConfig, weight_window, breast_bbox, iter_tiles


# -------------------------
# ckpt/model utils
# -------------------------

def load_ckpt(path: Path) -> dict:
    return torch.load(str(path), map_location="cpu")


def build_unet_from_ckpt(ckpt: dict, fallback_in_channels: int = 1) -> UNet:
    args = ckpt.get("args", {}) or {}
    in_ch = int(args.get("in_channels", fallback_in_channels))

    # Student ckpt (from our train_student.py) may not store in_channels explicitly.
    if "include_prob_input" in args or "include_unc_input" in args:
        in_ch = 1 + (1 if bool(args.get("include_prob_input", False)) else 0) + (1 if bool(args.get("include_unc_input", False)) else 0)

    model = UNet(
        in_channels=in_ch,
        out_channels=1,
        base_channels=int(args.get("base_channels", 32)),
        depth=int(args.get("depth", 4)),
        norm=str(args.get("norm", "bn")),
        dropout=float(args.get("dropout", 0.0)),
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    return model


def load_norm_cfg_from_ckpt_or_data(ckpt: Optional[dict], data_dir: Path, use_minmax: bool = False) -> NormConfig:
    if ckpt is not None and ckpt.get("norm_cfg", None) is not None:
        d = ckpt["norm_cfg"]
        # allow overriding minmax for vis if wanted
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


# -------------------------
# tiling for multi-channel input (CHW)
# -------------------------

@torch.no_grad()
def sliding_window_predict_proba_chw(
    model: torch.nn.Module,
    x_chw: np.ndarray,                # [C,H,W]
    mask_breast: Optional[np.ndarray],
    cfg: TilingConfig,
) -> np.ndarray:
    """
    Like src.tiling.sliding_window_predict_proba, but supports multi-channel CHW input.
    Returns prob map [H,W].
    """
    assert x_chw.ndim == 3, "x_chw must be [C,H,W]"
    C, H, W = x_chw.shape
    th = tw = int(cfg.tile_size)

    by0, by1, bx0, bx1 = breast_bbox(mask_breast, cfg.roi_pad, (H, W))

    win = weight_window(th, tw, cfg.weight_mode)
    acc = np.zeros((H, W), dtype=np.float32)
    wsum = np.zeros((H, W), dtype=np.float32)

    device = torch.device(cfg.device) if isinstance(cfg.device, str) else cfg.device
    model = model.to(device)
    model.eval()

    tiles_xy = list(iter_tiles(by0, by1, bx0, bx1, th, cfg.overlap))

    batch_tiles = []
    batch_meta = []  # (yy,xx, py0,py1,px0,px1)
    for (yy, xx) in tiles_xy:
        tile, (py0, py1, px0, px1) = _extract_tile_chw(x_chw, yy, xx, th, tw, pad_mode=cfg.pad_mode)
        batch_tiles.append(tile)
        batch_meta.append((yy, xx, py0, py1, px0, px1))
        if len(batch_tiles) == cfg.batch_size:
            _run_and_blend_chw(model, batch_tiles, batch_meta, win, acc, wsum, device, cfg.amp)
            batch_tiles, batch_meta = [], []

    if batch_tiles:
        _run_and_blend_chw(model, batch_tiles, batch_meta, win, acc, wsum, device, cfg.amp)

    prob = acc / (wsum + 1e-8)
    return np.clip(prob, 0.0, 1.0).astype(np.float32)


def _extract_tile_chw(
    x_chw: np.ndarray, y0: int, x0: int, th: int, tw: int, pad_mode: str = "reflect"
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Extract CHW tile with padding if out of bounds.
    Returns tile [C,th,tw] and slice (py0,py1,px0,px1) mapping to real pixels.
    """
    C, H, W = x_chw.shape
    y1 = y0 + th
    x1 = x0 + tw

    pad_top = max(0, -y0)
    pad_left = max(0, -x0)
    pad_bot = max(0, y1 - H)
    pad_right = max(0, x1 - W)

    yy0 = max(0, y0)
    xx0 = max(0, x0)
    yy1 = min(H, y1)
    xx1 = min(W, x1)

    tile = x_chw[:, yy0:yy1, xx0:xx1].astype(np.float32)
    if pad_top or pad_left or pad_bot or pad_right:
        tile = np.pad(tile, ((0, 0), (pad_top, pad_bot), (pad_left, pad_right)), mode=pad_mode)

    py0 = pad_top
    py1 = py0 + (yy1 - yy0)
    px0 = pad_left
    px1 = px0 + (xx1 - xx0)
    return tile, (py0, py1, px0, px1)


def _run_and_blend_chw(
    model: torch.nn.Module,
    batch_tiles: list,
    batch_meta: list,
    win: np.ndarray,
    acc: np.ndarray,
    wsum: np.ndarray,
    device: torch.device,
    amp: bool,
) -> None:
    x = np.stack(batch_tiles, axis=0)  # [B,C,H,W]
    x_t = torch.from_numpy(x).to(device=device, dtype=torch.float32)

    if device.type == "cuda" and amp:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = model(x_t)
    else:
        out = model(x_t)

    if out.ndim == 4:
        logits = out[:, 0]
    elif out.ndim == 3:
        logits = out
    else:
        raise ValueError(f"Unexpected model output shape: {tuple(out.shape)}")

    prob = torch.sigmoid(logits).detach().float().cpu().numpy()  # [B,H,W]

    for i, (yy, xx, py0, py1, px0, px1) in enumerate(batch_meta):
        tile_prob = prob[i]
        tile_win = win

        real_h = py1 - py0
        real_w = px1 - px0
        y0_img = max(0, yy)
        x0_img = max(0, xx)
        y1_img = min(acc.shape[0], yy + real_h)
        x1_img = min(acc.shape[1], xx + real_w)

        p_prob_crop = tile_prob[py0:py0 + (y1_img - y0_img), px0:px0 + (x1_img - x0_img)]
        p_win_crop = tile_win[py0:py0 + (y1_img - y0_img), px0:px0 + (x1_img - x0_img)]

        acc[y0_img:y1_img, x0_img:x1_img] += p_prob_crop * p_win_crop
        wsum[y0_img:y1_img, x0_img:x1_img] += p_win_crop


# -------------------------
# loading helpers
# -------------------------

def load_paths_from_split(data_dir: Path, split: str) -> List[Path]:
    p = data_dir / "splits" / f"{split}.txt"
    if not p.exists():
        raise FileNotFoundError(f"Missing split list: {p}")
    return [Path(x.strip()) for x in p.read_text().splitlines() if x.strip()]


def load_pseudo_maps(pseudo_root: Path, sample_id: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    p_prob = pseudo_root / "prob_mean" / f"{sample_id}.npz"
    p_unc = pseudo_root / "uncertainty" / f"{sample_id}.npz"
    prob = None
    unc = None
    if p_prob.exists():
        d = np.load(p_prob, allow_pickle=True)
        prob = d["prob"].astype(np.float32)
    if p_unc.exists():
        d = np.load(p_unc, allow_pickle=True)
        unc = d["unc"].astype(np.float32)
    return prob, unc


def normalize_for_display(img: np.ndarray, breast: Optional[np.ndarray]) -> np.ndarray:
    """Map image to [0,1] for visualization using ROI percentiles."""
    if breast is not None and (breast > 0).sum() > 100:
        roi = img[breast > 0]
    else:
        roi = img.reshape(-1)
    lo, hi = np.percentile(roi, [1, 99])
    if hi <= lo:
        lo, hi = float(img.min()), float(img.max() + 1e-6)
    x = np.clip(img, lo, hi)
    x = (x - lo) / (hi - lo + 1e-8)
    return x.astype(np.float32)


def overlay_mask(ax, img01: np.ndarray, mask: np.ndarray, title: str, alpha: float = 0.35):
    ax.imshow(img01, cmap="gray", vmin=0, vmax=1)
    m = (mask > 0).astype(np.uint8)
    if m.sum() > 0:
        ax.imshow(m, cmap="Reds", alpha=alpha, vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")


def show_map(ax, m: Optional[np.ndarray], title: str, cmap: str, vmin=None, vmax=None):
    if m is None:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=14)
        ax.set_title(title)
        ax.axis("off")
        return
    im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    return im


# -------------------------
# main visualization routine
# -------------------------

def make_figure(
    img01: np.ndarray,
    breast: np.ndarray,
    gt: np.ndarray,
    pseudo_prob: Optional[np.ndarray],
    pseudo_unc: Optional[np.ndarray],
    teacher_prob: Optional[np.ndarray],
    student_prob: Optional[np.ndarray],
    thr: float,
    title_prefix: str,
) -> plt.Figure:
    """
    Layout (2 rows x 4 cols):
      Row1: Image | GT overlay | Teacher prob | Teacher mask
      Row2: Pseudo prob | Pseudo unc | Student prob | Student mask
    """
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.reshape(2, 4)

    # Row 1
    axes[0, 0].imshow(img01, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title(f"{title_prefix}\nImage")
    axes[0, 0].axis("off")

    overlay_mask(axes[0, 1], img01, gt, "GT mask overlay")

    im_tp = show_map(axes[0, 2], teacher_prob, "Teacher prob", cmap="viridis", vmin=0.0, vmax=1.0)
    if teacher_prob is not None:
        teacher_mask = (teacher_prob >= thr).astype(np.uint8)
        overlay_mask(axes[0, 3], img01, teacher_mask, f"Teacher mask (thr={thr:.2f})")
    else:
        axes[0, 3].text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=14)
        axes[0, 3].set_title("Teacher mask")
        axes[0, 3].axis("off")

    # Row 2
    im_pp = show_map(axes[1, 0], pseudo_prob, "Pseudo prob", cmap="viridis", vmin=0.0, vmax=1.0)

    # For uncertainty, autoscale to a robust max so it's interpretable
    if pseudo_unc is not None:
        vmax_unc = float(np.percentile(pseudo_unc[np.isfinite(pseudo_unc)], 99)) if np.isfinite(pseudo_unc).any() else None
        im_pu = show_map(axes[1, 1], pseudo_unc, "Pseudo uncertainty", cmap="magma", vmin=0.0, vmax=vmax_unc)
    else:
        im_pu = show_map(axes[1, 1], None, "Pseudo uncertainty", cmap="magma")

    im_sp = show_map(axes[1, 2], student_prob, "Student prob", cmap="viridis", vmin=0.0, vmax=1.0)
    if student_prob is not None:
        student_mask = (student_prob >= thr).astype(np.uint8)
        overlay_mask(axes[1, 3], img01, student_mask, f"Student mask (thr={thr:.2f})")
    else:
        axes[1, 3].text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=14)
        axes[1, 3].set_title("Student mask")
        axes[1, 3].axis("off")

    # Colorbars for prob/unc if present
    # (keep them compact)
    if im_tp is not None:
        fig.colorbar(im_tp, ax=axes[0, 2], fraction=0.046, pad=0.04)
    if im_pp is not None:
        fig.colorbar(im_pp, ax=axes[1, 0], fraction=0.046, pad=0.04)
    if pseudo_unc is not None and im_pu is not None:
        fig.colorbar(im_pu, ax=axes[1, 1], fraction=0.046, pad=0.04)
    if im_sp is not None:
        fig.colorbar(im_sp, ax=axes[1, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--split", type=str, default=None, choices=["train", "val", "test"])
    ap.add_argument("--npz", type=str, default=None, help="Direct path to one .npz sample (overrides --split).")
    ap.add_argument("--n", type=int, default=8, help="Number of samples to visualize (only for --split).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="vis/qc")
    ap.add_argument("--show", action="store_true")

    ap.add_argument("--target_key", type=str, default="mask_mass", choices=["mask_mass", "mask_any"])

    # pseudo maps
    ap.add_argument("--pseudo_root", type=str, default="pseudo/teacher_train")

    # optional inference
    ap.add_argument("--teacher_ckpt", type=str, default=None)
    ap.add_argument("--student_ckpt", type=str, default=None)

    # tiling/inference
    ap.add_argument("--tile_size", type=int, default=512)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--tile_batch", type=int, default=4)
    ap.add_argument("--weight_mode", type=str, default="hann", choices=["hann", "gaussian", "uniform"])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--use_minmax", action="store_true", help="For vis/inference: use per-image minmax after clip (instead of global z-score).")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # choose samples
    if args.npz is not None:
        paths = [Path(args.npz)]
    else:
        if args.split is None:
            raise ValueError("Provide either --npz or --split.")
        all_paths = load_paths_from_split(data_dir, args.split)
        if len(all_paths) == 0:
            raise ValueError(f"No samples in split {args.split}")
        take = min(int(args.n), len(all_paths))
        idx = rng.choice(len(all_paths), size=take, replace=False)
        paths = [all_paths[int(i)] for i in idx]

    # load models if provided
    teacher_model = None
    student_model = None
    teacher_ckpt = None
    student_ckpt = None

    if args.teacher_ckpt:
        teacher_ckpt = load_ckpt(Path(args.teacher_ckpt))
        teacher_model = build_unet_from_ckpt(teacher_ckpt, fallback_in_channels=1).to(args.device).eval()

    if args.student_ckpt:
        student_ckpt = load_ckpt(Path(args.student_ckpt))
        student_model = build_unet_from_ckpt(student_ckpt, fallback_in_channels=1).to(args.device).eval()

    # normalization
    norm_cfg = load_norm_cfg_from_ckpt_or_data(teacher_ckpt or student_ckpt, data_dir, use_minmax=args.use_minmax)

    tile_cfg = TilingConfig(
        tile_size=args.tile_size,
        overlap=args.overlap,
        batch_size=args.tile_batch,
        weight_mode=args.weight_mode,
        device=args.device,
        amp=True,
    )

    pseudo_root = Path(args.pseudo_root) if args.pseudo_root else None

    for p in tqdm(paths, desc="Visualizing"):
        d = np.load(p, allow_pickle=True)
        img = d["image"].astype(np.float32)
        breast = d.get("mask_breast", np.ones_like(img, dtype=np.uint8)).astype(np.uint8)
        gt = d.get(args.target_key, d.get("mask_mass", d.get("mask_any"))).astype(np.uint8)

        # normalize for inference (exactly like eval)
        clip_vals = compute_clip_values(img, breast, norm_cfg.clip_p_lo, norm_cfg.clip_p_hi)
        img_norm = apply_norm(img, breast, norm_cfg, precomputed_clip=clip_vals)

        # display image in [0,1]
        img01 = normalize_for_display(img_norm, breast)

        sample_id = p.stem

        pseudo_prob, pseudo_unc = (None, None)
        if pseudo_root is not None:
            pseudo_prob, pseudo_unc = load_pseudo_maps(pseudo_root, sample_id)

        # teacher prediction
        teacher_prob = None
        if teacher_model is not None:
            # teacher takes image-only
            x_chw = img_norm[None, ...]  # [1,H,W]
            teacher_prob = sliding_window_predict_proba_chw(teacher_model, x_chw, breast, tile_cfg)

        # student prediction
        student_prob = None
        if student_model is not None:
            # build input according to student args (supports image-only and image+prob+unc)
            s_args = (student_ckpt.get("args", {}) if student_ckpt is not None else {}) or {}
            need_prob = bool(s_args.get("include_prob_input", False))
            need_unc = bool(s_args.get("include_unc_input", False))

            chans = [img_norm.astype(np.float32)]
            if need_prob:
                if pseudo_prob is None:
                    # still allow but will degrade; fill zeros
                    chans.append(np.zeros_like(img_norm, dtype=np.float32))
                else:
                    chans.append(pseudo_prob.astype(np.float32))
            if need_unc:
                if pseudo_unc is None:
                    chans.append(np.zeros_like(img_norm, dtype=np.float32))
                else:
                    chans.append(pseudo_unc.astype(np.float32))

            x_chw = np.stack(chans, axis=0)  # [C,H,W]
            student_prob = sliding_window_predict_proba_chw(student_model, x_chw, breast, tile_cfg)

        title_prefix = f"{sample_id}"
        fig = make_figure(
            img01=img01,
            breast=breast,
            gt=gt,
            pseudo_prob=pseudo_prob,
            pseudo_unc=pseudo_unc,
            teacher_prob=teacher_prob,
            student_prob=student_prob,
            thr=float(args.threshold),
            title_prefix=title_prefix,
        )

        out_path = out_dir / f"{sample_id}_compare.png"
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

        if args.show:
            # In notebooks, you may want to display; saving already done.
            from PIL import Image
            display(Image.open(out_path))  # type: ignore

    print(f"Saved visualizations to: {out_dir}")


if __name__ == "__main__":
    main()