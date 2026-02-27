#!/usr/bin/env python3
"""
Visual QC viewer v2 for INbreast NPZ (built by inbreast_build_v2.py).

Key fixes vs v1:
- Can overlay: mass OR calcification OR any.
- Calcifications can be tiny; supports zooming around lesion bounding box.
- Can also show ROI mask.

Deps:
  pip install numpy pandas matplotlib pillow tqdm
Optional (for contours):
  pip install scikit-image

usage example:
    python qc_inbreast.py \
    --out_dir ./data \
    --split train \
    --mask mass \
    --show_roi \
    --contours \
    --n 12 \
    --cols 4
    --seed 123

    python qc_inbreast.py \
    --out_dir ./data \
    --split unlabeled \
    --mask mass \
    --show_roi \
    --n 12 \
    --cols 4

    python qc_inbreast.py \
    --out_dir ./data \
    --split train \
    --mask mass \
    --zoom_on mass \
    --zoom_pad 80 \
    --n 12 \
    --cols 4
"""

from __future__ import annotations
import argparse
from pathlib import Path
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

_HAS_SKIMAGE = False
try:
    from skimage.measure import find_contours  # type: ignore
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False


def load_npz(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    img = d["image"].astype(np.float32)
    masks = {
        "mass": d["mask_mass"].astype(np.uint8) if "mask_mass" in d.files else None,
        "calc": d["mask_calcification"].astype(np.uint8) if "mask_calcification" in d.files else None,
        "any": d["mask_any"].astype(np.uint8) if "mask_any" in d.files else None,
    }
    breast = d["mask_breast"].astype(np.uint8) if "mask_breast" in d.files else None
    meta = {
        "case_id": str(d["case_id"]) if "case_id" in d.files else "",
        "laterality": str(d["laterality"]) if "laterality" in d.files else "",
        "view": str(d["view"]) if "view" in d.files else "",
    }
    return img, masks, breast, meta


def normalize_for_view(img: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0):
    lo = np.percentile(img, p_lo)
    hi = np.percentile(img, p_hi)
    x = np.clip(img, lo, hi)
    x = (x - lo) / max(1e-6, (hi - lo))
    return x


def bbox_from_mask(mask: np.ndarray, pad: int = 40):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    y0 = max(0, y0 - pad); x0 = max(0, x0 - pad)
    y1 = min(mask.shape[0] - 1, y1 + pad); x1 = min(mask.shape[1] - 1, x1 + pad)
    return y0, y1, x0, x1


def crop(img: np.ndarray, mask: np.ndarray | None, breast: np.ndarray | None, box):
    y0, y1, x0, x1 = box
    img_c = img[y0:y1+1, x0:x1+1]
    mask_c = mask[y0:y1+1, x0:x1+1] if mask is not None else None
    breast_c = breast[y0:y1+1, x0:x1+1] if breast is not None else None
    return img_c, mask_c, breast_c


def downsample(img: np.ndarray, mask: np.ndarray | None, breast: np.ndarray | None, max_side: int):
    h, w = img.shape
    s = max(h, w)
    if s <= max_side:
        return img, mask, breast
    scale = max_side / float(s)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    img_ds = np.array(Image.fromarray(img).resize((new_w, new_h), resample=Image.BILINEAR), dtype=np.float32)
    mask_ds = None
    breast_ds = None
    if mask is not None:
        mask_ds = np.array(Image.fromarray(mask).resize((new_w, new_h), resample=Image.NEAREST), dtype=np.uint8)
    if breast is not None:
        breast_ds = np.array(Image.fromarray(breast).resize((new_w, new_h), resample=Image.NEAREST), dtype=np.uint8)
    return img_ds, mask_ds, breast_ds


def overlay(ax, img: np.ndarray, mask: np.ndarray | None, breast: np.ndarray | None,
            title: str, show_roi: bool, alpha: float, contours: bool, mode: str):
    x = normalize_for_view(img)
    ax.imshow(x, cmap="gray")
    ax.axis("off")
    ax.set_title(title, fontsize=9)

    if show_roi and breast is not None:
        roi = (breast > 0).astype(np.float32)
        ax.imshow(np.dstack([np.zeros_like(roi), roi, roi]), alpha=0.15)

    if mask is None or not mask.any():
        return

    if contours and _HAS_SKIMAGE:
        cs = find_contours(mask.astype(np.float32), 0.5)
        for c in cs:
            ax.plot(c[:, 1], c[:, 0], linewidth=1.0)
    else:
        m = (mask > 0).astype(np.float32)
        if mode == "calc":
            # yellow overlay for calcifications
            ax.imshow(np.dstack([m, m, np.zeros_like(m)]), alpha=alpha)
        else:
            # red overlay for mass/any
            ax.imshow(np.dstack([m, np.zeros_like(m), np.zeros_like(m)]), alpha=alpha)


def get_npz_list(out_dir: Path, split: str):
    if split.lower() == "all":
        df = pd.read_csv(out_dir / "metadata.csv")
        return [Path(p) for p in df["npz_path"].astype(str).tolist()]
    sp = out_dir / "splits" / f"{split}.txt"
    return [Path(x.strip()) for x in sp.read_text().splitlines() if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split", default="train", help="train|val|test|all")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--max_side", type=int, default=1200)
    ap.add_argument("--show_roi", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--contours", action="store_true")
    ap.add_argument("--mask", choices=["mass", "calc", "any"], default="mass")
    ap.add_argument("--zoom_on", choices=["none", "mass", "calc", "any"], default="none",
                    help="Crop around selected mask bounding box before downsampling (useful for calcifications).")
    ap.add_argument("--zoom_pad", type=int, default=60)
    ap.add_argument("--save_dir", default="")
    ap.add_argument("--no_show", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    npz_paths = get_npz_list(out_dir, args.split)
    random.seed(args.seed)
    chosen = random.sample(npz_paths, k=min(args.n, len(npz_paths)))

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    cols = max(1, args.cols)
    rows = int(np.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for i, p in enumerate(tqdm(chosen, desc="QC")):
        img, masks, breast, meta = load_npz(p)
        mask = masks[args.mask]

        if args.zoom_on != "none":
            zmask = masks[args.zoom_on]
            if zmask is not None and zmask.any():
                box = bbox_from_mask(zmask, pad=args.zoom_pad)
                if box is not None:
                    img, mask, breast = crop(img, mask, breast, box)

        img, mask, breast = downsample(img, mask, breast, args.max_side)

        title = f"{p.stem}\ncase={meta['case_id']} {meta['laterality']} {meta['view']}  {args.mask}_px={int(mask.sum()) if mask is not None else 0}"
        overlay(axes[i], img, mask, breast, title, args.show_roi, args.alpha, args.contours, args.mask)

        if save_dir:
            out_png = save_dir / f"{p.stem}_{args.mask}_qc.png"
            f2, ax2 = plt.subplots(1, 1, figsize=(6, 6))
            overlay(ax2, img, mask, breast, title, args.show_roi, args.alpha, args.contours, args.mask)
            f2.tight_layout()
            f2.savefig(out_png, dpi=160)
            plt.close(f2)

    for j in range(len(chosen), len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
