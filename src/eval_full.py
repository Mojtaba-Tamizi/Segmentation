# src/eval_full.py
"""
Full-image stitched evaluation for a saved checkpoint.

Outputs:
- logs/metrics_<split>.csv (per-image)
- logs/summary_<split>.json

Example:
python -m src.eval_full --data_dir data --ckpt ckpt/teacher_best.pt --split test --device cuda
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.models import UNet
from src.patches import NormConfig, apply_norm, compute_clip_values
from src.tiling import TilingConfig, sliding_window_predict_proba
from src.metrics import compute_all_metrics, MetricConfig


def build_model_from_ckpt(ckpt: dict) -> UNet:
    args = ckpt.get("args", {})
    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=int(args.get("base_channels", 32)),
        depth=int(args.get("depth", 4)),
        norm=str(args.get("norm", "bn")),
        dropout=float(args.get("dropout", 0.0)),
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--target_key", type=str, default="mask_mass", choices=["mask_mass", "mask_any"])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--cap_mm", type=float, default=-1.0)
    ap.add_argument("--tile_size", type=int, default=512)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--tile_batch", type=int, default=4)
    ap.add_argument("--weight_mode", type=str, default="hann", choices=["hann", "gaussian", "uniform"])
    ap.add_argument("--max_images", type=int, default=-1)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    split_list = data_dir / "splits" / f"{args.split}.txt"
    assert split_list.exists(), f"Missing {split_list}"

    out_logs = Path("logs")
    out_logs.mkdir(exist_ok=True, parents=True)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = build_model_from_ckpt(ckpt).to(args.device)
    model.eval()

    # norm_cfg stored in ckpt, else fallback
    norm_dict = ckpt.get("norm_cfg", None)
    if norm_dict is not None:
        norm_cfg = NormConfig(**norm_dict)
    else:
        # fallback: use train-only normalization.json if present
        norm_path = data_dir / "normalization.json"
        if norm_path.exists():
            d = json.loads(norm_path.read_text())
            clip_lo, clip_hi = d.get("clip_percentiles", [1.0, 99.0])
            norm_cfg = NormConfig(
                clip_p_lo=float(clip_lo),
                clip_p_hi=float(clip_hi),
                use_global_zscore=True,
                global_mean=float(d.get("global_mean_after_clip", 0.0)),
                global_std=float(d.get("global_std_after_clip", 1.0)),
                use_minmax=False,
            )
        else:
            norm_cfg = NormConfig()

    tile_cfg = TilingConfig(
        tile_size=args.tile_size,
        overlap=args.overlap,
        batch_size=args.tile_batch,
        weight_mode=args.weight_mode,
        device=args.device,
    )

    cap_mm = None if args.cap_mm < 0 else float(args.cap_mm)
    mcfg = MetricConfig(threshold=args.threshold, cap_mm=cap_mm)

    paths = [Path(p.strip()) for p in split_list.read_text().splitlines() if p.strip()]
    if args.max_images > 0:
        paths = paths[: int(args.max_images)]

    rows: List[Dict] = []
    for p in tqdm(paths, desc=f"Eval full ({args.split})"):
        d = np.load(p, allow_pickle=True)
        img = d["image"].astype(np.float32)
        breast = d.get("mask_breast", np.ones_like(img, dtype=np.uint8)).astype(np.uint8)
        gt = d.get(args.target_key, d.get("mask_mass", d.get("mask_any"))).astype(np.uint8)

        rs = float(d.get("row_spacing_mm", -1.0))
        cs = float(d.get("col_spacing_mm", -1.0))

        clip_vals = compute_clip_values(img, breast, norm_cfg.clip_p_lo, norm_cfg.clip_p_hi)
        img_norm = apply_norm(img, breast, norm_cfg, precomputed_clip=clip_vals)

        prob = sliding_window_predict_proba(
            model=model,
            image=img_norm,
            mask_breast=breast,
            preprocess=None,
            cfg=tile_cfg,
        )

        met = compute_all_metrics(prob, gt, rs, cs, cfg=mcfg, input_is_prob=True)

        rows.append({
            "npz_path": str(p),
            "sample_id": p.stem,
            "dice": met["dice"],
            "iou": met["iou"],
            "hd95_mm": met["hd95_mm"],
            "assd_mm": met["assd_mm"],
            "used_fallback_spacing": met["used_fallback_spacing"],
            "gt_pos_pixels": int((gt > 0).sum()),
            "pred_pos_pixels": int((prob >= args.threshold).sum()),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_logs / f"metrics_{args.split}.csv", index=False)

    summary = {
        "split": args.split,
        "n_images": int(len(df)),
        "mean_dice": float(df["dice"].mean()) if len(df) else float("nan"),
        "mean_iou": float(df["iou"].mean()) if len(df) else float("nan"),
        "mean_hd95_mm": float(df["hd95_mm"].mean()) if len(df) else float("nan"),
        "mean_assd_mm": float(df["assd_mm"].mean()) if len(df) else float("nan"),
        "fail_dist_rate": float(np.mean(np.isinf(df["hd95_mm"].values))) if len(df) else float("nan"),
    }
    (out_logs / f"summary_{args.split}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()