# src/train_student.py
"""
Guided refinement student training using rich pseudo maps.

Key idea:
- Student input = image + selected pseudo guidance maps (e.g., prob, mi, boundary_band, edge)
- Target = pseudo prob (soft) or hard threshold (ablation)
- Loss weight = uncertainty weight * (1 + beta * boundary_focus) * optional edge weight
- Optional seed loss to enforce confident fg/bg cores

Example:
python -m src.train_student \
  --data_dir data \
  --pseudo_root pseudo/teacher_train \
  --input_maps prob,mi,boundary_band,edge_strength \
  --unc_map mi \
  --boundary_map boundary_band \
  --epochs 20 --steps_per_epoch 600 \
  --batch_size 4 --patch_size 512 \
  --device cuda
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models import UNet
from src.losses import BCEDiceLoss, LossConfig
from src.patches import InbreastPatchDataset, PatchConfig, NormConfig
from src.tiling import TilingConfig, sliding_window_predict_proba
from src.metrics import compute_all_metrics, MetricConfig


def seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def load_norm_cfg(data_dir: Path, use_minmax: bool) -> NormConfig:
    norm_path = data_dir / "normalization.json"
    if not norm_path.exists():
        return NormConfig(use_minmax=use_minmax, use_global_zscore=not use_minmax)

    d = json.loads(norm_path.read_text())
    clip_lo, clip_hi = d.get("clip_percentiles", [1.0, 99.0])
    mean = float(d.get("global_mean_after_clip", 0.0))
    std = float(d.get("global_std_after_clip", 1.0))
    return NormConfig(
        clip_p_lo=float(clip_lo),
        clip_p_hi=float(clip_hi),
        use_global_zscore=not use_minmax,
        global_mean=mean,
        global_std=std,
        use_minmax=use_minmax,
    )


@torch.no_grad()
def eval_full_images(
    model: torch.nn.Module,
    split_list: Path,
    norm_cfg: NormConfig,
    tile_cfg: TilingConfig,
    target_key: str,
    threshold: float,
    cap_mm: float | None,
    device: str,
    max_images: int | None = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    model.eval()
    model.to(device)

    paths = [Path(p.strip()) for p in split_list.read_text().splitlines() if p.strip()]
    if max_images is not None:
        paths = paths[: int(max_images)]

    from src.patches import apply_norm, compute_clip_values

    rows: List[Dict] = []
    for p in tqdm(paths, desc=f"Val full-image ({split_list.name})", leave=False):
        d = np.load(p, allow_pickle=True)
        img = d["image"].astype(np.float32)
        breast = d.get("mask_breast", np.ones_like(img, dtype=np.uint8)).astype(np.uint8)
        gt = d.get(target_key, d.get("mask_mass", d.get("mask_any"))).astype(np.uint8)
        rs = float(d.get("row_spacing_mm", -1.0))
        cs = float(d.get("col_spacing_mm", -1.0))

        clip_vals = compute_clip_values(img, breast, norm_cfg.clip_p_lo, norm_cfg.clip_p_hi)
        img_norm = apply_norm(img, breast, norm_cfg, precomputed_clip=clip_vals)

        # student inference is image-only OR multi-channel?
        # Here we assume student is image-only at inference unless you trained with extra inputs.
        # If you trained with extra inputs, keep inference deployable by setting input_maps to none at inference,
        # OR implement a cascade evaluator (teacher->maps->student). For now: image-only student recommended.
        if model is None:
            raise RuntimeError("Model missing")

        # If model expects >1 channels, we cannot infer without guidance maps.
        in_ch = next(model.parameters()).shape[1] if hasattr(model, "head") else None  # not reliable
        # robust check:
        expected_in = model.enc_blocks[0].conv1.in_channels  # type: ignore
        if expected_in != 1:
            raise RuntimeError(
                f"Student model expects {expected_in} input channels. "
                f"For deployable inference, train student with image-only OR add cascade eval."
            )

        prob = sliding_window_predict_proba(
            model=model,
            image=img_norm,
            mask_breast=breast,
            preprocess=None,
            cfg=tile_cfg,
        )

        mcfg = MetricConfig(threshold=threshold, cap_mm=cap_mm)
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
            "pred_pos_pixels": int((prob >= threshold).sum()),
        })

    df = pd.DataFrame(rows)
    summary = {
        "n_images": int(len(df)),
        "mean_dice": float(df["dice"].mean()) if len(df) else float("nan"),
        "mean_iou": float(df["iou"].mean()) if len(df) else float("nan"),
        "mean_hd95_mm": float(df["hd95_mm"].mean()) if len(df) else float("nan"),
        "mean_assd_mm": float(df["assd_mm"].mean()) if len(df) else float("nan"),
        "fail_dist_rate": float(np.mean(np.isinf(df["hd95_mm"].values))) if len(df) else float("nan"),
    }
    return df, summary


def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def build_index_map(pseudo_keys: List[str]) -> Dict[str, int]:
    # x channels: [image] + pseudo_keys in order
    return {k: 1 + i for i, k in enumerate(pseudo_keys)}


def bce_logits_masked(logits: torch.Tensor, target01: torch.Tensor, mask01: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    BCEWithLogits over masked pixels. logits/target/mask are [B,1,H,W].
    """
    if mask01.sum() < 1:
        return logits.new_tensor(0.0)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target01, reduction="none")
    loss = loss * mask01
    return loss.sum() / (mask01.sum() + eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--pseudo_root", type=str, default="pseudo/teacher_train")

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps_per_epoch", type=int, default=600)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--patch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", type=str, default="cuda")

    # model
    ap.add_argument("--base_channels", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--norm", type=str, default="bn")

    # what maps to feed as inputs (must exist in pseudo_root/all/<id>.npz)
    ap.add_argument("--input_maps", type=str, default="prob,mi,boundary_band,edge_strength",
                    help="comma-separated: e.g. prob,mi,boundary_band,edge_strength,dist_to_boundary,boundary_soft,entropy_mean,confidence,ambiguity")
    ap.add_argument("--unc_map", type=str, default="mi", help="unc map used for weighting: mi|unc_var|entropy_mean|entropy_mean_of_samples")
    ap.add_argument("--boundary_map", type=str, default="boundary_band", help="boundary weight: boundary_band|boundary_soft|dist_to_boundary|none")
    ap.add_argument("--edge_weight_gamma", type=float, default=0.0, help="if >0: multiply weights by (1 + gamma*edge_strength)")

    # pseudo target
    ap.add_argument("--pseudo_target", type=str, default="soft", choices=["soft", "hard"])
    ap.add_argument("--hard_thr", type=float, default=0.5)

    # weighting
    ap.add_argument("--unc_alpha", type=float, default=6.0, help="w_unc = exp(-alpha * unc_norm)")
    ap.add_argument("--drop_top_unc", type=float, default=0.0, help="drop top fraction by uncertainty (0.1 = drop top 10%)")
    ap.add_argument("--boundary_beta", type=float, default=1.5, help="w *= (1 + beta*boundary_focus)")
    ap.add_argument("--boundary_maxdist", type=float, default=20.0, help="if boundary_map=dist_to_boundary, focus = 1 - dt/maxdist")
    ap.add_argument("--min_w", type=float, default=0.0)
    ap.add_argument("--max_w", type=float, default=3.0)

    # optional seed loss
    ap.add_argument("--seed_loss_w", type=float, default=0.0, help="if >0 uses fg_seed/bg_seed to enforce cores")
    ap.add_argument("--seed_maps", type=str, default="fg_seed,bg_seed")

    # eval
    ap.add_argument("--eval_target_key", type=str, default="mask_mass", choices=["mask_mass", "mask_any"])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--cap_mm", type=float, default=-1.0)
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--max_val_images", type=int, default=-1)

    # normalization
    ap.add_argument("--use_minmax", action="store_true")

    # tiling
    ap.add_argument("--tile_overlap", type=float, default=0.5)
    ap.add_argument("--tile_batch", type=int, default=4)
    ap.add_argument("--weight_mode", type=str, default="hann", choices=["hann", "gaussian", "uniform"])

    args = ap.parse_args()
    seed_everything(args.seed)

    data_dir = Path(args.data_dir)
    pseudo_root = Path(args.pseudo_root)
    train_list = data_dir / "splits" / "train.txt"
    val_list = data_dir / "splits" / "val.txt"
    if not train_list.exists(): raise FileNotFoundError(f"Missing {train_list}")
    if not val_list.exists(): raise FileNotFoundError(f"Missing {val_list}")
    if not (pseudo_root / "all").exists():
        raise FileNotFoundError(f"Missing {pseudo_root/'all'}. Run gen_pseudo first.")

    out_ckpt = Path("ckpt"); out_ckpt.mkdir(exist_ok=True, parents=True)
    out_logs = Path("logs"); out_logs.mkdir(exist_ok=True, parents=True)

    norm_cfg = load_norm_cfg(data_dir, use_minmax=args.use_minmax)

    input_maps = parse_csv_list(args.input_maps)
    seed_maps = parse_csv_list(args.seed_maps) if args.seed_loss_w > 0 else []

    # maps needed for training: always need 'prob' as target
    needed = ["prob"]
    if args.unc_alpha > 0 or args.drop_top_unc > 0:
        needed.append(args.unc_map)
    if args.boundary_map.lower() != "none":
        needed.append(args.boundary_map)
    if args.edge_weight_gamma > 0:
        needed.append("edge_strength")
    needed += seed_maps
    needed += input_maps

    # de-duplicate while preserving order
    pseudo_keys: List[str] = []
    for k in needed:
        if k not in pseudo_keys:
            pseudo_keys.append(k)

    idx = build_index_map(pseudo_keys)

    patch_cfg = PatchConfig(
        patch_size=args.patch_size,
        p_unc=0.50,
        p_boundary=0.30,
        p_hardneg_student=0.20,
        min_breast_fraction=0.60,
        max_tries=25,
        pad_mode="reflect",
    )

    epoch_len = int(args.steps_per_epoch * args.batch_size)

    train_ds = InbreastPatchDataset(
        split_list=train_list,
        stage="student",
        patch_cfg=patch_cfg,
        norm_cfg=norm_cfg,
        target_key="mask_mass",
        epoch_len=epoch_len,
        seed=args.seed,
        cache_items=8,
        pseudo_root=pseudo_root,
        pseudo_keys=pseudo_keys,  # <-- requires patches.py update
        return_torch=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Student input channels: image + input_maps (not all loaded maps)
    in_ch = 1 + len(input_maps)
    model = UNet(
        in_channels=in_ch,
        out_channels=1,
        base_channels=args.base_channels,
        depth=args.depth,
        norm=args.norm,
        dropout=args.dropout,
    ).to(args.device)

    criterion = BCEDiceLoss(cfg=LossConfig(dice_weight=1.0, bce_weight=1.0, use_focal=False), pos_weight=None).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=args.device.startswith("cuda"))

    tile_cfg = TilingConfig(
        tile_size=args.patch_size,
        overlap=args.tile_overlap,
        batch_size=args.tile_batch,
        weight_mode=args.weight_mode,
        device=args.device,
    )
    cap_mm = None if args.cap_mm < 0 else float(args.cap_mm)
    max_val_images = None if args.max_val_images < 0 else int(args.max_val_images)

    best_dice = -1.0
    history: List[Dict] = []

    eps = 1e-6

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0

        pbar = tqdm(train_loader, desc=f"Student {epoch}/{args.epochs}", leave=True)
        for x, _y_gt, meta in pbar:
            x = x.to(args.device, non_blocking=True)  # [B, 1+len(pseudo_keys), H, W]

            # build student input
            xs = [x[:, 0:1]]
            for k in input_maps:
                if k not in idx:
                    raise RuntimeError(f"Requested input_map '{k}' not loaded. Loaded keys: {pseudo_keys}")
                xs.append(x[:, idx[k]:idx[k] + 1])
            x_in = torch.cat(xs, dim=1)

            # target from pseudo prob
            if "prob" not in idx:
                raise RuntimeError("Pseudo prob not loaded.")
            prob = x[:, idx["prob"]:idx["prob"] + 1].clamp(0.0, 1.0)
            if args.pseudo_target == "soft":
                y = prob
            else:
                y = (prob >= float(args.hard_thr)).float()

            # uncertainty map for weighting
            if (args.unc_alpha > 0) or (args.drop_top_unc > 0):
                if args.unc_map not in idx:
                    raise RuntimeError(f"unc_map '{args.unc_map}' not loaded.")
                unc = x[:, idx[args.unc_map]:idx[args.unc_map] + 1].clamp_min(0.0)

                # normalize unc per-sample using p95
                B = unc.shape[0]
                unc_flat = unc.view(B, -1)
                scale = torch.quantile(unc_flat, q=0.95, dim=1, keepdim=True).clamp_min(eps)  # [B,1]
                unc_norm = (unc_flat / scale).view_as(unc).clamp(0.0, 3.0)

                w_unc = torch.exp(-float(args.unc_alpha) * unc_norm) if args.unc_alpha > 0 else torch.ones_like(unc_norm)
                if args.drop_top_unc and args.drop_top_unc > 0:
                    q = 1.0 - float(args.drop_top_unc)
                    thr = torch.quantile(unc_flat, q=q, dim=1, keepdim=True)  # [B,1]
                    keep = (unc_flat <= thr).view_as(unc).float()
                    w_unc = w_unc * keep
            else:
                w_unc = torch.ones_like(y)

            # boundary focus
            bmap = args.boundary_map.lower()
            if bmap == "none":
                w_b = torch.ones_like(y)
            else:
                if args.boundary_map not in idx:
                    raise RuntimeError(f"boundary_map '{args.boundary_map}' not loaded.")
                b = x[:, idx[args.boundary_map]:idx[args.boundary_map] + 1]
                if bmap == "boundary_band":
                    # band is uint8 0/1 saved; treat as focus
                    focus = (b > 0.5).float()
                elif bmap == "boundary_soft":
                    focus = b.clamp(0.0, 1.0)
                elif bmap == "dist_to_boundary":
                    # focus = 1 - dt/maxdist
                    focus = (1.0 - b / max(eps, float(args.boundary_maxdist))).clamp(0.0, 1.0)
                else:
                    raise ValueError("boundary_map must be boundary_band|boundary_soft|dist_to_boundary|none")
                w_b = (1.0 + float(args.boundary_beta) * focus)

            # optional edge weighting
            if args.edge_weight_gamma > 0:
                if "edge_strength" not in idx:
                    raise RuntimeError("edge_strength not loaded but edge_weight_gamma>0")
                e = x[:, idx["edge_strength"]:idx["edge_strength"] + 1].clamp(0.0, 1.0)
                w_e = (1.0 + float(args.edge_weight_gamma) * e)
            else:
                w_e = torch.ones_like(y)

            w = (w_unc * w_b * w_e).clamp(min=float(args.min_w), max=float(args.max_w))

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.device.startswith("cuda")):
                logits = model(x_in)
                loss_main = criterion(logits, y, weight=w)

                loss = loss_main

                # optional seed loss
                if args.seed_loss_w > 0:
                    if "fg_seed" not in idx or "bg_seed" not in idx:
                        raise RuntimeError("seed_loss_w>0 requires fg_seed and bg_seed maps loaded.")
                    fg = (x[:, idx["fg_seed"]:idx["fg_seed"] + 1] > 0.5).float()
                    bg = (x[:, idx["bg_seed"]:idx["bg_seed"] + 1] > 0.5).float()

                    # enforce: fg -> 1, bg -> 0
                    loss_fg = bce_logits_masked(logits, torch.ones_like(logits), fg, eps=eps)
                    loss_bg = bce_logits_masked(logits, torch.zeros_like(logits), bg, eps=eps)
                    loss_seed = loss_fg + loss_bg
                    loss = loss + float(args.seed_loss_w) * loss_seed

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += float(loss.item())
            if (len(history) + 1) % 50 == 0:
                pbar.set_postfix(loss=float(loss.item()), lr=float(opt.param_groups[0]["lr"]))

        train_loss = running / max(1, len(train_loader))
        sched.step()

        # Validate (stitched full image) - requires deployable student (1-channel input)
        val_summary = {}
        if epoch % args.eval_every == 0:
            expected_in = model.enc_blocks[0].conv1.in_channels  # type: ignore
            if expected_in == 1:
                df_val, val_summary = eval_full_images(
                    model=model,
                    split_list=val_list,
                    norm_cfg=norm_cfg,
                    tile_cfg=tile_cfg,
                    target_key=args.eval_target_key,
                    threshold=args.threshold,
                    cap_mm=cap_mm,
                    device=args.device,
                    max_images=max_val_images,
                )
                df_val.to_csv(Path("logs") / "student_val_metrics.csv", index=False)
                (Path("logs") / "student_val_summary.json").write_text(json.dumps(val_summary, indent=2))

                val_dice = float(val_summary.get("mean_dice", float("nan")))
                if np.isfinite(val_dice) and val_dice > best_dice:
                    best_dice = val_dice
                    torch.save(
                        {
                            "epoch": epoch,
                            "best_dice": best_dice,
                            "model_state": model.state_dict(),
                            "args": vars(args),
                            "norm_cfg": norm_cfg.__dict__,
                            "pseudo_keys": pseudo_keys,
                            "input_maps": input_maps,
                        },
                        Path("ckpt") / "student_best.pt",
                    )
            else:
                # still save best based on training loss only (not ideal) if non-deployable
                pass

        torch.save(
            {
                "epoch": epoch,
                "best_dice": best_dice,
                "model_state": model.state_dict(),
                "args": vars(args),
                "norm_cfg": norm_cfg.__dict__,
                "pseudo_keys": pseudo_keys,
                "input_maps": input_maps,
            },
            Path("ckpt") / "student_last.pt",
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": float(opt.param_groups[0]["lr"]),
            "best_val_dice": best_dice,
        }
        row.update({f"val_{k}": v for k, v in val_summary.items()})
        history.append(row)
        pd.DataFrame(history).to_csv(Path("logs") / "student_train.csv", index=False)

        print(f"[Student Epoch {epoch}] train_loss={train_loss:.4f} "
              f"val_mean_dice={val_summary.get('mean_dice', None)} best={best_dice:.4f}")

    print(f"Done. Best student val Dice: {best_dice:.4f}. Saved to ckpt/student_best.pt")


if __name__ == "__main__":
    main()