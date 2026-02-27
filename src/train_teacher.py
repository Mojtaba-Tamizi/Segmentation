# src/train_teacher.py
"""
Stage-1 Teacher training (patch-based) + full-image stitched validation.

Key features:
- Train on randomly sampled patches (virtual-length dataset + epoch-aware sampler)
- Validate on full images using sliding-window stitching
- Stable AMP: forward in (bf16/fp16), loss computed in fp32 + optional grad clipping

Outputs:
- ckpt/teacher_best.pt
- ckpt/teacher_last.pt
- logs/teacher_train.csv
- logs/teacher_val_metrics.csv (per-image)
- logs/teacher_val_summary.json

Example:
python -m src.train_teacher \
  --data_dir data \
  --epochs 30 \
  --steps_per_epoch 500 \
  --batch_size 6 \
  --patch_size 512 \
  --lr 2e-4 \
  --device cuda \
  --amp --amp_dtype bf16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from src.models import UNet
from src.losses import BCEDiceLoss, LossConfig
from src.patches import InbreastPatchDataset, PatchConfig, NormConfig, apply_norm, compute_clip_values
from src.tiling import TilingConfig, sliding_window_predict_proba
from src.metrics import compute_all_metrics, MetricConfig


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Keep deterministic=False for speed; make reproducibility "good enough".
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def load_norm_cfg(data_dir: Path, use_minmax: bool) -> NormConfig:
    norm_path = data_dir / "normalization.json"
    if not norm_path.exists():
        # safe defaults if not present
        return NormConfig(
            clip_p_lo=1.0,
            clip_p_hi=99.0,
            use_global_zscore=not use_minmax,
            global_mean=0.0,
            global_std=1.0,
            use_minmax=use_minmax,
        )

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


class EpochRandomSampler(Sampler[int]):
    """Sample indices with replacement, deterministically changing per epoch."""

    def __init__(self, data_source, num_samples: int, seed: int = 0):
        self.data_source = data_source
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        n = len(self.data_source)
        idx = torch.randint(high=n, size=(self.num_samples,), generator=g)
        return iter(idx.tolist())

    def __len__(self) -> int:
        return self.num_samples


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
    """Full-image stitched evaluation. Returns (per-image df, summary dict)."""

    model.eval()
    model.to(device)

    paths = [Path(p.strip()) for p in split_list.read_text().splitlines() if p.strip()]
    if max_images is not None:
        paths = paths[: int(max_images)]

    rows: List[Dict] = []
    for p in tqdm(paths, desc=f"Val full-image ({split_list.name})", leave=False):
        d = np.load(p, allow_pickle=True)
        img = d["image"].astype(np.float32)
        breast = d.get("mask_breast", np.ones_like(img, dtype=np.uint8)).astype(np.uint8)
        gt = d.get(target_key, d.get("mask_mass", d.get("mask_any"))).astype(np.uint8)

        rs = float(d.get("row_spacing_mm", -1.0))
        cs = float(d.get("col_spacing_mm", -1.0))

        # normalize full image ONCE using breast ROI clip
        clip_vals = compute_clip_values(img, breast, norm_cfg.clip_p_lo, norm_cfg.clip_p_hi)
        img_norm = apply_norm(img, breast, norm_cfg, precomputed_clip=clip_vals)

        prob = sliding_window_predict_proba(
            model=model,
            image=img_norm,
            mask_breast=breast,
            preprocess=None,
            cfg=tile_cfg,
        )

        mcfg = MetricConfig(threshold=threshold, cap_mm=cap_mm)
        met = compute_all_metrics(prob, gt, rs, cs, cfg=mcfg, input_is_prob=True)

        rows.append(
            {
                "npz_path": str(p),
                "sample_id": p.stem,
                "dice": met["dice"],
                "iou": met["iou"],
                "hd95_mm": met["hd95_mm"],
                "assd_mm": met["assd_mm"],
                "used_fallback_spacing": met["used_fallback_spacing"],
                "gt_pos_pixels": int((gt > 0).sum()),
                "pred_pos_pixels": int((prob >= threshold).sum()),
            }
        )

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps_per_epoch", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=6)
    ap.add_argument("--patch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")

    # Model
    ap.add_argument("--base_channels", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--norm", type=str, default="bn")

    # Targets / normalization
    ap.add_argument("--target_key", type=str, default="mask_mass", choices=["mask_mass", "mask_any"])
    ap.add_argument(
        "--use_minmax",
        action="store_true",
        help="Use per-image minmax after clipping instead of global z-score",
    )

    # Loss
    ap.add_argument("--dice_w", type=float, default=1.0)
    ap.add_argument("--bce_w", type=float, default=1.0)
    ap.add_argument("--use_focal", action="store_true")
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--focal_alpha", type=float, default=-1.0, help="set -1 to disable alpha")
    ap.add_argument("--pos_weight", type=float, default=-1.0, help="set -1 to disable BCE pos_weight")

    # AMP / stability
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--grad_clip", type=float, default=1.0, help="clip grad norm; <=0 disables")

    # Full-image validation / tiling
    ap.add_argument("--tile_overlap", type=float, default=0.5)
    ap.add_argument("--tile_batch", type=int, default=4)
    ap.add_argument("--weight_mode", type=str, default="hann", choices=["hann", "gaussian", "uniform"])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--cap_mm", type=float, default=-1.0, help="cap inf distance metrics to this value; -1 disables")
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--max_val_images", type=int, default=-1, help="debug: limit val images; -1 disables")

    args = ap.parse_args()
    seed_everything(args.seed)

    data_dir = Path(args.data_dir)
    train_list = data_dir / "splits" / "train.txt"
    val_list = data_dir / "splits" / "val.txt"
    assert train_list.exists(), f"Missing {train_list}"
    assert val_list.exists(), f"Missing {val_list}"

    out_ckpt = Path("ckpt")
    out_logs = Path("logs")
    out_ckpt.mkdir(exist_ok=True, parents=True)
    out_logs.mkdir(exist_ok=True, parents=True)

    norm_cfg = load_norm_cfg(data_dir, use_minmax=args.use_minmax)

    patch_cfg = PatchConfig(
        patch_size=args.patch_size,
        p_pos=0.50,
        p_hardneg=0.25,
        min_breast_fraction=0.60,
        max_tries=20,
        pad_mode="reflect",
    )

    # We want exactly steps_per_epoch batches.
    samples_per_epoch = int(args.steps_per_epoch * args.batch_size)

    # Make dataset virtual-length so different indices yield different random patches across epochs.
    virtual_len = int(1_000_000_000)

    train_ds = InbreastPatchDataset(
        split_list=train_list,
        stage="teacher",
        patch_cfg=patch_cfg,
        norm_cfg=norm_cfg,
        target_key=args.target_key,
        epoch_len=virtual_len,
        seed=args.seed,
        cache_items=8,
        return_torch=True,
    )

    sampler = EpochRandomSampler(train_ds, num_samples=samples_per_epoch, seed=args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )

    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=args.base_channels,
        depth=args.depth,
        norm=args.norm,
        dropout=args.dropout,
    ).to(args.device)

    loss_cfg = LossConfig(
        dice_weight=args.dice_w,
        bce_weight=args.bce_w,
        use_focal=args.use_focal,
        focal_gamma=args.focal_gamma,
        focal_alpha=None if args.focal_alpha < 0 else float(args.focal_alpha),
    )
    pos_w = None if args.pos_weight < 0 else float(args.pos_weight)
    criterion = BCEDiceLoss(cfg=loss_cfg, pos_weight=pos_w).to(args.device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))

    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    use_amp = bool(args.amp and use_cuda)

    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    if use_amp and amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        print("[AMP] bf16 not supported on this GPU, falling back to fp16")
        amp_dtype = torch.float16

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    tile_cfg = TilingConfig(
        tile_size=args.patch_size,
        overlap=args.tile_overlap,
        batch_size=args.tile_batch,
        weight_mode=args.weight_mode,
        amp=use_amp,
        amp_dtype=("bf16" if amp_dtype == torch.bfloat16 else "fp16"),
        device=args.device,
    )

    cap_mm = None if args.cap_mm < 0 else float(args.cap_mm)
    max_val_images = None if args.max_val_images < 0 else int(args.max_val_images)

    best_dice = -1.0
    history_rows: List[Dict] = []

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch - 1)

        model.train()
        pbar = tqdm(train_loader, desc=f"Train epoch {epoch}/{args.epochs}", leave=True)
        running = 0.0

        # (optional) track sampling mix
        tag_counts = {"pos": 0, "hardneg": 0, "rand": 0, "uniform": 0, "na": 0, "boundary": 0, "unc": 0}

        for step_in_epoch, batch in enumerate(pbar, start=1):
            x, y, meta = batch
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)

            try:
                tags = meta["tag"]
                for t in tags:
                    tag_counts[str(t)] = tag_counts.get(str(t), 0) + 1
            except Exception:
                pass

            opt.zero_grad(set_to_none=True)

            # forward (possibly AMP)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(x)
            else:
                logits = model(x)

            # loss in fp32 for stability
            loss = criterion(logits.float(), y.float())

            if not torch.isfinite(loss):
                print(
                    "Non-finite loss!",
                    "loss=", float(loss.detach().cpu()),
                    "logits_range=", float(logits.min().detach().cpu()), float(logits.max().detach().cpu()),
                    "x_range=", float(x.min().detach().cpu()), float(x.max().detach().cpu()),
                )
                raise RuntimeError("Non-finite loss")

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()

            running += float(loss.detach().item())
            global_step += 1

            if step_in_epoch % 20 == 0:
                lr = float(opt.param_groups[0]["lr"])
                pbar.set_postfix(loss=running / max(1, step_in_epoch), lr=lr)

        train_loss = running / max(1, len(train_loader))
        lr = float(opt.param_groups[0]["lr"])
        sched.step()

        # full-image validation
        val_summary: Dict[str, float] = {}
        if (epoch % args.eval_every) == 0:
            model.eval()
            df_val, val_summary = eval_full_images(
                model=model,
                split_list=val_list,
                norm_cfg=norm_cfg,
                tile_cfg=tile_cfg,
                target_key=args.target_key,
                threshold=args.threshold,
                cap_mm=cap_mm,
                device=args.device,
                max_images=max_val_images,
            )

            df_val.to_csv(out_logs / "teacher_val_metrics.csv", index=False)
            (out_logs / "teacher_val_summary.json").write_text(json.dumps(val_summary, indent=2))

            val_dice = float(val_summary.get("mean_dice", float("nan")))
            if np.isfinite(val_dice) and val_dice > best_dice:
                best_dice = val_dice
                ckpt = {
                    "epoch": epoch,
                    "best_dice": best_dice,
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "norm_cfg": norm_cfg.__dict__,
                }
                torch.save(ckpt, out_ckpt / "teacher_best.pt")

        # always save last
        torch.save(
            {
                "epoch": epoch,
                "best_dice": best_dice,
                "model_state": model.state_dict(),
                "args": vars(args),
                "norm_cfg": norm_cfg.__dict__,
            },
            out_ckpt / "teacher_last.pt",
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": lr,
            "best_val_dice": best_dice,
        }
        row.update({f"val_{k}": v for k, v in val_summary.items()})
        history_rows.append(row)
        pd.DataFrame(history_rows).to_csv(out_logs / "teacher_train.csv", index=False)

        print(
            f"[Epoch {epoch}] train_loss={train_loss:.4f} "
            f"val_mean_dice={val_summary.get('mean_dice', None)} best={best_dice:.4f}"
        )

    print(f"Done. Best val Dice: {best_dice:.4f}. Saved to ckpt/teacher_best.pt")


if __name__ == "__main__":
    main()

