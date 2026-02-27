# src/tiling.py
"""Sliding-window inference + weighted stitching for large mammograms.

Features:
- ROI-aware tiling using mask_breast bounding box (optional)
- overlap + blending window to avoid seams
- batch tiles for speed
- optional MC Dropout inference (mean/var uncertainty)

Deps: numpy, torch
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Tuple, Union

import numpy as np
import torch


@dataclass
class TilingConfig:
    tile_size: int = 512
    overlap: float = 0.5  # 0..0.9 typical
    batch_size: int = 4
    weight_mode: str = "hann"  # "hann" or "gaussian" or "uniform"
    roi_pad: int = 16
    pad_mode: str = "reflect"
    amp: bool = True  # autocast on CUDA
    amp_dtype: str = "fp16"  # "fp16" or "bf16"
    device: Union[str, torch.device] = "cuda"


def breast_bbox(mask_breast: Optional[np.ndarray], pad: int, shape_hw: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """Return (y0,y1,x0,x1) inclusive-exclusive bbox within image bounds.

    If mask is None or empty -> full image.
    """
    H, W = shape_hw
    if mask_breast is None:
        return 0, H, 0, W
    ys, xs = np.where(mask_breast > 0)
    if ys.size == 0:
        return 0, H, 0, W
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(H, int(ys.max()) + 1 + pad)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(W, int(xs.max()) + 1 + pad)
    return y0, y1, x0, x1


def weight_window(h: int, w: int, mode: str = "hann") -> np.ndarray:
    """2D blending weights, peak at center, low at edges."""
    mode = mode.lower()
    if mode == "uniform":
        return np.ones((h, w), dtype=np.float32)

    if mode == "hann":
        wy = np.hanning(h).astype(np.float32)
        wx = np.hanning(w).astype(np.float32)
        win = np.outer(wy, wx).astype(np.float32)
        win = np.maximum(win, 1e-6)
        return win

    if mode == "gaussian":
        yy = np.arange(h, dtype=np.float32) - (h - 1) / 2.0
        xx = np.arange(w, dtype=np.float32) - (w - 1) / 2.0
        Y, X = np.meshgrid(yy, xx, indexing="ij")
        sigma_y = max(1.0, h / 8.0)
        sigma_x = max(1.0, w / 8.0)
        win = np.exp(-0.5 * ((Y / sigma_y) ** 2 + (X / sigma_x) ** 2)).astype(np.float32)
        win = np.maximum(win, 1e-6)
        return win

    raise ValueError(f"Unknown weight_mode: {mode}")


def _extract_tile(
    img: np.ndarray,
    y0: int,
    x0: int,
    th: int,
    tw: int,
    pad_mode: str = "reflect",
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Extract tile with padding if out of bounds.

    Returns (tile, (py0, py1, px0, px1)) where p* is the slice in the padded tile that maps to real image.
    """
    H, W = img.shape
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

    tile = img[yy0:yy1, xx0:xx1].astype(np.float32)
    if pad_top or pad_left or pad_bot or pad_right:
        tile = np.pad(tile, ((pad_top, pad_bot), (pad_left, pad_right)), mode=pad_mode)

    py0 = pad_top
    py1 = py0 + (yy1 - yy0)
    px0 = pad_left
    px1 = px0 + (xx1 - xx0)
    return tile, (py0, py1, px0, px1)


def iter_tiles(y0: int, y1: int, x0: int, x1: int, tile: int, overlap: float) -> Iterator[Tuple[int, int]]:
    """Yield top-left corners (yy,xx) covering bbox with given overlap."""
    stride = max(1, int(round(tile * (1.0 - overlap))))
    ys = list(range(y0, max(y0, y1 - tile) + 1, stride))
    xs = list(range(x0, max(x0, x1 - tile) + 1, stride))

    if len(ys) == 0:
        ys = [y0]
    if len(xs) == 0:
        xs = [x0]

    if ys[-1] != y1 - tile:
        ys.append(max(y0, y1 - tile))
    if xs[-1] != x1 - tile:
        xs.append(max(x0, x1 - tile))

    for yy in ys:
        for xx in xs:
            yield yy, xx


def enable_mc_dropout(model: torch.nn.Module) -> None:
    """Enable dropout layers during eval for MC Dropout. Keeps BatchNorm in eval."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            m.train()


def _resolve_amp_dtype(cfg: TilingConfig) -> torch.dtype:
    if cfg.amp_dtype.lower() == "bf16":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        # fallback if not supported
        return torch.float16
    return torch.float16


@torch.no_grad()
def sliding_window_predict_proba(
    model: torch.nn.Module,
    image: np.ndarray,  # [H,W]
    mask_breast: Optional[np.ndarray] = None,
    preprocess: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    cfg: TilingConfig = TilingConfig(),
) -> np.ndarray:
    """Return prob map float32 [H,W] in [0,1] for binary segmentation.

    Assumes model outputs logits [B,1,H,W] or [B,H,W].
    """
    assert image.ndim == 2, "image must be [H,W]"
    img = image.astype(np.float32)
    if preprocess is not None:
        img = preprocess(img)

    H, W = img.shape
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
        tile_np, (py0, py1, px0, px1) = _extract_tile(img, yy, xx, th, tw, pad_mode=cfg.pad_mode)
        batch_tiles.append(tile_np)
        batch_meta.append((yy, xx, py0, py1, px0, px1))
        if len(batch_tiles) == cfg.batch_size:
            _run_and_blend(model, batch_tiles, batch_meta, win, acc, wsum, device, cfg)
            batch_tiles, batch_meta = [], []

    if batch_tiles:
        _run_and_blend(model, batch_tiles, batch_meta, win, acc, wsum, device, cfg)

    prob = acc / (wsum + 1e-8)
    prob = np.clip(prob, 0.0, 1.0).astype(np.float32)
    return prob


def _run_and_blend(
    model: torch.nn.Module,
    batch_tiles: list,
    batch_meta: list,
    win: np.ndarray,
    acc: np.ndarray,
    wsum: np.ndarray,
    device: torch.device,
    cfg: TilingConfig,
) -> None:
    """Run model on a batch of tiles and blend into (acc,wsum) using win."""
    x = np.stack(batch_tiles, axis=0)  # [B,H,W]
    x_t = torch.from_numpy(x).to(device=device, dtype=torch.float32)[:, None, :, :]  # [B,1,H,W]

    if device.type == "cuda" and cfg.amp:
        dtype = _resolve_amp_dtype(cfg)
        with torch.autocast(device_type="cuda", dtype=dtype):
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

        p_prob_crop = tile_prob[py0 : py0 + (y1_img - y0_img), px0 : px0 + (x1_img - x0_img)]
        p_win_crop = tile_win[py0 : py0 + (y1_img - y0_img), px0 : px0 + (x1_img - x0_img)]

        acc[y0_img:y1_img, x0_img:x1_img] += p_prob_crop * p_win_crop
        wsum[y0_img:y1_img, x0_img:x1_img] += p_win_crop


@torch.no_grad()
def mc_dropout_predict_mean_var(
    model: torch.nn.Module,
    image: np.ndarray,
    mask_breast: Optional[np.ndarray] = None,
    preprocess: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    cfg: TilingConfig = TilingConfig(),
    K: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """MC Dropout on full image: returns (mean_prob, var_prob) float32 [H,W]."""
    device = torch.device(cfg.device) if isinstance(cfg.device, str) else cfg.device
    model = model.to(device)
    enable_mc_dropout(model)

    probs = []
    for _ in range(int(K)):
        p = sliding_window_predict_proba(model, image, mask_breast=mask_breast, preprocess=preprocess, cfg=cfg)
        probs.append(p)

    stack = np.stack(probs, axis=0).astype(np.float32)  # [K,H,W]
    mean = stack.mean(axis=0)
    var = stack.var(axis=0)
    return mean.astype(np.float32), var.astype(np.float32)

# # src/tiling.py
# """
# Sliding-window inference + weighted stitching for large mammograms.

# Features:
# - ROI-aware tiling using mask_breast bounding box (optional)
# - overlap + blending window to avoid seams
# - batch tiles for speed
# - optional MC Dropout inference (mean/var uncertainty)

# Deps: numpy, torch
# """

# from __future__ import annotations
# from dataclasses import dataclass
# from typing import Callable, Iterator, Optional, Tuple, Union

# import numpy as np
# import torch


# @dataclass
# class TilingConfig:
#     tile_size: int = 512
#     overlap: float = 0.5          # 0..0.9 typical
#     batch_size: int = 4
#     weight_mode: str = "hann"     # "hann" or "gaussian" or "uniform"
#     roi_pad: int = 16             # padding around breast bbox
#     pad_mode: str = "reflect"     # reflect is good for mammograms
#     amp: bool = True              # autocast on CUDA
#     device: Union[str, torch.device] = "cuda"


# def breast_bbox(mask_breast: Optional[np.ndarray], pad: int, shape_hw: Tuple[int, int]) -> Tuple[int, int, int, int]:
#     """
#     Returns (y0,y1,x0,x1) inclusive-exclusive bbox within image bounds.
#     If mask is None or empty -> full image.
#     """
#     H, W = shape_hw
#     if mask_breast is None:
#         return 0, H, 0, W
#     ys, xs = np.where(mask_breast > 0)
#     if ys.size == 0:
#         return 0, H, 0, W
#     y0 = max(0, int(ys.min()) - pad)
#     y1 = min(H, int(ys.max()) + 1 + pad)
#     x0 = max(0, int(xs.min()) - pad)
#     x1 = min(W, int(xs.max()) + 1 + pad)
#     return y0, y1, x0, x1


# def weight_window(h: int, w: int, mode: str = "hann") -> np.ndarray:
#     """2D blending weights, peak at center, low at edges."""
#     mode = mode.lower()
#     if mode == "uniform":
#         return np.ones((h, w), dtype=np.float32)

#     if mode == "hann":
#         wy = np.hanning(h).astype(np.float32)
#         wx = np.hanning(w).astype(np.float32)
#         win = np.outer(wy, wx).astype(np.float32)
#         win = np.maximum(win, 1e-6)
#         return win

#     if mode == "gaussian":
#         yy = np.arange(h, dtype=np.float32) - (h - 1) / 2.0
#         xx = np.arange(w, dtype=np.float32) - (w - 1) / 2.0
#         Y, X = np.meshgrid(yy, xx, indexing="ij")
#         sigma_y = max(1.0, h / 8.0)
#         sigma_x = max(1.0, w / 8.0)
#         win = np.exp(-0.5 * ((Y / sigma_y) ** 2 + (X / sigma_x) ** 2)).astype(np.float32)
#         win = np.maximum(win, 1e-6)
#         return win

#     raise ValueError(f"Unknown weight_mode: {mode}")


# def _extract_tile(img: np.ndarray, y0: int, x0: int, th: int, tw: int,
#                   pad_mode: str = "reflect") -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
#     """
#     Extract tile with padding if out of bounds.
#     Returns (tile, (py0, py1, px0, px1)) where p* is the slice in the padded tile that maps to real image.
#     """
#     H, W = img.shape
#     y1 = y0 + th
#     x1 = x0 + tw

#     pad_top = max(0, -y0)
#     pad_left = max(0, -x0)
#     pad_bot = max(0, y1 - H)
#     pad_right = max(0, x1 - W)

#     yy0 = max(0, y0)
#     xx0 = max(0, x0)
#     yy1 = min(H, y1)
#     xx1 = min(W, x1)

#     tile = img[yy0:yy1, xx0:xx1].astype(np.float32)
#     if pad_top or pad_left or pad_bot or pad_right:
#         tile = np.pad(tile, ((pad_top, pad_bot), (pad_left, pad_right)), mode=pad_mode)

#     # slice in tile corresponding to real image pixels
#     py0 = pad_top
#     py1 = py0 + (yy1 - yy0)
#     px0 = pad_left
#     px1 = px0 + (xx1 - xx0)
#     return tile, (py0, py1, px0, px1)


# def iter_tiles(y0: int, y1: int, x0: int, x1: int, tile: int, overlap: float) -> Iterator[Tuple[int, int]]:
#     """
#     Yield top-left corners (yy,xx) covering bbox with given overlap.
#     """
#     stride = max(1, int(round(tile * (1.0 - overlap))))
#     ys = list(range(y0, max(y0, y1 - tile) + 1, stride))
#     xs = list(range(x0, max(x0, x1 - tile) + 1, stride))
#     # ensure last tile hits the end
#     if len(ys) == 0:
#         ys = [y0]
#     if len(xs) == 0:
#         xs = [x0]
#     if ys[-1] != y1 - tile:
#         ys.append(max(y0, y1 - tile))
#     if xs[-1] != x1 - tile:
#         xs.append(max(x0, x1 - tile))

#     for yy in ys:
#         for xx in xs:
#             yield yy, xx


# def enable_mc_dropout(model: torch.nn.Module) -> None:
#     """
#     Enable dropout layers during eval for MC Dropout.
#     Keeps BatchNorm in eval.
#     """
#     model.eval()
#     for m in model.modules():
#         if isinstance(m, torch.nn.Dropout) or isinstance(m, torch.nn.Dropout2d) or isinstance(m, torch.nn.Dropout3d):
#             m.train()


# @torch.no_grad()
# def sliding_window_predict_proba(
#     model: torch.nn.Module,
#     image: np.ndarray,                    # [H,W] float32/float64
#     mask_breast: Optional[np.ndarray] = None,
#     preprocess: Optional[Callable[[np.ndarray], np.ndarray]] = None,
#     cfg: TilingConfig = TilingConfig(),
# ) -> np.ndarray:
#     """
#     Returns prob map float32 [H,W] in [0,1] for binary segmentation.
#     Assumes model outputs logits [B,1,H,W] or [B,H,W].
#     """
#     assert image.ndim == 2, "image must be [H,W]"
#     img = image.astype(np.float32)
#     if preprocess is not None:
#         img = preprocess(img)

#     H, W = img.shape
#     th = tw = int(cfg.tile_size)

#     by0, by1, bx0, bx1 = breast_bbox(mask_breast, cfg.roi_pad, (H, W))

#     win = weight_window(th, tw, cfg.weight_mode)
#     acc = np.zeros((H, W), dtype=np.float32)
#     wsum = np.zeros((H, W), dtype=np.float32)

#     device = torch.device(cfg.device) if isinstance(cfg.device, str) else cfg.device
#     model = model.to(device)
#     model.eval()

#     tiles_xy = list(iter_tiles(by0, by1, bx0, bx1, th, cfg.overlap))

#     batch_tiles = []
#     batch_meta = []  # (yy,xx, py0,py1,px0,px1)
#     for (yy, xx) in tiles_xy:
#         tile_np, (py0, py1, px0, px1) = _extract_tile(img, yy, xx, th, tw, pad_mode=cfg.pad_mode)
#         batch_tiles.append(tile_np)
#         batch_meta.append((yy, xx, py0, py1, px0, px1))
#         if len(batch_tiles) == cfg.batch_size:
#             _run_and_blend(model, batch_tiles, batch_meta, win, acc, wsum, device, cfg.amp)
#             batch_tiles, batch_meta = [], []

#     if batch_tiles:
#         _run_and_blend(model, batch_tiles, batch_meta, win, acc, wsum, device, cfg.amp)

#     prob = acc / (wsum + 1e-8)
#     prob = np.clip(prob, 0.0, 1.0).astype(np.float32)
#     return prob


# def _run_and_blend(
#     model: torch.nn.Module,
#     batch_tiles: list,
#     batch_meta: list,
#     win: np.ndarray,
#     acc: np.ndarray,
#     wsum: np.ndarray,
#     device: torch.device,
#     amp: bool,
# ) -> None:
#     """
#     Run model on a batch of tiles and blend into (acc,wsum) using win.
#     """
#     x = np.stack(batch_tiles, axis=0)  # [B,H,W]
#     x_t = torch.from_numpy(x).to(device=device, dtype=torch.float32)[:, None, :, :]  # [B,1,H,W]

#     if device.type == "cuda" and amp:
#         with torch.autocast(device_type="cuda", dtype=torch.float16):
#             out = model(x_t)
#     else:
#         out = model(x_t)

#     # logits -> prob
#     if out.ndim == 4:
#         logits = out[:, 0]
#     elif out.ndim == 3:
#         logits = out
#     else:
#         raise ValueError(f"Unexpected model output shape: {tuple(out.shape)}")

#     prob = torch.sigmoid(logits).detach().float().cpu().numpy()  # [B,H,W]

#     for i, (yy, xx, py0, py1, px0, px1) in enumerate(batch_meta):
#         # tile region that corresponds to real image pixels
#         tile_prob = prob[i]
#         tile_win = win

#         real_h = py1 - py0
#         real_w = px1 - px0
#         # coordinates in image
#         y0_img = max(0, yy)
#         x0_img = max(0, xx)
#         y1_img = min(acc.shape[0], yy + real_h)
#         x1_img = min(acc.shape[1], xx + real_w)

#         # corresponding crop in padded tile
#         p_prob_crop = tile_prob[py0:py0 + (y1_img - y0_img), px0:px0 + (x1_img - x0_img)]
#         p_win_crop = tile_win[py0:py0 + (y1_img - y0_img), px0:px0 + (x1_img - x0_img)]

#         acc[y0_img:y1_img, x0_img:x1_img] += p_prob_crop * p_win_crop
#         wsum[y0_img:y1_img, x0_img:x1_img] += p_win_crop


# @torch.no_grad()
# def mc_dropout_predict_mean_var(
#     model: torch.nn.Module,
#     image: np.ndarray,
#     mask_breast: Optional[np.ndarray] = None,
#     preprocess: Optional[Callable[[np.ndarray], np.ndarray]] = None,
#     cfg: TilingConfig = TilingConfig(),
#     K: int = 8,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     MC Dropout on full image:
#       returns (mean_prob, var_prob) float32 [H,W]
#     """
#     device = torch.device(cfg.device) if isinstance(cfg.device, str) else cfg.device
#     model = model.to(device)
#     enable_mc_dropout(model)

#     probs = []
#     for _ in range(int(K)):
#         p = sliding_window_predict_proba(model, image, mask_breast=mask_breast, preprocess=preprocess, cfg=cfg)
#         probs.append(p)
#     stack = np.stack(probs, axis=0).astype(np.float32)  # [K,H,W]
#     mean = stack.mean(axis=0)
#     var = stack.var(axis=0)
#     return mean.astype(np.float32), var.astype(np.float32)