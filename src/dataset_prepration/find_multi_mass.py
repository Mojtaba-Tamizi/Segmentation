#!/usr/bin/env python3
"""
Find images with multiple mass components in mask_mass and create a split list
for QC viewer: out_dir/splits/multi_mass.txt

Usage:
  python find_multi_mass.py --out_dir ./data --split train --min_area 200
  python find_multi_mass.py --out_dir ./data --split all --min_area 200 --out_name multi_mass

Notes:
- We count connected components in mask_mass (after filtering tiny components by area).
- If two masses touch in the raster mask, they will count as 1 component.

python find_multi_mass.py --out_dir ./data --split train --min_area 200

python qc_inbreast.py \
  --out_dir ./data \
  --split multi_mass \
  --mask mass \
  --zoom_on mass \
  --zoom_pad 120 \
  --show_roi \
  --contours \
  --n 12 \
  --cols 4

"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd

_HAS_CV2 = False
try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

_HAS_SKIMAGE = False
if not _HAS_CV2:
    try:
        from skimage.measure import label as sk_label  # type: ignore
        _HAS_SKIMAGE = True
    except Exception:
        _HAS_SKIMAGE = False


def get_npz_list(out_dir: Path, split: str) -> List[Path]:
    if split.lower() == "all":
        df = pd.read_csv(out_dir / "metadata.csv")
        return [Path(p) for p in df["npz_path"].astype(str).tolist()]
    sp = out_dir / "splits" / f"{split}.txt"
    return [Path(x.strip()) for x in sp.read_text().splitlines() if x.strip()]


def count_components(mask: np.ndarray, min_area: int, connectivity: int = 8) -> Tuple[int, List[int]]:
    """
    Returns (n_components_kept, areas_kept_sorted_desc)
    """
    m = (mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return 0, []

    if _HAS_CV2:
        # connectedComponents uses 4 or 8 connectivity
        conn = 8 if connectivity == 8 else 4
        n, labels = cv2.connectedComponents(m, connectivity=conn)
        # labels: 0 is background, 1..n-1 components
        areas = []
        for k in range(1, n):
            a = int((labels == k).sum())
            if a >= min_area:
                areas.append(a)
        areas.sort(reverse=True)
        return len(areas), areas

    if _HAS_SKIMAGE:
        conn = 2 if connectivity == 8 else 1  # skimage: 1=4-neigh, 2=8-neigh
        labels = sk_label(m, connectivity=conn)
        areas = []
        for k in range(1, int(labels.max()) + 1):
            a = int((labels == k).sum())
            if a >= min_area:
                areas.append(a)
        areas.sort(reverse=True)
        return len(areas), areas

    raise RuntimeError("Need opencv-python or scikit-image installed to count components.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split", default="train", help="train|val|test|unlabeled|all")
    ap.add_argument("--min_area", type=int, default=200, help="Ignore tiny components below this area (pixels).")
    ap.add_argument("--connectivity", choices=["4", "8"], default="8")
    ap.add_argument("--out_name", default="multi_mass", help="Name of output split file (without .txt)")
    ap.add_argument("--max_items", type=int, default=0, help="0 means no limit.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    npz_paths = get_npz_list(out_dir, args.split)

    rows = []
    kept_paths = []

    conn = 8 if args.connectivity == "8" else 4

    for p in npz_paths:
        d = np.load(p, allow_pickle=True)
        if "mask_mass" not in d.files:
            continue
        mask = d["mask_mass"].astype(np.uint8)
        ncomp, areas = count_components(mask, min_area=args.min_area, connectivity=conn)

        if ncomp >= 2:
            rows.append({
                "npz_path": str(p),
                "n_mass_components": int(ncomp),
                "areas_desc": ",".join(map(str, areas[:10])),
            })
            kept_paths.append(str(p))

    df = pd.DataFrame(rows).sort_values(["n_mass_components"], ascending=False)

    splits_dir = out_dir / "splits"
    splits_dir.mkdir(exist_ok=True)

    out_txt = splits_dir / f"{args.out_name}.txt"
    out_csv = splits_dir / f"{args.out_name}.csv"

    if args.max_items and args.max_items > 0:
        kept_paths = kept_paths[:args.max_items]
        df = df.head(args.max_items)

    out_txt.write_text("\n".join(kept_paths) + ("\n" if kept_paths else ""))
    df.to_csv(out_csv, index=False)

    print(f"Scanned {len(npz_paths)} images from split='{args.split}'")
    print(f"Found {len(kept_paths)} images with >=2 mass components (min_area={args.min_area}, conn={conn})")
    print(f"Wrote:\n  {out_txt}\n  {out_csv}")


if __name__ == "__main__":
    main()