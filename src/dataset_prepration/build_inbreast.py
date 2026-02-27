#!/usr/bin/env python3
"""
INbreast dataset builder v4.2 (DICOM + INbreast XML/plist -> npz), GROUP-level splits, normalization stats.

v4.2 change (per your clarification):
- Split outputs ONLY 4 disjoint sets:
    train / val / test  (labeled-only groups)
    unlabeled           (any group that has missing_xml/parse_failed; mixed groups go here entirely)
- No unlabeled_in_train/val/test lists.
- Strong sanity check: every sample assigned to exactly one of the 4 sets; group sets disjoint.

Build + Norm are same as v3 (with breast ROI non-empty guard + annotation_status tracking).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import pydicom
import plistlib

_HAS_CV2 = False
_HAS_SKIMAGE = False
try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

if not _HAS_CV2:
    try:
        from skimage.draw import polygon as sk_polygon  # type: ignore
        from skimage.measure import label as sk_label  # type: ignore
        _HAS_SKIMAGE = True
    except Exception:
        _HAS_SKIMAGE = False


# ---------------------------
# Helpers: IDs / DICOM fields
# ---------------------------

def extract_case_id_from_filename(fname: str) -> str:
    return Path(fname).name.split("_")[0]


def parse_laterality_view_from_filename(fname: str) -> Tuple[Optional[str], Optional[str]]:
    base = Path(fname).stem
    parts = base.split("_")
    lat, view = None, None
    for p in parts:
        if p in ("L", "R"):
            lat = p
        if p in ("CC", "MLO"):
            view = p
    return lat, view


def dicom_to_array(ds: pydicom.Dataset) -> np.ndarray:
    img = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    img = img * slope + intercept
    if getattr(ds, "PhotometricInterpretation", None) == "MONOCHROME1":
        img = img.max() - img
    return img


def get_pixel_spacing_mm(ds: pydicom.Dataset) -> Tuple[Optional[float], Optional[float]]:
    ps = getattr(ds, "PixelSpacing", None)
    if ps is None:
        return None, None
    try:
        return float(ps[0]), float(ps[1])
    except Exception:
        return None, None


def _safe_str(x) -> str:
    try:
        return str(x).strip()
    except Exception:
        return ""


def get_patient_id(ds: pydicom.Dataset, fallback: str) -> str:
    pid = _safe_str(getattr(ds, "PatientID", ""))
    if pid:
        return pid
    pname = _safe_str(getattr(ds, "PatientName", ""))
    if pname:
        return pname
    study = _safe_str(getattr(ds, "StudyInstanceUID", ""))
    if study:
        return f"STUDY_{study}"
    return fallback


def get_study_uid(ds: pydicom.Dataset, fallback: str) -> str:
    uid = _safe_str(getattr(ds, "StudyInstanceUID", ""))
    return uid if uid else fallback


def get_sop_uid(ds: pydicom.Dataset, fallback: str) -> str:
    uid = _safe_str(getattr(ds, "SOPInstanceUID", ""))
    return uid if uid else fallback


# ---------------------------
# INBreast XML/plist parsing
# ---------------------------

def load_plist(xml_path: Path) -> Dict:
    with xml_path.open("rb") as f:
        return plistlib.load(f, fmt=plistlib.FMT_XML)


def parse_rois(xml_path: Path) -> List[Dict]:
    plist = load_plist(xml_path)
    images = plist.get("Images", [])
    if not images:
        return []
    return images[0].get("ROIs", []) or []


def load_points_xy(point_list: List[str]) -> np.ndarray:
    pts = []
    for s in point_list:
        s = s.strip().strip("()")
        if not s:
            continue
        x_str, y_str = s.split(",")
        pts.append((float(x_str), float(y_str)))
    return np.array(pts, dtype=np.float32)


def draw_points(mask: np.ndarray, pts_xy: np.ndarray, radius: int = 0):
    H, W = mask.shape
    for x, y in pts_xy:
        xi = int(round(x))
        yi = int(round(y))
        if not (0 <= xi < W and 0 <= yi < H):
            continue
        if radius <= 0:
            mask[yi, xi] = 1
        else:
            y0, y1 = max(0, yi - radius), min(H - 1, yi + radius)
            x0, x1 = max(0, xi - radius), min(W - 1, xi + radius)
            yy, xx = np.ogrid[y0:y1 + 1, x0:x1 + 1]
            disk = (yy - yi) ** 2 + (xx - xi) ** 2 <= radius ** 2
            mask[y0:y1 + 1, x0:x1 + 1][disk] = 1


def fill_polygon(mask: np.ndarray, pts_xy: np.ndarray):
    H, W = mask.shape
    pts = np.round(pts_xy).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)

    if _HAS_CV2:
        cv2.fillPoly(mask, [pts.reshape((-1, 1, 2))], 1)
        return

    if _HAS_SKIMAGE:
        xs = pts[:, 0]
        ys = pts[:, 1]
        rr, cc = sk_polygon(ys, xs, shape=(H, W))
        mask[rr, cc] = 1
        return

    raise RuntimeError("Need opencv-python or scikit-image for polygon rasterization.")


def build_masks_from_rois(rois: List[Dict], shape_hw: Tuple[int, int], point_radius: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    H, W = shape_hw
    m_mass = np.zeros((H, W), dtype=np.uint8)
    m_calc = np.zeros((H, W), dtype=np.uint8)

    for roi in rois:
        name = str(roi.get("Name", "")).strip()
        pts_list = roi.get("Point_px", []) or []
        if len(pts_list) == 0:
            continue
        pts_xy = load_points_xy(pts_list)

        if name == "Mass":
            if len(pts_xy) <= 2:
                draw_points(m_mass, pts_xy, radius=point_radius)
            else:
                fill_polygon(m_mass, pts_xy)

        elif name in ("Calcification", "Cluster"):
            if len(pts_xy) <= 2:
                draw_points(m_calc, pts_xy, radius=point_radius)
            else:
                fill_polygon(m_calc, pts_xy)

    return m_mass, m_calc


# ---------------------------
# Breast ROI estimation
# ---------------------------

def breast_roi_mask(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, [1, 99])
    x = np.clip(img, lo, hi)
    x = (255.0 * (x - lo) / max(1e-6, (hi - lo))).astype(np.uint8)

    if _HAS_CV2:
        _, th = cv2.threshold(x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th = (th > 0).astype(np.uint8)
        num, labels = cv2.connectedComponents(th)
        if num <= 1:
            return th
        areas = [(labels == i).sum() for i in range(1, num)]
        biggest = 1 + int(np.argmax(areas))
        mask = (labels == biggest).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        return mask

    if _HAS_SKIMAGE:
        t = np.percentile(x, 10)
        th = (x > t).astype(np.uint8)
        lbl = sk_label(th)
        if lbl.max() == 0:
            return th
        areas = [(lbl == i).sum() for i in range(1, lbl.max() + 1)]
        biggest = 1 + int(np.argmax(areas))
        return (lbl == biggest).astype(np.uint8)

    return (img > np.percentile(img, 10)).astype(np.uint8)


def _ensure_nonempty_breast_mask(img: np.ndarray, breast: np.ndarray, min_pixels: int = 100) -> np.ndarray:
    if breast is None or breast.size == 0:
        return np.ones_like(img, dtype=np.uint8)
    if int(breast.sum()) < min_pixels:
        return np.ones_like(img, dtype=np.uint8)
    return breast.astype(np.uint8)


def maybe_flip_left(img: np.ndarray, *masks: np.ndarray, laterality: Optional[str], flip_left: bool):
    if flip_left and (laterality == "L"):
        img = np.fliplr(img).copy()
        masks = tuple(np.fliplr(m).copy() for m in masks)
    return (img,) + masks


# ---------------------------
# XML matching / annotation status
# ---------------------------

def find_xml_for_case(xml_dir: Path, case_id: str, stem: str) -> Optional[Path]:
    for name in (f"{case_id}.xml", f"{case_id}.XML", f"{stem}.xml", f"{stem}.XML"):
        p = xml_dir / name
        if p.exists():
            return p

    candidates = []
    candidates.extend(sorted(xml_dir.glob(f"{case_id}*.xml")))
    candidates.extend(sorted(xml_dir.glob(f"{case_id}*.XML")))
    candidates = [c for c in candidates if c.is_file()]

    if len(candidates) == 1:
        return candidates[0]
    return None


@dataclass
class SampleRecord:
    sample_id: str
    case_id: str
    patient_id: str
    study_uid: str
    sop_uid: str
    dicom_path: str
    xml_path: str
    annotation_status: str
    n_rois: int
    laterality: str
    view: str
    has_mass: int
    has_calcification: int
    height: int
    width: int
    row_spacing_mm: float
    col_spacing_mm: float
    npz_path: str


def cmd_build(args: argparse.Namespace) -> None:
    dicom_dir = Path(args.dicom_dir)
    xml_dir = Path(args.xml_dir)
    out_dir = Path(args.out_dir)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    dicoms = sorted(dicom_dir.rglob("*.dcm"))
    if not dicoms:
        raise FileNotFoundError(f"No .dcm files found under: {dicom_dir}")

    records: List[SampleRecord] = []

    for dcm_path in tqdm(dicoms, desc="Building samples v4.2"):
        case_id = extract_case_id_from_filename(dcm_path.name)
        lat, view = parse_laterality_view_from_filename(dcm_path.name)
        lat = lat or ""
        view = view or ""

        ds = pydicom.dcmread(str(dcm_path))
        img = dicom_to_array(ds)
        H, W = img.shape

        patient_id = get_patient_id(ds, fallback=case_id)
        study_uid = get_study_uid(ds, fallback=f"STUDY_FALLBACK_{case_id}")
        sop_uid = get_sop_uid(ds, fallback=f"SOP_FALLBACK_{dcm_path.stem}")

        breast = _ensure_nonempty_breast_mask(img, breast_roi_mask(img), min_pixels=100)

        xml_path = find_xml_for_case(xml_dir, case_id=case_id, stem=dcm_path.stem)
        xml_path_str = str(xml_path) if xml_path else ""

        annotation_status = "missing_xml"
        rois: List[Dict] = []
        n_rois = 0

        if xml_path and xml_path.exists():
            try:
                rois = parse_rois(xml_path)
                n_rois = int(len(rois))
                annotation_status = "parsed_empty" if n_rois == 0 else "parsed_nonempty"
            except Exception as e:
                print(f"[WARN] XML parse failed for {xml_path}: {e}")
                annotation_status = "parse_failed"
                rois = []
                n_rois = 0

        m_mass = np.zeros((H, W), dtype=np.uint8)
        m_calc = np.zeros((H, W), dtype=np.uint8)
        if annotation_status in ("parsed_empty", "parsed_nonempty"):
            m_mass, m_calc = build_masks_from_rois(rois, (H, W), point_radius=args.calc_point_radius)

        img, m_mass, m_calc, breast = maybe_flip_left(
            img, m_mass, m_calc, breast, laterality=lat, flip_left=args.flip_left
        )

        m_any = np.clip(m_mass + m_calc, 0, 1).astype(np.uint8)

        has_mass = int(m_mass.sum() > 0)
        has_calc = int(m_calc.sum() > 0)

        rs, cs = get_pixel_spacing_mm(ds)
        rs = float(rs) if rs is not None else -1.0
        cs = float(cs) if cs is not None else -1.0

        sample_id = dcm_path.stem
        npz_path = samples_dir / f"{sample_id}.npz"

        np.savez_compressed(
            npz_path,
            image=img.astype(np.float32),
            mask_mass=m_mass.astype(np.uint8),
            mask_calcification=m_calc.astype(np.uint8),
            mask_any=m_any.astype(np.uint8),
            mask_breast=breast.astype(np.uint8),

            case_id=case_id,
            patient_id=patient_id,
            study_uid=study_uid,
            sop_uid=sop_uid,

            laterality=lat,
            view=view,
            row_spacing_mm=rs,
            col_spacing_mm=cs,

            annotation_status=annotation_status,
            n_rois=n_rois,

            source_dicom=str(dcm_path),
            source_xml=xml_path_str,
        )

        records.append(SampleRecord(
            sample_id=sample_id,
            case_id=case_id,
            patient_id=patient_id,
            study_uid=study_uid,
            sop_uid=sop_uid,
            dicom_path=str(dcm_path),
            xml_path=xml_path_str,
            annotation_status=annotation_status,
            n_rois=n_rois,
            laterality=lat,
            view=view,
            has_mass=has_mass,
            has_calcification=has_calc,
            height=H,
            width=W,
            row_spacing_mm=rs,
            col_spacing_mm=cs,
            npz_path=str(npz_path),
        ))

    df = pd.DataFrame([r.__dict__ for r in records])
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "metadata.csv", index=False)

    print(f"Saved {len(df)} samples to: {samples_dir}")
    print(f"Metadata: {out_dir / 'metadata.csv'}")
    print("\nAnnotation status counts:")
    print(df["annotation_status"].value_counts(dropna=False).to_string())


def _group_split(group_ids: List[str], val_ratio: float, test_ratio: float, seed: int) -> Dict[str, List[str]]:
    if val_ratio < 0 or test_ratio < 0 or (val_ratio + test_ratio) >= 1.0:
        raise ValueError("val_ratio and test_ratio must be >= 0 and val_ratio + test_ratio must be < 1.0")

    rng = np.random.default_rng(seed)
    unique = np.array(sorted(set(group_ids)))
    rng.shuffle(unique)
    n = len(unique)

    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))

    test = unique[:n_test]
    val = unique[n_test:n_test + n_val]
    train = unique[n_test + n_val:]
    return {"train": train.tolist(), "val": val.tolist(), "test": test.tolist()}


def _sanity_4way(df: pd.DataFrame, group_col: str, groups: Dict[str, List[str]]) -> None:
    train = set(map(str, groups["train"]))
    val = set(map(str, groups["val"]))
    test = set(map(str, groups["test"]))
    unl = set(map(str, groups["unlabeled"]))

    # group disjointness
    if not train.isdisjoint(val): raise RuntimeError("Sanity failed: train/val overlap.")
    if not train.isdisjoint(test): raise RuntimeError("Sanity failed: train/test overlap.")
    if not val.isdisjoint(test):   raise RuntimeError("Sanity failed: val/test overlap.")
    if not unl.isdisjoint(train | val | test): raise RuntimeError("Sanity failed: unlabeled overlaps labeled splits.")

    # each sample assigned exactly once
    g = df[group_col].astype(str)
    a = (
        g.isin(train).astype(int)
        + g.isin(val).astype(int)
        + g.isin(test).astype(int)
        + g.isin(unl).astype(int)
    )
    if not (a == 1).all():
        bad = df.loc[a != 1, ["sample_id", group_col, "annotation_status", "npz_path"]].head(10)
        raise RuntimeError(
            "Sanity failed: some samples assigned to 0 or multiple sets.\n"
            f"Example rows:\n{bad.to_string(index=False)}"
        )
    print("Sanity checks: OK (4-way disjoint; every sample assigned exactly once).")


def cmd_split(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    meta_path = out_dir / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata.csv at {meta_path}. Run build first.")
    df = pd.read_csv(meta_path)

    if "annotation_status" not in df.columns:
        raise ValueError("metadata.csv missing 'annotation_status'. Re-run build with this script.")

    # grouping column
    if args.group_by == "patient":
        group_col = "patient_id"
    elif args.group_by == "study":
        group_col = "study_uid"
    elif args.group_by == "image":
        group_col = "sop_uid"
    else:
        raise ValueError(f"Unknown group_by: {args.group_by}")

    if group_col not in df.columns:
        raise ValueError(f"metadata.csv missing '{group_col}'. Re-run build with this script.")

    labeled_statuses = {"parsed_empty", "parsed_nonempty"}
    unlabeled_statuses = {"missing_xml", "parse_failed"}

    df["annotation_status"] = df["annotation_status"].astype(str)
    df[group_col] = df[group_col].astype(str)

    # Decide group membership by status:
    # - If a group has ANY unlabeled sample -> entire group goes to UNLABELED (your request: put aside).
    # - Else (no unlabeled, i.e., labeled-only) -> eligible for train/val/test split.
    grp = df.groupby(group_col)["annotation_status"]
    group_has_unl = grp.apply(lambda s: any(x in unlabeled_statuses for x in s.tolist()))
    group_has_lbl = grp.apply(lambda s: any(x in labeled_statuses for x in s.tolist()))

    unlabeled_groups = set(g for g in group_has_unl.index.tolist() if bool(group_has_unl.loc[g]))
    labeled_only_groups = set(
        g for g in group_has_lbl.index.tolist()
        if bool(group_has_lbl.loc[g]) and (not bool(group_has_unl.loc[g]))
    )
    mixed_groups = set(g for g in group_has_lbl.index.tolist() if bool(group_has_lbl.loc[g]) and bool(group_has_unl.loc[g]))

    if len(labeled_only_groups) == 0:
        raise RuntimeError("No labeled-only groups found. Check XML parsing / annotation_status.")

    split_lbl = _group_split(list(labeled_only_groups), args.val_ratio, args.test_ratio, args.seed)

    split_groups = {
        "train": split_lbl["train"],
        "val": split_lbl["val"],
        "test": split_lbl["test"],
        "unlabeled": sorted(unlabeled_groups),  # includes mixed groups by policy (entire group set aside)
    }

    # Sanity
    _sanity_4way(df, group_col, split_groups)

    # Write lists
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(exist_ok=True)

    def write_list(fname: str, paths: List[str]) -> None:
        (splits_dir / fname).write_text("\n".join(paths) + ("\n" if paths else ""))

    train_set = set(map(str, split_groups["train"]))
    val_set = set(map(str, split_groups["val"]))
    test_set = set(map(str, split_groups["test"]))
    unl_set = set(map(str, split_groups["unlabeled"]))

    df_train = df[df[group_col].isin(train_set) & df["annotation_status"].isin(labeled_statuses)]
    df_val = df[df[group_col].isin(val_set) & df["annotation_status"].isin(labeled_statuses)]
    df_test = df[df[group_col].isin(test_set) & df["annotation_status"].isin(labeled_statuses)]
    df_unl = df[df[group_col].isin(unl_set)]  # all samples in unlabeled groups (labeled or not), by policy

    write_list("train.txt", df_train["npz_path"].astype(str).tolist())
    write_list("val.txt", df_val["npz_path"].astype(str).tolist())
    write_list("test.txt", df_test["npz_path"].astype(str).tolist())
    write_list("unlabeled.txt", df_unl["npz_path"].astype(str).tolist())

    # Save splits.json
    payload = {
        "group_by": args.group_by,
        "group_col": group_col,
        "seed": int(args.seed),
        "val_ratio": float(args.val_ratio),
        "test_ratio": float(args.test_ratio),
        "labeled_statuses": sorted(labeled_statuses),
        "unlabeled_statuses": sorted(unlabeled_statuses),
        "groups": split_groups,
        "counts": {
            "n_samples_total": int(len(df)),
            "n_samples_train_labeled": int(len(df_train)),
            "n_samples_val_labeled": int(len(df_val)),
            "n_samples_test_labeled": int(len(df_test)),
            "n_samples_unlabeled_pool": int(len(df_unl)),
            "n_groups_total": int(df[group_col].nunique()),
            "n_groups_labeled_only": int(len(labeled_only_groups)),
            "n_groups_unlabeled_pool": int(len(unlabeled_groups)),
            "n_groups_mixed": int(len(mixed_groups)),
        }
    }
    with (out_dir / "splits.json").open("w") as f:
        json.dump(payload, f, indent=2)

    # Print summary
    def info(name: str, dfx: pd.DataFrame):
        n_imgs = len(dfx)
        n_groups = int(dfx[group_col].nunique())
        n_mass = int(dfx["has_mass"].sum()) if "has_mass" in dfx.columns else -1
        n_calc = int(dfx["has_calcification"].sum()) if "has_calcification" in dfx.columns else -1
        print(f"{name}: {n_imgs} images | {n_groups} {args.group_by}s | mass+ {n_mass} | calc+ {n_calc}")

    info("train", df_train)
    info("val", df_val)
    info("test", df_test)
    print(f"unlabeled: {len(df_unl)} images (all samples from unlabeled/mixed groups, set aside)")

    print(f"Wrote: {out_dir / 'splits.json'} and {splits_dir}/train.txt val.txt test.txt unlabeled.txt")


def cmd_norm(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    split_path = out_dir / "splits" / f"{args.split_name}.txt"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split list: {split_path}. Run split first.")
    npz_paths = [Path(p.strip()) for p in split_path.read_text().splitlines() if p.strip()]

    rng = np.random.default_rng(args.seed)
    samples = []
    for p in tqdm(npz_paths, desc=f"Computing norm ({args.split_name})"):
        d = np.load(p, allow_pickle=True)
        img = d["image"].astype(np.float32)
        breast = d["mask_breast"].astype(np.uint8) if "mask_breast" in d.files else None
        if breast is None or breast.size == 0 or int(breast.sum()) < 100:
            roi = img.reshape(-1)
        else:
            roi = img[breast > 0]
        if roi.size < 100:
            continue
        lo = np.percentile(roi, args.clip_p_lo)
        hi = np.percentile(roi, args.clip_p_hi)
        roi = np.clip(roi, lo, hi)
        take = min(args.max_pixels_per_image, roi.size)
        idx = rng.choice(roi.size, size=take, replace=False)
        samples.append(roi.reshape(-1)[idx])

    if not samples:
        raise RuntimeError("No pixels collected for normalization. Check mask_breast quality / split list.")

    xs = np.concatenate(samples).astype(np.float32)
    norm = {
        "split_used": args.split_name,
        "clip_percentiles": [args.clip_p_lo, args.clip_p_hi],
        "global_mean_after_clip": float(xs.mean()),
        "global_std_after_clip": float(xs.std() + 1e-8),
        "n_pixels_used": int(xs.size),
        "max_pixels_per_image": int(args.max_pixels_per_image),
        "seed": int(args.seed),
    }
    with (out_dir / "normalization.json").open("w") as f:
        json.dump(norm, f, indent=2)
    print(f"Saved: {out_dir / 'normalization.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build")
    p_build.add_argument("--dicom_dir", required=True)
    p_build.add_argument("--xml_dir", required=True)
    p_build.add_argument("--out_dir", required=True)
    p_build.add_argument("--flip_left", action="store_true")
    p_build.add_argument("--calc_point_radius", type=int, default=0)
    p_build.set_defaults(func=cmd_build)

    p_split = sub.add_parser("split")
    p_split.add_argument("--out_dir", required=True)
    p_split.add_argument("--group_by", default="patient", choices=["patient", "study", "image"])
    p_split.add_argument("--val_ratio", type=float, default=0.15)
    p_split.add_argument("--test_ratio", type=float, default=0.15)
    p_split.add_argument("--seed", type=int, default=42)
    p_split.set_defaults(func=cmd_split)

    p_norm = sub.add_parser("norm")
    p_norm.add_argument("--out_dir", required=True)
    p_norm.add_argument("--split_name", default="train", choices=["train", "val", "test"])
    p_norm.add_argument("--clip_p_lo", type=float, default=1.0)
    p_norm.add_argument("--clip_p_hi", type=float, default=99.0)
    p_norm.add_argument("--max_pixels_per_image", type=int, default=200_000)
    p_norm.add_argument("--seed", type=int, default=42)
    p_norm.set_defaults(func=cmd_norm)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()