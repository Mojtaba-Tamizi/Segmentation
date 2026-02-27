#!/usr/bin/env bash
set -euo pipefail

# preprocess.sh
# Runs: build -> split (with unlabeled.txt) -> norm(train)

SCRIPT="src/build_inbreast.py"

DICOM_DIR="dataset/AllDICOMs"
XML_DIR="dataset/AllXML"
OUT_DIR="./data"

GROUP_BY="patient"
VAL_RATIO="0.15"
TEST_RATIO="0.15"
SEED="42"

CALC_POINT_RADIUS="1"

SPLIT_NAME="train"
CLIP_P_LO="1"
CLIP_P_HI="99"
MAX_PIXELS_PER_IMAGE="2000000"

echo "=== INBreast pipeline ==="
echo "Script:    ${SCRIPT}"
echo "DICOM_DIR: ${DICOM_DIR}"
echo "XML_DIR:   ${XML_DIR}"
echo "OUT_DIR:   ${OUT_DIR}"
echo "Group by:  ${GROUP_BY}"
echo "Seed:      ${SEED}"
echo

# Basic checks
if [[ ! -f "${SCRIPT}" ]]; then
  echo "ERROR: ${SCRIPT} not found."
  exit 1
fi
if [[ ! -d "${DICOM_DIR}" ]]; then
  echo "ERROR: DICOM_DIR not found: ${DICOM_DIR}"
  exit 1
fi
if [[ ! -d "${XML_DIR}" ]]; then
  echo "ERROR: XML_DIR not found: ${XML_DIR}"
  exit 1
fi

echo "=== Step 1/3: build ==="
python "${SCRIPT}" build \
  --dicom_dir "${DICOM_DIR}" \
  --xml_dir "${XML_DIR}" \
  --out_dir "${OUT_DIR}" \
  --flip_left \
  --calc_point_radius "${CALC_POINT_RADIUS}"

echo
echo "=== Step 2/3: split (train/val/test + unlabeled) ==="
python "${SCRIPT}" split \
  --out_dir "${OUT_DIR}" \
  --group_by "${GROUP_BY}" \
  --val_ratio "${VAL_RATIO}" \
  --test_ratio "${TEST_RATIO}" \
  --seed "${SEED}"

echo
echo "=== Step 3/3: norm (train only) ==="
python "${SCRIPT}" norm \
  --out_dir "${OUT_DIR}" \
  --split_name "${SPLIT_NAME}" \
  --clip_p_lo "${CLIP_P_LO}" \
  --clip_p_hi "${CLIP_P_HI}" \
  --max_pixels_per_image "${MAX_PIXELS_PER_IMAGE}" \
  --seed "${SEED}"

echo
echo "✅ Done. Outputs in: ${OUT_DIR}"
echo " - ${OUT_DIR}/metadata.csv"
echo " - ${OUT_DIR}/splits.json"
echo " - ${OUT_DIR}/splits/train.txt val.txt test.txt unlabeled.txt"
echo " - ${OUT_DIR}/normalization.json"

# Optional: show counts if available
if [[ -f "${OUT_DIR}/splits/train.txt" ]]; then
  echo
  echo "=== Split file counts ==="
  wc -l "${OUT_DIR}/splits/train.txt" "${OUT_DIR}/splits/val.txt" "${OUT_DIR}/splits/test.txt" "${OUT_DIR}/splits/unlabeled.txt" 2>/dev/null || true
fi