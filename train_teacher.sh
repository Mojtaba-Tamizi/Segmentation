#!/usr/bin/env bash
set -euo pipefail

# -----------------------
# Config (edit as needed)
# -----------------------
DATA_DIR="data"          # contains splits/train.txt and splits/val.txt and normalization.json (optional)
DEVICE="cuda"            # "cuda" or "cpu"
SEED=42

EPOCHS=30
STEPS_PER_EPOCH=500
BATCH_SIZE=6
PATCH_SIZE=512

LR=3e-5
WEIGHT_DECAY=1e-5
NUM_WORKERS=4

# UNet
BASE_CHANNELS=32
DEPTH=4
NORM="bn"
DROPOUT=0.1

# Target
TARGET_KEY="mask_mass"   # "mask_mass" or "mask_any"

# Validation tiling
TILE_OVERLAP=0.5
TILE_BATCH=4
WEIGHT_MODE="hann"       # hann|gaussian|uniform
THRESH=0.5

# AMP (recommended bf16 if supported)
AMP="--amp"
AMP_DTYPE="fp16"         # bf16|fp16
GRAD_CLIP=3.0

# Distance metric cap (optional; -1 disables)
CAP_MM=-1

# How often to run full-image validation
EVAL_EVERY=1

# Debug: limit val images (-1 disables)
MAX_VAL_IMAGES=-1

# -----------------------
# Run
# -----------------------
python -m src.train_teacher \
  --data_dir "${DATA_DIR}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --epochs "${EPOCHS}" \
  --steps_per_epoch "${STEPS_PER_EPOCH}" \
  --batch_size "${BATCH_SIZE}" \
  --patch_size "${PATCH_SIZE}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --num_workers "${NUM_WORKERS}" \
  --base_channels "${BASE_CHANNELS}" \
  --depth "${DEPTH}" \
  --norm "${NORM}" \
  --dropout "${DROPOUT}" \
  --target_key "${TARGET_KEY}" \
  --tile_overlap "${TILE_OVERLAP}" \
  --tile_batch "${TILE_BATCH}" \
  --weight_mode "${WEIGHT_MODE}" \
  --threshold "${THRESH}" \
  --cap_mm "${CAP_MM}" \
  --eval_every "${EVAL_EVERY}" \
  --max_val_images "${MAX_VAL_IMAGES}" \
  ${AMP} \
  --amp_dtype "${AMP_DTYPE}" \
  --grad_clip "${GRAD_CLIP}" \
  --use_focal
  --pos_weight 10 \
