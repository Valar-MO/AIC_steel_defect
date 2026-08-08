#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/autodl-tmp/AIC_steel_defect
PYTHON=/root/miniconda3/envs/aic-steel/bin/python
CHECKPOINT=/root/autodl-tmp/D-FINE/output/dfine_hgnetv2_l_hierarchical_aic/best_stg1.pth
OUTPUT="$PROJECT/outputs/dinov2_teacher_upper_bound_v1"

cd "$PROJECT"
mkdir -p "$OUTPUT/cache" "$OUTPUT/logs"

"$PYTHON" scripts/cache_dfine_verifier_candidates.py \
  --checkpoint "$CHECKPOINT" \
  --split val \
  --output "$OUTPUT/cache/val_candidates_s005.pt" \
  --min-selection 0.05 \
  --batch 16 \
  --workers 2 \
  --overwrite

"$PYTHON" scripts/cache_dfine_verifier_candidates.py \
  --checkpoint "$CHECKPOINT" \
  --split train \
  --output "$OUTPUT/cache/train_candidates_s005.pt" \
  --min-selection 0.05 \
  --batch 16 \
  --workers 2 \
  --overwrite

echo CACHE_COMPLETE
