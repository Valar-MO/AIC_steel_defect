#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/autodl-tmp/AIC_steel_defect
PYTHON=/root/miniconda3/envs/aic-steel/bin/python
OUTPUT="$PROJECT/outputs/dinov2_teacher_upper_bound_v1"
TRAIN_CACHE="$OUTPUT/cache/train_candidates_s005.pt"
VAL_CACHE="$OUTPUT/cache/val_candidates_s005.pt"
LOG="$OUTPUT/logs/train_dino_and_fusion.log"

cd "$PROJECT"
mkdir -p "$OUTPUT/logs"
exec > >(tee "$LOG") 2>&1

COMMON_ARGS=(
  --train-cache "$TRAIN_CACHE"
  --val-cache "$VAL_CACHE"
  --dinov2-weights /root/autodl-tmp/weights/dinov2-base
  --image-size 384
  --tight-expand 1.0
  --context-expand 1.8
  --min-crop-size 128
  --batch 208
  --workers 16
  --samples-per-epoch 60000
  --epochs 8
  --head-only-epochs 1
  --unfreeze-last-blocks 4
  --lr 0.0002
  --backbone-lr 0.00001
  --rank-weight 0.5
  --rank-margin 0.5
  --residual-weight 0.001
  --patience 4
  --amp-dtype bfloat16
)

echo "START mode=dino"
"$PYTHON" scripts/train_dfine_verifier_teacher.py \
  --mode dino \
  --output "$OUTPUT/train_dino" \
  "${COMMON_ARGS[@]}" \
  --overwrite

echo "START mode=query_dino"
"$PYTHON" scripts/train_dfine_verifier_teacher.py \
  --mode query_dino \
  --output "$OUTPUT/train_query_dino" \
  "${COMMON_ARGS[@]}" \
  --overwrite

echo TRAIN_COMPLETE
