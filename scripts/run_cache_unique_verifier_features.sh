#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/autodl-tmp/AIC_steel_defect
PYTHON=/root/miniconda3/envs/aic-steel/bin/python
OUTPUT="$PROJECT/outputs/dinov2_teacher_upper_bound_v1"
CHECKPOINT="$OUTPUT/train_query_dino/best.pt"
LOG="$OUTPUT/logs/cache_unique_features.log"

cd "$PROJECT"
exec > >(tee "$LOG") 2>&1

if [[ ! -f "$OUTPUT/features/val_query_dino_ap.pt" ]]; then
  "$PYTHON" scripts/score_dfine_verifier_candidates.py \
    --mode query_dino \
    --cache "$OUTPUT/cache/val_candidates_unique.pt" \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT/features/val_query_dino_ap.pt" \
    --batch 768 --workers 16 --save-features --overwrite
fi

"$PYTHON" scripts/score_dfine_verifier_candidates.py \
  --mode query_dino \
  --cache "$OUTPUT/cache/train_candidates_unique.pt" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT/features/train_query_dino_ap.pt" \
  --batch 768 --workers 16 --save-features --overwrite

echo UNIQUE_FEATURE_CACHE_COMPLETE
