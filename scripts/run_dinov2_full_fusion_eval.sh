#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/autodl-tmp/AIC_steel_defect
PYTHON=/root/miniconda3/envs/aic-steel/bin/python
DFINE_CHECKPOINT=/root/autodl-tmp/D-FINE/output/dfine_hgnetv2_l_hierarchical_aic/best_stg1.pth
DINO_CHECKPOINT="$PROJECT/outputs/dinov2_teacher_upper_bound_v1/train_query_dino/best.pt"
OUTPUT="$PROJECT/outputs/dinov2_teacher_upper_bound_v1/full_fusion_v1"

cd "$PROJECT"
mkdir -p "$OUTPUT"
exec > >(tee "$OUTPUT/run.log") 2>&1

"$PYTHON" scripts/predict_hierarchical_dfine.py \
  --checkpoint "$DFINE_CHECKPOINT" \
  --mode whole \
  --output "$OUTPUT/whole_raw.json" \
  --candidate-cache "$OUTPUT/whole_cache.pt" \
  --cache-min-selection 0.05 \
  --imgsz 1024 --batch 4 --conf 0.001 \
  --overwrite

"$PYTHON" scripts/predict_hierarchical_dfine.py \
  --checkpoint "$DFINE_CHECKPOINT" \
  --mode tiles \
  --output "$OUTPUT/tiles_raw.json" \
  --candidate-cache "$OUTPUT/tiles_cache.pt" \
  --cache-min-selection 0.05 \
  --imgsz 1024 --tile-size 2048 --overlap 512 \
  --batch 4 --conf 0.001 \
  --overwrite

"$PYTHON" scripts/score_dfine_verifier_candidates.py \
  --mode query_dino \
  --cache "$OUTPUT/whole_cache.pt" \
  --checkpoint "$DINO_CHECKPOINT" \
  --output "$OUTPUT/whole_dino_scores.pt" \
  --batch 768 --workers 8 --overwrite

"$PYTHON" scripts/score_dfine_verifier_candidates.py \
  --mode query_dino \
  --cache "$OUTPUT/tiles_cache.pt" \
  --checkpoint "$DINO_CHECKPOINT" \
  --output "$OUTPUT/tiles_dino_scores.pt" \
  --batch 768 --workers 8 --overwrite

"$PYTHON" scripts/merge_predictions.py \
  --whole "$OUTPUT/whole_raw.json" \
  --tiles "$OUTPUT/tiles_raw.json" \
  --classes configs/classes.yaml \
  --min-score 0.30 --nms-iou 0.80 --max-det 300 \
  --output "$OUTPUT/baseline_fused.json" \
  --report "$OUTPUT/baseline_fused_merge_report.json" \
  --overwrite

for setting in balanced recall; do
  if [[ "$setting" == balanced ]]; then
    gate=0.40
  else
    gate=0.0075
  fi

  "$PYTHON" scripts/apply_dinov2_defectness_gate.py \
    --predictions "$OUTPUT/whole_raw.json" \
    --verifier-scores "$OUTPUT/whole_dino_scores.pt" \
    --gate-threshold "$gate" \
    --output "$OUTPUT/${setting}_whole_gated.json" \
    --overwrite
  "$PYTHON" scripts/apply_dinov2_defectness_gate.py \
    --predictions "$OUTPUT/tiles_raw.json" \
    --verifier-scores "$OUTPUT/tiles_dino_scores.pt" \
    --gate-threshold "$gate" \
    --output "$OUTPUT/${setting}_tiles_gated.json" \
    --overwrite
  "$PYTHON" scripts/merge_predictions.py \
    --whole "$OUTPUT/${setting}_whole_gated.json" \
    --tiles "$OUTPUT/${setting}_tiles_gated.json" \
    --classes configs/classes.yaml \
    --min-score 0.30 --nms-iou 0.80 --max-det 300 \
    --output "$OUTPUT/${setting}_fused.json" \
    --report "$OUTPUT/${setting}_fused_merge_report.json" \
    --overwrite
done

for setting in baseline balanced recall; do
  "$PYTHON" scripts/compare_hierarchical_score_modes.py \
    --predictions "$OUTPUT/${setting}_fused.json" \
    --manifest splits/v1/split_manifest.csv \
    --classes configs/classes.yaml --split val \
    --score-modes objectness \
    --score-threshold 0.30 \
    --match-iou 0.50 --background-iou 0.10 \
    --output-dir "$OUTPUT/${setting}_evaluation" \
    --overwrite
done

echo FULL_FUSION_COMPLETE
