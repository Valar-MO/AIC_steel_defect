#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/AIC_steel_defect
P=/root/miniconda3/envs/aic-steel/bin/python
ROOT=outputs/hierarchical_objectness_dual_v1
INPUT="$ROOT/checkpoint0000_ema/raw_whole.json"
OUT="$ROOT/checkpoint0000_ema/nms_scan"
mkdir -p "$OUT"

candidates=()
for nms_iou in 0.50 0.60 0.70 0.80 0.90; do
  stem="nms${nms_iou/./}"
  prediction="$OUT/$stem/raw_whole.json"
  "$P" scripts/apply_prediction_nms.py \
    --input "$INPUT" --output "$prediction" \
    --nms-iou "$nms_iou" --overwrite
  "$P" scripts/compare_hierarchical_score_modes.py \
    --predictions "$prediction" \
    --manifest splits/v1/split_manifest.csv --classes configs/classes.yaml \
    --split val --score-modes objectness --score-threshold 0.10 \
    --match-iou 0.50 --background-iou 0.10 \
    --output-dir "$OUT/$stem/results" --overwrite
  candidates+=("$prediction")
done

"$P" scripts/compare_objectness_rank_checkpoints.py \
  --baseline outputs/hierarchical_score_mode_diagnosis_v1/raw_whole.json \
  --candidates "${candidates[@]}" \
  --manifest splits/v1/split_manifest.csv --classes configs/classes.yaml \
  --split val --score-threshold 0.10 --match-iou 0.50 \
  --background-iou 0.10 --output "$OUT/nms_comparison.json"
