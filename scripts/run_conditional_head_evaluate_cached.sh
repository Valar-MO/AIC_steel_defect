#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/AIC_steel_defect
P=/root/miniconda3/envs/aic-steel/bin/python
O=outputs/conditional_head_ablation_v1
RAW=outputs/dfine_best_stg2/val_fused_nms080.json
rm -f "$O/DONE" "$O/FAILED"
trap 'touch "$O/FAILED"' ERR
for variant in conditional conditional_margin conditional_hard; do
  echo "[$(date '+%F %T')] apply and evaluate $variant"
  "$P" scripts/apply_conditional_head_cache.py \
    --cache "$O/cache/raw_val_min003.pt" --predictions "$RAW" \
    --checkpoint "$O/heads/${variant}.pt" --output "$O/val_${variant}.json" --overwrite
  "$P" scripts/evaluate_predictions.py \
    --predictions "$O/val_${variant}.json" \
    --manifest splits/v1/split_manifest.csv --classes configs/classes.yaml --split val \
    --score-threshold 0.001 --output-dir "$O/evaluations/${variant}" --overwrite
done
touch "$O/DONE"
echo "[$(date '+%F %T')] complete"
