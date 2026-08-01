#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/AIC_steel_defect
O=outputs/dfine_best_stg2
P=/root/miniconda3/envs/aic-steel/bin/python
rm -f "$O/FUSION_SWEEP_DONE"
for pair in 070:0.70 080:0.80; do
  tag=${pair%%:*}
  iou=${pair##*:}
  "$P" scripts/merge_predictions.py \
    --whole "$O/val_whole.json" --tiles "$O/val_tiles.json" \
    --classes configs/defect.yaml --min-score 0.001 --nms-iou "$iou" \
    --tile-border-policy keep --max-det 1000 \
    --output "$O/val_fused_nms${tag}.json" \
    --report "$O/val_fused_nms${tag}_report.json" --overwrite \
    >"$O/logs/fusion_nms${tag}.log" 2>&1
  "$P" scripts/evaluate_agnostic_proposals.py \
    --predictions "$O/val_fused_nms${tag}.json" \
    --manifest splits/v1/split_manifest.csv --classes configs/classes.yaml --split val \
    --score-thresholds 0.001 0.003 0.005 0.01 0.03 0.05 0.1 \
    --iou-thresholds 0.3 0.5 --max-proposals-per-image 1000 \
    --output-dir "$O/coverage_fused_nms${tag}" --overwrite \
    >"$O/logs/eval_fused_nms${tag}.log" 2>&1
done
touch "$O/FUSION_SWEEP_DONE"
