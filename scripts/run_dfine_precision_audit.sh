#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/AIC_steel_defect
P=/root/miniconda3/envs/aic-steel/bin/python
O=outputs/dfine_best_stg2
rm -f "$O/PRECISION_AUDIT_DONE" "$O/PRECISION_AUDIT_FAILED"
trap 'touch "$O/PRECISION_AUDIT_FAILED"' ERR
for mode in whole tiles fused_nms080; do
  echo "[$(date '+%F %T')] audit $mode"
  "$P" scripts/analyze_dfine_proposal_precision.py \
    --predictions "$O/val_${mode}.json" \
    --manifest splits/v1/split_manifest.csv --classes configs/classes.yaml --split val \
    --match-iou 0.5 --near-iou 0.3 --background-iou 0.1 \
    --score-thresholds 0.001 0.003 0.005 0.01 0.03 0.05 0.1 0.2 0.3 0.5 \
    --output-dir "$O/precision_audit_${mode}" --overwrite \
    >"$O/logs/precision_audit_${mode}.log" 2>&1
done
touch "$O/PRECISION_AUDIT_DONE"
echo "[$(date '+%F %T')] audits complete"
