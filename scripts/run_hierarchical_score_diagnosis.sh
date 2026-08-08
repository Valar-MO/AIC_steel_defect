#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/AIC_steel_defect
P=/root/miniconda3/envs/aic-steel/bin/python
CHECKPOINT=${1:-/root/autodl-tmp/D-FINE/output/dfine_hgnetv2_l_hierarchical_aic/best_stg1.pth}
O=${2:-outputs/hierarchical_score_mode_diagnosis_v1}
CONFIG=configs/dfine/custom/objects365/dfine_hgnetv2_l_hierarchical_aic.yml

mkdir -p "$O"
"$P" scripts/predict_hierarchical_dfine.py \
  --config "$CONFIG" --checkpoint "$CHECKPOINT" \
  --manifest splits/v1/split_manifest.csv --classes configs/classes.yaml \
  --split val --mode whole --output "$O/raw_whole.json" \
  --imgsz 1024 --batch 4 --conf 0.001 --overwrite

"$P" scripts/compare_hierarchical_score_modes.py \
  --predictions "$O/raw_whole.json" \
  --manifest splits/v1/split_manifest.csv --classes configs/classes.yaml \
  --split val --score-threshold 0.10 --match-iou 0.50 --background-iou 0.10 \
  --output-dir "$O/results" --overwrite
