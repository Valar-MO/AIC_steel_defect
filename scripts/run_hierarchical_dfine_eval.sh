#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/AIC_steel_defect
P=/root/miniconda3/envs/aic-steel/bin/python
CHECKPOINT=${1:?usage: run_hierarchical_dfine_eval.sh CHECKPOINT [OUTPUT_DIR]}
O=${2:-outputs/hierarchical_dfine_val}
CONFIG=configs/dfine/custom/objects365/dfine_hgnetv2_l_hierarchical_aic.yml

mkdir -p "$O"
"$P" scripts/install_hierarchical_dfine.py
"$P" scripts/predict_hierarchical_dfine.py --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" --mode whole --output "$O/whole.json" \
  --imgsz 1024 --batch 4 --conf 0.001 --overwrite
"$P" scripts/predict_hierarchical_dfine.py --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" --mode tiles --output "$O/tiles.json" \
  --imgsz 1024 --tile-size 2048 --overlap 512 --batch 4 --conf 0.001 --overwrite
"$P" scripts/merge_predictions.py --whole "$O/whole.json" --tiles "$O/tiles.json" \
  --classes configs/classes.yaml --output "$O/fused_nms080.json" \
  --nms-iou 0.80 --min-score 0.001 --max-det 300 --overwrite
"$P" scripts/evaluate_predictions.py --predictions "$O/fused_nms080.json" \
  --manifest splits/v1/split_manifest.csv --classes configs/classes.yaml --split val \
  --score-threshold 0.1 --output-dir "$O/evaluation" --overwrite
