#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/AIC_steel_defect
P=/root/miniconda3/envs/aic-steel/bin/python
ROOT=outputs/hierarchical_selection_gated_v1
CONFIG=configs/dfine/custom/objects365/dfine_hgnetv2_l_hierarchical_selection_gated_v1.yml
SOURCE=/root/autodl-tmp/D-FINE/output/dfine_hgnetv2_l_hierarchical_selection_gated_v1/checkpoint0000.pth
RAW="$ROOT/raw_checkpoints/checkpoint0000_raw.pth"
OUT="$ROOT/checkpoint0000_raw"

mkdir -p "$ROOT/raw_checkpoints" "$OUT"
"$P" analysis/hierarchical_gtaware_e1_1/extract_raw_checkpoint.py \
  "$SOURCE" "$RAW"
"$P" scripts/predict_hierarchical_dfine.py \
  --config "$CONFIG" --checkpoint "$RAW" \
  --manifest splits/v1/split_manifest.csv --classes configs/classes.yaml \
  --split val --mode whole --output "$OUT/raw_whole.json" \
  --imgsz 1024 --batch 32 --conf 0.001 --overwrite
"$P" scripts/compare_hierarchical_score_modes.py \
  --predictions "$OUT/raw_whole.json" \
  --manifest splits/v1/split_manifest.csv --classes configs/classes.yaml \
  --split val \
  --score-modes objectness selection selection_quality \
    selection_defectness defectness joint \
  --score-threshold 0.10 --match-iou 0.50 --background-iou 0.10 \
  --output-dir "$OUT/results" --overwrite
find "$OUT/results" -name predictions.json -type f -delete
rm -f "$RAW"
