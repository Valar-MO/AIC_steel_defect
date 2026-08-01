#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/autodl-tmp/AIC_steel_defect
PYTHON=/root/miniconda3/envs/aic-steel/bin/python
DFINE=/root/autodl-tmp/D-FINE
CONFIG=configs/dfine/custom/objects365/dfine_hgnetv2_l_obj2aic_defect.yml
CHECKPOINT=$DFINE/output/dfine_hgnetv2_l_obj2aic_defect/best_stg2.pth
OUTPUT=$PROJECT/outputs/dfine_best_stg2

cd "$PROJECT"
mkdir -p "$OUTPUT/logs"
rm -f "$OUTPUT/PIPELINE_DONE" "$OUTPUT/PIPELINE_FAILED"
trap 'touch "$OUTPUT/PIPELINE_FAILED"' ERR

echo "[$(date '+%F %T')] whole inference"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" scripts/predict_dfine_agnostic.py \
  --dfine-dir "$DFINE" --config "$CONFIG" --checkpoint "$CHECKPOINT" \
  --manifest splits/v1/split_manifest.csv --split val --mode whole \
  --imgsz 1024 --batch 32 --conf 0.001 --max-det-per-view 300 \
  --output "$OUTPUT/val_whole.json" --overwrite \
  >"$OUTPUT/logs/whole.log" 2>&1

echo "[$(date '+%F %T')] tile inference"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" scripts/predict_dfine_agnostic.py \
  --dfine-dir "$DFINE" --config "$CONFIG" --checkpoint "$CHECKPOINT" \
  --manifest splits/v1/split_manifest.csv --split val --mode tiles \
  --imgsz 1024 --tile-size 2048 --overlap 512 --batch 32 \
  --conf 0.001 --max-det-per-view 300 \
  --output "$OUTPUT/val_tiles.json" --overwrite \
  >"$OUTPUT/logs/tiles.log" 2>&1

echo "[$(date '+%F %T')] fusion"
"$PYTHON" scripts/merge_predictions.py \
  --whole "$OUTPUT/val_whole.json" --tiles "$OUTPUT/val_tiles.json" \
  --classes configs/defect.yaml --min-score 0.001 --nms-iou 0.50 \
  --tile-border-policy keep --max-det 1000 \
  --output "$OUTPUT/val_fused.json" --report "$OUTPUT/val_fused_report.json" \
  --overwrite >"$OUTPUT/logs/fusion.log" 2>&1

for mode in whole tiles fused; do
  echo "[$(date '+%F %T')] evaluate $mode"
  "$PYTHON" scripts/evaluate_agnostic_proposals.py \
    --predictions "$OUTPUT/val_${mode}.json" \
    --manifest splits/v1/split_manifest.csv --classes configs/classes.yaml \
    --split val --score-thresholds 0.001 0.003 0.005 0.01 0.03 0.05 0.1 \
    --iou-thresholds 0.3 0.5 --max-proposals-per-image 1000 \
    --output-dir "$OUTPUT/coverage_${mode}" --overwrite \
    >"$OUTPUT/logs/eval_${mode}.log" 2>&1
done

touch "$OUTPUT/PIPELINE_DONE"
echo "[$(date '+%F %T')] pipeline complete"
