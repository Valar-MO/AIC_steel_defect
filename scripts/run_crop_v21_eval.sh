#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/AIC_steel_defect
P=/root/miniconda3/envs/aic-steel/bin/python
O=outputs/dfine_crop_residual_v21_e4_min003
BEST=/root/autodl-tmp/runs/crop_cls/dfine_dinov2_base_518_residual_token_v21/best.pt
RAW=outputs/dfine_best_stg2/val_fused_nms080.json

mkdir -p "$O/evaluations"
rm -f "$O/DONE" "$O/FAILED"
exec > >(tee "$O/pipeline.log") 2>&1
trap 'touch "$O/FAILED"' ERR

echo "[$(date '+%F %T')] classify proposals"
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 "$P" scripts/apply_dfine_crop_classifier.py \
  --predictions "$RAW" \
  --checkpoint "$BEST" \
  --dinov2-weights /root/autodl-tmp/weights/dinov2-base \
  --manifest splits/v1/split_manifest.csv \
  --classes configs/classes.yaml \
  --output "$O/val_keep.json" \
  --image-size 518 --tight-expand 1.0 --context-expand 1.8 --min-crop-size 128 \
  --min-det-score 0.03 --batch 256 --workers 8 \
  --score-mode keep --amp --amp-dtype bfloat16 --overwrite

for mode in det_obj det_obj_softcls det_obj_cls; do
  echo "[$(date '+%F %T')] rescore $mode"
  "$P" scripts/rescore_dfine_crop_predictions.py \
    --input "$O/val_keep.json" --output "$O/val_${mode}.json" \
    --mode "$mode" --overwrite
done

for mode in keep det_obj det_obj_softcls det_obj_cls; do
  echo "[$(date '+%F %T')] evaluate $mode"
  "$P" scripts/evaluate_predictions.py \
    --predictions "$O/val_${mode}.json" \
    --manifest splits/v1/split_manifest.csv --classes configs/classes.yaml --split val \
    --score-threshold 0.001 \
    --output-dir "$O/evaluations/$mode" --overwrite
done

touch "$O/DONE"
echo "[$(date '+%F %T')] complete"
