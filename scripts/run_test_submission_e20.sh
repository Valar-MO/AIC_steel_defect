#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/AIC_steel_defect
P=/root/miniconda3/envs/aic-steel/bin/python
DFINE=/root/autodl-tmp/D-FINE
CFG=configs/dfine/custom/objects365/dfine_hgnetv2_l_obj2aic_defect.yml
DET_CKPT=$DFINE/output/dfine_hgnetv2_l_obj2aic_defect/best_stg2.pth
CLS_CKPT=/root/autodl-tmp/runs/crop_cls/dfine_dinov2_base_518_dual_v1/best.pt
TEST=/root/autodl-tmp/datasets/AIC_steel_defect/initial_test
O=outputs/test_dfine_crop_e20_detobj010
S=/root/autodl-tmp/submissions/dfine_crop_e20_detobj010

mkdir -p "$O" "$S"
rm -f "$O/DONE" "$O/FAILED"
trap 'touch "$O/FAILED"' ERR

echo "[$(date '+%F %T')] D-FINE whole"
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 "$P" scripts/predict_dfine_agnostic.py \
  --dfine-dir "$DFINE" --config "$CFG" --checkpoint "$DET_CKPT" \
  --image-dir "$TEST" --mode whole --imgsz 1024 --batch 32 \
  --conf 0.001 --max-det-per-view 300 --output "$O/test_whole.json" --overwrite

echo "[$(date '+%F %T')] D-FINE tiles"
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 "$P" scripts/predict_dfine_agnostic.py \
  --dfine-dir "$DFINE" --config "$CFG" --checkpoint "$DET_CKPT" \
  --image-dir "$TEST" --mode tiles --imgsz 1024 --tile-size 2048 --overlap 512 \
  --batch 32 --conf 0.001 --max-det-per-view 300 \
  --output "$O/test_tiles.json" --overwrite

echo "[$(date '+%F %T')] fuse"
"$P" scripts/merge_predictions.py \
  --whole "$O/test_whole.json" --tiles "$O/test_tiles.json" \
  --classes configs/defect.yaml --min-score 0.001 --nms-iou 0.80 \
  --tile-border-policy keep --max-det 1000 \
  --output "$O/test_fused_nms080.json" --report "$O/test_fused_nms080_report.json" --overwrite

echo "[$(date '+%F %T')] DINOv2 classify"
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 "$P" scripts/apply_dfine_crop_classifier.py \
  --predictions "$O/test_fused_nms080.json" --checkpoint "$CLS_CKPT" \
  --dinov2-weights /root/autodl-tmp/weights/dinov2-base \
  --classes configs/classes.yaml --output "$O/test_keep.json" \
  --image-size 518 --tight-expand 1.0 --context-expand 1.8 --min-crop-size 128 \
  --min-det-score 0.03 --batch 128 --workers 8 \
  --score-mode keep --amp --amp-dtype bfloat16 --overwrite

echo "[$(date '+%F %T')] det_obj rescore"
"$P" scripts/rescore_dfine_crop_predictions.py \
  --input "$O/test_keep.json" --output "$O/test_det_obj.json" \
  --mode det_obj --overwrite

echo "[$(date '+%F %T')] build submission"
"$P" build_submission.py \
  --raw "$O/test_det_obj.json" --classes configs/classes.yaml \
  --min-score 0.10 --output "$S/submission.json" \
  --report "$S/submission_report.json" --overwrite

echo "[$(date '+%F %T')] validate submission"
"$P" validate_submission.py \
  --submission "$S/submission.json" --raw "$O/test_det_obj.json" \
  --test-dir "$TEST" --classes configs/classes.yaml \
  --report "$S/validation_report.json"

touch "$O/DONE"
echo "[$(date '+%F %T')] complete: $S/submission.json"
