#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/AIC_steel_defect
P=/root/miniconda3/envs/aic-steel/bin/python
O=outputs/conditional_head_ablation_v1
C=$O/cache
H=$O/heads
BEST=/root/autodl-tmp/runs/crop_cls/dfine_dinov2_base_518_residual_token_v21/best.pt
RAW=outputs/dfine_best_stg2/val_fused_nms080.json

mkdir -p "$C" "$H" "$O/evaluations"
rm -f "$O/DONE" "$O/FAILED"
exec > >(tee "$O/pipeline.log") 2>&1
trap 'touch "$O/FAILED"' ERR

cache_curated() {
  local split=$1
  echo "[$(date '+%F %T')] cache curated $split"
  PYTHONUNBUFFERED=1 "$P" scripts/cache_dfine_crop_features.py \
    --data /root/autodl-tmp/datasets/AIC_dfine_crop_dual_base_v1 --split "$split" \
    --checkpoint "$BEST" --output "$C/${split}.pt" \
    --image-size 518 --batch 256 --workers 8 --amp-dtype bfloat16 --overwrite
}

cache_curated train
cache_curated val

echo "[$(date '+%F %T')] cache full raw val proposals"
PYTHONUNBUFFERED=1 "$P" scripts/cache_dfine_crop_features.py \
  --predictions "$RAW" --manifest splits/v1/split_manifest.csv \
  --checkpoint "$BEST" --output "$C/raw_val_min003.pt" \
  --image-size 518 --min-det-score 0.03 --batch 256 --workers 8 \
  --amp-dtype bfloat16 --overwrite

echo "[$(date '+%F %T')] train cached-feature head ablations"
PYTHONUNBUFFERED=1 "$P" scripts/train_conditional_verifier_head.py \
  --train-cache "$C/train.pt" --val-cache "$C/val.pt" --output "$H" \
  --variants conditional conditional_margin conditional_hard \
  --hidden 256 --batch 1024 --epochs 40 --lr 0.001 --patience 8 \
  --lambda-ce 0.5 --lambda-margin 0.2 --margin 0.5 \
  --negative-focal-gamma 2.0 --positive-fraction 0.5 --hard-negative-fraction 0.75 \
  --overwrite

for variant in conditional conditional_margin conditional_hard; do
  echo "[$(date '+%F %T')] apply and evaluate $variant"
  "$P" scripts/apply_conditional_head_cache.py \
    --cache "$C/raw_val_min003.pt" --predictions "$RAW" \
    --checkpoint "$H/${variant}.pt" --output "$O/val_${variant}.json" --overwrite
  "$P" scripts/evaluate_predictions.py \
    --predictions "$O/val_${variant}.json" \
    --manifest splits/v1/split_manifest.csv --classes configs/classes.yaml --split val \
    --score-threshold 0.001 --output-dir "$O/evaluations/${variant}" --overwrite
done

touch "$O/DONE"
echo "[$(date '+%F %T')] complete"
