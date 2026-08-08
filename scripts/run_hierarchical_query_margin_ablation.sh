#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/autodl-tmp/AIC_steel_defect
PYTHON=/root/miniconda3/envs/aic-steel/bin/python
CHECKPOINT=/root/autodl-tmp/D-FINE/output/dfine_hgnetv2_l_hierarchical_aic/best_stg1.pth
CONFIG=configs/dfine/custom/objects365/dfine_hgnetv2_l_hierarchical_aic.yml
ROOT=outputs/hierarchical_query_margin_v1

cd "$PROJECT"
"$PYTHON" scripts/install_hierarchical_dfine.py
mkdir -p "$ROOT/cache" "$ROOT/heads" "$ROOT/predictions" "$ROOT/evaluations"

"$PYTHON" scripts/cache_hierarchical_query_features.py \
  --config "$CONFIG" --checkpoint "$CHECKPOINT" --split train \
  --output "$ROOT/cache/train.pt" --batch 32 --workers 0 --overwrite

"$PYTHON" scripts/cache_hierarchical_query_features.py \
  --config "$CONFIG" --checkpoint "$CHECKPOINT" --split val \
  --output "$ROOT/cache/val.pt" --predictions "$ROOT/predictions/baseline.json" \
  --batch 32 --workers 0 --conf 0.001 --overwrite

"$PYTHON" scripts/evaluate_predictions.py \
  --predictions "$ROOT/predictions/baseline.json" \
  --manifest splits/v1/split_manifest.csv --classes configs/classes.yaml --split val \
  --score-threshold 0.1 --iou-threshold 0.5 \
  --output-dir "$ROOT/baseline_evaluation" --overwrite

"$PYTHON" scripts/train_hierarchical_query_classifier.py \
  --train-cache "$ROOT/cache/train.pt" --val-cache "$ROOT/cache/val.pt" \
  --output "$ROOT/heads" --variants H0 H1 H2 --seeds 42 43 44 \
  --batch 512 --epochs 40 --lr 0.0005 --weight-decay 0.000125 \
  --label-smoothing 0.05 --margin 0.2 --margin-weight 0.2 \
  --patience 8 --overwrite

for variant in H0 H1 H2; do
  for seed in 42 43 44; do
    name="${variant}_seed${seed}"
    "$PYTHON" scripts/apply_hierarchical_query_classifier.py \
      --cache "$ROOT/cache/val.pt" \
      --predictions "$ROOT/predictions/baseline.json" \
      --checkpoint "$ROOT/heads/${name}.pt" \
      --output "$ROOT/predictions/${name}.json" --overwrite
    "$PYTHON" scripts/evaluate_predictions.py \
      --predictions "$ROOT/predictions/${name}.json" \
      --manifest splits/v1/split_manifest.csv --classes configs/classes.yaml --split val \
      --score-threshold 0.1 --iou-threshold 0.5 \
      --output-dir "$ROOT/evaluations/${name}" --overwrite
  done
done

"$PYTHON" scripts/summarize_hierarchical_query_ablation.py \
  --root "$ROOT" --output "$ROOT/summary.json"
