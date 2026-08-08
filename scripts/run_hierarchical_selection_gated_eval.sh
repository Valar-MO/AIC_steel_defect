#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/AIC_steel_defect
P=/root/miniconda3/envs/aic-steel/bin/python
TRAIN_DIR=${1:-/root/autodl-tmp/D-FINE/output/dfine_hgnetv2_l_hierarchical_selection_gated_v1}
OUT_ROOT=${2:-outputs/hierarchical_selection_gated_v1}
CONFIG=configs/dfine/custom/objects365/dfine_hgnetv2_l_hierarchical_selection_gated_v1.yml

mkdir -p "$OUT_ROOT"
"$P" scripts/install_hierarchical_dfine.py

# Evaluate every EMA checkpoint. Raw-vs-EMA is run only for the selected best
# epoch afterwards, avoiding needless duplicate 246 MiB checkpoint files.
for checkpoint in "$TRAIN_DIR"/checkpoint*.pth; do
  [[ -f "$checkpoint" ]] || continue
  stem=$(basename "$checkpoint" .pth)
  output="$OUT_ROOT/${stem}_ema"
  mkdir -p "$output"
  if [[ -f "$output/results/comparison_summary.json" ]]; then
    find "$output/results" -name predictions.json -type f -delete
    continue
  fi
  "$P" scripts/predict_hierarchical_dfine.py \
    --config "$CONFIG" --checkpoint "$checkpoint" \
    --manifest splits/v1/split_manifest.csv --classes configs/classes.yaml \
    --split val --mode whole --output "$output/raw_whole.json" \
    --imgsz 1024 --batch 32 --conf 0.001 --overwrite
  "$P" scripts/compare_hierarchical_score_modes.py \
    --predictions "$output/raw_whole.json" \
    --manifest splits/v1/split_manifest.csv --classes configs/classes.yaml \
    --split val \
    --score-modes objectness selection selection_quality \
      selection_defectness defectness joint \
    --score-threshold 0.10 --match-iou 0.50 --background-iou 0.10 \
    --output-dir "$output/results" --overwrite
  # These six rescored payloads are fully reproducible from raw_whole.json
  # and otherwise consume ~630 MiB per checkpoint.
  find "$output/results" -name predictions.json -type f -delete
done

mapfile -t candidates < <(find "$OUT_ROOT" -mindepth 2 -maxdepth 2 \
  -path '*/raw_whole.json' -type f | sort)
"$P" scripts/compare_objectness_rank_checkpoints.py \
  --baseline outputs/hierarchical_score_mode_diagnosis_v1/raw_whole.json \
  --candidates "${candidates[@]}" \
  --manifest splits/v1/split_manifest.csv --classes configs/classes.yaml \
  --split val --score-threshold 0.10 --match-iou 0.50 \
  --background-iou 0.10 --output "$OUT_ROOT/checkpoint_comparison.json"
