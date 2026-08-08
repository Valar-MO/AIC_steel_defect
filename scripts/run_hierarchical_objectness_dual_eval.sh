#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/AIC_steel_defect
P=/root/miniconda3/envs/aic-steel/bin/python
TRAIN_DIR=${1:-/root/autodl-tmp/D-FINE/output/dfine_hgnetv2_l_hierarchical_objectness_dual_v1}
OUT_ROOT=${2:-outputs/hierarchical_objectness_dual_v1}
CONFIG=configs/dfine/custom/objects365/dfine_hgnetv2_l_hierarchical_objectness_dual_v1.yml

mkdir -p "$OUT_ROOT/raw_checkpoints"
"$P" scripts/install_hierarchical_dfine.py

for checkpoint in "$TRAIN_DIR"/checkpoint*.pth; do
  [[ -f "$checkpoint" ]] || continue
  stem=$(basename "$checkpoint" .pth)
  for variant in ema raw; do
    input="$checkpoint"
    if [[ "$variant" == raw ]]; then
      input="$OUT_ROOT/raw_checkpoints/${stem}_raw.pth"
      "$P" analysis/hierarchical_gtaware_e1_1/extract_raw_checkpoint.py \
        "$checkpoint" "$input"
    fi
    output="$OUT_ROOT/${stem}_${variant}"
    mkdir -p "$output"
    "$P" scripts/predict_hierarchical_dfine.py \
      --config "$CONFIG" --checkpoint "$input" \
      --manifest splits/v1/split_manifest.csv --classes configs/classes.yaml \
      --split val --mode whole --output "$output/raw_whole.json" \
      --imgsz 1024 --batch 32 --conf 0.001 --overwrite
    "$P" scripts/compare_hierarchical_score_modes.py \
      --predictions "$output/raw_whole.json" \
      --manifest splits/v1/split_manifest.csv --classes configs/classes.yaml \
      --split val --score-modes objectness defectness --score-threshold 0.10 \
      --match-iou 0.50 --background-iou 0.10 \
      --output-dir "$output/results" --overwrite
  done
done

mapfile -t candidates < <(find "$OUT_ROOT" -mindepth 2 -maxdepth 2 \
  -path '*/raw_whole.json' -type f | sort)
"$P" scripts/compare_objectness_rank_checkpoints.py \
  --baseline outputs/hierarchical_score_mode_diagnosis_v1/raw_whole.json \
  --candidates "${candidates[@]}" \
  --manifest splits/v1/split_manifest.csv --classes configs/classes.yaml \
  --split val --score-threshold 0.10 --match-iou 0.50 \
  --background-iou 0.10 --output "$OUT_ROOT/checkpoint_comparison.json"
