#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/autodl-tmp/AIC_steel_defect
DFINE=/root/autodl-tmp/D-FINE
PYTHON=/root/miniconda3/envs/aic-steel/bin/python
CONFIG=configs/dfine/custom/objects365/dfine_hgnetv2_l_hierarchical_gtaware_e1.yml
TUNING=$DFINE/output/dfine_hgnetv2_l_hierarchical_aic/best_stg1.pth
OUTPUT=$DFINE/output/dfine_hgnetv2_l_hierarchical_gtaware_e1

cd "$PROJECT"
"$PYTHON" scripts/install_hierarchical_dfine.py
"$PYTHON" scripts/build_gtaware_dfine_dataset.py --overwrite

cd "$DFINE"
mkdir -p "$OUTPUT"
exec > >(tee "$OUTPUT/train.log") 2>&1
PYTHONUNBUFFERED=1 "$PYTHON" train.py -c "$CONFIG" -t "$TUNING" --use-amp --seed 42
