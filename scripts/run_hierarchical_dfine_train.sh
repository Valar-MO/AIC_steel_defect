#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/autodl-tmp/AIC_steel_defect
DFINE=/root/autodl-tmp/D-FINE
PYTHON=/root/miniconda3/envs/aic-steel/bin/python
CONFIG=configs/dfine/custom/objects365/dfine_hgnetv2_l_hierarchical_aic.yml
TUNING=$DFINE/output/dfine_hgnetv2_l_obj2aic_defect/best_stg2.pth
OUTPUT=$DFINE/output/dfine_hgnetv2_l_hierarchical_aic

cd "$PROJECT"
"$PYTHON" scripts/install_hierarchical_dfine.py
"$PYTHON" scripts/convert_mixed_yolo_to_coco_hierarchical.py --overwrite

cd "$DFINE"
mkdir -p "$OUTPUT"
exec > >(tee "$OUTPUT/train.log") 2>&1
PYTHONUNBUFFERED=1 "$PYTHON" train.py -c "$CONFIG" -t "$TUNING" --use-amp --seed 42

