#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/autodl-tmp/AIC_steel_defect
DFINE=/root/autodl-tmp/D-FINE
PYTHON=/root/miniconda3/envs/aic-steel/bin/python
CONFIG=configs/dfine/custom/objects365/dfine_hgnetv2_l_hierarchical_p2_e2.yml
SOURCE=$DFINE/output/dfine_hgnetv2_l_hierarchical_aic/best_stg1.pth
OUTPUT=$DFINE/output/dfine_hgnetv2_l_hierarchical_p2_e2
MIGRATED=$OUTPUT/initial_p2_migrated.pth

cd "$PROJECT"
"$PYTHON" scripts/install_hierarchical_dfine.py
mkdir -p "$OUTPUT"
"$PYTHON" scripts/migrate_hierarchical_dfine_p2.py \
  --source "$SOURCE" --output "$MIGRATED" --overwrite

cd "$DFINE"
exec > >(tee "$OUTPUT/train.log") 2>&1
PYTHONUNBUFFERED=1 "$PYTHON" train.py -c "$CONFIG" -t "$MIGRATED" --use-amp --seed 42
