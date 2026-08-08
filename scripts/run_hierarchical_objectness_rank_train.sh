#!/usr/bin/env bash
set -euo pipefail

# Large 4K batches transferred by PyTorch workers can exceed the container's
# conservative per-process file-descriptor limit before the first GPU step.
ulimit -n 65535

PROJECT=/root/autodl-tmp/AIC_steel_defect
DFINE=/root/autodl-tmp/D-FINE
PYTHON=/root/miniconda3/envs/aic-steel/bin/python
CONFIG=configs/dfine/custom/objects365/dfine_hgnetv2_l_hierarchical_objectness_rank.yml
TUNING=${1:-$DFINE/output/dfine_hgnetv2_l_hierarchical_aic/best_stg1.pth}
OUTPUT=$DFINE/output/dfine_hgnetv2_l_hierarchical_objectness_rank_v1

cd "$PROJECT"
"$PYTHON" scripts/install_hierarchical_dfine.py

cd "$DFINE"
mkdir -p "$OUTPUT"
exec > >(tee "$OUTPUT/train.log") 2>&1
PYTHONUNBUFFERED=1 "$PYTHON" train.py -c "$CONFIG" -t "$TUNING" --use-amp --seed 42
