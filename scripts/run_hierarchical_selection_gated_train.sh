#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/AIC_steel_defect
ulimit -n 65535
P=/root/miniconda3/envs/aic-steel/bin/python
DFINE=/root/autodl-tmp/D-FINE
CONFIG=configs/dfine/custom/objects365/dfine_hgnetv2_l_hierarchical_selection_gated_v1.yml
CHECKPOINT=${1:-$DFINE/output/dfine_hgnetv2_l_hierarchical_aic/best_stg1.pth}

"$P" scripts/install_hierarchical_dfine.py
cd "$DFINE"
"$P" train.py -c "$CONFIG" -t "$CHECKPOINT" --use-amp --seed 42
