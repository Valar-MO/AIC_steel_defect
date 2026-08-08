#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/AIC_steel_defect
OUT=/root/autodl-tmp/D-FINE/output/dfine_hgnetv2_l_hierarchical_objectness_dual_v1
CHECKPOINT=${1:-/root/autodl-tmp/D-FINE/output/dfine_hgnetv2_l_hierarchical_aic/best_stg1.pth}
mkdir -p "$OUT"
bash scripts/run_hierarchical_objectness_dual_train.sh "$CHECKPOINT" \
  2>&1 | tee "$OUT/train.log"

