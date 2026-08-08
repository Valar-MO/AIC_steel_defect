#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/AIC_steel_defect
mkdir -p outputs/hierarchical_objectness_rank_v1
while tmux has-session -t obj_rank_v1 2>/dev/null; do
  sleep 30
done
bash scripts/run_hierarchical_objectness_rank_eval.sh \
  > outputs/hierarchical_objectness_rank_v1/eval.log 2>&1

