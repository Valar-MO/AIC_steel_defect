#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/AIC_steel_defect
P=/root/miniconda3/envs/aic-steel/bin/python
BASE=outputs/conditional_head_ablation_v1
O=outputs/proposal_relation_v1
RAW=outputs/dfine_best_stg2/val_fused_nms080.json
DATA=/root/autodl-tmp/datasets/AIC_dfine_crop_dual_base_v1

mkdir -p "$O/heads" "$O/predictions" "$O/evaluations"
rm -f "$O/DONE" "$O/FAILED"
exec > >(tee "$O/pipeline.log") 2>&1
trap 'touch "$O/FAILED"' ERR

"$P" scripts/train_proposal_relation_head.py \
  --train-cache "$BASE/cache/train.pt" --val-cache "$BASE/cache/val.pt" \
  --data "$DATA" --output "$O/heads" --dimension 256 --heads 8 --layers 2 \
  --batch-images 16 --max-proposals 64 --epochs 24 --lr 0.0003 \
  --lambda-class 0.5 --lambda-rank 0.25 --lambda-residual 0.001 \
  --hard-negative-fraction 0.25 --overwrite

# Epoch zero must reproduce V2.1 exactly.  Evaluate every fourth epoch plus the
# curated best on all 78,848 raw proposals; choose architecture by real AP, not crop F1.
for checkpoint in "$O/heads/epoch_000.pt" "$O/heads/epoch_004.pt" \
                  "$O/heads/epoch_008.pt" "$O/heads/epoch_012.pt" \
                  "$O/heads/epoch_016.pt" "$O/heads/epoch_020.pt" \
                  "$O/heads/epoch_024.pt" "$O/heads/best_curated.pt"; do
  name=$(basename "$checkpoint" .pt)
  "$P" scripts/apply_proposal_relation_cache.py \
    --cache "$BASE/cache/raw_val_min003.pt" --predictions "$RAW" \
    --checkpoint "$checkpoint" --output "$O/predictions/${name}.json" \
    --batch-images 16 --max-proposals 256 --overwrite
  "$P" scripts/evaluate_predictions.py \
    --predictions "$O/predictions/${name}.json" \
    --manifest splits/v1/split_manifest.csv --classes configs/classes.yaml --split val \
    --score-threshold 0.1 --output-dir "$O/evaluations/${name}" --overwrite
done

"$P" -c 'import json,pathlib; root=pathlib.Path("outputs/proposal_relation_v1/evaluations"); rows=[]
for p in root.glob("*/evaluation_report.json"):
 r=json.loads(p.read_text())["overall"]; rows.append({"checkpoint":p.parent.name,**r})
rows.sort(key=lambda x:x["map50"],reverse=True); out={"ranking":rows,"best":rows[0]}; pathlib.Path("outputs/proposal_relation_v1/formal_ranking.json").write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))'

touch "$O/DONE"

