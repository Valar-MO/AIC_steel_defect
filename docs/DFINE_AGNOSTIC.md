# D-FINE-L class-agnostic AIC pipeline

This pipeline keeps the previously generated mixed dataset and changes only
the detector target: all nine defect classes become one zero-based `defect`
category. The original full-image validation split is retained.

## 1. Convert existing mixed labels (images are not copied)

```bash
cd /root/autodl-tmp/AIC_steel_defect
python scripts/convert_mixed_yolo_to_coco_agnostic.py
```

## 2. Install project configs into the official D-FINE checkout

```bash
python scripts/setup_dfine_aic.py --dfine-dir /root/autodl-tmp/D-FINE
```

## 3. Train

Run from the official D-FINE checkout. `-t` is required for pretrained tuning;
do not use `-r`, which means resuming a same-architecture training run.

```bash
cd /root/autodl-tmp/D-FINE
CUDA_VISIBLE_DEVICES=0 python train.py \
  -c configs/dfine/custom/objects365/dfine_hgnetv2_l_obj2aic_defect.yml \
  -t /root/autodl-tmp/weights/dfine/dfine_l_obj2coco_e25.pth \
  --use-amp --seed 42
```

The first configuration uses 1024×1024 input and total batch size 4. Validation
contains only the 640 original full images. Mixed training contains the old
whole images, regular positive tiles, rare-centered tiles, and random
background tiles. The recorded old dataset did not contain hard negatives.

## 4. Whole-image and tiled inference

Run both modes at a deliberately low score threshold:

```bash
cd /root/autodl-tmp/AIC_steel_defect
python scripts/predict_dfine_agnostic.py \
  --mode whole --checkpoint /path/to/best_stg2.pth \
  --output outputs/dfine/val_whole.json

python scripts/predict_dfine_agnostic.py \
  --mode tiles --checkpoint /path/to/best_stg2.pth \
  --tile-size 2048 --overlap 512 \
  --output outputs/dfine/val_tiles.json
```

Fuse with the existing merger. The initial recall audit keeps border boxes;
border penalties should only be considered after measuring their TP/FN effect.

```bash
python scripts/merge_predictions.py \
  --whole outputs/dfine/val_whole.json \
  --tiles outputs/dfine/val_tiles.json \
  --classes configs/defect.yaml --min-score 0.001 \
  --tile-border-policy keep --nms-iou 0.50 --max-det 1000 \
  --output outputs/dfine/val_fused.json
```

## 5. Proposal coverage gate

```bash
python scripts/evaluate_agnostic_proposals.py \
  --predictions outputs/dfine/val_fused.json \
  --output-dir outputs/dfine/coverage --overwrite
```

Compare `coverage_summary.csv` and `coverage_by_group.csv` against the existing
DINOv2-RTDETR proposal audit. Proceed to the nine-class crop classifier only if
the fused D-FINE proposals improve recall, especially at IoU 0.5 and for tiny
objects, without requiring an impractical number of proposals per image.
