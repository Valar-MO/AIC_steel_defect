# DINOv2 + RT-DETR competition pipeline

## Train

```bash
python scripts/train_dinov2_rtdetr.py \
  --config configs/dinov2_rtdetr_vits14.yaml
```

Useful modes:

- `--resume RUN/last.pt` restores model, optimizer, scheduler, scaler and early-stop state.
- `--init-checkpoint RUN/best_loss.pt` starts a new optimization stage from model weights only.
- `--unfreeze-last-blocks 4 --backbone-lr 1e-5` fine-tunes the final DINOv2 blocks.
- `--save-every 5` writes `epoch_005.pt`, `epoch_010.pt`, and so on.

The trainer writes atomic `last.pt`, `best_loss.pt`, periodic checkpoints,
`history.jsonl`, `history.csv`, `loss_curve.png`, and peak GPU memory per epoch.

## Tiled validation

```bash
python scripts/predict_dinov2_rtdetr_tiles.py \
  --checkpoint RUN/epoch_010.pt \
  --dinov2-weights /root/autodl-tmp/weights/dinov2-small \
  --rtdetr-weights /root/autodl-tmp/weights/rtdetr-r50vd-hf \
  --output /root/autodl-tmp/evaluations/epoch_010.json \
  --tile-size 2048 --overlap 512 --imgsz 1008 --batch 24 --conf 0.001

python scripts/evaluate_predictions.py \
  --predictions /root/autodl-tmp/evaluations/epoch_010.json \
  --score-threshold 0.05 \
  --output-dir /root/autodl-tmp/evaluations/epoch_010_metrics
```

The report contains mAP50, mAP50-95, Precision, Recall, F1, F2, per-class
AP/Recall, and a confidence-threshold sweep.

## Select a checkpoint by an official metric

```bash
python scripts/select_best_dinov2_checkpoint.py \
  --checkpoints RUN/epoch_*.pt \
  --dinov2-weights /root/autodl-tmp/weights/dinov2-small \
  --rtdetr-weights /root/autodl-tmp/weights/rtdetr-r50vd-hf \
  --output-dir /root/autodl-tmp/evaluations/checkpoint_selection \
  --metric map50_95
```

This produces `checkpoint_ranking.json` and `best_map50_95.pt`.

## Test inference and official JSON

```bash
python scripts/predict_dinov2_rtdetr_tiles.py \
  --checkpoint /path/to/best_map50_95.pt \
  --dinov2-weights /root/autodl-tmp/weights/dinov2-small \
  --rtdetr-weights /root/autodl-tmp/weights/rtdetr-r50vd-hf \
  --image-dir /path/to/test/images \
  --output /root/autodl-tmp/submissions/dinov2/raw_predictions.json \
  --tile-size 2048 --overlap 512 --imgsz 1008 --batch 24 --conf 0.001

python scripts/build_submission.py \
  --raw /root/autodl-tmp/submissions/dinov2/raw_predictions.json \
  --output /root/autodl-tmp/submissions/dinov2/submission.json \
  --report /root/autodl-tmp/submissions/dinov2/submission_report.json \
  --min-score 0.05 --overwrite
```
