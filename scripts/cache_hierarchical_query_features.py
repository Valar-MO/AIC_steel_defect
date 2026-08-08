#!/usr/bin/env python3
"""Cache final-decoder query features and class-agnostic matches from D-FINE."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.nn import functional as F
from torchvision.ops import box_convert, box_iou


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dfine-dir", type=Path, default=Path("/root/autodl-tmp/D-FINE"))
    parser.add_argument(
        "--config",
        default="configs/dfine/custom/objects365/dfine_hgnetv2_l_hierarchical_aic.yml",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path,
                        help="Also write baseline whole-image predictions.")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-batches", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolved_config_path(dfine_dir: Path, config: str) -> Path:
    path = Path(config)
    return path if path.is_absolute() else dfine_dir / path


def load_classes(path: Path) -> list[str]:
    names = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("names")
    if not isinstance(names, list) or len(names) != 9:
        raise ValueError("Expected exactly nine class names")
    return [str(name) for name in names]


def load_coco_image_names(annotation_path: Path) -> dict[int, str]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    return {int(row["id"]): Path(row["file_name"]).name for row in payload["images"]}


def target_to_device(target: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in target.items()
    }


def image_identifier(target: dict) -> int:
    value = target["image_id"]
    return int(value.flatten()[0].item()) if isinstance(value, torch.Tensor) else int(value)


def matched_ious(outputs: dict, targets: list[dict], indices) -> list[torch.Tensor]:
    rows = []
    for batch_index, (source, target_index) in enumerate(indices):
        if not len(source):
            rows.append(torch.empty(0, device=outputs["pred_boxes"].device))
            continue
        predicted = box_convert(
            outputs["pred_boxes"][batch_index, source], in_fmt="cxcywh", out_fmt="xyxy"
        )
        expected = box_convert(
            targets[batch_index]["boxes"][target_index], in_fmt="cxcywh", out_fmt="xyxy"
        )
        rows.append(box_iou(predicted, expected).diag())
    return rows


def main():
    args = parse_args()
    if args.batch <= 0 or args.workers < 0:
        raise ValueError("batch must be positive and workers non-negative")
    if not 0 <= args.conf <= 1:
        raise ValueError("conf must be in [0,1]")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")
    if args.predictions and args.predictions.exists() and not args.overwrite:
        raise FileExistsError(f"{args.predictions} exists; pass --overwrite")
    classes = load_classes(args.classes)
    dfine_dir = args.dfine_dir.resolve()
    sys.path.insert(0, str(dfine_dir))
    from src.core import YAMLConfig  # pylint: disable=import-outside-toplevel

    config_path = resolved_config_path(dfine_dir, args.config)
    cfg = YAMLConfig(str(config_path), resume=str(args.checkpoint))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    loader_name = f"{args.split}_dataloader"
    loader_cfg = cfg.yaml_cfg[loader_name]
    loader_cfg["total_batch_size"] = args.batch
    loader_cfg["num_workers"] = args.workers
    loader_cfg["shuffle"] = False
    loader_cfg["drop_last"] = False
    # Cache deterministic features for both splits.  Reuse the exact validation
    # resize/normalization instead of random training augmentation.
    if args.split == "train":
        loader_cfg["dataset"]["transforms"] = copy.deepcopy(
            cfg.yaml_cfg["val_dataloader"]["dataset"]["transforms"]
        )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    model = cfg.model
    model.load_state_dict(state, strict=True)
    device = torch.device(args.device)
    model = model.deploy().to(device).eval()
    matcher = cfg.criterion.matcher.to(device).eval()
    loader = getattr(cfg, loader_name)
    postprocessor = cfg.postprocessor.to(device).eval()

    annotation_path = Path(loader_cfg["dataset"]["ann_file"])
    image_names = load_coco_image_names(annotation_path)
    score_head = model.decoder.dec_score_head[-1]
    captured: list[torch.Tensor] = []

    def capture_input(_module, inputs):
        captured.append(inputs[0].detach())

    handle = score_head.register_forward_pre_hook(capture_input)
    matched_features = []
    matched_targets = []
    matched_baseline_logits = []
    matched_objectness = []
    matched_iou_rows = []
    matched_image_ids = []
    prediction_features = []
    prediction_locations = []
    prediction_baseline_logits = []
    prediction_scores = []
    prediction_labels = []
    prediction_boxes = []
    prediction_image_names = []
    prediction_query_indices = []
    records = []
    processed_images = 0
    started = time.time()

    try:
        for step, (images, targets) in enumerate(loader, 1):
            if args.limit_batches is not None and step > args.limit_batches:
                break
            images = images.to(device, non_blocking=True)
            targets = [target_to_device(target, device) for target in targets]
            captured.clear()
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                outputs = model(images)
            if not captured:
                raise RuntimeError("Final decoder score-head hook did not capture query features")
            features = captured[-1]
            if features.shape[:2] != outputs["pred_boxes"].shape[:2]:
                raise RuntimeError(
                    f"Feature/query shape mismatch: {tuple(features.shape)} vs "
                    f"{tuple(outputs['pred_boxes'].shape)}"
                )
            assignment = matcher(outputs, targets)["indices"]
            ious = matched_ious(outputs, targets, assignment)
            objectness = outputs["pred_objectness_logits"].sigmoid().squeeze(-1)
            class_logits = outputs["pred_class_logits"]
            class_probability = F.softmax(class_logits.float(), dim=-1)

            for batch_index, ((source, target_index), target, match_iou) in enumerate(
                zip(assignment, targets, ious, strict=True)
            ):
                labels = target["labels"][target_index].long()
                if len(labels) and (labels.min() < 0 or labels.max() >= len(classes)):
                    raise ValueError(f"Invalid target labels: {labels.tolist()}")
                matched_features.append(features[batch_index, source].cpu().to(torch.float16))
                matched_targets.append(labels.cpu())
                matched_baseline_logits.append(
                    class_logits[batch_index, source].cpu().to(torch.float16)
                )
                matched_objectness.append(objectness[batch_index, source].cpu().to(torch.float16))
                matched_iou_rows.append(match_iou.cpu().to(torch.float16))
                image_id = image_identifier(target)
                matched_image_ids.extend([image_id] * len(source))

                if image_id not in image_names:
                    raise KeyError(f"COCO image id {image_id} missing from annotations")
                original_size = target["orig_size"].reshape(1, 2)
                processed = postprocessor(
                    {key: value[batch_index:batch_index + 1]
                     for key, value in outputs.items() if isinstance(value, torch.Tensor)},
                    original_size,
                )[0]
                scores = processed["scores"]
                order = scores.argsort(descending=True)
                record_index = len(records)
                predictions = []
                for query_index in order.tolist():
                    score = float(scores[query_index])
                    if score < args.conf:
                        continue
                    label = int(processed["labels"][query_index])
                    probability = float(class_probability[batch_index, query_index, label])
                    prediction_index = len(predictions)
                    predictions.append({
                        "class_id": label,
                        "category_name": classes[label],
                        "bbox_xyxy": [float(value) for value in
                                      processed["boxes"][query_index].cpu().tolist()],
                        "score": score,
                        "objectness": score,
                        "class_probability": probability,
                        "source": "whole",
                        "query_index": int(query_index),
                    })
                    prediction_features.append(
                        features[batch_index, query_index].cpu().to(torch.float16)
                    )
                    prediction_baseline_logits.append(
                        class_logits[batch_index, query_index].cpu().to(torch.float16)
                    )
                    prediction_locations.append((record_index, prediction_index))
                    prediction_scores.append(score)
                    prediction_labels.append(label)
                    prediction_boxes.append(
                        processed["boxes"][query_index].cpu().to(torch.float32)
                    )
                    prediction_image_names.append(image_names[image_id])
                    prediction_query_indices.append(int(query_index))
                records.append({
                    "image_id": image_names[image_id],
                    "coco_image_id": image_id,
                    "predictions": predictions,
                })
            processed_images += len(images)
            if step % 20 == 0 or step == len(loader):
                print(
                    f"cache split={args.split} step={step}/{len(loader)} "
                    f"images={processed_images}", flush=True
                )
    finally:
        handle.remove()

    def concatenate(rows, shape, dtype):
        return torch.cat(rows) if rows else torch.empty(shape, dtype=dtype)

    def stack(rows, shape, dtype):
        return torch.stack(rows) if rows else torch.empty(shape, dtype=dtype)

    cache = {
        "version": 1,
        "architecture": "hierarchical_dfine_final_decoder_queries",
        "split": args.split,
        "classes": classes,
        "feature_dim": int(score_head.class_weight.shape[1]),
        "checkpoint": str(args.checkpoint.resolve()),
        "config": str(config_path.resolve()),
        "image_count": processed_images,
        "matched_features": concatenate(
            matched_features, (0, score_head.class_weight.shape[1]), torch.float16
        ),
        "matched_class_targets": concatenate(matched_targets, (0,), torch.long),
        "matched_baseline_class_logits": concatenate(
            matched_baseline_logits, (0, len(classes)), torch.float16
        ),
        "matched_objectness": concatenate(matched_objectness, (0,), torch.float16),
        "matched_iou": concatenate(matched_iou_rows, (0,), torch.float16),
        "matched_image_ids": torch.tensor(matched_image_ids, dtype=torch.long),
        "detector_class_weight": score_head.class_weight.detach().cpu().float(),
        "detector_class_bias": score_head.class_bias.detach().cpu().float(),
        "elapsed_seconds": time.time() - started,
    }
    cache.update({
        "prediction_features": stack(
            prediction_features, (0, score_head.class_weight.shape[1]), torch.float16
        ),
        "prediction_baseline_class_logits": stack(
            prediction_baseline_logits, (0, len(classes)), torch.float16
        ),
        "prediction_locations": torch.tensor(prediction_locations, dtype=torch.long),
        "prediction_scores": torch.tensor(prediction_scores, dtype=torch.float32),
        "prediction_labels": torch.tensor(prediction_labels, dtype=torch.long),
        "prediction_boxes": stack(prediction_boxes, (0, 4), torch.float32),
        "prediction_image_names": prediction_image_names,
        "prediction_query_indices": torch.tensor(prediction_query_indices, dtype=torch.long),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.overwrite:
        args.output.unlink()
    torch.save(cache, args.output)

    if args.predictions:
        if args.predictions.exists() and args.overwrite:
            args.predictions.unlink()
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": {
                "classes": classes,
                "architecture": "hierarchical_dfine",
                "checkpoint": str(args.checkpoint.resolve()),
                "split": "val",
                "mode": "whole",
                "image_count": len(records),
                "confidence": args.conf,
                "prediction_count": sum(len(row["predictions"]) for row in records),
                "score_mode": "objectness",
            },
            "images": records,
        }
        args.predictions.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "predictions": str(args.predictions.resolve()) if args.predictions else None,
        "split": args.split,
        "images": processed_images,
        "matched_queries": len(cache["matched_features"]),
        "cached_predictions": len(cache.get("prediction_features", [])),
        "feature_dim": cache["feature_dim"],
        "elapsed_seconds": round(cache["elapsed_seconds"], 3),
    }, indent=2))


if __name__ == "__main__":
    main()
