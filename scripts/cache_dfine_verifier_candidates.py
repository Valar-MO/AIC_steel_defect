#!/usr/bin/env python3
"""Cache fixed Hierarchical D-FINE predictions and final decoder query features.

The cache is the immutable candidate universe for the DINOv2 teacher upper-bound
experiment.  Boxes and top-1 classes never change between verifier variants.
"""

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


def parse_args() -> argparse.Namespace:
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
    parser.add_argument(
        "--min-selection",
        type=float,
        default=0.05,
        help="Immutable baseline Selection floor. It must cover the largest target recall.",
    )
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-batches", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_classes(path: Path) -> list[str]:
    names = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("names")
    if not isinstance(names, list) or len(names) != 9:
        raise ValueError("Expected exactly nine class names")
    return [str(name) for name in names]


def image_id(target: dict) -> int:
    value = target["image_id"]
    return int(value.flatten()[0].item()) if isinstance(value, torch.Tensor) else int(value)


def to_device(target: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in target.items()
    }


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.min_selection < 1.0:
        raise ValueError("min-selection must be in [0,1)")
    if args.batch < 1 or args.workers < 0:
        raise ValueError("batch must be positive and workers non-negative")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")

    classes = load_classes(args.classes)
    dfine_dir = args.dfine_dir.resolve()
    sys.path.insert(0, str(dfine_dir))
    from src.core import YAMLConfig  # pylint: disable=import-outside-toplevel

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = dfine_dir / config_path
    cfg = YAMLConfig(str(config_path), resume=str(args.checkpoint))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    loader_name = f"{args.split}_dataloader"
    loader_cfg = cfg.yaml_cfg[loader_name]
    loader_cfg["total_batch_size"] = args.batch
    loader_cfg["num_workers"] = args.workers
    loader_cfg["shuffle"] = False
    loader_cfg["drop_last"] = False
    if args.split == "train":
        # Fixed outputs require deterministic resize/normalization, not train augmentation.
        loader_cfg["dataset"]["transforms"] = copy.deepcopy(
            cfg.yaml_cfg["val_dataloader"]["dataset"]["transforms"]
        )

    annotation_path = Path(loader_cfg["dataset"]["ann_file"])
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    image_rows = {int(row["id"]): row for row in coco["images"]}
    annotations_by_image: dict[int, list[dict]] = {}
    for annotation in coco.get("annotations", []):
        if int(annotation.get("iscrowd", 0)):
            continue
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
    image_root = Path(loader_cfg["dataset"]["img_folder"])

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    model = cfg.model
    model.load_state_dict(state, strict=True)
    device = torch.device(args.device)
    model = model.deploy().to(device).eval()
    loader = getattr(cfg, loader_name)
    score_head = model.decoder.dec_score_head[-1]
    captured: list[torch.Tensor] = []

    def capture_query(_module, inputs):
        captured.append(inputs[0].detach())

    handle = score_head.register_forward_pre_hook(capture_query)
    features, boxes, selection_logits, class_logits = [], [], [], []
    candidate_image_index, query_indices = [], []
    max_ious, nearest_labels = [], []
    images = []
    started = time.time()
    try:
        for step, (batch_images, targets) in enumerate(loader, 1):
            if args.limit_batches is not None and step > args.limit_batches:
                break
            batch_images = batch_images.to(device, non_blocking=True)
            targets = [to_device(target, device) for target in targets]
            captured.clear()
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                outputs = model(batch_images)
            if not captured:
                raise RuntimeError("Final decoder score-head hook captured no query features")
            query_features = captured[-1]
            objectness_logits = outputs["pred_objectness_logits"].squeeze(-1)
            probabilities = objectness_logits.sigmoid()
            predicted_boxes = outputs["pred_boxes"]
            predicted_classes = outputs["pred_class_logits"]
            if query_features.shape[:2] != predicted_boxes.shape[:2]:
                raise RuntimeError("Query feature and prediction shapes do not align")

            for batch_index, target in enumerate(targets):
                coco_id = image_id(target)
                row = image_rows[coco_id]
                source_path = image_root / row["file_name"]
                if not source_path.is_file():
                    raise FileNotFoundError(source_path)
                width, height = int(row["width"]), int(row["height"])
                image_index = len(images)
                images.append({
                    "coco_image_id": coco_id,
                    "image_id": Path(row["file_name"]).name,
                    "source_path": str(source_path.resolve()),
                    "width": width,
                    "height": height,
                })

                keep = torch.where(probabilities[batch_index] >= args.min_selection)[0]
                if not len(keep):
                    continue
                normalized_xyxy = box_convert(
                    predicted_boxes[batch_index, keep], in_fmt="cxcywh", out_fmt="xyxy"
                )
                pixel_scale = torch.tensor(
                    [width, height, width, height],
                    dtype=normalized_xyxy.dtype,
                    device=normalized_xyxy.device,
                )
                pixel_boxes = normalized_xyxy * pixel_scale

                # Read GT directly from the COCO source in original-image pixels.  The
                # current validation transform deliberately leaves target boxes as
                # resized xyxy pixels, while training transforms normally convert them
                # to normalized cxcywh.  Using transformed targets here would therefore
                # make the cache label convention depend on the selected split.
                image_annotations = annotations_by_image.get(coco_id, [])
                if image_annotations:
                    target_xywh = torch.tensor(
                        [annotation["bbox"] for annotation in image_annotations],
                        dtype=torch.float32,
                        device=device,
                    )
                    target_xyxy = box_convert(target_xywh, in_fmt="xywh", out_fmt="xyxy")
                    target_labels = torch.tensor(
                        [int(annotation["category_id"]) for annotation in image_annotations],
                        dtype=torch.long,
                        device=device,
                    )
                    overlaps = box_iou(pixel_boxes.float(), target_xyxy)
                    best_iou, best_target = overlaps.max(dim=1)
                    best_label = target_labels[best_target]
                else:
                    best_iou = torch.zeros(len(keep), device=device)
                    best_label = torch.full(
                        (len(keep),), -1, dtype=torch.long, device=device
                    )

                features.append(query_features[batch_index, keep].cpu().to(torch.float16))
                boxes.append(pixel_boxes.cpu().to(torch.float32))
                selection_logits.append(
                    objectness_logits[batch_index, keep].cpu().to(torch.float16)
                )
                class_logits.append(
                    predicted_classes[batch_index, keep].cpu().to(torch.float16)
                )
                candidate_image_index.append(
                    torch.full((len(keep),), image_index, dtype=torch.int32)
                )
                query_indices.append(keep.cpu().to(torch.int16))
                max_ious.append(best_iou.cpu().to(torch.float16))
                nearest_labels.append(best_label.cpu().to(torch.int16))

            if step % 20 == 0 or step == len(loader):
                count = sum(len(item) for item in features)
                print(
                    f"cache split={args.split} step={step}/{len(loader)} "
                    f"images={len(images)} candidates={count}",
                    flush=True,
                )
    finally:
        handle.remove()

    def cat(rows, shape, dtype):
        return torch.cat(rows, dim=0) if rows else torch.empty(shape, dtype=dtype)

    feature_dim = int(score_head.class_weight.shape[1])
    cache = {
        "version": 1,
        "architecture": "fixed_hierarchical_dfine_verifier_candidates",
        "split": args.split,
        "classes": classes,
        "feature_dim": feature_dim,
        "checkpoint": str(args.checkpoint.resolve()),
        "config": str(config_path.resolve()),
        "annotation_path": str(annotation_path.resolve()),
        "min_selection": args.min_selection,
        "images": images,
        "query_features": cat(features, (0, feature_dim), torch.float16),
        "boxes_xyxy": cat(boxes, (0, 4), torch.float32),
        "selection_logits": cat(selection_logits, (0,), torch.float16),
        "class_logits": cat(class_logits, (0, len(classes)), torch.float16),
        "candidate_image_index": cat(candidate_image_index, (0,), torch.int32),
        "query_indices": cat(query_indices, (0,), torch.int16),
        "max_iou": cat(max_ious, (0,), torch.float16),
        "nearest_gt_label": cat(nearest_labels, (0,), torch.int16),
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()
    torch.save(cache, args.output)
    labels = cache["max_iou"]
    print(json.dumps({
        "output": str(args.output.resolve()),
        "split": args.split,
        "images": len(images),
        "candidates": len(labels),
        "positive_iou_ge_050": int((labels >= 0.5).sum()),
        "background_iou_lt_010": int((labels < 0.1).sum()),
        "ambiguous": int(((labels >= 0.1) & (labels < 0.5)).sum()),
        "feature_dim": feature_dim,
        "min_selection": args.min_selection,
        "elapsed_seconds": round(cache["elapsed_seconds"], 3),
    }, indent=2))


if __name__ == "__main__":
    main()
