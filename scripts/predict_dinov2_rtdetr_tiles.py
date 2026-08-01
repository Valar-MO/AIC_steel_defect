#!/usr/bin/env python3
"""Sliding-window inference for a DINOv2 + Transformers RT-DETR checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision.ops import batched_nms
from torchvision.transforms.functional import pil_to_tensor


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models import build_dinov2_rtdetr


MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dinov2-weights", type=Path, required=True)
    parser.add_argument("--rtdetr-weights", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("splits/v1/split_manifest.csv"))
    parser.add_argument("--image-dir", type=Path,
                        help="Unlabeled test image directory; overrides manifest/split.")
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=2048)
    parser.add_argument("--overlap", type=int, default=512)
    parser.add_argument("--imgsz", type=int, default=1008)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--max-det-per-tile", type=int, default=300)
    parser.add_argument("--max-det-per-image", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-images", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_classes(path: Path) -> list[str]:
    names = yaml.safe_load(path.read_text(encoding="utf-8")).get("names")
    if not isinstance(names, list) or not names:
        raise ValueError(f"Invalid classes file: {path}")
    return [str(name) for name in names]


def load_images(manifest: Path, split: str, limit: int | None,
                image_dir: Path | None = None) -> list[Path]:
    if image_dir is not None:
        suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        paths = sorted(path for path in image_dir.rglob("*")
                       if path.is_file() and path.suffix.lower() in suffixes)
        if not paths:
            raise ValueError(f"No images under {image_dir}")
        names = [path.name for path in paths]
        if len(names) != len(set(names)):
            raise ValueError("Image basenames must be unique")
        return paths[:limit] if limit else paths
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        paths = [Path(row["image_path"]) for row in csv.DictReader(handle) if row["split"] == split]
    if not paths:
        raise ValueError(f"No {split!r} images in {manifest}")
    return paths[:limit] if limit else paths


def axis_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def make_tiles(width: int, height: int, tile_size: int, overlap: int) -> list[tuple[int, int, int, int]]:
    stride = tile_size - overlap
    return [
        (x, y, min(x + tile_size, width), min(y + tile_size, height))
        for y in axis_starts(height, tile_size, stride)
        for x in axis_starts(width, tile_size, stride)
    ]


def preprocess(images: list[Image.Image], image_size: int) -> torch.Tensor:
    tensors = []
    for image in images:
        resized = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        tensor = pil_to_tensor(resized).float().div_(255.0)
        tensors.append((tensor - MEAN) / STD)
    return torch.stack(tensors)


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    center_x, center_y, width, height = boxes.unbind(-1)
    return torch.stack(
        (center_x - width / 2, center_y - height / 2,
         center_x + width / 2, center_y + height / 2), dim=-1
    )


def decode_batch(outputs, tiles, num_classes: int, confidence: float, max_det: int):
    probabilities = outputs.logits.sigmoid()
    decoded = []
    for batch_index, tile in enumerate(tiles):
        flat = probabilities[batch_index].flatten()
        count = min(max_det, flat.numel())
        scores, indices = flat.topk(count)
        keep = scores >= confidence
        scores, indices = scores[keep], indices[keep]
        classes = indices.remainder(num_classes)
        queries = indices.div(num_classes, rounding_mode="floor")
        boxes = cxcywh_to_xyxy(outputs.pred_boxes[batch_index, queries]).clamp_(0, 1)
        tile_width, tile_height = tile[2] - tile[0], tile[3] - tile[1]
        boxes[:, (0, 2)] *= tile_width
        boxes[:, (1, 3)] *= tile_height
        boxes[:, (0, 2)] += tile[0]
        boxes[:, (1, 3)] += tile[1]
        decoded.append((boxes, scores, classes))
    return decoded


def main() -> None:
    args = parse_args()
    if args.imgsz % 14:
        raise ValueError("imgsz must be divisible by 14")
    if args.tile_size <= 0 or not 0 <= args.overlap < args.tile_size:
        raise ValueError("Invalid tile size/overlap")
    if not 0 <= args.conf <= 1 or not 0 <= args.nms_iou <= 1:
        raise ValueError("Invalid confidence/NMS threshold")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}")

    classes = load_classes(args.classes)
    images = load_images(args.manifest, args.split, args.limit_images, args.image_dir)
    device = torch.device(args.device)
    model = build_dinov2_rtdetr(
        args.dinov2_weights,
        num_classes=len(classes),
        freeze_backbone=True,
        rtdetr_weights=args.rtdetr_weights,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()

    started = time.time()
    records = []
    total_tiles = total_predictions = 0
    class_counts = Counter({name: 0 for name in classes})
    amp_enabled = device.type == "cuda"
    for image_index, image_path in enumerate(images, 1):
        with Image.open(image_path) as opened:
            source = opened.convert("RGB")
        width, height = source.size
        tiles = make_tiles(width, height, args.tile_size, args.overlap)
        all_boxes, all_scores, all_classes = [], [], []
        for offset in range(0, len(tiles), args.batch):
            tile_batch = tiles[offset : offset + args.batch]
            crops = [source.crop(tile) for tile in tile_batch]
            pixel_values = preprocess(crops, args.imgsz).to(device, non_blocking=True)
            pixel_mask = torch.ones(
                (len(crops), args.imgsz, args.imgsz), dtype=torch.bool, device=device
            )
            with torch.inference_mode(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled
            ):
                outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            for boxes, scores, class_ids in decode_batch(
                outputs, tile_batch, len(classes), args.conf, args.max_det_per_tile
            ):
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_classes.append(class_ids)
        total_tiles += len(tiles)
        if all_boxes:
            boxes = torch.cat(all_boxes)
            scores = torch.cat(all_scores)
            class_ids = torch.cat(all_classes)
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes, scores, class_ids = boxes[valid], scores[valid], class_ids[valid]
            keep = batched_nms(boxes, scores, class_ids, args.nms_iou)
            keep = keep[: args.max_det_per_image]
            boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]
            predictions = []
            for box, score, class_id in zip(
                boxes.cpu().tolist(), scores.cpu().tolist(), class_ids.cpu().tolist(), strict=True
            ):
                class_id = int(class_id)
                predictions.append({
                    "class_id": class_id,
                    "category_name": classes[class_id],
                    "bbox_xyxy": [float(value) for value in box],
                    "score": float(score),
                })
                class_counts[classes[class_id]] += 1
            total_predictions += len(predictions)
        else:
            predictions = []
        records.append({
            "image_id": image_path.name,
            "source_path": str(image_path),
            "width": width,
            "height": height,
            "predictions": predictions,
        })
        if image_index % 20 == 0 or image_index == len(images):
            print(f"Processed {image_index}/{len(images)} images", flush=True)

    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]) + 1,
        "checkpoint_step": int(checkpoint["step"]),
        "classes": classes,
        "image_count": len(images),
        "tile_count": total_tiles,
        "tile_size": args.tile_size,
        "overlap": args.overlap,
        "imgsz": args.imgsz,
        "confidence": args.conf,
        "nms_iou": args.nms_iou,
        "prediction_count": total_predictions,
        "class_prediction_counts": dict(class_counts),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"images": records, "metadata": metadata}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
