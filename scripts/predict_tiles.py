#!/usr/bin/env python3
"""Run sliding-window RT-DETR inference and save global-coordinate predictions."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
DEFAULT_WEIGHTS = Path(
    "/root/autodl-tmp/runs/rtdetr/rtdetr_l_1024_baseline_v2/weights/best.pt"
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--image-dir", type=Path, required=True)
    p.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tile-size", type=int, default=1024)
    p.add_argument("--overlap", type=int, default=384)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default="0")
    p.add_argument("--border-margin", type=float, default=8.0)
    p.add_argument("--limit-images", type=int)
    p.add_argument("--estimate-only", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_classes(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        names = yaml.safe_load(f).get("names")
    if not isinstance(names, list) or not names or len(names) != len(set(names)):
        raise ValueError(f"Invalid classes file: {path}")
    return [str(x) for x in names]


def find_images(root: Path, limit: int | None) -> list[Path]:
    images = sorted(p.resolve() for p in root.rglob("*")
                    if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"No images under {root}")
    names = [p.name for p in images]
    if len(names) != len(set(names)):
        raise ValueError("Image basenames must be unique")
    return images[:limit] if limit is not None else images


def axis_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def make_tiles(width: int, height: int, tile_size: int, overlap: int):
    stride = tile_size - overlap
    xs = axis_starts(width, tile_size, stride)
    ys = axis_starts(height, tile_size, stride)
    return [(x, y, min(x + tile_size, width), min(y + tile_size, height))
            for y in ys for x in xs]


def touches_internal_border(box, tile, width, height, margin):
    x1, y1, x2, y2 = box
    tx1, ty1, tx2, ty2 = tile
    return bool(
        (tx1 > 0 and x1 <= margin) or
        (ty1 > 0 and y1 <= margin) or
        (tx2 < width and x2 >= (tx2 - tx1) - margin) or
        (ty2 < height and y2 >= (ty2 - ty1) - margin)
    )


def estimate(images, tile_size, overlap):
    counts, dimensions = [], Counter()
    for path in images:
        with Image.open(path) as image:
            width, height = image.size
        dimensions[f"{width}x{height}"] += 1
        counts.append(len(make_tiles(width, height, tile_size, overlap)))
    return {
        "image_count": len(images),
        "tile_count": sum(counts),
        "tiles_per_image_min": min(counts),
        "tiles_per_image_max": max(counts),
        "tiles_per_image_mean": sum(counts) / len(counts),
        "image_dimensions": dict(dimensions),
    }


def main():
    args = parse_args()
    if args.tile_size <= 0 or not 0 <= args.overlap < args.tile_size:
        raise ValueError("Require tile_size > 0 and 0 <= overlap < tile_size")
    if args.batch <= 0 or args.max_det <= 0 or not 0 <= args.conf <= 1:
        raise ValueError("Invalid batch/max_det/conf")
    if args.border_margin < 0 or args.border_margin * 2 >= args.tile_size:
        raise ValueError("Invalid border margin")
    classes = load_classes(args.classes)
    images = find_images(args.image_dir.resolve(), args.limit_images)
    plan = estimate(images, args.tile_size, args.overlap)
    plan.update({"tile_size": args.tile_size, "overlap": args.overlap,
                 "stride": args.tile_size - args.overlap,
                 "maximum_possible_predictions": plan["tile_count"] * args.max_det})
    print(json.dumps({"inference_plan": plan}, ensure_ascii=False, indent=2))
    if args.estimate_only:
        return
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output}; pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from ultralytics import RTDETR

    model = RTDETR(str(args.weights.resolve()))
    model_names = ([str(model.names[i]) for i in range(len(model.names))]
                   if isinstance(model.names, dict) else [str(x) for x in model.names])
    if model_names != classes:
        raise ValueError(f"Model classes differ from config: {model_names} != {classes}")

    started = time.time()
    total_predictions = border_predictions = total_tiles = 0
    class_counts = Counter({name: 0 for name in classes})
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write('{"images":[')
        first_record = True
        for image_index, image_path in enumerate(images, start=1):
            with Image.open(image_path) as source:
                source = source.convert("RGB")
                width, height = source.size
                tiles = make_tiles(width, height, args.tile_size, args.overlap)
                predictions = []
                tile_records = []
                for offset in range(0, len(tiles), args.batch):
                    tile_batch = tiles[offset:offset + args.batch]
                    crops = [source.crop(tile) for tile in tile_batch]
                    results = model.predict(source=crops, imgsz=args.imgsz, conf=args.conf,
                                            max_det=args.max_det, device=args.device,
                                            verbose=False)
                    if len(results) != len(tile_batch):
                        raise RuntimeError("Model returned a different number of tile results")
                    for local_index, (tile, result) in enumerate(zip(tile_batch, results, strict=True)):
                        tile_index = offset + local_index
                        tile_records.append({"tile_index": tile_index, "tile_xyxy": list(tile)})
                        if result.boxes is None:
                            continue
                        xyxy = result.boxes.xyxy.detach().cpu().tolist()
                        scores = result.boxes.conf.detach().cpu().tolist()
                        class_ids = result.boxes.cls.detach().cpu().to(dtype=int).tolist()
                        for box, score, class_id in zip(xyxy, scores, class_ids, strict=True):
                            if not 0 <= class_id < len(classes):
                                raise RuntimeError(f"Invalid class ID {class_id}")
                            internal_border = touches_internal_border(
                                box, tile, width, height, args.border_margin)
                            global_box = [box[0] + tile[0], box[1] + tile[1],
                                          box[2] + tile[0], box[3] + tile[1]]
                            predictions.append({
                                "class_id": class_id,
                                "category_name": classes[class_id],
                                "bbox_xyxy": [float(v) for v in global_box],
                                "score": float(score),
                                "source": "tile",
                                "source_tile_index": tile_index,
                                "touches_internal_border": internal_border,
                            })
                            class_counts[classes[class_id]] += 1
                            total_predictions += 1
                            border_predictions += int(internal_border)
                total_tiles += len(tiles)
                predictions.sort(key=lambda item: item["score"], reverse=True)
                record = {"image_id": image_path.name, "source_path": str(image_path),
                          "width": width, "height": height, "tiles": tile_records,
                          "predictions": predictions}
                if not first_record:
                    stream.write(",")
                json.dump(record, stream, ensure_ascii=False, separators=(",", ":"))
                first_record = False
            if image_index % 10 == 0 or image_index == len(images):
                print(f"Processed {image_index}/{len(images)} images, {total_tiles} tiles")
        metadata = {
            "weights": str(args.weights.resolve()), "image_dir": str(args.image_dir.resolve()),
            "classes": classes, "image_count": len(images), "tile_count": total_tiles,
            "tile_size": args.tile_size, "overlap": args.overlap, "imgsz": args.imgsz,
            "conf": args.conf, "max_det": args.max_det, "batch": args.batch,
            "border_margin": args.border_margin, "prediction_count": total_predictions,
            "internal_border_prediction_count": border_predictions,
            "class_prediction_counts": dict(class_counts),
            "elapsed_seconds": round(time.time() - started, 3),
        }
        stream.write('],"metadata":')
        json.dump(metadata, stream, ensure_ascii=False, separators=(",", ":"))
        stream.write("}")
    temporary.replace(output)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
