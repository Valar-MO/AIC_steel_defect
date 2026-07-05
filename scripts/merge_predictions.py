#!/usr/bin/env python3
"""Merge whole-image and sliding-window predictions with class-wise NMS."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--whole", type=Path, help="Whole-image raw prediction JSON")
    p.add_argument("--tiles", type=Path, help="Tile raw prediction JSON")
    p.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    p.add_argument("--class-thresholds", type=Path,
                   help="YAML mapping from every class name to score threshold")
    p.add_argument("--min-score", type=float, default=0.001,
                   help="Fallback threshold when class thresholds are omitted")
    p.add_argument("--nms-iou", type=float, default=0.50)
    p.add_argument("--tile-border-policy", choices=("keep", "drop", "penalize"),
                   default="penalize")
    p.add_argument("--tile-border-score-factor", type=float, default=0.50)
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_classes(path):
    with path.open(encoding="utf-8") as f:
        names = yaml.safe_load(f).get("names")
    if not isinstance(names, list) or not names or len(names) != len(set(names)):
        raise ValueError(f"Invalid classes file: {path}")
    return [str(x) for x in names]


def load_thresholds(path, classes, fallback):
    if path is None:
        return {name: fallback for name in classes}
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if isinstance(data, dict) and "thresholds" in data:
        data = data["thresholds"]
    if not isinstance(data, dict) or set(data) != set(classes):
        raise ValueError("Class threshold YAML must define every class exactly once")
    thresholds = {name: float(data[name]) for name in classes}
    if any(not 0 <= value <= 1 for value in thresholds.values()):
        raise ValueError("Class thresholds must be in [0, 1]")
    return thresholds


def load_raw(path, classes):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("images"), list):
        raise ValueError(f"Unsupported raw prediction schema: {path}")
    metadata_classes = payload.get("metadata", {}).get("classes")
    if metadata_classes is not None and metadata_classes != classes:
        raise ValueError(f"Classes differ in {path}")
    records = {}
    for record in payload["images"]:
        image_id = Path(str(record["image_id"])).name
        if image_id in records:
            raise ValueError(f"Duplicate image record in {path}: {image_id}")
        records[image_id] = record
    return records


def nms(boxes, scores, threshold):
    if not boxes:
        return []
    boxes = np.asarray(boxes, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        ix1, iy1 = np.maximum(x1[current], x1[rest]), np.maximum(y1[current], y1[rest])
        ix2, iy2 = np.minimum(x2[current], x2[rest]), np.minimum(y2[current], y2[rest])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        union = areas[current] + areas[rest] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = rest[iou <= threshold]
    return keep


def validate_prediction(pred, classes, width, height):
    name = str(pred["category_name"])
    if name not in classes:
        raise ValueError(f"Unknown class {name!r}")
    class_id = classes.index(name)
    if "class_id" in pred and int(pred["class_id"]) != class_id:
        raise ValueError("Class ID/name mismatch")
    score = float(pred["score"])
    box = [float(v) for v in pred["bbox_xyxy"]]
    if not math.isfinite(score) or not 0 <= score <= 1 or len(box) != 4:
        raise ValueError("Invalid prediction score or box")
    box = [max(0.0, min(box[0], width)), max(0.0, min(box[1], height)),
           max(0.0, min(box[2], width)), max(0.0, min(box[3], height))]
    if not all(math.isfinite(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
        return None
    return class_id, name, score, box


def main():
    args = parse_args()
    if args.whole is None and args.tiles is None:
        raise ValueError("Provide --whole, --tiles, or both")
    if not 0 <= args.min_score <= 1 or not 0 <= args.nms_iou <= 1:
        raise ValueError("Score and IoU thresholds must be in [0, 1]")
    if not 0 <= args.tile_border_score_factor <= 1 or args.max_det <= 0:
        raise ValueError("Invalid border score factor or max_det")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}; pass --overwrite")
    classes = load_classes(args.classes)
    thresholds = load_thresholds(args.class_thresholds, classes, args.min_score)
    sources = []
    if args.whole:
        sources.append(("whole", load_raw(args.whole, classes)))
    if args.tiles:
        sources.append(("tile", load_raw(args.tiles, classes)))
    image_sets = [set(records) for _, records in sources]
    if any(ids != image_sets[0] for ids in image_sets[1:]):
        raise ValueError("Whole and tile prediction files contain different image sets")

    counters = Counter()
    class_counts = Counter({name: 0 for name in classes})
    output_images = []
    for image_id in sorted(image_sets[0]):
        reference = sources[0][1][image_id]
        width, height = int(reference["width"]), int(reference["height"])
        candidates = defaultdict(list)
        for source_name, records in sources:
            record = records[image_id]
            if (int(record["width"]), int(record["height"])) != (width, height):
                raise ValueError(f"Image dimensions differ for {image_id}")
            for pred in record.get("predictions", []):
                counters[f"input_{source_name}"] += 1
                parsed = validate_prediction(pred, classes, width, height)
                if parsed is None:
                    counters["invalid_dropped"] += 1
                    continue
                class_id, name, score, box = parsed
                is_border = source_name == "tile" and bool(pred.get("touches_internal_border"))
                if is_border and args.tile_border_policy == "drop":
                    counters["border_dropped"] += 1
                    continue
                adjusted_score = score
                if is_border and args.tile_border_policy == "penalize":
                    adjusted_score *= args.tile_border_score_factor
                    counters["border_penalized"] += 1
                if adjusted_score < thresholds[name]:
                    counters["threshold_dropped"] += 1
                    continue
                candidates[class_id].append({
                    "class_id": class_id, "category_name": name,
                    "bbox_xyxy": box, "score": adjusted_score,
                    "original_score": score, "source": source_name,
                    "touches_internal_border": is_border,
                })

        merged = []
        for class_id, items in candidates.items():
            keep = nms([item["bbox_xyxy"] for item in items],
                       [item["score"] for item in items], args.nms_iou)
            counters["nms_dropped"] += len(items) - len(keep)
            merged.extend(items[index] for index in keep)
        merged.sort(key=lambda item: item["score"], reverse=True)
        if len(merged) > args.max_det:
            counters["max_det_dropped"] += len(merged) - args.max_det
            merged = merged[:args.max_det]
        for item in merged:
            class_counts[item["category_name"]] += 1
        output_images.append({"image_id": image_id,
                              "source_path": reference.get("source_path"),
                              "width": width, "height": height,
                              "predictions": merged})

    metadata = {
        "classes": classes, "image_count": len(output_images),
        "whole_predictions": str(args.whole.resolve()) if args.whole else None,
        "tile_predictions": str(args.tiles.resolve()) if args.tiles else None,
        "class_thresholds": thresholds, "nms_iou": args.nms_iou,
        "tile_border_policy": args.tile_border_policy,
        "tile_border_score_factor": args.tile_border_score_factor,
        "max_det": args.max_det,
        "prediction_count": sum(len(r["predictions"]) for r in output_images),
        "class_prediction_counts": dict(class_counts), "processing_counts": dict(counters),
    }
    payload = {"metadata": metadata, "images": output_images}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                           encoding="utf-8")
    report = args.report or args.output.with_name(args.output.stem + "_report.json")
    report.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
