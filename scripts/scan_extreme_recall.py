#!/usr/bin/env python3
"""Scan the class-aware validation recall ceiling without a precision constraint.

The input files are compact Top-K raw predictions from
``predict_hierarchical_dfine.py --class-topk K``.  For each requested Top-K,
NMS IoU and per-image output cap, this script fuses whole/tile predictions once
and evaluates many score thresholds.  Top-K copies use the detector's
defectness score by default, which deliberately measures the recall available
when class ambiguity is allowed to create extra false positives.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from evaluate_predictions import (
    box_iou,
    load_ground_truth,
    match_predictions,
    point_metrics,
)


def comma_numbers(value: str, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--whole", type=Path, required=True)
    parser.add_argument("--tiles", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("splits/v1/split_manifest.csv"))
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--topks", default="1,2,3")
    parser.add_argument("--nms-ious", default="0.8,0.9,0.95,1.0")
    parser.add_argument("--max-dets", default="300,600,1000,100000")
    parser.add_argument(
        "--thresholds", default="0,0.0001,0.001,0.003,0.01,0.03,0.1,0.3,0.5,0.59"
    )
    parser.add_argument("--tile-border-policy", choices=("keep", "penalize", "drop"),
                        default="keep")
    parser.add_argument("--tile-border-score-factor", type=float, default=0.5)
    parser.add_argument("--score-mode", choices=("objectness", "joint"),
                        default="objectness")
    parser.add_argument("--matching-iou", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_classes(path: Path) -> list[str]:
    names = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("names")
    if not isinstance(names, list) or not names or len(names) != len(set(names)):
        raise ValueError(f"Invalid classes file: {path}")
    return [str(name) for name in names]


def load_records(path: Path, classes: list[str]):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("images"), list):
        raise ValueError(f"Unsupported raw prediction file: {path}")
    metadata_classes = payload.get("metadata", {}).get("classes")
    if metadata_classes is not None and metadata_classes != classes:
        raise ValueError(f"Class metadata differs in {path}")
    records = {}
    for record in payload["images"]:
        image_id = Path(str(record["image_id"])).name
        if image_id in records:
            raise ValueError(f"Duplicate image record: {image_id}")
        records[image_id] = record
    return records, payload.get("metadata", {})


def nms(items, threshold: float):
    if not items:
        return []
    order = np.argsort(np.asarray([item[0] for item in items], dtype=np.float64))[::-1]
    if threshold >= 1.0:
        return [items[int(index)] for index in order]
    boxes = np.asarray([item[2] for item in items], dtype=np.float64)
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    keep = []
    while order.size:
        current = int(order[0])
        keep.append(items[current])
        if order.size == 1:
            break
        rest = order[1:]
        ix1 = np.maximum(x1[current], x1[rest])
        iy1 = np.maximum(y1[current], y1[rest])
        ix2 = np.minimum(x2[current], x2[rest])
        iy2 = np.minimum(y2[current], y2[rest])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        union = areas[current] + areas[rest] - inter
        ious = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = rest[ious <= threshold]
    return keep


def clean_box(box, width, height):
    if not isinstance(box, list) or len(box) != 4:
        return None
    values = [float(value) for value in box]
    if not all(math.isfinite(value) for value in values):
        return None
    values = [max(0.0, min(values[0], width)), max(0.0, min(values[1], height)),
              max(0.0, min(values[2], width)), max(0.0, min(values[3], height))]
    return tuple(values) if values[2] > values[0] and values[3] > values[1] else None


def expand_record(record, source: str, topk: int, class_count: int,
                  border_policy: str, border_factor: float, score_mode: str):
    width, height = int(record["width"]), int(record["height"])
    by_class = defaultdict(list)
    for prediction in record.get("predictions", []):
        is_border = source == "tile" and bool(prediction.get("touches_internal_border"))
        if is_border and border_policy == "drop":
            continue
        box = clean_box(prediction.get("bbox_xyxy"), width, height)
        if box is None:
            continue
        objectness = float(prediction.get("objectness", prediction["score"]))
        class_ids = prediction.get("class_ids_ranked", [int(prediction["class_id"])])
        probabilities = prediction.get(
            "class_probabilities_ranked", [float(prediction.get("class_probability", 1.0))]
        )
        if len(class_ids) < topk or len(probabilities) < topk:
            raise ValueError(
                f"Prediction contains only {len(class_ids)} ranked classes, requested Top-{topk}"
            )
        for rank in range(topk):
            class_id = int(class_ids[rank])
            if not 0 <= class_id < class_count:
                raise ValueError(f"Invalid class id {class_id}")
            score = objectness if score_mode == "objectness" else objectness * float(probabilities[rank])
            if is_border and border_policy == "penalize":
                score *= border_factor
            by_class[class_id].append((score, class_id, box))
    return by_class


def fuse_setting(whole_records, tile_records, topk, nms_iou, class_count,
                 border_policy, border_factor, score_mode):
    merged_by_image = {}
    for image_index, image_id in enumerate(sorted(whole_records), start=1):
        whole = whole_records[image_id]
        tile = tile_records[image_id]
        if (int(whole["width"]), int(whole["height"])) != (
                int(tile["width"]), int(tile["height"])):
            raise ValueError(f"Image sizes differ for {image_id}")
        candidates = defaultdict(list)
        for class_id, items in expand_record(
                whole, "whole", topk, class_count, border_policy, border_factor,
                score_mode).items():
            candidates[class_id].extend(items)
        for class_id, items in expand_record(
                tile, "tile", topk, class_count, border_policy, border_factor,
                score_mode).items():
            candidates[class_id].extend(items)
        merged = []
        for items in candidates.values():
            merged.extend(nms(items, nms_iou))
        merged.sort(key=lambda item: item[0], reverse=True)
        merged_by_image[image_id] = merged
        if image_index % 100 == 0 or image_index == len(whole_records):
            print(f"  fused {image_index}/{len(whole_records)} images", flush=True)
    return merged_by_image


def evaluate_setting(merged_by_image, max_det, thresholds, classes, gt, matching_iou):
    predictions = defaultdict(list)
    for image_id, merged in merged_by_image.items():
        for score, class_id, box in merged[:max_det]:
            predictions[class_id].append((score, image_id, box))
    outcomes, gt_counts = {}, {}
    for class_id in range(len(classes)):
        predictions[class_id].sort(key=lambda item: item[0], reverse=True)
        gt_counts[class_id] = sum(len(items) for items in gt[class_id].values())
        outcomes[class_id] = match_predictions(
            predictions[class_id], gt[class_id], matching_iou
        )
    rows = []
    for threshold in thresholds:
        points = [point_metrics(outcomes[class_id], gt_counts[class_id], threshold)
                  for class_id in range(len(classes))]
        tp = sum(point["tp"] for point in points)
        fp = sum(point["fp"] for point in points)
        fn = sum(point["fn"] for point in points)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        row = {
            "max_det": max_det, "threshold": threshold,
            "tp": tp, "fp": fp, "fn": fn,
            "prediction_count": tp + fp,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall)
                  if precision + recall else 0.0,
        }
        row.update({f"recall_{name}": points[index]["recall"]
                    for index, name in enumerate(classes)})
        rows.append(row)
    return rows


def localization_coverage(records_by_source, gt, class_count, thresholds, matching_iou):
    """Per-GT diagnostic upper bound that ignores class and one-to-one conflicts."""
    all_gt = defaultdict(list)
    for class_id in range(class_count):
        for image_id, boxes in gt[class_id].items():
            all_gt[image_id].extend(boxes)
    rows = []
    for threshold in thresholds:
        covered = total = 0
        for image_id, gt_boxes in all_gt.items():
            predictions = []
            for records in records_by_source:
                record = records[image_id]
                width, height = int(record["width"]), int(record["height"])
                for pred in record.get("predictions", []):
                    score = float(pred.get("objectness", pred["score"]))
                    box = clean_box(pred.get("bbox_xyxy"), width, height)
                    if score >= threshold and box is not None:
                        predictions.append(box)
            for gt_box in gt_boxes:
                total += 1
                covered += int(any(box_iou(pred_box, gt_box) >= matching_iou
                                   for pred_box in predictions))
        rows.append({"threshold": threshold, "covered_gt": covered,
                     "ground_truth_count": total,
                     "localization_coverage": covered / total if total else 0.0})
    return rows


def main():
    args = parse_args()
    output_csv = args.output_dir / "extreme_recall_scan.csv"
    if output_csv.exists() and not args.overwrite:
        raise FileExistsError(f"{output_csv} exists; pass --overwrite")
    classes = load_classes(args.classes)
    class_to_id = {name: index for index, name in enumerate(classes)}
    gt, _, image_ids = load_ground_truth(args.manifest, args.split, class_to_id)
    whole_records, whole_metadata = load_records(args.whole, classes)
    tile_records, tile_metadata = load_records(args.tiles, classes)
    if set(whole_records) != image_ids or set(tile_records) != image_ids:
        raise ValueError("Prediction image sets differ from the requested split")

    topks = comma_numbers(args.topks, int)
    nms_ious = comma_numbers(args.nms_ious, float)
    max_dets = comma_numbers(args.max_dets, int)
    thresholds = comma_numbers(args.thresholds, float)
    if any(not 1 <= value <= len(classes) for value in topks):
        raise ValueError("Invalid Top-K values")
    if any(not 0 <= value <= 1 for value in nms_ious + thresholds):
        raise ValueError("Thresholds and NMS IoUs must be in [0, 1]")

    rows = []
    for topk in topks:
        for nms_iou in nms_ious:
            print(f"Top-{topk}, NMS={nms_iou:.2f}", flush=True)
            merged = fuse_setting(
                whole_records, tile_records, topk, nms_iou, len(classes),
                args.tile_border_policy, args.tile_border_score_factor, args.score_mode,
            )
            for max_det in max_dets:
                setting_rows = evaluate_setting(
                    merged, max_det, thresholds, classes, gt, args.matching_iou
                )
                for row in setting_rows:
                    row.update({"topk": topk, "nms_iou": nms_iou,
                                "tile_border_policy": args.tile_border_policy,
                                "score_mode": args.score_mode})
                rows.extend(setting_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["topk", "nms_iou", "max_det", "threshold",
                  "tile_border_policy", "score_mode", "tp", "fp", "fn",
                  "prediction_count", "precision", "recall", "f1"]
    fieldnames.extend(f"recall_{name}" for name in classes)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    localization = localization_coverage(
        [whole_records, tile_records], gt, len(classes), thresholds, args.matching_iou
    )
    best_recall = max(rows, key=lambda row: (row["recall"], row["precision"]))
    summary = {
        "metadata": {
            "whole": str(args.whole.resolve()), "tiles": str(args.tiles.resolve()),
            "whole_metadata": whole_metadata, "tile_metadata": tile_metadata,
            "ground_truth_count": sum(
                len(items) for class_gt in gt.values() for items in class_gt.values()
            ),
            "matching_iou": args.matching_iou,
        },
        "best_recall": best_recall,
        "localization_coverage": localization,
        "outputs": str(output_csv.resolve()),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
