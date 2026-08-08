#!/usr/bin/env python3
"""Evaluate raw detection predictions against the fixed VOC validation split.

The evaluator uses confidence-ranked, one-to-one matching independently for
each class. AP uses 101-point interpolated precision at IoU 0.50:0.95.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import yaml


IOU_THRESHOLDS = tuple(round(0.50 + 0.05 * i, 2) for i in range(10))
DEFAULT_SWEEP = (0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1,
                 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True,
                   help="Raw JSON written by predict_rtdetr.py")
    p.add_argument("--manifest", type=Path, default=Path("splits/v1/split_manifest.csv"))
    p.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    p.add_argument("--split", default="val")
    p.add_argument("--score-threshold", type=float, default=0.25)
    p.add_argument("--iou-threshold", type=float, default=0.50)
    p.add_argument("--output-dir", type=Path,
                   default=Path("outputs/evaluation/whole_val"))
    p.add_argument("--allow-missing-images", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_classes(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        names = yaml.safe_load(f).get("names")
    if not isinstance(names, list) or not names or len(names) != len(set(names)):
        raise ValueError(f"Invalid class names in {path}")
    return [str(x) for x in names]


def clean_voc_boxes(xml_path: Path, class_to_id: dict[str, int]):
    root = ET.parse(xml_path).getroot()
    width = int(float(root.findtext("size/width", "0")))
    height = int(float(root.findtext("size/height", "0")))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in {xml_path}")
    boxes, seen = [], set()
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in class_to_id:
            raise ValueError(f"Unknown class {name!r} in {xml_path}")
        node = obj.find("bndbox")
        if node is None:
            continue
        try:
            raw = tuple(float(node.findtext(k, "nan"))
                        for k in ("xmin", "ymin", "xmax", "ymax"))
        except ValueError:
            continue
        if not all(math.isfinite(v) for v in raw):
            continue
        signature = (name, *raw)
        if signature in seen:
            continue
        seen.add(signature)
        x1, y1, x2, y2 = raw
        box = (max(0.0, min(x1, width)), max(0.0, min(y1, height)),
               max(0.0, min(x2, width)), max(0.0, min(y2, height)))
        if box[2] > box[0] and box[3] > box[1]:
            boxes.append((class_to_id[name], box))
    return width, height, boxes


def load_ground_truth(manifest: Path, split: str, class_to_id: dict[str, int]):
    with manifest.open(encoding="utf-8-sig", newline="") as f:
        rows = [row for row in csv.DictReader(f) if row["split"] == split]
    if not rows:
        raise ValueError(f"No {split!r} rows in {manifest}")
    gt = defaultdict(lambda: defaultdict(list))
    sizes, image_ids = {}, set()
    for row in rows:
        image_id = Path(row["image_path"]).name
        if image_id in image_ids:
            raise ValueError(f"Duplicate image basename in manifest: {image_id}")
        image_ids.add(image_id)
        width, height, boxes = clean_voc_boxes(Path(row["xml_path"]), class_to_id)
        sizes[image_id] = (width, height)
        for class_id, box in boxes:
            gt[class_id][image_id].append(box)
    return gt, sizes, image_ids


def load_predictions(path: Path, classes: list[str], sizes, expected_images,
                     allow_missing: bool):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "images" in payload:
        metadata_classes = payload.get("metadata", {}).get("classes")
        if metadata_classes is not None and metadata_classes != classes:
            raise ValueError("Prediction metadata classes differ from classes config")
        records = payload["images"]
    elif isinstance(payload, list):
        grouped = defaultdict(list)
        for item in payload:
            grouped[item["image_id"]].append({
                "category_name": item["category_name"],
                "bbox_xyxy": item.get("bbox_xyxy", item.get("bbox")),
                "score": item["score"],
            })
        records = [{"image_id": k, "predictions": v} for k, v in grouped.items()]
    else:
        raise ValueError("Unsupported prediction JSON schema")

    class_to_id = {name: i for i, name in enumerate(classes)}
    predictions = defaultdict(list)
    seen_images = set()
    invalid_boxes = clipped_boxes = 0
    for record in records:
        image_id = Path(str(record["image_id"])).name
        if image_id not in expected_images:
            raise ValueError(f"Prediction contains image outside evaluation split: {image_id}")
        if image_id in seen_images:
            raise ValueError(f"Duplicate prediction image record: {image_id}")
        seen_images.add(image_id)
        width, height = sizes[image_id]
        for pred in record.get("predictions", []):
            name = str(pred["category_name"])
            if name not in class_to_id:
                raise ValueError(f"Unknown predicted class {name!r} in {image_id}")
            if "class_id" in pred and int(pred["class_id"]) != class_to_id[name]:
                raise ValueError(f"Class ID/name mismatch in {image_id}")
            score = float(pred["score"])
            raw_box = pred.get("bbox_xyxy", pred.get("bbox"))
            if (not math.isfinite(score) or not 0 <= score <= 1 or
                    not isinstance(raw_box, list) or len(raw_box) != 4):
                raise ValueError(f"Invalid prediction in {image_id}: {pred}")
            box = tuple(float(v) for v in raw_box)
            if not all(math.isfinite(v) for v in box):
                invalid_boxes += 1
                continue
            clean = (max(0.0, min(box[0], width)), max(0.0, min(box[1], height)),
                     max(0.0, min(box[2], width)), max(0.0, min(box[3], height)))
            clipped_boxes += int(clean != box)
            if clean[2] <= clean[0] or clean[3] <= clean[1]:
                invalid_boxes += 1
                continue
            predictions[class_to_id[name]].append((score, image_id, clean))
    missing = expected_images - seen_images
    if missing and not allow_missing:
        raise ValueError(f"Predictions missing {len(missing)} images; examples: {sorted(missing)[:5]}")
    for items in predictions.values():
        items.sort(key=lambda x: x[0], reverse=True)
    return predictions, seen_images, invalid_boxes, clipped_boxes


def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_predictions(predictions, gt_by_image, iou_threshold):
    matched = {image_id: set() for image_id in gt_by_image}
    outcomes = []
    for score, image_id, box in predictions:
        candidates = gt_by_image.get(image_id, [])
        best_index, best_iou = -1, -1.0
        for index, gt_box in enumerate(candidates):
            if index in matched.setdefault(image_id, set()):
                continue
            iou = box_iou(box, gt_box)
            if iou > best_iou:
                best_index, best_iou = index, iou
        is_tp = best_index >= 0 and best_iou >= iou_threshold
        if is_tp:
            matched[image_id].add(best_index)
        outcomes.append((score, int(is_tp)))
    return outcomes


def ap_101(outcomes, gt_count):
    if gt_count == 0:
        return None
    tp = fp = 0
    recalls, precisions = [], []
    for _, is_tp in outcomes:
        tp += is_tp
        fp += 1 - is_tp
        recalls.append(tp / gt_count)
        precisions.append(tp / (tp + fp))
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    total = 0.0
    for step in range(101):
        recall_level = step / 100
        total += max((p for r, p in zip(recalls, precisions) if r >= recall_level), default=0.0)
    return total / 101


def point_metrics(outcomes, gt_count, score_threshold):
    kept = [(s, tp) for s, tp in outcomes if s >= score_threshold]
    tp = sum(v for _, v in kept)
    fp = len(kept) - tp
    fn = gt_count - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / gt_count if gt_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision,
            "recall": recall, "f1": f1, "f2": f2, "prediction_count": len(kept)}


def best_f1(outcomes, gt_count):
    best = {"threshold": 1.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    tp = 0
    for index, (score, is_tp) in enumerate(outcomes, start=1):
        tp += is_tp
        precision, recall = tp / index, tp / gt_count if gt_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best["f1"]:
            best = {"threshold": score, "precision": precision, "recall": recall, "f1": f1}
    return best


def best_f2(outcomes, gt_count):
    best = {"threshold": 1.0, "precision": 0.0, "recall": 0.0, "f2": 0.0}
    tp = 0
    for index, (score, is_tp) in enumerate(outcomes, start=1):
        tp += is_tp
        precision, recall = tp / index, tp / gt_count if gt_count else 0.0
        f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall else 0.0
        if f2 > best["f2"]:
            best = {"threshold": score, "precision": precision, "recall": recall, "f2": f2}
    return best


def rounded(value):
    return None if value is None else round(value, 8)


def main():
    args = parse_args()
    if not 0 <= args.score_threshold <= 1 or not 0 <= args.iou_threshold <= 1:
        raise ValueError("Thresholds must be in [0, 1]")
    report_path = args.output_dir / "evaluation_report.json"
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {report_path}; pass --overwrite")
    classes = load_classes(args.classes)
    class_to_id = {name: i for i, name in enumerate(classes)}
    gt, sizes, image_ids = load_ground_truth(args.manifest, args.split, class_to_id)
    predictions, seen, invalid, clipped = load_predictions(
        args.predictions, classes, sizes, image_ids, args.allow_missing_images)

    per_class, outcomes_at_point = {}, {}
    for class_id, name in enumerate(classes):
        gt_count = sum(len(v) for v in gt[class_id].values())
        pred = predictions[class_id]
        aps = {}
        for threshold in IOU_THRESHOLDS:
            outcomes = match_predictions(pred, gt[class_id], threshold)
            aps[f"{threshold:.2f}"] = ap_101(outcomes, gt_count)
            if threshold == args.iou_threshold:
                outcomes_at_point[class_id] = outcomes
        if class_id not in outcomes_at_point:
            outcomes_at_point[class_id] = match_predictions(pred, gt[class_id], args.iou_threshold)
        point = point_metrics(outcomes_at_point[class_id], gt_count, args.score_threshold)
        per_class[name] = {
            "ground_truth_count": gt_count,
            **{k: rounded(v) if isinstance(v, float) else v for k, v in point.items()},
            "ap50": rounded(aps["0.50"]),
            "ap50_95": rounded(sum(v for v in aps.values() if v is not None) /
                               sum(v is not None for v in aps.values())),
            "ap_by_iou": {k: rounded(v) for k, v in aps.items()},
            "best_f1_at_iou": {k: rounded(v) for k, v in
                                best_f1(outcomes_at_point[class_id], gt_count).items()},
            "best_f2_at_iou": {k: rounded(v) for k, v in
                                best_f2(outcomes_at_point[class_id], gt_count).items()},
        }

    totals = {key: sum(per_class[name][key] for name in classes)
              for key in ("tp", "fp", "fn", "prediction_count", "ground_truth_count")}
    p = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else 0.0
    r = totals["tp"] / totals["ground_truth_count"] if totals["ground_truth_count"] else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    f2 = 5 * p * r / (4 * p + r) if p + r else 0.0
    overall = {**totals, "precision": rounded(p), "recall": rounded(r),
               "f1": rounded(f1), "f2": rounded(f2),
               "map50": rounded(sum(per_class[n]["ap50"] for n in classes) / len(classes)),
               "map50_95": rounded(sum(per_class[n]["ap50_95"] for n in classes) / len(classes))}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sweep_rows = []
    for threshold in DEFAULT_SWEEP:
        class_points = [point_metrics(outcomes_at_point[i], per_class[name]["ground_truth_count"], threshold)
                        for i, name in enumerate(classes)]
        tp, fp, fn = (sum(row[k] for row in class_points) for k in ("tp", "fp", "fn"))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        row = {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn,
               "precision": precision, "recall": recall,
               "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
               "f2": 5 * precision * recall / (4 * precision + recall) if precision + recall else 0.0}
        row.update({f"recall_{name}": class_points[i]["recall"] for i, name in enumerate(classes)})
        sweep_rows.append(row)
    with (args.output_dir / "threshold_sweep.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep_rows[0]))
        writer.writeheader()
        writer.writerows(sweep_rows)

    report = {
        "metadata": {"predictions": str(args.predictions.resolve()),
                     "manifest": str(args.manifest.resolve()), "split": args.split,
                     "evaluated_images": len(image_ids), "prediction_image_records": len(seen),
                     "score_threshold": args.score_threshold,
                     "matching_iou_threshold": args.iou_threshold,
                     "ap_method": "101-point interpolated precision",
                     "ap_iou_thresholds": list(IOU_THRESHOLDS),
                     "invalid_predictions_dropped": invalid,
                     "predictions_clipped_to_image": clipped},
        "overall": overall, "per_class": per_class,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"overall": overall,
                      "best_f1_thresholds": {n: per_class[n]["best_f1_at_iou"]["threshold"]
                                             for n in classes},
                      "best_f2_thresholds": {n: per_class[n]["best_f2_at_iou"]["threshold"]
                                             for n in classes},
                      "outputs": str(args.output_dir.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
