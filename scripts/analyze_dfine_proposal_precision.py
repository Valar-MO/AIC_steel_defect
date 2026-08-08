#!/usr/bin/env python3
"""Audit class-agnostic proposals with greedy one-to-one GT matching.

Each prediction is classified as one of:
  tp          matches an as-yet unmatched GT at --match-iou;
  duplicate   overlaps a GT at --match-iou, but all eligible GTs are matched;
  near_miss   best IoU is in [--near-iou, --match-iou);
  weak_overlap best IoU is in [--background-iou, --near-iou);
  background  best IoU is below --background-iou (including images without GT).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import yaml


DEFAULT_THRESHOLDS = (0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5)
OUTCOMES = ("tp", "duplicate", "near_miss", "weak_overlap", "background")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("splits/v1/split_manifest.csv"))
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--near-iou", type=float, default=0.3)
    parser.add_argument("--background-iou", type=float, default=0.1)
    parser.add_argument("--score-thresholds", nargs="+", type=float, default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_classes(path: Path) -> list[str]:
    names = yaml.safe_load(path.read_text(encoding="utf-8"))["names"]
    if not isinstance(names, list) or not names or len(names) != len(set(names)):
        raise ValueError(f"Invalid classes: {path}")
    return [str(name) for name in names]


def box_iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def size_bucket(box, image_area: float) -> str:
    ratio = ((box[2] - box[0]) * (box[3] - box[1])) / image_area
    if ratio < 0.001:
        return "tiny_lt_0.1pct"
    if ratio < 0.01:
        return "small_0.1_to_1pct"
    if ratio < 0.10:
        return "medium_1_to_10pct"
    return "large_ge_10pct"


def load_ground_truth(manifest: Path, split: str, classes: list[str]):
    class_set = set(classes)
    by_image, sizes, expected = defaultdict(list), {}, set()
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == split]
    if not rows:
        raise ValueError(f"No rows for split {split!r}")
    next_gt_id = 0
    for row in rows:
        image_id = Path(row["image_path"]).name
        if image_id in expected:
            raise ValueError(f"Duplicate image in manifest: {image_id}")
        expected.add(image_id)
        root = ET.parse(row["xml_path"]).getroot()
        width = int(float(root.findtext("size/width", "0")))
        height = int(float(root.findtext("size/height", "0")))
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image dimensions: {row['xml_path']}")
        sizes[image_id] = (width, height)
        seen = set()
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip()
            if name not in class_set:
                raise ValueError(f"Unknown class {name!r} in {row['xml_path']}")
            node = obj.find("bndbox")
            if node is None:
                continue
            raw = tuple(float(node.findtext(key, "nan")) for key in ("xmin", "ymin", "xmax", "ymax"))
            signature = (name, *raw)
            if signature in seen or not all(math.isfinite(value) for value in raw):
                continue
            seen.add(signature)
            box = (
                max(0.0, min(raw[0], width)), max(0.0, min(raw[1], height)),
                max(0.0, min(raw[2], width)), max(0.0, min(raw[3], height)),
            )
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            by_image[image_id].append({
                "gt_id": next_gt_id, "class_name": name, "box": box,
                "size_bucket": size_bucket(box, width * height),
            })
            next_gt_id += 1
    return by_image, sizes, expected


def clean_box(raw, width: int, height: int):
    if not isinstance(raw, list) or len(raw) != 4:
        return None
    try:
        values = tuple(float(value) for value in raw)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in values):
        return None
    box = (
        max(0.0, min(values[0], width)), max(0.0, min(values[1], height)),
        max(0.0, min(values[2], width)), max(0.0, min(values[3], height)),
    )
    return box if box[2] > box[0] and box[3] > box[1] else None


def load_predictions(path: Path, sizes, expected):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("images"), list):
        raise ValueError("Expected raw prediction JSON with top-level images")
    records, invalid = {}, 0
    for record in payload["images"]:
        image_id = Path(str(record["image_id"])).name
        if image_id not in expected or image_id in records:
            raise ValueError(f"Unexpected or duplicate image: {image_id}")
        width, height = sizes[image_id]
        predictions = []
        for index, pred in enumerate(record.get("predictions", [])):
            try:
                score = float(pred["score"])
            except (KeyError, TypeError, ValueError):
                invalid += 1
                continue
            box = clean_box(pred.get("bbox_xyxy", pred.get("bbox")), width, height)
            if box is None or not math.isfinite(score) or not 0 <= score <= 1:
                invalid += 1
                continue
            predictions.append({
                "pred_index": index, "score": score, "box": box,
                "source": str(pred.get("source", "unknown")),
                "touches_internal_border": bool(pred.get("touches_internal_border", False)),
            })
        records[image_id] = sorted(predictions, key=lambda item: item["score"], reverse=True)
    missing = expected - set(records)
    if missing:
        raise ValueError(f"Predictions missing {len(missing)} images")
    return records, invalid


def classify_predictions(records, gt_by_image, match_iou, near_iou, background_iou):
    audit_rows, matched_gt = [], set()
    for image_id in sorted(records):
        ground_truth = gt_by_image[image_id]
        matched_in_image = set()
        for rank, pred in enumerate(records[image_id], 1):
            overlaps = [(box_iou(pred["box"], gt["box"]), gt) for gt in ground_truth]
            overlaps.sort(key=lambda pair: pair[0], reverse=True)
            best_iou, best_gt = overlaps[0] if overlaps else (0.0, None)
            eligible_unmatched = [
                (overlap, gt) for overlap, gt in overlaps
                if overlap >= match_iou and gt["gt_id"] not in matched_in_image
            ]
            matched = eligible_unmatched[0][1] if eligible_unmatched else None
            if matched is not None:
                outcome = "tp"
                matched_in_image.add(matched["gt_id"])
                matched_gt.add(matched["gt_id"])
            elif best_iou >= match_iou:
                outcome = "duplicate"
            elif best_iou >= near_iou:
                outcome = "near_miss"
            elif best_iou >= background_iou:
                outcome = "weak_overlap"
            else:
                outcome = "background"
            reference_gt = matched or best_gt
            audit_rows.append({
                "image_id": image_id, "pred_index": pred["pred_index"], "rank_in_image": rank,
                "image_has_gt": int(bool(ground_truth)),
                "score": pred["score"], "source": pred["source"],
                "touches_internal_border": int(pred["touches_internal_border"]),
                "outcome": outcome, "best_iou": round(best_iou, 8),
                "matched_gt_id": matched["gt_id"] if matched else "",
                "reference_gt_id": reference_gt["gt_id"] if reference_gt else "",
                "reference_class": reference_gt["class_name"] if reference_gt else "",
                "reference_size": reference_gt["size_bucket"] if reference_gt else "",
                "x1": pred["box"][0], "y1": pred["box"][1],
                "x2": pred["box"][2], "y2": pred["box"][3],
            })
    return audit_rows, matched_gt


def safe_ratio(numerator, denominator):
    return round(numerator / denominator, 8) if denominator else 0.0


def summarize_rows(rows, gt_count: int, image_count: int, label: str, value) -> dict:
    counts = Counter(row["outcome"] for row in rows)
    predictions = len(rows)
    tp = counts["tp"]
    output = {
        label: value, "predictions": predictions,
        **{outcome: counts[outcome] for outcome in OUTCOMES},
        "strict_precision": safe_ratio(tp, predictions),
        "recall": safe_ratio(tp, gt_count),
        "f1": safe_ratio(2 * tp, 2 * tp + (predictions - tp) + (gt_count - tp)),
        "duplicates_per_tp": safe_ratio(counts["duplicate"], tp),
        "avg_predictions_per_image": safe_ratio(predictions, image_count),
    }
    return output


def cumulative_rows(audit_rows, thresholds, gt_count, image_count):
    return [
        summarize_rows(
            [row for row in audit_rows if row["score"] >= threshold],
            gt_count, image_count, "score_threshold", threshold,
        )
        for threshold in sorted(set(thresholds))
    ]


def score_bin_rows(audit_rows, thresholds, gt_count, image_count):
    edges = sorted(set([0.0, *thresholds, 1.0]))
    output = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        rows = [
            row for row in audit_rows
            if lower <= row["score"] < upper or (upper == 1.0 and row["score"] == 1.0)
        ]
        summary = summarize_rows(rows, gt_count, image_count, "score_min", lower)
        summary["score_max"] = upper
        output.append(summary)
    return output


def group_rows(audit_rows, gt_by_image, threshold, field, output_name):
    kept = [row for row in audit_rows if row["score"] >= threshold]
    gt_counts = Counter()
    for ground_truth in gt_by_image.values():
        for gt in ground_truth:
            key = gt["class_name"] if field == "reference_class" else gt["size_bucket"]
            gt_counts[key] += 1
    keys = sorted(set(gt_counts) | {str(row[field]) for row in kept if row[field] != ""})
    result = []
    for key in keys:
        rows = [row for row in kept if row[field] == key]
        summary = summarize_rows(rows, gt_counts[key], len(gt_by_image), output_name, key)
        result.append(summary)
    return result


def source_rows(audit_rows, threshold, gt_count, image_count):
    kept = [row for row in audit_rows if row["score"] >= threshold]
    sources = sorted(set(row["source"] for row in kept))
    return [
        summarize_rows([row for row in kept if row["source"] == source], gt_count, image_count,
                       "source", source)
        for source in sources
    ]


def categorical_rows(audit_rows, threshold, field, label, gt_count, image_count):
    kept = [row for row in audit_rows if row["score"] >= threshold]
    values = sorted(set(row[field] for row in kept), key=str)
    return [
        summarize_rows([row for row in kept if row[field] == value], gt_count, image_count,
                       label, value)
        for value in values
    ]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not 0 <= args.background_iou < args.near_iou < args.match_iou <= 1:
        raise ValueError("Require background_iou < near_iou < match_iou")
    if any(not 0 <= value <= 1 for value in args.score_thresholds):
        raise ValueError("Score thresholds must be in [0, 1]")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    classes = load_classes(args.classes)
    gt_by_image, sizes, expected = load_ground_truth(args.manifest, args.split, classes)
    records, invalid = load_predictions(args.predictions, sizes, expected)
    audit_rows, matched_gt = classify_predictions(
        records, gt_by_image, args.match_iou, args.near_iou, args.background_iou
    )
    gt_count = sum(len(items) for items in gt_by_image.values())
    thresholds = sorted(set(args.score_thresholds))
    cumulative = cumulative_rows(audit_rows, thresholds, gt_count, len(expected))
    bins = score_bin_rows(audit_rows, thresholds, gt_count, len(expected))
    operating_threshold = 0.03 if 0.03 in thresholds else thresholds[0]
    by_class = group_rows(audit_rows, gt_by_image, operating_threshold,
                          "reference_class", "class_name")
    by_size = group_rows(audit_rows, gt_by_image, operating_threshold,
                         "reference_size", "size_bucket")
    by_source = source_rows(audit_rows, operating_threshold, gt_count, len(expected))
    by_image_status = categorical_rows(
        audit_rows, operating_threshold, "image_has_gt", "image_has_gt", gt_count, len(expected)
    )
    tile_rows = [row for row in audit_rows if row["source"] == "tile"]
    by_tile_border = categorical_rows(
        tile_rows, operating_threshold, "touches_internal_border", "touches_internal_border",
        gt_count, len(expected),
    )

    write_csv(args.output_dir / "prediction_audit.csv", audit_rows)
    write_csv(args.output_dir / "cumulative_by_threshold.csv", cumulative)
    write_csv(args.output_dir / "disjoint_score_bins.csv", bins)
    write_csv(args.output_dir / "by_class_at_operating_threshold.csv", by_class)
    write_csv(args.output_dir / "by_size_at_operating_threshold.csv", by_size)
    write_csv(args.output_dir / "by_source_at_operating_threshold.csv", by_source)
    write_csv(args.output_dir / "by_image_status_at_operating_threshold.csv", by_image_status)
    write_csv(args.output_dir / "by_tile_border_at_operating_threshold.csv", by_tile_border)
    report = {
        "metadata": {
            "predictions": str(args.predictions.resolve()), "manifest": str(args.manifest.resolve()),
            "split": args.split, "images": len(expected), "gt": gt_count,
            "valid_predictions": len(audit_rows), "invalid_predictions": invalid,
            "match_iou": args.match_iou, "near_iou": args.near_iou,
            "background_iou": args.background_iou,
            "operating_threshold_for_groups": operating_threshold,
            "matching": "per-image score-descending greedy one-to-one",
        },
        "cumulative_by_threshold": cumulative,
        "matched_gt_at_minimum_loaded_score": len(matched_gt),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
