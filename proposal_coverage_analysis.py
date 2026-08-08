#!/usr/bin/env python3
"""Analyze detector proposal coverage upper bound.

This is intentionally class-agnostic: a GT object is considered covered if any
detector proposal, regardless of predicted class, overlaps it above an IoU
threshold.  The goal is to answer whether a second-stage crop
verifier/classifier has enough proposals to work with.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evaluate_predictions import box_iou, load_classes, load_ground_truth


DEFAULT_SCORE_THRESHOLDS = (0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1)
DEFAULT_IOU_THRESHOLDS = (0.30, 0.50, 0.75)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("splits/v1/split_manifest.csv"))
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--score-thresholds", nargs="+", type=float, default=list(DEFAULT_SCORE_THRESHOLDS))
    parser.add_argument("--iou-thresholds", nargs="+", type=float, default=list(DEFAULT_IOU_THRESHOLDS))
    parser.add_argument("--near-miss-low", type=float, default=0.30)
    parser.add_argument("--near-miss-high", type=float, default=0.50)
    parser.add_argument("--max-proposals-per-image", type=int, default=0,
                        help="Optional top-K by score per image. 0 means no top-K cap.")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/proposal_coverage_analysis"))
    parser.add_argument("--allow-missing-images", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    values = [*args.score_thresholds, *args.iou_thresholds, args.near_miss_low, args.near_miss_high]
    if any(not 0 <= value <= 1 for value in values):
        raise ValueError("All score/IoU thresholds must be in [0, 1]")
    if len(args.score_thresholds) != len(set(args.score_thresholds)):
        raise ValueError("--score-thresholds must be unique")
    if len(args.iou_thresholds) != len(set(args.iou_thresholds)):
        raise ValueError("--iou-thresholds must be unique")
    if args.near_miss_low >= args.near_miss_high:
        raise ValueError("--near-miss-low must be < --near-miss-high")
    if args.max_proposals_per_image < 0:
        raise ValueError("--max-proposals-per-image must be >= 0")


def clean_box(raw_box: Any, width: int, height: int):
    if not isinstance(raw_box, list) or len(raw_box) != 4:
        return None
    try:
        box = tuple(float(v) for v in raw_box)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in box):
        return None
    clean = (
        max(0.0, min(box[0], width)),
        max(0.0, min(box[1], height)),
        max(0.0, min(box[2], width)),
        max(0.0, min(box[3], height)),
    )
    return clean if clean[2] > clean[0] and clean[3] > clean[1] else None


def load_prediction_records(path: Path, classes: list[str], sizes, expected_images: set[str], allow_missing: bool):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "images" not in payload:
        raise ValueError("Expected prediction JSON with top-level 'images'")
    metadata_classes = payload.get("metadata", {}).get("classes")
    if metadata_classes is not None and list(metadata_classes) != classes:
        raise ValueError("Prediction metadata classes differ from classes config")

    records = {}
    invalid = clipped = 0
    for record in payload["images"]:
        image_id = Path(str(record["image_id"])).name
        if image_id not in expected_images:
            raise ValueError(f"Prediction contains image outside split: {image_id}")
        if image_id in records:
            raise ValueError(f"Duplicate prediction image record: {image_id}")
        width, height = sizes[image_id]
        proposals = []
        for pred_index, pred in enumerate(record.get("predictions", [])):
            score = float(pred.get("score", 0.0))
            if not math.isfinite(score):
                invalid += 1
                continue
            box = clean_box(pred.get("bbox_xyxy", pred.get("bbox")), width, height)
            if box is None:
                invalid += 1
                continue
            raw = pred.get("bbox_xyxy", pred.get("bbox"))
            clipped += int(tuple(float(v) for v in raw) != box)
            proposals.append({
                "pred_index": pred_index,
                "score": score,
                "box": box,
                "class_name": str(pred.get("category_name", "")),
                "class_id": int(pred.get("class_id", -1)),
            })
        proposals.sort(key=lambda item: item["score"], reverse=True)
        records[image_id] = proposals
    missing = expected_images - set(records)
    if missing and not allow_missing:
        raise ValueError(f"Predictions missing {len(missing)} images; examples: {sorted(missing)[:5]}")
    return records, invalid, clipped


def flatten_ground_truth(gt, classes: list[str]):
    rows = []
    for class_id, name in enumerate(classes):
        for image_id, boxes in gt[class_id].items():
            for gt_index, box in enumerate(boxes):
                rows.append({
                    "image_id": image_id,
                    "gt_index": gt_index,
                    "class_id": class_id,
                    "class_name": name,
                    "box": box,
                })
    rows.sort(key=lambda row: (row["image_id"], row["class_id"], row["gt_index"]))
    return rows


def proposals_for_image(records, image_id: str, score_threshold: float, max_per_image: int):
    proposals = [item for item in records.get(image_id, []) if item["score"] >= score_threshold]
    if max_per_image > 0:
        proposals = proposals[:max_per_image]
    return proposals


def best_proposal_for_gt(gt_box, proposals):
    best = None
    best_iou = 0.0
    for proposal in proposals:
        iou = box_iou(gt_box, proposal["box"])
        if iou > best_iou or (iou == best_iou and best is not None and proposal["score"] > best["score"]):
            best = proposal
            best_iou = iou
    return best, best_iou


def rounded(value):
    return round(float(value), 8)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    score_thresholds = sorted(args.score_thresholds)
    iou_thresholds = sorted(args.iou_thresholds)
    classes = load_classes(args.classes)
    class_to_id = {name: index for index, name in enumerate(classes)}
    gt, sizes, image_ids = load_ground_truth(args.manifest, args.split, class_to_id)
    prediction_records, invalid, clipped = load_prediction_records(
        args.predictions, classes, sizes, image_ids, args.allow_missing_images)
    gt_rows = flatten_ground_truth(gt, classes)
    gt_count = len(gt_rows)
    per_class_gt = Counter(row["class_name"] for row in gt_rows)

    threshold_rows = []
    per_class_rows = []
    detail_rows = []

    for score_threshold in score_thresholds:
        best_by_key = {}
        proposal_counts = []
        for image_id in image_ids:
            proposal_counts.append(
                len(proposals_for_image(prediction_records, image_id, score_threshold, args.max_proposals_per_image))
            )

        for gt_row in gt_rows:
            proposals = proposals_for_image(
                prediction_records, gt_row["image_id"], score_threshold, args.max_proposals_per_image)
            best, best_iou = best_proposal_for_gt(gt_row["box"], proposals)
            key = (gt_row["image_id"], gt_row["class_id"], gt_row["gt_index"])
            best_by_key[key] = (best, best_iou)

        for iou_threshold in iou_thresholds:
            covered = 0
            near_miss = 0
            low_miss = 0
            class_covered = Counter()
            class_near = Counter()
            class_low = Counter()
            for gt_row in gt_rows:
                key = (gt_row["image_id"], gt_row["class_id"], gt_row["gt_index"])
                best, best_iou = best_by_key[key]
                if best_iou >= iou_threshold:
                    covered += 1
                    class_covered[gt_row["class_name"]] += 1
                elif args.near_miss_low <= best_iou < args.near_miss_high:
                    near_miss += 1
                    class_near[gt_row["class_name"]] += 1
                else:
                    low_miss += 1
                    class_low[gt_row["class_name"]] += 1

            threshold_rows.append({
                "score_threshold": score_threshold,
                "iou_threshold": iou_threshold,
                "gt_count": gt_count,
                "covered": covered,
                "missed": gt_count - covered,
                "coverage": rounded(covered / gt_count if gt_count else 0.0),
                "near_miss_0.3_0.5": near_miss,
                "low_miss_lt_0.3_or_empty": low_miss,
                "avg_proposals_per_image": rounded(sum(proposal_counts) / max(1, len(proposal_counts))),
                "max_proposals_per_image": max(proposal_counts) if proposal_counts else 0,
            })

            for class_name in classes:
                total = per_class_gt[class_name]
                cov = class_covered[class_name]
                per_class_rows.append({
                    "score_threshold": score_threshold,
                    "iou_threshold": iou_threshold,
                    "class_name": class_name,
                    "gt_count": total,
                    "covered": cov,
                    "missed": total - cov,
                    "coverage": rounded(cov / total if total else 0.0),
                    "near_miss_0.3_0.5": class_near[class_name],
                    "low_miss_lt_0.3_or_empty": class_low[class_name],
                })

        # Detailed rows are written once per score threshold with best IoU,
        # because the same best proposal can be interpreted at multiple IoUs.
        for gt_row in gt_rows:
            key = (gt_row["image_id"], gt_row["class_id"], gt_row["gt_index"])
            best, best_iou = best_by_key[key]
            detail_rows.append({
                "score_threshold": score_threshold,
                "image_id": gt_row["image_id"],
                "gt_index": gt_row["gt_index"],
                "class_name": gt_row["class_name"],
                "best_iou": rounded(best_iou),
                "best_score": rounded(best["score"]) if best else "",
                "best_pred_class": best["class_name"] if best else "",
                "best_pred_index": best["pred_index"] if best else "",
                "covered_iou_0.3": int(best_iou >= 0.3),
                "covered_iou_0.5": int(best_iou >= 0.5),
                "covered_iou_0.75": int(best_iou >= 0.75),
                "miss_type": (
                    "covered_0.5" if best_iou >= 0.5 else
                    "near_miss_0.3_0.5" if best_iou >= 0.3 else
                    "low_miss_lt_0.3_or_empty"
                ),
            })

    write_csv(args.output_dir / "coverage_by_threshold.csv", threshold_rows, [
        "score_threshold", "iou_threshold", "gt_count", "covered", "missed", "coverage",
        "near_miss_0.3_0.5", "low_miss_lt_0.3_or_empty",
        "avg_proposals_per_image", "max_proposals_per_image",
    ])
    write_csv(args.output_dir / "coverage_by_class.csv", per_class_rows, [
        "score_threshold", "iou_threshold", "class_name", "gt_count", "covered", "missed",
        "coverage", "near_miss_0.3_0.5", "low_miss_lt_0.3_or_empty",
    ])
    write_csv(args.output_dir / "gt_best_proposals.csv", detail_rows, [
        "score_threshold", "image_id", "gt_index", "class_name", "best_iou", "best_score",
        "best_pred_class", "best_pred_index", "covered_iou_0.3", "covered_iou_0.5",
        "covered_iou_0.75", "miss_type",
    ])

    selected = [
        row for row in threshold_rows
        if row["score_threshold"] in (0.001, 0.003, 0.005, 0.01, 0.03, 0.05)
    ]
    summary = {
        "metadata": {
            "predictions": str(args.predictions.resolve()),
            "manifest": str(args.manifest.resolve()),
            "split": args.split,
            "gt_count": gt_count,
            "score_thresholds": score_thresholds,
            "iou_thresholds": iou_thresholds,
            "max_proposals_per_image": args.max_proposals_per_image,
            "invalid_predictions_dropped": invalid,
            "predictions_clipped_to_image": clipped,
        },
        "coverage": threshold_rows,
        "outputs": {
            "coverage_by_threshold": str((args.output_dir / "coverage_by_threshold.csv").resolve()),
            "coverage_by_class": str((args.output_dir / "coverage_by_class.csv").resolve()),
            "gt_best_proposals": str((args.output_dir / "gt_best_proposals.csv").resolve()),
        },
    }
    (args.output_dir / "coverage_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "selected_coverage": selected,
        "outputs": str(args.output_dir.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
