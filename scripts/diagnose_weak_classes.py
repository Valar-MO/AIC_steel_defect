#!/usr/bin/env python3
"""Diagnose weak detection classes by separating localization and classification errors.

The script uses the same cleaned VOC ground truth and prediction schema as
``evaluate_predictions.py``.  It reports conventional class-wise TP/FP/FN,
GT-centric confusion, and class-agnostic proposal coverage at several IoUs.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from evaluate_predictions import (
    box_iou,
    load_classes,
    load_ground_truth,
    load_predictions,
)


DEFAULT_WEAK_CLASSES = ("huashang", "qilie", "yanghuatiepi", "zonglie")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("splits/v1/split_manifest.csv"))
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--target-classes", nargs="+", default=list(DEFAULT_WEAK_CLASSES))
    parser.add_argument("--score-threshold", type=float, default=0.25,
                        help="Score used for conventional TP/FP/FN and confusion.")
    parser.add_argument("--proposal-score-threshold", type=float, default=0.05,
                        help="Lower score used to ask whether any class localized a GT.")
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--coverage-ious", nargs="+", type=float,
                        default=[0.10, 0.30, 0.50, 0.75])
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/error_analysis/weak_classes"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace, classes: list[str]) -> None:
    unknown = sorted(set(args.target_classes) - set(classes))
    if unknown:
        raise ValueError(f"Unknown target classes: {unknown}")
    if len(args.target_classes) != len(set(args.target_classes)):
        raise ValueError("target classes must be unique")
    values = [args.score_threshold, args.proposal_score_threshold,
              args.match_iou, *args.coverage_ious]
    if any(not 0 <= value <= 1 for value in values):
        raise ValueError("All score and IoU thresholds must be in [0, 1]")
    if args.proposal_score_threshold > args.score_threshold:
        raise ValueError("proposal score threshold must not exceed evaluation score threshold")
    if not args.coverage_ious or len(args.coverage_ious) != len(set(args.coverage_ious)):
        raise ValueError("coverage IoUs must be non-empty and unique")


def flatten_ground_truth(gt, class_ids: set[int]):
    by_image = defaultdict(list)
    for class_id in class_ids:
        for image_id, boxes in gt[class_id].items():
            for index, box in enumerate(boxes):
                by_image[image_id].append({
                    "class_id": class_id,
                    "gt_index": index,
                    "box": box,
                })
    return by_image


def flatten_predictions(predictions, minimum_score: float):
    by_image = defaultdict(list)
    for class_id, items in predictions.items():
        for score, image_id, box in items:
            if score >= minimum_score:
                by_image[image_id].append({
                    "class_id": class_id,
                    "score": score,
                    "box": box,
                })
    for items in by_image.values():
        items.sort(key=lambda item: item["score"], reverse=True)
    return by_image


def greedy_class_matches(class_predictions, gt_by_image, score_threshold, iou_threshold):
    """Return standard score-ordered one-to-one matches for one class."""
    matched = defaultdict(set)
    outcomes = []
    for score, image_id, box in class_predictions:
        if score < score_threshold:
            continue
        candidates = gt_by_image.get(image_id, [])
        best_index, best_iou = -1, -1.0
        for index, gt_box in enumerate(candidates):
            if index in matched[image_id]:
                continue
            iou = box_iou(box, gt_box)
            if iou > best_iou:
                best_index, best_iou = index, iou
        is_tp = best_index >= 0 and best_iou >= iou_threshold
        if is_tp:
            matched[image_id].add(best_index)
        outcomes.append({
            "image_id": image_id,
            "score": score,
            "bbox_xyxy": list(box),
            "outcome": "TP" if is_tp else "FP",
            "matched_gt_index": best_index if is_tp else None,
            "best_unmatched_iou": max(best_iou, 0.0),
        })
    return outcomes, matched


def best_candidate(gt_box, candidates, class_id=None):
    eligible = [item for item in candidates
                if class_id is None or item["class_id"] == class_id]
    if not eligible:
        return None, 0.0
    # IoU is primary for localization diagnosis; score breaks ties.
    item = max(eligible, key=lambda pred: (box_iou(gt_box, pred["box"]), pred["score"]))
    return item, box_iou(gt_box, item["box"])


def rounded(value):
    return round(float(value), 8)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    classes = load_classes(args.classes)
    validate_args(args, classes)
    class_to_id = {name: index for index, name in enumerate(classes)}
    target_ids = {class_to_id[name] for name in args.target_classes}

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gt, sizes, image_ids = load_ground_truth(args.manifest, args.split, class_to_id)
    predictions, seen_images, invalid, clipped = load_predictions(
        args.predictions, classes, sizes, image_ids, allow_missing=False)
    proposal_by_image = flatten_predictions(predictions, args.proposal_score_threshold)
    evaluation_by_image = flatten_predictions(predictions, args.score_threshold)
    target_gt_by_image = flatten_ground_truth(gt, target_ids)
    all_gt_by_image = flatten_ground_truth(gt, set(range(len(classes))))

    per_class = {}
    prediction_rows = []
    matched_by_class = {}
    for name in args.target_classes:
        class_id = class_to_id[name]
        outcomes, matched = greedy_class_matches(
            predictions[class_id], gt[class_id], args.score_threshold, args.match_iou)
        gt_count = sum(len(items) for items in gt[class_id].values())
        tp = sum(row["outcome"] == "TP" for row in outcomes)
        fp = len(outcomes) - tp
        fn = gt_count - tp
        matched_by_class[class_id] = matched
        per_class[name] = {
            "ground_truth_count": gt_count,
            "prediction_count": len(outcomes),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": rounded(tp / (tp + fp) if tp + fp else 0.0),
            "recall": rounded(tp / gt_count if gt_count else 0.0),
        }
        for row in outcomes:
            nearest = None
            nearest_iou = 0.0
            for candidate in all_gt_by_image.get(row["image_id"], []):
                iou = box_iou(row["bbox_xyxy"], candidate["box"])
                if iou > nearest_iou:
                    nearest, nearest_iou = candidate, iou
            nearest_name = classes[nearest["class_id"]] if nearest else ""
            if row["outcome"] == "TP":
                fp_reason = ""
            elif nearest_iou < args.match_iou:
                fp_reason = "background_or_poor_localization"
            elif nearest and nearest["class_id"] != class_id:
                fp_reason = "wrong_class"
            else:
                fp_reason = "duplicate_same_class"
            prediction_rows.append({
                "predicted_class": name,
                **row,
                "nearest_gt_class": nearest_name,
                "nearest_gt_iou": rounded(nearest_iou),
                "fp_reason": fp_reason,
            })

    gt_rows = []
    confusion = {name: Counter() for name in args.target_classes}
    coverage_counts = {
        name: {str(iou): {"any_class": 0, "same_class": 0}
               for iou in args.coverage_ious}
        for name in args.target_classes
    }
    failure_counts = {name: Counter() for name in args.target_classes}

    for image_id, gt_items in target_gt_by_image.items():
        for gt_item in gt_items:
            class_id = gt_item["class_id"]
            class_name = classes[class_id]
            gt_box = gt_item["box"]
            proposal_any, proposal_any_iou = best_candidate(
                gt_box, proposal_by_image.get(image_id, []))
            proposal_same, proposal_same_iou = best_candidate(
                gt_box, proposal_by_image.get(image_id, []), class_id)
            eval_any, eval_any_iou = best_candidate(
                gt_box, evaluation_by_image.get(image_id, []))
            eval_same, eval_same_iou = best_candidate(
                gt_box, evaluation_by_image.get(image_id, []), class_id)

            is_tp = gt_item["gt_index"] in matched_by_class[class_id].get(image_id, set())
            wrong_overlaps = [item for item in evaluation_by_image.get(image_id, [])
                              if item["class_id"] != class_id
                              and box_iou(gt_box, item["box"]) >= args.match_iou]
            wrong = max(wrong_overlaps, key=lambda item: item["score"], default=None)

            if is_tp:
                failure = "TP"
                confusion_label = class_name
            elif wrong is not None:
                failure = "wrong_class"
                confusion_label = classes[wrong["class_id"]]
            elif proposal_any_iou >= args.match_iou:
                failure = "low_score_or_duplicate"
                confusion_label = "unmatched_localized"
            elif proposal_any_iou >= min(args.coverage_ious):
                failure = "poor_localization"
                confusion_label = "poor_localization"
            else:
                failure = "no_proposal"
                confusion_label = "background"
            failure_counts[class_name][failure] += 1
            confusion[class_name][confusion_label] += 1

            for threshold in args.coverage_ious:
                coverage_counts[class_name][str(threshold)]["any_class"] += int(
                    proposal_any_iou >= threshold)
                coverage_counts[class_name][str(threshold)]["same_class"] += int(
                    proposal_same_iou >= threshold)

            gt_rows.append({
                "image_id": image_id,
                "true_class": class_name,
                "gt_index": gt_item["gt_index"],
                "gt_bbox_xyxy": json.dumps(list(gt_box)),
                "status": failure,
                "confusion_label": confusion_label,
                "best_any_class": classes[proposal_any["class_id"]] if proposal_any else "",
                "best_any_score": rounded(proposal_any["score"]) if proposal_any else "",
                "best_any_iou": rounded(proposal_any_iou),
                "best_same_score": rounded(proposal_same["score"]) if proposal_same else "",
                "best_same_iou": rounded(proposal_same_iou),
                "eval_best_any_class": classes[eval_any["class_id"]] if eval_any else "",
                "eval_best_any_score": rounded(eval_any["score"]) if eval_any else "",
                "eval_best_any_iou": rounded(eval_any_iou),
                "eval_best_same_score": rounded(eval_same["score"]) if eval_same else "",
                "eval_best_same_iou": rounded(eval_same_iou),
            })

    for name in args.target_classes:
        total = per_class[name]["ground_truth_count"]
        coverage = {}
        for threshold in args.coverage_ious:
            counts = coverage_counts[name][str(threshold)]
            coverage[str(threshold)] = {
                "any_class_count": counts["any_class"],
                "any_class_recall": rounded(counts["any_class"] / total if total else 0),
                "same_class_count": counts["same_class"],
                "same_class_recall": rounded(counts["same_class"] / total if total else 0),
            }
        per_class[name]["proposal_coverage"] = coverage
        per_class[name]["failure_modes"] = dict(failure_counts[name])
        per_class[name]["confusion"] = dict(confusion[name])

    report = {
        "metadata": {
            "predictions": str(args.predictions.resolve()),
            "manifest": str(args.manifest.resolve()),
            "split": args.split,
            "target_classes": args.target_classes,
            "score_threshold": args.score_threshold,
            "proposal_score_threshold": args.proposal_score_threshold,
            "match_iou": args.match_iou,
            "coverage_ious": args.coverage_ious,
            "evaluated_images": len(image_ids),
            "prediction_image_records": len(seen_images),
            "invalid_predictions_dropped": invalid,
            "predictions_clipped_to_image": clipped,
        },
        "per_class": per_class,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.output_dir / "ground_truth_diagnostics.csv", gt_rows,
              ["image_id", "true_class", "gt_index", "gt_bbox_xyxy", "status",
               "confusion_label", "best_any_class", "best_any_score", "best_any_iou",
               "best_same_score", "best_same_iou", "eval_best_any_class",
               "eval_best_any_score", "eval_best_any_iou", "eval_best_same_score",
               "eval_best_same_iou"])
    write_csv(args.output_dir / "target_prediction_outcomes.csv", prediction_rows,
              ["predicted_class", "image_id", "score", "bbox_xyxy", "outcome",
               "matched_gt_index", "best_unmatched_iou", "nearest_gt_class",
               "nearest_gt_iou", "fp_reason"])
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
