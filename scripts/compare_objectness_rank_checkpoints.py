#!/usr/bin/env python3
"""Compare objectness-ranking checkpoints against an unchanged baseline ranking."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from compare_hierarchical_score_modes import (
    load_normalized_predictions,
    match_and_decompose,
)
from evaluate_predictions import load_classes, load_ground_truth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, default=Path("splits/v1/split_manifest.csv"))
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--score-threshold", type=float, default=0.10)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--background-iou", type=float, default=0.10)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def audit_operating_point(rows: list[dict], gt_counts: list[int],
                          classes: list[str], count: int) -> dict:
    """Measure one exact audit prefix, including deterministic tie handling."""
    selected = rows[:count]
    counts = Counter(row["outcome"] for row in selected)
    fp = len(selected) - counts["tp"]
    per_class_recall = {}
    for class_id, name in enumerate(classes):
        class_tp = sum(
            row["outcome"] == "tp" and row["predicted_class_id"] == class_id
            for row in selected
        )
        per_class_recall[name] = class_tp / gt_counts[class_id] if gt_counts[class_id] else 0.0
    return {
        "prediction_count": len(selected),
        "tp": counts["tp"],
        "fp": fp,
        "fn": sum(gt_counts) - counts["tp"],
        "precision": counts["tp"] / len(selected) if selected else 0.0,
        "recall": counts["tp"] / sum(gt_counts) if sum(gt_counts) else 0.0,
        "threshold": selected[-1]["score"] if selected else 1.0,
        "per_class_recall": per_class_recall,
        "fp_buckets": {
            name: {
                "count": counts[name],
                "share_of_fp": counts[name] / fp if fp else 0.0,
            }
            for name in ("background", "duplicate", "wrong_class", "localization")
        },
    }


def audit_first_reaching_recall(rows: list[dict], gt_counts: list[int],
                                classes: list[str], target: float) -> dict | None:
    target_tp = round(target * sum(gt_counts))
    tp = 0
    for index, row in enumerate(rows, 1):
        tp += row["outcome"] == "tp"
        if tp >= target_tp:
            return audit_operating_point(rows, gt_counts, classes, index)
    return None


def main() -> None:
    args = parse_args()
    classes = load_classes(args.classes)
    class_to_id = {name: index for index, name in enumerate(classes)}
    gt, sizes, image_ids = load_ground_truth(args.manifest, args.split, class_to_id)
    gt_counts = [sum(len(items) for items in gt[index].values())
                 for index in range(len(classes))]

    baseline_predictions = load_normalized_predictions(
        args.baseline, "objectness", classes, sizes, image_ids
    )
    baseline_fixed, _ = match_and_decompose(
        baseline_predictions, gt, classes, args.score_threshold,
        args.match_iou, args.background_iou,
    )
    _, baseline_all_rows = match_and_decompose(
        baseline_predictions, gt, classes, 0.0, args.match_iou, args.background_iou
    )
    baseline_budget = baseline_fixed["prediction_count"]
    baseline_recall = baseline_fixed["tp"] / sum(gt_counts)
    baseline_equal_recall = audit_first_reaching_recall(
        baseline_all_rows, gt_counts, classes, baseline_recall
    )

    results = []
    for candidate_path in args.candidates:
        predictions = load_normalized_predictions(
            candidate_path, "objectness", classes, sizes, image_ids
        )
        fixed, _ = match_and_decompose(
            predictions, gt, classes, args.score_threshold,
            args.match_iou, args.background_iou,
        )
        _, all_rows = match_and_decompose(
            predictions, gt, classes, 0.0, args.match_iou, args.background_iou
        )
        equal_budget = audit_operating_point(
            all_rows, gt_counts, classes, baseline_budget
        )
        equal_recall = audit_first_reaching_recall(
            all_rows, gt_counts, classes, baseline_recall
        )
        results.append({
            "name": candidate_path.parent.name,
            "predictions": str(candidate_path.resolve()),
            "fixed_threshold": fixed,
            "at_baseline_prediction_budget": equal_budget,
            "first_point_reaching_baseline_recall": equal_recall,
        })

    payload = {
        "control": {
            "baseline": str(args.baseline.resolve()),
            "score_threshold": args.score_threshold,
            "match_iou": args.match_iou,
            "background_iou": args.background_iou,
            "ground_truth_count": sum(gt_counts),
            "baseline_fixed_threshold": baseline_fixed,
            "baseline_prediction_budget": baseline_budget,
            "baseline_recall": baseline_recall,
            "baseline_first_point_reaching_fixed_recall": baseline_equal_recall,
        },
        "candidates": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [
            "name", "fixed_tp", "fixed_fp", "fixed_recall",
            "budget_tp", "budget_fp", "budget_recall",
            "recall_target_predictions", "recall_target_fp",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in results:
            fixed = item["fixed_threshold"]
            budget = item["at_baseline_prediction_budget"]
            recall_point = item["first_point_reaching_baseline_recall"]
            writer.writerow({
                "name": item["name"],
                "fixed_tp": fixed["tp"], "fixed_fp": fixed["fp"],
                "fixed_recall": fixed["tp"] / sum(gt_counts),
                "budget_tp": budget["tp"], "budget_fp": budget["fp"],
                "budget_recall": budget["recall"],
                "recall_target_predictions": (
                    recall_point["prediction_count"] if recall_point else ""
                ),
                "recall_target_fp": recall_point["fp"] if recall_point else "",
            })
    print(json.dumps({"output": str(args.output), "csv": str(csv_path),
                      "candidate_count": len(results)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
