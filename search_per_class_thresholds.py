#!/usr/bin/env python3
"""Search class-specific score thresholds for a prediction JSON.

This is a post-processing helper: AP/mAP are still computed from the full
ranked predictions, while fixed-threshold metrics such as precision, recall,
F1, and F2 are recomputed after applying one threshold per class.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_predictions import (  # noqa: E402
    IOU_THRESHOLDS,
    ap_101,
    best_f1,
    best_f2,
    load_classes,
    load_ground_truth,
    load_predictions,
    match_predictions,
    point_metrics,
    rounded,
)


DEFAULT_THRESHOLDS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("splits/v1/split_manifest.csv"))
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--initial-threshold", type=float, default=0.05)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument(
        "--objective",
        choices=("f1", "f2", "macro_f1", "macro_f2"),
        default="f2",
        help="Metric optimized by greedy per-class threshold search.",
    )
    parser.add_argument("--min-overall-precision", type=float, default=0.0)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--allow-missing-images", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_thresholds(values: list[float]) -> list[float]:
    output = sorted(set(float(v) for v in values))
    if not output or output[0] < 0 or output[-1] > 1:
        raise ValueError("Thresholds must be non-empty and in [0, 1]")
    return output


def f_scores(precision: float, recall: float) -> tuple[float, float]:
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall else 0.0
    return f1, f2


def summarize(classes: list[str], per_class_points: dict[str, dict]) -> dict:
    totals = {
        key: sum(per_class_points[name][key] for name in classes)
        for key in ("tp", "fp", "fn", "prediction_count")
    }
    totals["ground_truth_count"] = sum(per_class_points[name]["ground_truth_count"] for name in classes)
    precision = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else 0.0
    recall = totals["tp"] / totals["ground_truth_count"] if totals["ground_truth_count"] else 0.0
    f1, f2 = f_scores(precision, recall)
    macro_f1 = sum(per_class_points[name]["f1"] for name in classes) / len(classes)
    macro_f2 = sum(per_class_points[name]["f2"] for name in classes) / len(classes)
    return {
        **totals,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "macro_f1": macro_f1,
        "macro_f2": macro_f2,
    }


def evaluate_thresholds(
    classes: list[str],
    outcomes_by_class: dict[int, list[tuple[float, int]]],
    gt_counts: dict[int, int],
    thresholds_by_class: dict[str, float],
) -> tuple[dict, dict]:
    per_class = {}
    for class_id, name in enumerate(classes):
        point = point_metrics(outcomes_by_class[class_id], gt_counts[class_id], thresholds_by_class[name])
        precision = point["precision"]
        recall = point["recall"]
        f1, f2 = f_scores(precision, recall)
        per_class[name] = {
            "threshold": thresholds_by_class[name],
            "ground_truth_count": gt_counts[class_id],
            **point,
            "f1": f1,
            "f2": f2,
        }
    overall = summarize(classes, per_class)
    return overall, per_class


def score_candidate(overall: dict, objective: str, min_precision: float) -> tuple[float, float, float]:
    if overall["precision"] < min_precision:
        return (-1.0, overall["precision"], overall["recall"])
    return (overall[objective], overall["precision"], overall["recall"])


def compute_ap_report(classes, predictions, gt, gt_counts, outcomes_at_point, thresholds_by_class):
    per_class = {}
    for class_id, name in enumerate(classes):
        aps = {}
        for threshold in IOU_THRESHOLDS:
            outcomes = match_predictions(predictions[class_id], gt[class_id], threshold)
            aps[f"{threshold:.2f}"] = ap_101(outcomes, gt_counts[class_id])
        point = point_metrics(
            outcomes_at_point[class_id],
            gt_counts[class_id],
            thresholds_by_class[name],
        )
        per_class[name] = {
            "threshold": thresholds_by_class[name],
            "ground_truth_count": gt_counts[class_id],
            **{k: rounded(v) if isinstance(v, float) else v for k, v in point.items()},
            "ap50": rounded(aps["0.50"]),
            "ap50_95": rounded(sum(v for v in aps.values() if v is not None) /
                               sum(v is not None for v in aps.values())),
            "ap_by_iou": {k: rounded(v) for k, v in aps.items()},
            "best_f1_at_iou": {k: rounded(v) for k, v in best_f1(outcomes_at_point[class_id], gt_counts[class_id]).items()},
            "best_f2_at_iou": {k: rounded(v) for k, v in best_f2(outcomes_at_point[class_id], gt_counts[class_id]).items()},
        }
    totals = {
        key: sum(per_class[name][key] for name in classes)
        for key in ("tp", "fp", "fn", "prediction_count", "ground_truth_count")
    }
    precision = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else 0.0
    recall = totals["tp"] / totals["ground_truth_count"] if totals["ground_truth_count"] else 0.0
    f1, f2 = f_scores(precision, recall)
    overall = {
        **totals,
        "precision": rounded(precision),
        "recall": rounded(recall),
        "f1": rounded(f1),
        "f2": rounded(f2),
        "macro_f1": rounded(sum(per_class[n]["f1"] for n in classes) / len(classes)),
        "macro_f2": rounded(sum(per_class[n]["f2"] for n in classes) / len(classes)),
        "map50": rounded(sum(per_class[n]["ap50"] for n in classes) / len(classes)),
        "map50_95": rounded(sum(per_class[n]["ap50_95"] for n in classes) / len(classes)),
    }
    return overall, per_class


def main() -> None:
    args = parse_args()
    thresholds = validate_thresholds(args.thresholds)
    if args.passes <= 0:
        raise ValueError("--passes must be positive")
    if not 0 <= args.initial_threshold <= 1 or not 0 <= args.min_overall_precision <= 1:
        raise ValueError("Initial threshold and precision constraint must be in [0, 1]")
    report_path = args.output_dir / "threshold_search_report.json"
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {report_path}; pass --overwrite")

    classes = load_classes(args.classes)
    class_to_id = {name: i for i, name in enumerate(classes)}
    gt, sizes, image_ids = load_ground_truth(args.manifest, args.split, class_to_id)
    predictions, seen, invalid, clipped = load_predictions(
        args.predictions, classes, sizes, image_ids, args.allow_missing_images
    )
    gt_counts = {class_id: sum(len(v) for v in gt[class_id].values()) for class_id in range(len(classes))}
    outcomes_at_point = {
        class_id: match_predictions(predictions[class_id], gt[class_id], args.iou_threshold)
        for class_id in range(len(classes))
    }

    current = {name: args.initial_threshold for name in classes}
    search_rows = []
    overall, per_class = evaluate_thresholds(classes, outcomes_at_point, gt_counts, current)
    best_score = score_candidate(overall, args.objective, args.min_overall_precision)
    search_rows.append({"pass": 0, "class": "__initial__", "threshold": args.initial_threshold, **overall})

    for pass_index in range(1, args.passes + 1):
        changed = False
        for name in classes:
            class_best_threshold = current[name]
            class_best_overall = overall
            class_best_per_class = per_class
            class_best_score = best_score
            for threshold in thresholds:
                candidate = dict(current)
                candidate[name] = threshold
                candidate_overall, candidate_per_class = evaluate_thresholds(
                    classes, outcomes_at_point, gt_counts, candidate
                )
                candidate_score = score_candidate(
                    candidate_overall, args.objective, args.min_overall_precision
                )
                search_rows.append({
                    "pass": pass_index,
                    "class": name,
                    "threshold": threshold,
                    **candidate_overall,
                })
                if candidate_score > class_best_score:
                    class_best_score = candidate_score
                    class_best_threshold = threshold
                    class_best_overall = candidate_overall
                    class_best_per_class = candidate_per_class
            if class_best_threshold != current[name]:
                current[name] = class_best_threshold
                overall = class_best_overall
                per_class = class_best_per_class
                best_score = class_best_score
                changed = True
        if not changed:
            break

    final_overall, final_per_class = compute_ap_report(
        classes, predictions, gt, gt_counts, outcomes_at_point, current
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "per_class_thresholds.json").write_text(
        json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = {
        "metadata": {
            "predictions": str(args.predictions.resolve()),
            "manifest": str(args.manifest.resolve()),
            "split": args.split,
            "objective": args.objective,
            "threshold_candidates": thresholds,
            "initial_threshold": args.initial_threshold,
            "min_overall_precision": args.min_overall_precision,
            "matching_iou_threshold": args.iou_threshold,
            "evaluated_images": len(image_ids),
            "prediction_image_records": len(seen),
            "invalid_predictions_dropped": invalid,
            "predictions_clipped_to_image": clipped,
        },
        "thresholds": current,
        "overall": final_overall,
        "per_class": final_per_class,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    with (args.output_dir / "threshold_search_trace.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(search_rows[0]))
        writer.writeheader()
        writer.writerows(search_rows)
    with (args.output_dir / "per_class_threshold_report.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["class", "threshold", "ground_truth_count", "prediction_count", "tp", "fp", "fn",
                  "precision", "recall", "f1", "f2", "ap50", "ap50_95"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in classes:
            row = {"class": name, **final_per_class[name]}
            writer.writerow({field: row.get(field) for field in fields})

    print(json.dumps({
        "thresholds": current,
        "overall": final_overall,
        "outputs": str(args.output_dir.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
