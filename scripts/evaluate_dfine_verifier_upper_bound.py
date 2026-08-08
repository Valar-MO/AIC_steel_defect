#!/usr/bin/env python3
"""Evaluate fixed-output verifier rankings over PR, recall grids and FP buckets."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from compare_hierarchical_score_modes import (
    first_point_reaching_recall,
    load_normalized_predictions,
    match_and_decompose,
    operating_point,
    ranked_outcomes,
    run_canonical_evaluator,
)
from evaluate_predictions import load_classes, load_ground_truth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--score",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Repeat for baseline/query/dino/query_dino score tensors.",
    )
    parser.add_argument("--manifest", type=Path, default=Path("splits/v1/split_manifest.csv"))
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recall-grid", type=float, nargs="+", default=[0.70, 0.74, 0.78, 0.80])
    parser.add_argument("--baseline-threshold", type=float, default=0.10)
    parser.add_argument("--fixed-budget", type=int, default=6213)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--background-iou", type=float, default=0.10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_scores(items: list[str]) -> dict[str, Path]:
    result = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected NAME=PATH, got {item!r}")
        name, raw_path = item.split("=", 1)
        if not name or name in result:
            raise ValueError(f"Invalid or duplicate score name {name!r}")
        result[name] = Path(raw_path)
    if "baseline" not in result:
        raise ValueError("Scores must include baseline=PATH")
    return result


def write_predictions(cache: dict, scores: torch.Tensor, output: Path, name: str) -> None:
    if len(scores) != len(cache["selection_logits"]):
        raise ValueError(f"{name} score count differs from cache")
    probabilities = scores.double().sigmoid()
    class_probability = cache["class_logits"].float().softmax(dim=1)
    confidence, labels = class_probability.max(dim=1)
    grouped = [[] for _ in cache["images"]]
    for index in range(len(scores)):
        image_index = int(cache["candidate_image_index"][index])
        label = int(labels[index])
        # bf16/float16 verifier logits contain many exact ties.  The canonical
        # evaluator and the decomposition code otherwise resolve a cutoff tie in
        # different stable orders.  A candidate-index epsilon is far below the
        # score resolution and only supplies one shared deterministic tie-break.
        ranking_score = float(
            probabilities[index] + (len(scores) - index) * 1e-12
        )
        grouped[image_index].append({
            "class_id": label,
            "category_name": cache["classes"][label],
            "bbox_xyxy": [float(value) for value in cache["boxes_xyxy"][index]],
            "score": ranking_score,
            "objectness": ranking_score,
            "class_probability": float(confidence[index]),
            "query_index": int(cache["query_indices"][index]),
            "candidate_index": index,
            "source": "whole",
        })
    records = []
    for image, predictions in zip(cache["images"], grouped, strict=True):
        predictions.sort(key=lambda row: row["score"], reverse=True)
        records.append({
            "image_id": image["image_id"],
            "source_path": image["source_path"],
            "width": image["width"],
            "height": image["height"],
            "predictions": predictions,
        })
    payload = {
        "metadata": {
            "classes": cache["classes"],
            "architecture": "fixed_dfine_dinov2_teacher_upper_bound",
            "variant": name,
            "candidate_floor": cache["min_selection"],
            "prediction_count": len(scores),
            "score_mode": "verifier_logit",
            "fixed_boxes_and_classes": True,
        },
        "images": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def point_with_decomposition(
    predictions, rankings, gt, classes, gt_counts, target_recall, match_iou, background_iou
):
    point = first_point_reaching_recall(rankings, gt_counts, target_recall)
    if point is None:
        return {"reachable": False, "target_recall": target_recall}
    selected = sorted(predictions, key=lambda item: item.score, reverse=True)[
        :point["prediction_count"]
    ]
    decomposition, _ = match_and_decompose(
        selected, gt, classes, 0.0, match_iou, background_iou
    )
    if decomposition["tp"] != point["tp"] or decomposition["fp"] != point["fp"]:
        raise AssertionError("Operating-point and FP-decomposition matching disagree")
    return {
        "reachable": True,
        "target_recall": target_recall,
        **point,
        "fp_buckets": decomposition["fp_buckets"],
        "per_class": decomposition["per_class"],
    }


def pr_rows(rankings, gt_total: int) -> list[dict]:
    rows, tp = [], 0
    last_tp = -1
    for count, (score, is_tp, _class_id) in enumerate(rankings, 1):
        tp += is_tp
        if tp == last_tp and count % 100 != 0:
            continue
        last_tp = tp
        rows.append({
            "threshold": score,
            "prediction_count": count,
            "tp": tp,
            "fp": count - tp,
            "precision": tp / count,
            "recall": tp / gt_total,
        })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_pr(curves: dict[str, list[dict]], path: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 6))
    for name, rows in curves.items():
        axis.plot(
            [row["recall"] for row in rows],
            [row["precision"] for row in rows],
            label=name,
            linewidth=1.8,
        )
    axis.set(
        xlabel="Recall",
        ylabel="Precision",
        title="Fixed D-FINE outputs: verifier precision-recall",
        xlim=(0.0, 0.85),
        ylim=(0.0, 1.0),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    score_paths = parse_scores(args.score)
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output} is not empty; pass --overwrite")
    args.output.mkdir(parents=True, exist_ok=True)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    classes = load_classes(args.classes)
    if classes != cache["classes"]:
        raise ValueError("Cache/classes mismatch")
    class_to_id = {name: index for index, name in enumerate(classes)}
    gt, sizes, image_ids = load_ground_truth(
        args.manifest, args.split, class_to_id
    )
    gt_counts = [
        sum(len(items) for items in gt[index].values())
        for index in range(len(classes))
    ]
    gt_total = sum(gt_counts)
    variants, curves = {}, {}
    for name, score_path in score_paths.items():
        score_payload = torch.load(score_path, map_location="cpu", weights_only=False)
        prediction_path = args.output / name / "predictions.json"
        write_predictions(cache, score_payload["scores"], prediction_path, name)
        report = run_canonical_evaluator(
            prediction_path,
            args.manifest,
            args.classes,
            args.split,
            0.0,
            args.match_iou,
            args.output / name / "canonical",
        )
        predictions = load_normalized_predictions(
            prediction_path, "objectness", classes, sizes, image_ids
        )
        rankings = ranked_outcomes(predictions, gt, classes, args.match_iou)
        grid = {
            f"{recall:.2f}": point_with_decomposition(
                predictions, rankings, gt, classes, gt_counts, recall,
                args.match_iou, args.background_iou,
            )
            for recall in args.recall_grid
        }
        target_804 = point_with_decomposition(
            predictions, rankings, gt, classes, gt_counts, 804 / gt_total,
            args.match_iou, args.background_iou,
        )
        fixed_budget = operating_point(rankings, gt_counts, args.fixed_budget)
        fixed_threshold, _ = match_and_decompose(
            predictions, gt, classes, args.baseline_threshold,
            args.match_iou, args.background_iou,
        )
        curve = pr_rows(rankings, gt_total)
        curves[name] = curve
        write_csv(args.output / name / "pr_curve.csv", curve)
        variants[name] = {
            "score_source": str(score_path.resolve()),
            "canonical": report["overall"],
            "recall_grid": grid,
            "at_tp_804": target_804,
            "at_fixed_budget": fixed_budget,
            "at_fixed_threshold_010": fixed_threshold,
            "scoring_runtime": {
                "elapsed_seconds": score_payload.get("elapsed_seconds"),
                "peak_cuda_gib": score_payload.get("peak_cuda_gib"),
            },
        }

    baseline = variants["baseline"]
    decisions = {}
    weak_classes = ("qilie", "huashang")
    for name, result in variants.items():
        if name == "baseline":
            continue
        background_reductions = []
        for recall in args.recall_grid:
            key = f"{recall:.2f}"
            base_point = baseline["recall_grid"][key]
            candidate_point = result["recall_grid"][key]
            if base_point.get("reachable") and candidate_point.get("reachable"):
                base_background = base_point["fp_buckets"]["background"]["count"]
                candidate_background = candidate_point["fp_buckets"]["background"]["count"]
                background_reductions.append(
                    1.0 - candidate_background / max(1, base_background)
                )
        average_background_reduction = (
            sum(background_reductions) / len(background_reductions)
            if background_reductions else float("-inf")
        )
        base_804, candidate_804 = baseline["at_tp_804"], result["at_tp_804"]
        total_fp_reduction = (
            1.0 - candidate_804["fp"] / max(1, base_804["fp"])
            if base_804.get("reachable") and candidate_804.get("reachable")
            else float("-inf")
        )
        weak_ok = bool(candidate_804.get("reachable"))
        weak_delta = {}
        if weak_ok:
            for class_name in weak_classes:
                base_tp = base_804["per_class"][class_name]["tp"]
                candidate_tp = candidate_804["per_class"][class_name]["tp"]
                weak_delta[class_name] = candidate_tp - base_tp
                weak_ok &= candidate_tp >= base_tp
        ap_delta = (
            result["canonical"]["map50"] - baseline["canonical"]["map50"]
        )
        budget_tp_delta = (
            result["at_fixed_budget"]["tp"] - baseline["at_fixed_budget"]["tp"]
        )
        checks = {
            "average_background_fp_reduction_ge_15pct":
                average_background_reduction >= 0.15,
            "tp804_total_fp_reduction_ge_10pct": total_fp_reduction >= 0.10,
            "fixed_budget_tp_above_baseline": budget_tp_delta > 0,
            "weak_classes_not_lower_at_tp804": weak_ok,
            "map50_delta_ge_0005": ap_delta >= 0.005,
        }
        decisions[name] = {
            "continue_to_distillation": all(checks.values()),
            "checks": checks,
            "average_background_fp_reduction": average_background_reduction,
            "tp804_total_fp_reduction": total_fp_reduction,
            "fixed_budget_tp_delta": budget_tp_delta,
            "weak_class_tp_delta_at_tp804": weak_delta,
            "map50_delta": ap_delta,
        }

    summary = {
        "metadata": {
            "cache": str(args.cache.resolve()),
            "candidate_floor": cache["min_selection"],
            "candidate_count": len(cache["selection_logits"]),
            "ground_truth_count": gt_total,
            "recall_grid": args.recall_grid,
            "fixed_budget": args.fixed_budget,
            "fixed_boxes_and_classes": True,
            "decision_rule": (
                "15% mean background-FP reduction over recall grid; 10% total-FP "
                "reduction at TP804; fixed-budget TP gain; no qilie/huashang TP loss; "
                "and +0.005 mAP50"
            ),
        },
        "variants": variants,
        "decisions": decisions,
    }
    (args.output / "comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_pr(curves, args.output / "pr_curves.png")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "candidate_count": len(cache["selection_logits"]),
        "decisions": decisions,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
