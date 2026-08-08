#!/usr/bin/env python3
"""Evaluate DINOv2 defectness as a gate while preserving Selection ranking.

The verifier score is never used as the final detection score.  For pre-NMS
gating, candidates below the verifier threshold are removed and the survivors
are sorted/NMSed only by the original Hierarchical D-FINE Selection score.  A
post-NMS variant is included as a safer, less powerful control.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from compare_hierarchical_score_modes import (
    Prediction,
    first_point_reaching_recall,
    load_normalized_predictions,
    match_and_decompose,
    ranked_outcomes,
    run_canonical_evaluator,
)
from evaluate_predictions import box_iou, load_classes, load_ground_truth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--verifier-scores",
        type=Path,
        required=True,
        help="Query+DINO score payload produced by score_dfine_verifier.py.",
    )
    parser.add_argument("--manifest", type=Path, default=Path("splits/v1/split_manifest.csv"))
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-threshold", type=float, default=0.10)
    parser.add_argument("--nms-iou", type=float, default=0.80)
    parser.add_argument(
        "--class-aware-nms",
        action="store_true",
        help="Match the ordinary whole/tile fusion path by suppressing only "
             "same-class boxes.",
    )
    parser.add_argument(
        "--gate-thresholds",
        default="0,0.001,0.003,0.005,0.01,0.02,0.03,0.05,0.075,0.1,"
                "0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
    )
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--background-iou", type=float, default=0.10)
    parser.add_argument("--max-recall-drop", type=float, default=0.005)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_thresholds(raw: str) -> list[float]:
    thresholds = sorted({float(value.strip()) for value in raw.split(",") if value.strip()})
    if not thresholds or any(not 0.0 <= value <= 1.0 for value in thresholds):
        raise ValueError("Gate thresholds must be comma-separated values in [0,1]")
    return thresholds


def candidate_rows(cache: dict, verifier_probability: torch.Tensor) -> list[dict]:
    selection = cache["selection_logits"].double().sigmoid()
    class_probability = cache["class_logits"].float().softmax(dim=1)
    class_confidence, labels = class_probability.max(dim=1)
    rows = []
    count = len(selection)
    for index in range(len(selection)):
        image_index = int(cache["candidate_image_index"][index])
        label = int(labels[index])
        rows.append({
            "candidate_index": index,
            "image_index": image_index,
            "image_id": Path(str(cache["images"][image_index]["image_id"])).name,
            "class_id": label,
            "category_name": cache["classes"][label],
            "bbox_xyxy": [float(value) for value in cache["boxes_xyxy"][index]],
            # bf16 Selection logits contain many exact ties.  This epsilon is
            # far below model-score resolution and supplies one deterministic
            # order shared by NMS, ranked matching and FP decomposition.
            "selection_score": float(selection[index] + (count - index) * 1e-12),
            "verifier_probability": float(verifier_probability[index]),
            "class_probability": float(class_confidence[index]),
            "query_index": int(cache["query_indices"][index]),
        })
    return rows


def nms(rows: list[dict], threshold: float, class_aware: bool) -> list[dict]:
    ordered = sorted(
        rows,
        key=lambda row: (-row["selection_score"], row["candidate_index"]),
    )
    kept = []
    for row in ordered:
        if any(
            (not class_aware or selected["class_id"] == row["class_id"])
            and box_iou(row["bbox_xyxy"], selected["bbox_xyxy"]) > threshold
            for selected in kept
        ):
            continue
        kept.append(row)
    return kept


def gated_rows(
    grouped: dict[int, list[dict]],
    gate_threshold: float,
    nms_iou: float,
    class_aware: bool,
    order: str,
) -> list[dict]:
    result = []
    for rows in grouped.values():
        if order == "pre_nms":
            eligible = [
                row for row in rows
                if row["verifier_probability"] >= gate_threshold
            ]
            result.extend(nms(eligible, nms_iou, class_aware))
        elif order == "post_nms":
            result.extend(
                row for row in nms(rows, nms_iou, class_aware)
                if row["verifier_probability"] >= gate_threshold
            )
        else:
            raise ValueError(order)
    return result


def as_predictions(rows: list[dict]) -> list[Prediction]:
    return [
        Prediction(
            prediction_id=row["candidate_index"],
            image_id=row["image_id"],
            class_id=row["class_id"],
            class_name=row["category_name"],
            box=tuple(row["bbox_xyxy"]),
            objectness=row["selection_score"],
            class_confidence=row["class_probability"],
            score=row["selection_score"],
        )
        for row in rows
    ]


def equal_recall_point(
    predictions: list[Prediction],
    gt,
    classes: list[str],
    gt_counts: list[int],
    target_recall: float,
    match_iou: float,
    background_iou: float,
) -> dict:
    rankings = ranked_outcomes(predictions, gt, classes, match_iou)
    point = first_point_reaching_recall(rankings, gt_counts, target_recall)
    if point is None:
        return {"reachable": False, "target_recall": target_recall}
    selected = sorted(
        predictions, key=lambda item: (-item.score, item.prediction_id)
    )[:point["prediction_count"]]
    decomposition, _ = match_and_decompose(
        selected, gt, classes, 0.0, match_iou, background_iou
    )
    if decomposition["tp"] != point["tp"]:
        raise AssertionError("Ranked and decomposed TP counts disagree")
    return {
        "reachable": True,
        "selection_threshold": point["threshold"],
        **{key: point[key] for key in (
            "prediction_count", "tp", "fp", "fn", "precision", "recall", "f1"
        )},
        "fp_buckets": decomposition["fp_buckets"],
        "per_class": decomposition["per_class"],
    }


def compact_point(point: dict) -> dict:
    result = {
        key: point.get(key) for key in (
            "reachable", "selection_threshold", "prediction_count",
            "tp", "fp", "fn", "precision", "recall", "f1",
        ) if key in point
    }
    if point.get("reachable", True) and "fp_buckets" in point:
        result["fp_buckets"] = {
            name: values["count"] for name, values in point["fp_buckets"].items()
        }
    return result


def add_rates(point: dict, gt_total: int) -> dict:
    """Add ordinary point metrics to an FP-decomposition summary."""
    point = dict(point)
    tp, fp = int(point["tp"]), int(point["fp"])
    point["fn"] = gt_total - tp
    point["precision"] = tp / (tp + fp) if tp + fp else 0.0
    point["recall"] = tp / gt_total if gt_total else 0.0
    precision, recall = point["precision"], point["recall"]
    point["f1"] = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    return point


def write_prediction_payload(
    cache: dict,
    rows: list[dict],
    output: Path,
    *,
    variant: str,
    gate_threshold: float | None,
    nms_iou: float,
    class_aware: bool,
) -> None:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["image_index"]].append(row)
    records = []
    for image_index, image in enumerate(cache["images"]):
        predictions = []
        for row in sorted(
            grouped[image_index],
            key=lambda item: (-item["selection_score"], item["candidate_index"]),
        ):
            predictions.append({
                "class_id": row["class_id"],
                "category_name": row["category_name"],
                "bbox_xyxy": row["bbox_xyxy"],
                "score": row["selection_score"],
                "objectness": row["selection_score"],
                "selection_probability": row["selection_score"],
                "dino_defectness_probability": row["verifier_probability"],
                "class_probability": row["class_probability"],
                "query_index": row["query_index"],
                "candidate_index": row["candidate_index"],
                "source": "whole",
            })
        records.append({
            "image_id": image["image_id"],
            "source_path": image["source_path"],
            "width": image["width"],
            "height": image["height"],
            "predictions": predictions,
        })
    metadata = {
        "classes": cache["classes"],
        "architecture": "hierarchical_dfine_with_query_dino_defectness_gate",
        "variant": variant,
        "score_mode": "selection",
        "score_formula": "original_selection_probability",
        "gate_formula": (
            None if gate_threshold is None
            else f"sigmoid(query_dino_logit) >= {gate_threshold:g}"
        ),
        "nms_iou": nms_iou,
        "nms_class_aware": class_aware,
        "prediction_count": len(rows),
        "fixed_boxes_and_classes": True,
        "verifier_never_used_for_ranking": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"metadata": metadata, "images": records}, ensure_ascii=False),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not 0 <= args.selection_threshold <= 1:
        raise ValueError("--selection-threshold must be in [0,1]")
    if not 0 <= args.nms_iou <= 1:
        raise ValueError("--nms-iou must be in [0,1]")
    if not 0 <= args.background_iou < args.match_iou <= 1:
        raise ValueError("Require 0 <= background-iou < match-iou <= 1")
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output} is not empty; pass --overwrite")
    args.output.mkdir(parents=True, exist_ok=True)

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    score_payload = torch.load(
        args.verifier_scores, map_location="cpu", weights_only=False
    )
    verifier_scores = score_payload["scores"].double()
    if len(verifier_scores) != len(cache["selection_logits"]):
        raise ValueError("Verifier score count differs from candidate cache")
    verifier_probability = verifier_scores.sigmoid()

    classes = load_classes(args.classes)
    if classes != cache["classes"]:
        raise ValueError("Cache/classes mismatch")
    class_to_id = {name: index for index, name in enumerate(classes)}
    gt, sizes, image_ids = load_ground_truth(args.manifest, args.split, class_to_id)
    gt_counts = [
        sum(len(items) for items in gt[index].values())
        for index in range(len(classes))
    ]

    rows = candidate_rows(cache, verifier_probability)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["image_index"]].append(row)
    baseline_rows = []
    for image_rows in grouped.values():
        baseline_rows.extend(nms(
            image_rows, args.nms_iou, args.class_aware_nms
        ))
    baseline_predictions = as_predictions(baseline_rows)
    baseline_fixed, _ = match_and_decompose(
        baseline_predictions, gt, classes, args.selection_threshold,
        args.match_iou, args.background_iou,
    )
    baseline_fixed = add_rates(baseline_fixed, sum(gt_counts))
    baseline_recall = baseline_fixed["recall"]
    baseline_equal = equal_recall_point(
        baseline_predictions, gt, classes, gt_counts, baseline_recall,
        args.match_iou, args.background_iou,
    )

    scan_rows, detailed = [], {}
    generated: dict[tuple[str, float], list[dict]] = {}
    for order in ("pre_nms", "post_nms"):
        for threshold in parse_thresholds(args.gate_thresholds):
            retained_rows = gated_rows(
                grouped, threshold, args.nms_iou,
                args.class_aware_nms, order,
            )
            generated[(order, threshold)] = retained_rows
            predictions = as_predictions(retained_rows)
            fixed, _ = match_and_decompose(
                predictions, gt, classes, args.selection_threshold,
                args.match_iou, args.background_iou,
            )
            fixed = add_rates(fixed, sum(gt_counts))
            equal = equal_recall_point(
                predictions, gt, classes, gt_counts, baseline_recall,
                args.match_iou, args.background_iou,
            )
            weak_ok = False
            if equal.get("reachable"):
                weak_ok = all(
                    equal["per_class"][name]["tp"]
                    >= baseline_equal["per_class"][name]["tp"]
                    for name in ("qilie", "huashang")
                )
            background_reduction = None
            total_fp_reduction = None
            if equal.get("reachable"):
                base_background = baseline_equal["fp_buckets"]["background"]["count"]
                candidate_background = equal["fp_buckets"]["background"]["count"]
                background_reduction = 1.0 - candidate_background / max(1, base_background)
                total_fp_reduction = 1.0 - equal["fp"] / max(1, baseline_equal["fp"])
            key = f"{order}_gate_{threshold:g}"
            detailed[key] = {
                "order": order,
                "gate_threshold": threshold,
                "candidate_count_after_gate_and_nms": len(retained_rows),
                "fixed_selection_010": fixed,
                "equal_baseline_recall": equal,
                "weak_class_guardrail": weak_ok,
                "background_fp_reduction_at_equal_recall": background_reduction,
                "total_fp_reduction_at_equal_recall": total_fp_reduction,
            }
            scan_rows.append({
                "order": order,
                "gate_threshold": threshold,
                "candidate_count": len(retained_rows),
                "fixed_tp": fixed["tp"],
                "fixed_fp": fixed["fp"],
                "fixed_recall": fixed["recall"],
                "fixed_precision": fixed["precision"],
                "fixed_background_fp": fixed["fp_buckets"]["background"]["count"],
                "equal_recall_reachable": equal.get("reachable", False),
                "equal_selection_threshold": equal.get("selection_threshold"),
                "equal_tp": equal.get("tp"),
                "equal_fp": equal.get("fp"),
                "equal_background_fp": (
                    equal.get("fp_buckets", {}).get("background", {}).get("count")
                    if equal.get("reachable") else None
                ),
                "equal_background_reduction": background_reduction,
                "equal_total_fp_reduction": total_fp_reduction,
                "weak_class_guardrail": weak_ok,
            })

    eligible = [
        row for row in scan_rows
        if row["equal_recall_reachable"] and row["weak_class_guardrail"]
    ]
    best = min(
        eligible,
        key=lambda row: (
            row["equal_fp"],
            row["equal_background_fp"],
            -row["gate_threshold"],
        ),
        default=None,
    )
    fixed_eligible = [
        row for row in scan_rows
        if row["fixed_recall"] >= baseline_recall - args.max_recall_drop
    ]
    best_fixed = min(
        fixed_eligible,
        key=lambda row: (
            row["fixed_fp"],
            row["fixed_background_fp"],
            -row["gate_threshold"],
        ),
        default=None,
    )

    selected_variants = {"baseline_nms": baseline_rows}
    selected_metadata = {
        "baseline_nms": {"gate_threshold": None, "order": "baseline"}
    }
    for name, row in (("best_equal_recall", best), ("best_fixed_threshold", best_fixed)):
        if row is None:
            continue
        key = (row["order"], row["gate_threshold"])
        selected_variants[name] = generated[key]
        selected_metadata[name] = {
            "gate_threshold": row["gate_threshold"], "order": row["order"]
        }

    canonical = {}
    for name, variant_rows in selected_variants.items():
        info = selected_metadata[name]
        prediction_path = args.output / name / "predictions.json"
        write_prediction_payload(
            cache, variant_rows, prediction_path,
            variant=name,
            gate_threshold=info["gate_threshold"],
            nms_iou=args.nms_iou,
            class_aware=args.class_aware_nms,
        )
        report = run_canonical_evaluator(
            prediction_path, args.manifest, args.classes, args.split,
            args.selection_threshold, args.match_iou,
            args.output / name / "evaluation",
        )
        normalized = load_normalized_predictions(
            prediction_path, "objectness", classes, sizes, image_ids
        )
        decomposition, audit_rows = match_and_decompose(
            normalized, gt, classes, args.selection_threshold,
            args.match_iou, args.background_iou,
        )
        (args.output / name / "fp_decomposition.json").write_text(
            json.dumps(decomposition, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_csv(args.output / name / "prediction_audit.csv", audit_rows)
        canonical[name] = {
            "selection": info,
            "overall": report["overall"],
            "fp_buckets_at_selection_threshold": {
                bucket: values["count"]
                for bucket, values in decomposition["fp_buckets"].items()
            },
        }

    write_csv(args.output / "gate_scan.csv", scan_rows)
    summary = {
        "metadata": {
            "cache": str(args.cache.resolve()),
            "verifier_scores": str(args.verifier_scores.resolve()),
            "candidate_count": len(rows),
            "ground_truth_count": sum(gt_counts),
            "selection_threshold": args.selection_threshold,
            "nms_iou": args.nms_iou,
            "class_aware_nms": args.class_aware_nms,
            "verifier_role": "binary gate only; never ranking score",
            "verifier_scoring_runtime": {
                "elapsed_seconds": score_payload.get("elapsed_seconds"),
                "peak_cuda_gib": score_payload.get("peak_cuda_gib"),
            },
        },
        "baseline": {
            "fixed_selection_threshold": baseline_fixed,
            "equal_recall_reference": baseline_equal,
        },
        "best_equal_recall": best,
        "best_fixed_threshold": best_fixed,
        "canonical_selected_variants": canonical,
        "scan": detailed,
    }
    (args.output / "comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "baseline": compact_point(baseline_fixed),
        "best_equal_recall": best,
        "best_fixed_threshold": best_fixed,
        "canonical_selected_variants": canonical,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
