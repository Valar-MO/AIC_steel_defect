#!/usr/bin/env python3
"""Compare Hierarchical D-FINE objectness and joint-score rankings without training.

The input must be the raw JSON written by ``predict_hierarchical_dfine.py``.
For every prediction it keeps the box and top-1 class unchanged and only replaces
``score`` with one of:

* objectness: model output score (the dual model stores defectness * quality);
* defectness: sigmoid(defectness logit), stored as ``defectness_probability``;
* joint: objectness * top-1 conditional class probability.

Besides invoking the project's canonical evaluator for both score modes, this
script assigns every fixed-threshold prediction to one mutually exclusive bucket:
TP, duplicate, wrong_class, localization, or background.  The four FP buckets
therefore reconcile exactly to the evaluator's FP total.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from evaluate_predictions import box_iou, load_classes, load_ground_truth


SCORE_MODES = (
    "objectness", "selection", "selection_quality",
    "selection_defectness", "defectness", "joint",
)
FP_BUCKETS = ("background", "duplicate", "wrong_class", "localization")


@dataclass(frozen=True)
class Prediction:
    prediction_id: int
    image_id: str
    class_id: int
    class_name: str
    box: tuple[float, float, float, float]
    objectness: float
    class_confidence: float
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("splits/v1/split_manifest.csv"))
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--score-modes", nargs="+", choices=SCORE_MODES,
        default=list(SCORE_MODES),
        help="Modes to evaluate. Use 'objectness' alone for legacy JSON files "
             "that did not store class_probability.",
    )
    parser.add_argument("--score-threshold", type=float, default=0.10)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument(
        "--background-iou", type=float, default=0.10,
        help="FPs below this maximum IoU to every GT are pure background; "
             "remaining sub-match-IoU FPs are localization errors.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def score_prediction(prediction: dict, mode: str) -> float:
    if mode not in SCORE_MODES:
        raise ValueError(f"Unknown score mode: {mode}")
    objectness = float(prediction.get("objectness", prediction.get("score", math.nan)))
    if not math.isfinite(objectness) or not 0.0 <= objectness <= 1.0:
        raise ValueError(f"Invalid objectness: {objectness}")
    if mode == "objectness":
        return objectness
    if mode in {"selection", "selection_quality", "selection_defectness"}:
        selection = float(prediction.get("selection_probability", math.nan))
        if not math.isfinite(selection) or not 0.0 <= selection <= 1.0:
            raise ValueError(
                "Raw predictions must contain a finite "
                "selection_probability in [0, 1]"
            )
        if mode == "selection":
            return selection
        gate_name = (
            "quality_probability"
            if mode == "selection_quality" else "defectness_probability"
        )
        gate = float(prediction.get(gate_name, math.nan))
        if not math.isfinite(gate) or not 0.0 <= gate <= 1.0:
            raise ValueError(f"Invalid {gate_name}: {gate}")
        return selection * gate
    if mode == "defectness":
        defectness = float(prediction.get("defectness_probability", math.nan))
        if not math.isfinite(defectness) or not 0.0 <= defectness <= 1.0:
            raise ValueError(
                "Raw predictions must contain a finite "
                "defectness_probability in [0, 1]"
            )
        return defectness
    confidence = float(prediction.get("class_probability", math.nan))
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError(
            "Raw predictions must contain a finite class_probability in [0, 1]"
        )
    return objectness * confidence


def write_rescored_payload(payload: dict, mode: str, output: Path) -> None:
    metadata = dict(payload.get("metadata", {}))
    metadata["score_mode"] = mode
    formulas = {
        "objectness": "selection * defectness * quality",
        "selection": "selection_probability",
        "selection_quality": "selection_probability * quality_probability",
        "selection_defectness": (
            "selection_probability * defectness_probability"
        ),
        "defectness": "defectness_probability",
        "joint": "objectness * class_probability",
    }
    metadata["score_formula"] = formulas[mode]
    records = []
    for record in payload["images"]:
        updated = dict(record)
        predictions = []
        for prediction in record.get("predictions", []):
            item = dict(prediction)
            item["source_score"] = float(prediction.get("score", 0.0))
            item["score"] = score_prediction(prediction, mode)
            predictions.append(item)
        updated["predictions"] = predictions
        records.append(updated)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"metadata": metadata, "images": records}, ensure_ascii=False),
        encoding="utf-8",
    )


def run_canonical_evaluator(
    predictions: Path,
    manifest: Path,
    classes: Path,
    split: str,
    score_threshold: float,
    match_iou: float,
    output_dir: Path,
) -> dict:
    evaluator = Path(__file__).with_name("evaluate_predictions.py")
    command = [
        sys.executable, str(evaluator),
        "--predictions", str(predictions),
        "--manifest", str(manifest),
        "--classes", str(classes),
        "--split", split,
        "--score-threshold", str(score_threshold),
        "--iou-threshold", str(match_iou),
        "--output-dir", str(output_dir),
        "--overwrite",
    ]
    subprocess.run(command, check=True)
    return json.loads((output_dir / "evaluation_report.json").read_text(encoding="utf-8"))


def load_normalized_predictions(
    path: Path,
    mode: str,
    classes: list[str],
    sizes: dict[str, tuple[int, int]],
    expected_images: set[str],
) -> list[Prediction]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("images")
    if not isinstance(records, list):
        raise ValueError("Expected prediction payload with an images list")
    class_to_id = {name: index for index, name in enumerate(classes)}
    seen, normalized = set(), []
    prediction_id = 0
    for record in records:
        image_id = Path(str(record["image_id"])).name
        if image_id not in expected_images:
            raise ValueError(f"Prediction outside split: {image_id}")
        if image_id in seen:
            raise ValueError(f"Duplicate image record: {image_id}")
        seen.add(image_id)
        width, height = sizes[image_id]
        for raw in record.get("predictions", []):
            name = str(raw["category_name"])
            if name not in class_to_id:
                raise ValueError(f"Unknown class {name!r}")
            class_id = class_to_id[name]
            if "class_id" in raw and int(raw["class_id"]) != class_id:
                raise ValueError(f"Class id/name mismatch in {image_id}")
            values = raw.get("bbox_xyxy", raw.get("bbox"))
            if not isinstance(values, list) or len(values) != 4:
                raise ValueError(f"Invalid box in {image_id}: {values}")
            box = tuple(float(value) for value in values)
            if not all(math.isfinite(value) for value in box):
                continue
            box = (
                max(0.0, min(box[0], width)), max(0.0, min(box[1], height)),
                max(0.0, min(box[2], width)), max(0.0, min(box[3], height)),
            )
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            objectness = float(raw.get("objectness", raw.get("score", math.nan)))
            confidence = float(raw.get("class_probability", math.nan))
            score = score_prediction(raw, mode)
            normalized.append(Prediction(
                prediction_id=prediction_id, image_id=image_id, class_id=class_id,
                class_name=name, box=box, objectness=objectness,
                class_confidence=confidence, score=score,
            ))
            prediction_id += 1
    missing = expected_images - seen
    if missing:
        raise ValueError(f"Predictions miss {len(missing)} images")
    return normalized


def flatten_ground_truth(gt, classes: list[str]):
    by_image = defaultdict(list)
    counts = [0] * len(classes)
    for class_id in range(len(classes)):
        for image_id, boxes in gt[class_id].items():
            for index, box in enumerate(boxes):
                by_image[image_id].append((class_id, index, tuple(box)))
                counts[class_id] += 1
    return by_image, counts


def match_and_decompose(
    predictions: Iterable[Prediction],
    gt,
    classes: list[str],
    score_threshold: float,
    match_iou: float,
    background_iou: float,
):
    """Return evaluator-compatible TP assignments and mutually exclusive FP reasons."""
    retained = [prediction for prediction in predictions
                if prediction.score >= score_threshold]
    by_class = defaultdict(list)
    for prediction in retained:
        by_class[prediction.class_id].append(prediction)
    for items in by_class.values():
        items.sort(key=lambda item: item.score, reverse=True)

    outcomes: dict[int, dict] = {}
    for class_id in range(len(classes)):
        matched = defaultdict(set)
        for prediction in by_class[class_id]:
            candidates = gt[class_id].get(prediction.image_id, [])
            best_index, best_iou = -1, -1.0
            for index, box in enumerate(candidates):
                if index in matched[prediction.image_id]:
                    continue
                overlap = box_iou(prediction.box, box)
                if overlap > best_iou:
                    best_index, best_iou = index, overlap
            if best_index >= 0 and best_iou >= match_iou:
                matched[prediction.image_id].add(best_index)
                outcomes[prediction.prediction_id] = {
                    "outcome": "tp", "matched_gt_class": class_id,
                    "matched_gt_index": best_index, "best_iou": best_iou,
                }

    gt_by_image, gt_counts = flatten_ground_truth(gt, classes)
    for prediction in retained:
        if prediction.prediction_id in outcomes:
            continue
        overlaps = [
            (box_iou(prediction.box, box), class_id, gt_index)
            for class_id, gt_index, box in gt_by_image.get(prediction.image_id, [])
        ]
        same = [row for row in overlaps if row[1] == prediction.class_id]
        different = [row for row in overlaps if row[1] != prediction.class_id]
        best_same = max(same, default=(0.0, -1, -1))
        best_different = max(different, default=(0.0, -1, -1))
        best_any = max(overlaps, default=(0.0, -1, -1))
        if best_same[0] >= match_iou:
            reason, reference = "duplicate", best_same
        elif best_different[0] >= match_iou:
            reason, reference = "wrong_class", best_different
        elif best_any[0] >= background_iou:
            reason, reference = "localization", best_any
        else:
            reason, reference = "background", best_any
        outcomes[prediction.prediction_id] = {
            "outcome": reason,
            "matched_gt_class": reference[1] if reference[1] >= 0 else None,
            "matched_gt_index": reference[2] if reference[2] >= 0 else None,
            "best_iou": reference[0],
            "best_same_class_iou": best_same[0],
            "best_different_class_iou": best_different[0],
        }

    rows = []
    for prediction in sorted(retained, key=lambda item: item.score, reverse=True):
        detail = outcomes[prediction.prediction_id]
        gt_class = detail.get("matched_gt_class")
        rows.append({
            "prediction_id": prediction.prediction_id,
            "image_id": prediction.image_id,
            "predicted_class": prediction.class_name,
            "predicted_class_id": prediction.class_id,
            "score": prediction.score,
            "objectness": prediction.objectness,
            "class_confidence": prediction.class_confidence,
            "outcome": detail["outcome"],
            "reference_gt_class": classes[gt_class] if gt_class is not None else "",
            "reference_gt_class_id": "" if gt_class is None else gt_class,
            "reference_gt_index": "" if detail.get("matched_gt_index") is None
                                      else detail["matched_gt_index"],
            "best_iou": detail["best_iou"],
            "best_same_class_iou": detail.get("best_same_class_iou", detail["best_iou"]),
            "best_different_class_iou": detail.get("best_different_class_iou", 0.0),
        })

    counts = Counter(row["outcome"] for row in rows)
    per_class = {}
    for class_id, name in enumerate(classes):
        class_rows = [row for row in rows if row["predicted_class_id"] == class_id]
        class_counts = Counter(row["outcome"] for row in class_rows)
        tp = class_counts["tp"]
        gt_count = gt_counts[class_id]
        fp = len(class_rows) - tp
        per_class[name] = {
            "ground_truth_count": gt_count, "prediction_count": len(class_rows),
            "tp": tp, "fp": fp, "fn": gt_count - tp,
            "recall": tp / gt_count if gt_count else 0.0,
            "precision": tp / len(class_rows) if class_rows else 0.0,
            **{bucket: class_counts[bucket] for bucket in FP_BUCKETS},
        }
    fp_total = sum(counts[bucket] for bucket in FP_BUCKETS)
    summary = {
        "score_threshold": score_threshold,
        "match_iou": match_iou,
        "background_iou": background_iou,
        "prediction_count": len(rows),
        "tp": counts["tp"],
        "fp": fp_total,
        "fn": sum(gt_counts) - counts["tp"],
        "fp_buckets": {
            bucket: {
                "count": counts[bucket],
                "share_of_fp": counts[bucket] / fp_total if fp_total else 0.0,
            }
            for bucket in FP_BUCKETS
        },
        "per_class": per_class,
    }
    if summary["tp"] + summary["fp"] != summary["prediction_count"]:
        raise AssertionError("TP/FP decomposition does not reconcile to prediction count")
    return summary, rows


def ranked_outcomes(predictions, gt, classes: list[str], match_iou: float):
    """Return all ranked TP flags using the same per-class greedy matching as evaluation."""
    by_class = defaultdict(list)
    for prediction in predictions:
        by_class[prediction.class_id].append(prediction)
    result = []
    for class_id in range(len(classes)):
        items = sorted(by_class[class_id], key=lambda item: item.score, reverse=True)
        matched = defaultdict(set)
        for prediction in items:
            candidates = gt[class_id].get(prediction.image_id, [])
            best_index, best_iou = -1, -1.0
            for index, box in enumerate(candidates):
                if index in matched[prediction.image_id]:
                    continue
                overlap = box_iou(prediction.box, box)
                if overlap > best_iou:
                    best_index, best_iou = index, overlap
            is_tp = best_index >= 0 and best_iou >= match_iou
            if is_tp:
                matched[prediction.image_id].add(best_index)
            result.append((prediction.score, int(is_tp), class_id))
    return sorted(result, key=lambda row: row[0], reverse=True)


def operating_point(rows, gt_counts: list[int], count: int) -> dict:
    selected = rows[:max(0, min(count, len(rows)))]
    tp = sum(row[1] for row in selected)
    fp = len(selected) - tp
    recalls = []
    for class_id, gt_count in enumerate(gt_counts):
        class_tp = sum(is_tp for _, is_tp, predicted_class in selected
                       if predicted_class == class_id)
        recalls.append(class_tp / gt_count if gt_count else 0.0)
    precision = tp / len(selected) if selected else 0.0
    recall = tp / sum(gt_counts) if sum(gt_counts) else 0.0
    threshold = selected[-1][0] if selected else 1.0
    return {
        "threshold": threshold, "prediction_count": len(selected),
        "tp": tp, "fp": fp, "fn": sum(gt_counts) - tp,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
              if precision + recall else 0.0,
        "per_class_recall": recalls,
    }


def best_point(rows, gt_counts: list[int], metric: str) -> dict:
    if metric not in {"f1", "f2"}:
        raise ValueError(metric)
    gt_total, tp, best = sum(gt_counts), 0, None
    for index, (_, is_tp, _) in enumerate(rows, 1):
        tp += is_tp
        precision = tp / index
        recall = tp / gt_total if gt_total else 0.0
        value = (2 * precision * recall / (precision + recall)
                 if metric == "f1" and precision + recall else
                 5 * precision * recall / (4 * precision + recall)
                 if metric == "f2" and 4 * precision + recall else 0.0)
        if best is None or value > best[0]:
            best = (value, index)
    point = operating_point(rows, gt_counts, best[1] if best else 0)
    point[metric] = best[0] if best else 0.0
    return point


def first_point_reaching_recall(rows, gt_counts: list[int], target_recall: float) -> dict | None:
    target_tp = math.ceil(target_recall * sum(gt_counts) - 1e-12)
    tp = 0
    for index, (_, is_tp, _) in enumerate(rows, 1):
        tp += is_tp
        if tp >= target_tp:
            return operating_point(rows, gt_counts, index)
    return None


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.score_threshold <= 1.0:
        raise ValueError("score-threshold must be in [0, 1]")
    if not 0.0 <= args.background_iou < args.match_iou <= 1.0:
        raise ValueError("Require 0 <= background-iou < match-iou <= 1")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} is not empty; pass --overwrite")

    payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("images"), list):
        raise ValueError("Expected raw hierarchical prediction JSON")
    classes = load_classes(args.classes)
    class_to_id = {name: index for index, name in enumerate(classes)}
    gt, sizes, image_ids = load_ground_truth(args.manifest, args.split, class_to_id)
    gt_counts = [sum(len(items) for items in gt[index].values())
                 for index in range(len(classes))]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    score_modes = tuple(dict.fromkeys(args.score_modes))
    reports, audits, rankings = {}, {}, {}
    for mode in score_modes:
        mode_dir = args.output_dir / mode
        prediction_path = mode_dir / "predictions.json"
        write_rescored_payload(payload, mode, prediction_path)
        reports[mode] = run_canonical_evaluator(
            prediction_path, args.manifest, args.classes, args.split,
            args.score_threshold, args.match_iou, mode_dir / "evaluation",
        )
        normalized = load_normalized_predictions(
            args.predictions, mode, classes, sizes, image_ids,
        )
        audit, audit_rows = match_and_decompose(
            normalized, gt, classes, args.score_threshold,
            args.match_iou, args.background_iou,
        )
        official = reports[mode]["overall"]
        for key in ("tp", "fp", "fn", "prediction_count"):
            if audit[key] != official[key]:
                raise AssertionError(
                    f"{mode} audit {key}={audit[key]} != evaluator {official[key]}"
                )
        audits[mode] = audit
        rankings[mode] = ranked_outcomes(normalized, gt, classes, args.match_iou)
        (mode_dir / "fp_decomposition.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_csv(mode_dir / "prediction_audit.csv", audit_rows)

    baseline_mode = "objectness" if "objectness" in reports else score_modes[0]
    baseline = reports[baseline_mode]["overall"]
    baseline_budget = int(baseline["prediction_count"])
    comparison = {
        "metadata": {
            "source_predictions": str(args.predictions.resolve()),
            "manifest": str(args.manifest.resolve()), "split": args.split,
            "classes": classes, "score_threshold": args.score_threshold,
            "match_iou": args.match_iou, "background_iou": args.background_iou,
            "comparison_control": "same checkpoint, boxes and top-1 labels; score only",
        },
        "fixed_threshold": {
            mode: {
                "overall": reports[mode]["overall"],
                "fp_decomposition": audits[mode]["fp_buckets"],
                "per_class": reports[mode]["per_class"],
            }
            for mode in score_modes
        },
        "best_operating_points": {
            mode: {
                "best_f1": best_point(rankings[mode], gt_counts, "f1"),
                "best_f2": best_point(rankings[mode], gt_counts, "f2"),
            }
            for mode in score_modes
        },
    }
    if "objectness" in reports and "joint" in reports:
        comparison["equal_prediction_budget"] = {
            mode: operating_point(rankings[mode], gt_counts, baseline_budget)
            for mode in score_modes
        }
        comparison["joint_at_objectness_fixed_recall"] = first_point_reaching_recall(
            rankings["joint"], gt_counts, float(baseline["recall"])
        )
        for mode in score_modes:
            equal = comparison["equal_prediction_budget"][mode]
            equal["per_class_recall"] = {
                name: equal["per_class_recall"][index]
                for index, name in enumerate(classes)
            }
        joint_recall = comparison["joint_at_objectness_fixed_recall"]
        if joint_recall is not None:
            joint_recall["per_class_recall"] = {
                name: joint_recall["per_class_recall"][index]
                for index, name in enumerate(classes)
            }
    for mode in score_modes:
        for name in ("best_f1", "best_f2"):
            point = comparison["best_operating_points"][mode][name]
            point["per_class_recall"] = {
                class_name: point["per_class_recall"][index]
                for index, class_name in enumerate(classes)
            }

    output = args.output_dir / "comparison_summary.json"
    output.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "fixed_threshold": {
            mode: reports[mode]["overall"] for mode in score_modes
        },
        "fp_decomposition": {
            mode: audits[mode]["fp_buckets"] for mode in score_modes
        },
        "equal_prediction_budget": comparison.get("equal_prediction_budget"),
        "output": str(output.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
