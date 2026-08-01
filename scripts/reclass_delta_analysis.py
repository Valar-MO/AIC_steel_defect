#!/usr/bin/env python3
"""Analyze whether crop-classifier reclassification changes help or hurt.

The script compares a reclassified prediction file against its embedded
``original_*`` fields.  It reconstructs two prediction sets:

1. before reclassification: original class and original detector score
2. after reclassification: current class and current score

Both sets are matched to VOC ground truth using the same score-ranked,
class-wise, one-to-one matching policy as ``evaluate_predictions.py``.  For
each changed prediction, the script reports whether the change turned an FP
into a TP, a TP into an FP, or had no point-metric effect.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True,
                        help="Reclassified predictions JSON written by reclassify_predictions.py")
    parser.add_argument("--manifest", type=Path, default=Path("splits/v1/split_manifest.csv"))
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/reclass_delta_analysis"))
    parser.add_argument("--allow-missing-original", action="store_true",
                        help="Skip changed predictions without original_* fields instead of failing.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_records(path: Path, classes: list[str]) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "images" not in payload:
        raise ValueError("Expected reclassified prediction JSON with top-level 'images'")
    metadata_classes = payload.get("metadata", {}).get("classes")
    if metadata_classes is not None and list(metadata_classes) != classes:
        raise ValueError("Prediction metadata classes differ from classes config")
    return payload["images"]


def clean_box(raw_box: Any, width: int, height: int):
    if not isinstance(raw_box, list) or len(raw_box) != 4:
        return None
    box = tuple(float(v) for v in raw_box)
    if not all(math.isfinite(v) for v in box):
        return None
    clean = (
        max(0.0, min(box[0], width)),
        max(0.0, min(box[1], height)),
        max(0.0, min(box[2], width)),
        max(0.0, min(box[3], height)),
    )
    return clean if clean[2] > clean[0] and clean[3] > clean[1] else None


def changed_prediction(pred: dict[str, Any]) -> bool:
    if pred.get("crop_classifier", {}).get("reclassified") is True:
        return True
    original = pred.get("original_category_name")
    return original is not None and str(original) != str(pred.get("category_name"))


def flatten_records(
    records: list[dict[str, Any]],
    *,
    classes: list[str],
    sizes: dict[str, tuple[int, int]],
    expected_images: set[str],
    use_original: bool,
):
    class_to_id = {name: index for index, name in enumerate(classes)}
    predictions = defaultdict(list)
    metadata = {}
    invalid = 0
    missing_original = 0

    for image_index, record in enumerate(records):
        image_id = Path(str(record["image_id"])).name
        if image_id not in expected_images:
            raise ValueError(f"Prediction contains image outside evaluation split: {image_id}")
        width, height = sizes[image_id]
        for pred_index, pred in enumerate(record.get("predictions", [])):
            key = f"{image_index}:{pred_index}"
            if use_original:
                name = pred.get("original_category_name", pred.get("category_name"))
                score = pred.get("original_score", pred.get("score"))
                if changed_prediction(pred) and (
                    "original_category_name" not in pred or "original_score" not in pred
                ):
                    missing_original += 1
            else:
                name = pred.get("category_name")
                score = pred.get("score")

            name = str(name)
            if name not in class_to_id:
                raise ValueError(f"Unknown predicted class {name!r} in {image_id}")
            score = float(score)
            box = clean_box(pred.get("bbox_xyxy", pred.get("bbox")), width, height)
            if box is None or not math.isfinite(score):
                invalid += 1
                continue
            predictions[class_to_id[name]].append((score, image_id, box, key))
            metadata[key] = {
                "image_id": image_id,
                "bbox_xyxy": list(box),
                "pred": pred,
            }

    for items in predictions.values():
        items.sort(key=lambda row: row[0], reverse=True)
    return predictions, metadata, invalid, missing_original


def match_with_keys(class_predictions, gt_by_image, score_threshold: float, iou_threshold: float):
    matched = defaultdict(set)
    outcomes = {}
    for score, image_id, box, key in class_predictions:
        if score < score_threshold:
            outcomes[key] = {
                "kept": False,
                "outcome": "DROP",
                "score": score,
                "matched_gt_index": None,
                "best_iou": 0.0,
            }
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
        outcomes[key] = {
            "kept": True,
            "outcome": "TP" if is_tp else "FP",
            "score": score,
            "matched_gt_index": best_index if is_tp else None,
            "best_iou": max(best_iou, 0.0),
        }
    return outcomes


def build_outcomes(predictions, gt, classes: list[str], score_threshold: float, iou_threshold: float):
    outcomes = {}
    for class_id, _name in enumerate(classes):
        outcomes.update(match_with_keys(predictions[class_id], gt[class_id], score_threshold, iou_threshold))
    return outcomes


def classify_delta(before: dict[str, Any], after: dict[str, Any]) -> str:
    before_tp = before["outcome"] == "TP"
    after_tp = after["outcome"] == "TP"
    if not before_tp and after_tp:
        return "good_change"
    if before_tp and not after_tp:
        return "bad_change"
    if before_tp and after_tp:
        return "tp_to_tp"
    if before["kept"] and not after["kept"]:
        return "dropped_after"
    if not before["kept"] and after["kept"]:
        return "revived_after"
    return "neutral_fp"


def rounded(value: float | None):
    return None if value is None else round(float(value), 8)


def mean(values: list[float]):
    return sum(values) / len(values) if values else None


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not 0 <= args.score_threshold <= 1 or not 0 <= args.match_iou <= 1:
        raise ValueError("score-threshold and match-iou must be in [0, 1]")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    classes = load_classes(args.classes)
    class_to_id = {name: index for index, name in enumerate(classes)}
    gt, sizes, image_ids = load_ground_truth(args.manifest, args.split, class_to_id)
    records = load_records(args.predictions, classes)

    before_predictions, before_meta, before_invalid, missing_original = flatten_records(
        records, classes=classes, sizes=sizes, expected_images=image_ids, use_original=True)
    after_predictions, after_meta, after_invalid, _ = flatten_records(
        records, classes=classes, sizes=sizes, expected_images=image_ids, use_original=False)
    if missing_original and not args.allow_missing_original:
        raise ValueError(f"{missing_original} changed predictions miss original_* fields")

    before_outcomes = build_outcomes(before_predictions, gt, classes, args.score_threshold, args.match_iou)
    after_outcomes = build_outcomes(after_predictions, gt, classes, args.score_threshold, args.match_iou)

    detail_rows = []
    pair_stats = defaultdict(lambda: {
        "count": 0,
        "good_change": 0,
        "bad_change": 0,
        "tp_to_tp": 0,
        "neutral_fp": 0,
        "dropped_after": 0,
        "revived_after": 0,
        "det_scores": [],
        "new_scores": [],
        "cls_conf": [],
        "before_iou": [],
        "after_iou": [],
    })
    overall = Counter()

    for image_index, record in enumerate(records):
        for pred_index, pred in enumerate(record.get("predictions", [])):
            if not changed_prediction(pred):
                continue
            key = f"{image_index}:{pred_index}"
            if key not in before_outcomes or key not in after_outcomes:
                continue
            original_name = str(pred.get("original_category_name", pred.get("category_name")))
            final_name = str(pred.get("category_name"))
            pair = f"{original_name}->{final_name}"
            before = before_outcomes[key]
            after = after_outcomes[key]
            delta = classify_delta(before, after)
            overall[delta] += 1
            stats = pair_stats[pair]
            stats["count"] += 1
            stats[delta] += 1
            old_score = float(pred.get("original_score", pred.get("score")))
            new_score = float(pred.get("score"))
            cls_conf = pred.get("crop_classifier", {}).get("top_confidence")
            stats["det_scores"].append(old_score)
            stats["new_scores"].append(new_score)
            if cls_conf is not None:
                stats["cls_conf"].append(float(cls_conf))
            stats["before_iou"].append(float(before["best_iou"]))
            stats["after_iou"].append(float(after["best_iou"]))
            detail_rows.append({
                "image_id": Path(str(record["image_id"])).name,
                "prediction_key": key,
                "pair": pair,
                "from_class": original_name,
                "to_class": final_name,
                "delta": delta,
                "before_outcome": before["outcome"],
                "after_outcome": after["outcome"],
                "before_kept": before["kept"],
                "after_kept": after["kept"],
                "before_iou": rounded(before["best_iou"]),
                "after_iou": rounded(after["best_iou"]),
                "old_score": rounded(old_score),
                "new_score": rounded(new_score),
                "score_delta": rounded(new_score - old_score),
                "classifier_conf": rounded(float(cls_conf)) if cls_conf is not None else None,
                "bbox_xyxy": json.dumps(pred.get("bbox_xyxy", pred.get("bbox")), ensure_ascii=False),
            })

    pair_rows = []
    for pair, stats in pair_stats.items():
        good = stats["good_change"]
        bad = stats["bad_change"]
        net = good - bad
        pair_rows.append({
            "pair": pair,
            "count": stats["count"],
            "good_change": good,
            "bad_change": bad,
            "tp_to_tp": stats["tp_to_tp"],
            "neutral_fp": stats["neutral_fp"],
            "dropped_after": stats["dropped_after"],
            "revived_after": stats["revived_after"],
            "net_gain": net,
            "good_rate": rounded(good / stats["count"]) if stats["count"] else 0.0,
            "bad_rate": rounded(bad / stats["count"]) if stats["count"] else 0.0,
            "avg_det_score": rounded(mean(stats["det_scores"])),
            "avg_new_score": rounded(mean(stats["new_scores"])),
            "avg_score_delta": rounded(mean([n - d for d, n in zip(stats["det_scores"], stats["new_scores"])])),
            "avg_cls_conf": rounded(mean(stats["cls_conf"])),
            "avg_before_iou": rounded(mean(stats["before_iou"])),
            "avg_after_iou": rounded(mean(stats["after_iou"])),
        })
    pair_rows.sort(key=lambda row: (row["net_gain"], row["good_change"], -row["bad_change"], row["count"]), reverse=True)

    suggested_allow = [
        row["pair"] for row in pair_rows
        if row["good_change"] > row["bad_change"] and row["good_change"] >= 1
    ]
    suggested_block = [
        row["pair"] for row in pair_rows
        if row["bad_change"] > row["good_change"] and row["bad_change"] >= 1
    ]

    write_csv(args.output_dir / "changed_predictions.csv", detail_rows, [
        "image_id", "prediction_key", "pair", "from_class", "to_class", "delta",
        "before_outcome", "after_outcome", "before_kept", "after_kept",
        "before_iou", "after_iou", "old_score", "new_score", "score_delta",
        "classifier_conf", "bbox_xyxy",
    ])
    write_csv(args.output_dir / "pair_delta_summary.csv", pair_rows, [
        "pair", "count", "good_change", "bad_change", "tp_to_tp", "neutral_fp",
        "dropped_after", "revived_after", "net_gain", "good_rate", "bad_rate",
        "avg_det_score", "avg_new_score", "avg_score_delta", "avg_cls_conf",
        "avg_before_iou", "avg_after_iou",
    ])

    summary = {
        "metadata": {
            "predictions": str(args.predictions.resolve()),
            "manifest": str(args.manifest.resolve()),
            "split": args.split,
            "score_threshold": args.score_threshold,
            "match_iou": args.match_iou,
            "changed_predictions": len(detail_rows),
            "invalid_before": before_invalid,
            "invalid_after": after_invalid,
            "missing_original_changed_predictions": missing_original,
        },
        "overall": {
            "good_change": overall["good_change"],
            "bad_change": overall["bad_change"],
            "tp_to_tp": overall["tp_to_tp"],
            "neutral_fp": overall["neutral_fp"],
            "dropped_after": overall["dropped_after"],
            "revived_after": overall["revived_after"],
            "net_gain": overall["good_change"] - overall["bad_change"],
        },
        "suggested_allow_pairs": suggested_allow,
        "suggested_block_pairs": suggested_block,
        "top_pairs": pair_rows[:30],
    }
    (args.output_dir / "delta_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "overall": summary["overall"],
        "suggested_allow_pairs": suggested_allow,
        "suggested_block_pairs": suggested_block,
        "outputs": str(args.output_dir.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
