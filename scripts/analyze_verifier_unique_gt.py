#!/usr/bin/env python3
"""Class-agnostic unique-GT ranking audit for fixed verifier candidates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from evaluate_predictions import load_classes, load_ground_truth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--score", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--manifest", type=Path, default=Path("splits/v1/split_manifest.csv"))
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--recall-grid", type=float, nargs="+", default=[0.70, 0.74, 0.78, 0.80])
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--background-iou", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def box_iou(left, right) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def parse_score_items(items: list[str]) -> dict[str, Path]:
    result = {}
    for item in items:
        name, separator, raw_path = item.partition("=")
        if not separator or not name or name in result:
            raise ValueError(f"Invalid score item {item!r}")
        result[name] = Path(raw_path)
    return result


def rank_unique_gt(cache: dict, scores: torch.Tensor, gt_by_image: dict, match_iou: float):
    if len(scores) != len(cache["boxes_xyxy"]):
        raise ValueError("Score/cache length mismatch")
    order = sorted(
        range(len(scores)),
        key=lambda index: (float(scores[index]), -index),
        reverse=True,
    )
    matched: dict[str, set[int]] = {}
    rows = []
    for index in order:
        image_index = int(cache["candidate_image_index"][index])
        image_id = cache["images"][image_index]["image_id"]
        candidate = [float(value) for value in cache["boxes_xyxy"][index]]
        targets = gt_by_image.get(image_id, [])
        used = matched.setdefault(image_id, set())
        best_index, best_overlap = -1, -1.0
        for target_index, target in enumerate(targets):
            if target_index in used:
                continue
            overlap = box_iou(candidate, target)
            if overlap > best_overlap:
                best_index, best_overlap = target_index, overlap
        is_tp = best_index >= 0 and best_overlap >= match_iou
        if is_tp:
            used.add(best_index)
        rows.append({
            "candidate_index": index,
            "score": float(scores[index]),
            "tp": int(is_tp),
            "max_iou": float(cache["max_iou"][index]),
        })
    return rows


def point(rows: list[dict], gt_total: int, target_recall: float, background_iou: float):
    required = math.ceil(target_recall * gt_total - 1e-12)
    tp = 0
    selected = []
    for row in rows:
        selected.append(row)
        tp += row["tp"]
        if tp >= required:
            break
    if tp < required:
        return {"reachable": False, "target_recall": target_recall}
    buckets = {"background": 0, "localization": 0, "duplicate": 0}
    for row in selected:
        if row["tp"]:
            continue
        if row["max_iou"] < background_iou:
            buckets["background"] += 1
        elif row["max_iou"] < 0.5:
            buckets["localization"] += 1
        else:
            buckets["duplicate"] += 1
    return {
        "reachable": True,
        "target_recall": target_recall,
        "prediction_count": len(selected),
        "tp": tp,
        "fp": len(selected) - tp,
        "precision": tp / len(selected),
        "recall": tp / gt_total,
        "fp_buckets": buckets,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    classes = load_classes(args.classes)
    class_to_id = {name: index for index, name in enumerate(classes)}
    gt, _sizes, _images = load_ground_truth(args.manifest, args.split, class_to_id)
    gt_by_image: dict[str, list] = {}
    for class_id in range(len(classes)):
        for image_id, boxes in gt[class_id].items():
            gt_by_image.setdefault(image_id, []).extend(boxes)
    gt_total = sum(len(boxes) for boxes in gt_by_image.values())
    variants = {}
    for name, path in parse_score_items(args.score).items():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        ranked = rank_unique_gt(cache, payload["scores"], gt_by_image, args.match_iou)
        variants[name] = {
            "maximum_unique_gt_recall": sum(row["tp"] for row in ranked) / gt_total,
            "recall_grid": {
                f"{recall:.2f}": point(
                    ranked, gt_total, recall, args.background_iou
                )
                for recall in args.recall_grid
            },
        }
    baseline = variants.get("baseline")
    if baseline:
        for name, variant in variants.items():
            if name == "baseline":
                continue
            reductions = []
            for recall in args.recall_grid:
                key = f"{recall:.2f}"
                base_point = baseline["recall_grid"][key]
                candidate_point = variant["recall_grid"][key]
                if base_point.get("reachable") and candidate_point.get("reachable"):
                    reductions.append(
                        1.0
                        - candidate_point["fp_buckets"]["background"]
                        / max(1, base_point["fp_buckets"]["background"])
                    )
            variant["mean_background_fp_reduction_vs_baseline"] = (
                sum(reductions) / len(reductions) if reductions else None
            )
    result = {
        "metadata": {
            "ground_truth_count": gt_total,
            "candidate_count": len(cache["selection_logits"]),
            "classes_ignored": True,
            "each_gt_matches_at_most_once": True,
            "recall_grid": args.recall_grid,
        },
        "variants": variants,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
