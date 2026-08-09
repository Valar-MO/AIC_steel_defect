#!/usr/bin/env python3
"""Official validation report for merged GL-Cascade predictions, including background FP."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_predictions import ap_101, box_iou, load_classes, load_ground_truth, load_predictions, match_predictions, point_metrics


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("splits/v1/split_manifest.csv"))
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    parser.add_argument("--split", choices=("val",), default="val")
    parser.add_argument("--score-threshold", type=float, default=.30)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    opt = args(); classes = load_classes(opt.classes); class_to_id = {name: index for index, name in enumerate(classes)}
    gt, sizes, image_ids = load_ground_truth(opt.manifest, opt.split, class_to_id)
    predictions, _, invalid, clipped = load_predictions(opt.predictions, classes, sizes, image_ids, allow_missing=False)
    per_class, total, ap50_values = {}, {"tp": 0, "fp": 0, "fn": 0, "background_fp": 0}, []
    weak = {"huashang", "qilie", "zonglie"}
    for class_id, name in enumerate(classes):
        gt_count = sum(len(value) for value in gt[class_id].values())
        outcomes = match_predictions(predictions[class_id], gt[class_id], .50)
        point = point_metrics(outcomes, gt_count, opt.score_threshold)
        background = 0
        for score, image_id, box in predictions[class_id]:
            if score < opt.score_threshold:
                continue
            all_gt = [item for grouped in gt.values() for item in grouped.get(image_id, [])]
            if max((box_iou(box, item) for item in all_gt), default=0.) < .10:
                background += 1
        ap50 = ap_101(outcomes, gt_count); ap50_values.append(ap50)
        per_class[name] = {**point, "ap50": ap50, "background_fp": background}
        for key in total:
            total[key] += point.get(key, 0) if key != "background_fp" else background
    precision = total["tp"] / max(1, total["tp"] + total["fp"]); recall = total["tp"] / max(1, total["tp"] + total["fn"])
    report = {"metadata": {"split": "val", "score_threshold": opt.score_threshold, "invalid_predictions": invalid, "clipped_predictions": clipped},
              "overall": {**total, "precision": precision, "recall": recall,
                          "f1": 2 * precision * recall / max(1e-12, precision + recall),
                          "f2": 5 * precision * recall / max(1e-12, 4 * precision + recall),
                          "map50": sum(ap50_values) / len(ap50_values)},
              "weak_class_recall": {name: per_class[name]["recall"] for name in weak}, "per_class": per_class}
    opt.output.parent.mkdir(parents=True, exist_ok=True)
    opt.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"overall": report["overall"], "weak_class_recall": report["weak_class_recall"], "output": str(opt.output.resolve())}, ensure_ascii=False))

if __name__ == "__main__":
    main()
