#!/usr/bin/env python3
"""Convert raw RT-DETR predictions to the official competition JSON schema."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import yaml


DEFAULT_DIR = Path("/root/autodl-tmp/submissions/rtdetr_l_1024_baseline")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, default=DEFAULT_DIR / "raw_predictions.json")
    p.add_argument("--output", type=Path, default=DEFAULT_DIR / "submission.json")
    p.add_argument("--report", type=Path, default=DEFAULT_DIR / "submission_report.json")
    p.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    p.add_argument("--min-score", type=float, default=0.001)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.min_score <= 1:
        raise ValueError("--min-score must be in [0, 1]")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}; pass --overwrite to replace it")
    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    with args.classes.open(encoding="utf-8") as f:
        classes = [str(x) for x in yaml.safe_load(f)["names"]]
    if raw.get("metadata", {}).get("classes") != classes:
        raise ValueError("Raw prediction classes differ from configs/classes.yaml")

    submission = []
    class_counts = Counter({name: 0 for name in classes})
    clipped = dropped = threshold_dropped = 0
    image_count = images_with_predictions = 0
    for image in raw.get("images", []):
        image_count += 1
        width, height = int(image["width"]), int(image["height"])
        kept_for_image = 0
        for prediction in image.get("predictions", []):
            score = float(prediction["score"])
            if score < args.min_score:
                threshold_dropped += 1
                continue
            name = prediction["category_name"]
            class_id = int(prediction["class_id"])
            if not 0 <= class_id < len(classes) or classes[class_id] != name:
                raise ValueError(f"Invalid class mapping in {image['image_id']}: {class_id}, {name}")
            values = [float(v) for v in prediction["bbox_xyxy"]]
            if len(values) != 4 or not all(math.isfinite(v) for v in values):
                dropped += 1
                continue
            raw_box = tuple(values)
            xmin = max(0, min(width, math.floor(values[0])))
            ymin = max(0, min(height, math.floor(values[1])))
            xmax = max(0, min(width, math.ceil(values[2])))
            ymax = max(0, min(height, math.ceil(values[3])))
            clipped += int((xmin, ymin, xmax, ymax) != raw_box)
            if xmin >= xmax or ymin >= ymax:
                dropped += 1
                continue
            submission.append({
                "image_id": image["image_id"],
                "category_name": name,
                "bbox": [xmin, ymin, xmax, ymax],
                "score": round(score, 8),
            })
            class_counts[name] += 1
            kept_for_image += 1
        images_with_predictions += int(kept_for_image > 0)

    submission.sort(key=lambda item: (item["image_id"], -item["score"], item["category_name"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(submission, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "raw_predictions": str(args.raw.resolve()),
        "submission": str(args.output.resolve()),
        "min_score": args.min_score,
        "processed_images": image_count,
        "images_with_predictions": images_with_predictions,
        "images_without_predictions": image_count - images_with_predictions,
        "prediction_count": len(submission),
        "class_prediction_counts": dict(class_counts),
        "threshold_dropped": threshold_dropped,
        "invalid_boxes_dropped": dropped,
        "boxes_clipped_or_integerized": clipped,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
