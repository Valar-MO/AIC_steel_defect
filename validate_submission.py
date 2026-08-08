#!/usr/bin/env python3
"""Validate an official-format submission against the local initial test set."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from PIL import Image
import yaml


DEFAULT_DIR = Path("/root/autodl-tmp/submissions/rtdetr_l_1024_baseline")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
REQUIRED_KEYS = {"image_id", "category_name", "bbox", "score"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--submission", type=Path, default=DEFAULT_DIR / "submission.json")
    p.add_argument("--raw", type=Path, default=DEFAULT_DIR / "raw_predictions.json")
    p.add_argument("--test-dir", type=Path,
                   default=Path("/root/autodl-tmp/datasets/AIC_steel_defect/initial_test"))
    p.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    p.add_argument("--report", type=Path, default=DEFAULT_DIR / "validation_report.json")
    return p.parse_args()


def main():
    args = parse_args()
    test_images = sorted(p.resolve() for p in args.test_dir.rglob("*")
                         if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    by_name = {p.name: p for p in test_images}
    errors, warnings = [], []
    if len(by_name) != len(test_images):
        errors.append("Test image basenames are not unique")
    with args.classes.open(encoding="utf-8") as f:
        classes = [str(x) for x in yaml.safe_load(f)["names"]]
    class_set = set(classes)
    submission = json.loads(args.submission.read_text(encoding="utf-8"))
    if not isinstance(submission, list):
        errors.append("Top-level JSON must be a list")
        submission = []

    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    raw_ids = [item.get("image_id") for item in raw.get("images", [])]
    if len(raw_ids) != len(test_images) or set(raw_ids) != set(by_name):
        errors.append("Raw inference does not cover every test image exactly once")

    sizes, class_counts, image_counts = {}, Counter(), Counter()
    seen_exact = set()
    scores = []
    for index, item in enumerate(submission):
        prefix = f"record[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{prefix}: must be a dictionary")
            continue
        if set(item) != REQUIRED_KEYS:
            errors.append(f"{prefix}: keys must be exactly {sorted(REQUIRED_KEYS)}")
            continue
        image_id, name = item["image_id"], item["category_name"]
        if image_id not in by_name:
            errors.append(f"{prefix}: unknown image_id {image_id!r}")
            continue
        if name not in class_set:
            errors.append(f"{prefix}: unknown category {name!r}")
        bbox = item["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4 or not all(isinstance(v, int) for v in bbox):
            errors.append(f"{prefix}: bbox must be four integers")
            continue
        if image_id not in sizes:
            with Image.open(by_name[image_id]) as image:
                sizes[image_id] = image.size
        width, height = sizes[image_id]
        xmin, ymin, xmax, ymax = bbox
        if not (0 <= xmin < xmax <= width and 0 <= ymin < ymax <= height):
            errors.append(f"{prefix}: bbox outside {width}x{height}: {bbox}")
        score = item["score"]
        if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score):
            errors.append(f"{prefix}: score must be finite numeric")
        elif not 0 <= score <= 1:
            errors.append(f"{prefix}: score outside [0,1]: {score}")
        else:
            scores.append(float(score))
        signature = (image_id, name, tuple(bbox), score)
        if signature in seen_exact:
            warnings.append(f"{prefix}: exact duplicate prediction")
        seen_exact.add(signature)
        class_counts[name] += 1
        image_counts[image_id] += 1

    if len(submission) == 0:
        warnings.append("Submission contains no predictions")
    report = {
        "status": "passed" if not errors else "failed",
        "test_image_count": len(test_images),
        "raw_processed_image_count": len(raw_ids),
        "submission_prediction_count": len(submission),
        "images_with_predictions": len(image_counts),
        "images_without_predictions": len(test_images) - len(image_counts),
        "class_prediction_counts": dict(class_counts),
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
        "errors": errors,
        "warnings": warnings[:100],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
