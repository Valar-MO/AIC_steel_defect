#!/usr/bin/env python3
"""Validate mixed-dataset labels, provenance, leakage, and sample balance."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image, ImageDraw


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", type=Path,
                   default=Path("/root/autodl-tmp/datasets/AIC_steel_defect_mixed"))
    p.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    p.add_argument("--visualize-count", type=int, default=24)
    p.add_argument("--max-patch-box-fraction", type=float, default=0.98)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    root = args.dataset_dir.resolve()
    with args.classes.open(encoding="utf-8") as f:
        classes = [str(x) for x in yaml.safe_load(f)["names"]]
    with (root / "sample_manifest.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    errors, warnings = [], []
    source_splits = defaultdict(set)
    sample_counts, instance_counts = Counter(), Counter({name: 0 for name in classes})
    parsed_samples = []
    seen_paths = set()
    for row in rows:
        sample = Path(row["sample_path"])
        label = Path(row["label_path"])
        split, sample_type = row["source_split"], row["sample_type"]
        source_splits[row["source_image_id"]].add(split)
        sample_counts[f"{split}_{sample_type}"] += 1
        if str(sample) in seen_paths:
            errors.append(f"Duplicate sample path: {sample}")
        seen_paths.add(str(sample))
        if not sample.is_file() or not label.is_file():
            errors.append(f"Missing image/label pair: {sample}, {label}")
            continue
        expected_image_dir = root / "images" / split
        expected_label_dir = root / "labels" / split
        if sample.parent != expected_image_dir or label.parent != expected_label_dir:
            errors.append(f"Path/split mismatch: {sample}")
        with Image.open(sample) as image:
            width, height = image.size
        boxes = []
        for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"Bad label fields: {label}:{line_number}")
                continue
            try:
                class_id = int(parts[0]); x, y, w, h = map(float, parts[1:])
            except ValueError:
                errors.append(f"Non-numeric label: {label}:{line_number}")
                continue
            if not 0 <= class_id < len(classes):
                errors.append(f"Invalid class ID: {label}:{line_number}")
                continue
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1 and
                    x - w / 2 >= -1e-6 and y - h / 2 >= -1e-6 and
                    x + w / 2 <= 1 + 1e-6 and y + h / 2 <= 1 + 1e-6):
                errors.append(f"Invalid normalized box: {label}:{line_number}")
                continue
            if sample_type != "whole" and (w >= args.max_patch_box_fraction or
                                            h >= args.max_patch_box_fraction):
                errors.append(
                    f"Extreme patch box (w={w:.6f}, h={h:.6f}): {label}:{line_number}"
                )
                continue
            instance_counts[classes[class_id]] += 1
            boxes.append((class_id, ((x-w/2)*width, (y-h/2)*height,
                                     (x+w/2)*width, (y+h/2)*height)))
        if int(row["bbox_count"]) != len(boxes):
            errors.append(f"Manifest bbox count mismatch: {sample}")
        if sample_type in {"random_background", "hard_negative"} and boxes:
            errors.append(f"Background sample has labels: {sample}")
        parsed_samples.append((row, sample, boxes))

    leaks = sorted(name for name, splits in source_splits.items() if len(splits) > 1)
    if leaks:
        errors.append(f"Source-image leakage across train/val: {leaks[:10]}")
    yaml_data = yaml.safe_load((root / "steel.yaml").read_text(encoding="utf-8"))
    if yaml_data.get("train") != "images/train" or yaml_data.get("val") != "images/val":
        errors.append("steel.yaml does not point to the expected train/val folders")
    actual_train = len(list((root / "images/train").iterdir()))
    actual_val = len(list((root / "images/val").iterdir()))
    manifest_train = sum(row["source_split"] == "train" for row in rows)
    manifest_val = sum(row["source_split"] == "val" for row in rows)
    if (actual_train, actual_val) != (manifest_train, manifest_val):
        errors.append("Image directory counts differ from sample manifest")

    visualization_dir = root / "validation_visualizations"
    if visualization_dir.exists():
        shutil.rmtree(visualization_dir)
    visualization_dir.mkdir(exist_ok=True)
    rng = random.Random(args.seed)
    positives = [item for item in parsed_samples if item[2]]
    by_type = defaultdict(list)
    for item in positives:
        by_type[item[0]["sample_type"]].append(item)
    for items in by_type.values():
        rng.shuffle(items)
    selected = []
    # Round-robin sampling prevents numerous full images from hiding the patch
    # types whose coordinate conversion most needs visual inspection.
    while len(selected) < min(args.visualize_count, len(positives)):
        added = False
        for sample_type in sorted(by_type):
            if by_type[sample_type] and len(selected) < args.visualize_count:
                selected.append(by_type[sample_type].pop())
                added = True
        if not added:
            break
    colors = ["#ff4040", "#00d070", "#40a0ff", "#ffd040", "#e060ff",
              "#00d8d8", "#ff8020", "#90e040", "#f060a0"]
    for index, (row, sample, boxes) in enumerate(selected):
        with Image.open(sample) as opened:
            image = opened.convert("RGB")
        draw = ImageDraw.Draw(image)
        for class_id, box in boxes:
            draw.rectangle(box, outline=colors[class_id % len(colors)], width=5)
            draw.text((box[0] + 4, box[1] + 4), classes[class_id],
                      fill=colors[class_id % len(colors)], stroke_width=1, stroke_fill="black")
        image.thumbnail((1400, 1000))
        image.save(visualization_dir / f"{index:03d}_{row['sample_type']}_{sample.name}", quality=90)

    report = {
        "status": "passed" if not errors else "failed", "dataset_dir": str(root),
        "manifest_samples": len(rows), "train_images": actual_train, "val_images": actual_val,
        "sample_counts": dict(sample_counts), "class_instance_counts": dict(instance_counts),
        "source_leakage_count": len(leaks), "visualizations": len(selected),
        "errors": errors, "warnings": warnings,
    }
    (root / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
