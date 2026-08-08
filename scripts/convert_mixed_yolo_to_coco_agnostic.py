#!/usr/bin/env python3
"""Convert the existing AIC mixed YOLO dataset to class-agnostic COCO.

The conversion reuses every previously generated whole image, positive tile,
rare-class centered crop, and background tile. It does not recrop images.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mixed-dir", type=Path,
        default=Path("/root/autodl-tmp/datasets/AIC_steel_defect_mixed"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("/root/autodl-tmp/datasets/AIC_steel_defect_mixed_coco_agnostic"),
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "sample_path", "label_path", "source_image_id", "source_split",
        "sample_type", "crop_xyxy", "classes",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Mixed manifest lacks fields: {sorted(required)}")
    if any(row["source_split"] not in {"train", "val"} for row in rows):
        raise ValueError("Mixed manifest contains a split other than train/val")
    sample_names = [Path(row["sample_path"]).name for row in rows]
    if len(sample_names) != len(set(sample_names)):
        raise ValueError("Mixed sample basenames must be globally unique")
    return rows


def read_yolo_labels(path: Path, width: int, height: int) -> list[dict]:
    annotations = []
    if not path.is_file():
        raise FileNotFoundError(path)
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Expected five YOLO fields in {path}:{line_number}")
        try:
            source_class_id = int(fields[0])
            cx, cy, bw, bh = (float(value) for value in fields[1:])
        except ValueError as error:
            raise ValueError(f"Invalid YOLO row in {path}:{line_number}") from error
        if source_class_id < 0 or not all(math.isfinite(v) for v in (cx, cy, bw, bh)):
            raise ValueError(f"Invalid YOLO values in {path}:{line_number}")
        x1 = max(0.0, min((cx - bw / 2) * width, float(width)))
        y1 = max(0.0, min((cy - bh / 2) * height, float(height)))
        x2 = max(0.0, min((cx + bw / 2) * width, float(width)))
        y2 = max(0.0, min((cy + bh / 2) * height, float(height)))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Degenerate YOLO box in {path}:{line_number}")
        annotations.append({
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "area": (x2 - x1) * (y2 - y1),
            "source_class_id": source_class_id,
        })
    return annotations


def convert(mixed_dir: Path, output_dir: Path) -> dict:
    mixed_dir, output_dir = mixed_dir.resolve(), output_dir.resolve()
    rows = load_manifest(mixed_dir / "sample_manifest.csv")
    annotations_dir = output_dir / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    sample_types = Counter()
    source_classes = Counter()
    split_report = {}
    next_annotation_id = 1

    for split in ("train", "val"):
        images, annotations = [], []
        split_rows = [row for row in rows if row["source_split"] == split]
        for image_id, row in enumerate(split_rows, 1):
            image_path, label_path = Path(row["sample_path"]), Path(row["label_path"])
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            with Image.open(image_path) as image:
                width, height = image.size
            boxes = read_yolo_labels(label_path, width, height)
            declared_count = int(row.get("bbox_count", len(boxes)))
            if declared_count != len(boxes):
                raise ValueError(
                    f"bbox_count mismatch for {image_path.name}: {declared_count} vs {len(boxes)}"
                )
            sample_type = row["sample_type"]
            sample_types[f"{split}_{sample_type}"] += 1
            for name in filter(None, row["classes"].split("|")):
                source_classes[name] += 1
            try:
                relative_image = image_path.absolute().relative_to(mixed_dir)
            except ValueError as error:
                raise ValueError(f"Sample path is outside mixed dataset: {image_path}") from error
            images.append({
                "id": image_id,
                # Use paths relative to the mixed dataset root; no duplicated images.
                # Do not resolve: whole images are symlinks into the original dataset.
                "file_name": str(relative_image).replace("\\", "/"),
                "width": width,
                "height": height,
                "source_image_id": row["source_image_id"],
                "sample_type": sample_type,
                "crop_xyxy": json.loads(row["crop_xyxy"]) if row["crop_xyxy"] else None,
                "source_classes": row["classes"].split("|") if row["classes"] else [],
            })
            for box in boxes:
                annotations.append({
                    "id": next_annotation_id,
                    "image_id": image_id,
                    "category_id": 0,
                    "bbox": box["bbox"],
                    "area": box["area"],
                    "iscrowd": 0,
                    "source_class_id": box["source_class_id"],
                    "sample_type": sample_type,
                })
                next_annotation_id += 1

        payload = {
            "info": {"description": "AIC mixed dataset, class-agnostic defect detection"},
            "licenses": [],
            "images": images,
            "annotations": annotations,
            # D-FINE directly consumes zero-based IDs when remapping is disabled.
            "categories": [{"id": 0, "name": "defect", "supercategory": "defect"}],
        }
        output_path = annotations_dir / f"instances_{split}.json"
        output_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        split_report[split] = {
            "images": len(images),
            "annotations": len(annotations),
            "empty_images": len(images) - len({item["image_id"] for item in annotations}),
            "annotation_file": str(output_path),
        }

    report = {
        "mixed_dir": str(mixed_dir),
        "output_dir": str(output_dir),
        "image_root": str(mixed_dir),
        "category": {"id": 0, "name": "defect"},
        "splits": split_report,
        "sample_type_counts": dict(sample_types),
        "source_class_presence_counts": dict(source_classes),
        "reuse_policy": "annotations only; images remain in the existing mixed dataset",
        "validation_policy": "whole original validation images only",
    }
    (output_dir / "conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    args = parse_args()
    print(json.dumps(convert(args.mixed_dir, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
