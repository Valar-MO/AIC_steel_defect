#!/usr/bin/env python3
"""Reuse the existing mixed images while restoring their original nine classes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image
import yaml

try:
    from .convert_mixed_yolo_to_coco_agnostic import load_manifest, read_yolo_labels
except ImportError:  # Direct script execution on the training server.
    from convert_mixed_yolo_to_coco_agnostic import load_manifest, read_yolo_labels


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixed-dir", type=Path,
                        default=Path("/root/autodl-tmp/datasets/AIC_steel_defect_mixed"))
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "/root/autodl-tmp/datasets/AIC_steel_defect_mixed_coco_hierarchical"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def convert(mixed_dir: Path, classes_path: Path, output_dir: Path, overwrite=False):
    names = (yaml.safe_load(classes_path.read_text(encoding="utf-8")) or {}).get("names")
    if not isinstance(names, list) or len(names) != 9 or len(set(names)) != 9:
        raise ValueError("Hierarchical D-FINE requires exactly nine unique classes")
    mixed_dir, output_dir = mixed_dir.resolve(), output_dir.resolve()
    rows = load_manifest(mixed_dir / "sample_manifest.csv")
    annotations_dir = output_dir / "annotations"
    if annotations_dir.exists() and not overwrite:
        raise FileExistsError(f"{annotations_dir} exists; pass --overwrite")
    annotations_dir.mkdir(parents=True, exist_ok=True)
    next_annotation_id, reports = 1, {}
    total_counts = Counter()
    for split in ("train", "val"):
        images, annotations, counts = [], [], Counter()
        split_rows = [row for row in rows if row["source_split"] == split]
        for image_id, row in enumerate(split_rows, 1):
            image_path, label_path = Path(row["sample_path"]), Path(row["label_path"])
            with Image.open(image_path) as image:
                width, height = image.size
            boxes = read_yolo_labels(label_path, width, height)
            relative = image_path.absolute().relative_to(mixed_dir)
            images.append({
                "id": image_id, "file_name": str(relative).replace("\\", "/"),
                "width": width, "height": height, "source_image_id": row["source_image_id"],
                "sample_type": row["sample_type"],
            })
            for box in boxes:
                class_id = int(box["source_class_id"])
                if not 0 <= class_id < len(names):
                    raise ValueError(f"Invalid class {class_id} in {label_path}")
                annotations.append({
                    "id": next_annotation_id, "image_id": image_id,
                    "category_id": class_id, "bbox": box["bbox"], "area": box["area"],
                    "iscrowd": 0, "sample_type": row["sample_type"],
                })
                next_annotation_id += 1; counts[names[class_id]] += 1
        payload = {
            "info": {"description": "AIC mixed dataset for Hierarchical D-FINE"},
            "licenses": [], "images": images, "annotations": annotations,
            "categories": [{"id": index, "name": name, "supercategory": "defect"}
                           for index, name in enumerate(names)],
        }
        path = annotations_dir / f"instances_{split}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        reports[split] = {"images": len(images), "annotations": len(annotations),
                          "empty_images": len(images)-len({x["image_id"] for x in annotations}),
                          "class_counts": dict(counts), "annotation_file": str(path)}
        total_counts.update(counts)
    report = {"classes": names, "splits": reports, "total_class_counts": dict(total_counts),
              "image_root": str(mixed_dir), "output_dir": str(output_dir),
              "reuse_policy": "annotations only; existing mixed images are reused"}
    (output_dir/"conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    args = parse_args()
    print(json.dumps(convert(args.mixed_dir, args.classes, args.output_dir, args.overwrite),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
