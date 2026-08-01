#!/usr/bin/env python3
"""Convert the fixed AIC VOC split to class-agnostic COCO for D-FINE."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("splits/v1/split_manifest.csv"))
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/root/autodl-tmp/datasets/AIC_steel_defect_coco_agnostic"),
    )
    parser.add_argument(
        "--image-storage",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
        help="How images are materialized in the COCO tree (default: symlink).",
    )
    return parser.parse_args()


def load_classes(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        names = yaml.safe_load(handle).get("names")
    if not isinstance(names, list) or not names or len(names) != len(set(names)):
        raise ValueError(f"Invalid class list in {path}")
    return [str(name) for name in names]


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image_path", "xml_path", "split"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest lacks required fields: {sorted(required)}")
    if any(row["split"] not in {"train", "val"} for row in rows):
        raise ValueError("Manifest contains a split other than train/val")
    basenames = [Path(row["image_path"]).name for row in rows]
    if len(basenames) != len(set(basenames)):
        raise ValueError("Image basenames must be unique across the manifest")
    return rows


def parse_voc(path: Path, known_classes: set[str]) -> tuple[int, int, list[dict], Counter]:
    root = ET.parse(path).getroot()
    width = int(float(root.findtext("size/width", "0")))
    height = int(float(root.findtext("size/height", "0")))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in {path}")

    boxes: list[dict] = []
    stats = Counter()
    seen: set[tuple] = set()
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in known_classes:
            raise ValueError(f"Unknown class {name!r} in {path}")
        node = obj.find("bndbox")
        if node is None:
            stats["invalid"] += 1
            continue
        try:
            raw = tuple(float(node.findtext(key, "nan")) for key in ("xmin", "ymin", "xmax", "ymax"))
        except ValueError:
            stats["invalid"] += 1
            continue
        if not all(math.isfinite(value) for value in raw):
            stats["invalid"] += 1
            continue
        signature = (name, *raw)
        if signature in seen:
            stats["duplicates"] += 1
            continue
        seen.add(signature)

        x1, y1, x2, y2 = raw
        clean = (
            min(max(x1, 0.0), float(width)),
            min(max(y1, 0.0), float(height)),
            min(max(x2, 0.0), float(width)),
            min(max(y2, 0.0), float(height)),
        )
        stats["clipped"] += int(clean != raw)
        x1, y1, x2, y2 = clean
        if x2 <= x1 or y2 <= y1:
            stats["invalid"] += 1
            continue
        boxes.append({
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "area": (x2 - x1) * (y2 - y1),
            "source_category_name": name,
        })
    return width, height, boxes, stats


def materialize_image(source: Path, destination: Path, storage: str) -> None:
    source = source.resolve()
    if destination.is_symlink() or destination.exists():
        if storage == "symlink" and destination.is_symlink() and destination.resolve() == source:
            return
        if storage == "hardlink" and not destination.is_symlink():
            try:
                if os.path.samefile(source, destination):
                    return
            except OSError:
                pass
        if storage == "copy" and not destination.is_symlink():
            if destination.stat().st_size == source.stat().st_size:
                return
        raise FileExistsError(f"Refusing to replace existing image: {destination}")
    if storage == "symlink":
        destination.symlink_to(source)
    elif storage == "hardlink":
        os.link(source, destination)
    elif storage == "copy":
        import shutil

        shutil.copy2(source, destination)
    else:  # Defensive guard for direct callers.
        raise ValueError(f"Unsupported image storage: {storage}")


def convert(
    manifest: Path, classes_path: Path, output_dir: Path, image_storage: str = "symlink"
) -> dict:
    if image_storage not in {"symlink", "hardlink", "copy"}:
        raise ValueError(f"Unsupported image storage: {image_storage}")
    classes = load_classes(classes_path)
    rows = load_manifest(manifest)
    output_dir = output_dir.resolve()
    annotations_dir = output_dir / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    totals = Counter()
    original_classes = Counter({name: 0 for name in classes})
    expected_images: dict[str, set[str]] = {"train": set(), "val": set()}
    split_reports: dict[str, dict] = {}
    next_annotation_id = 1

    for split in ("train", "val"):
        image_dir = output_dir / "images" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        images: list[dict] = []
        annotations: list[dict] = []
        split_rows = [row for row in rows if row["split"] == split]
        for image_id, row in enumerate(split_rows, start=1):
            image_path = Path(row["image_path"])
            xml_path = Path(row["xml_path"])
            if not image_path.is_file() or not xml_path.is_file():
                raise FileNotFoundError(f"Missing image/XML pair: {image_path}, {xml_path}")
            if image_path.name in expected_images[split]:
                raise ValueError(f"Duplicate basename in {split}: {image_path.name}")
            expected_images[split].add(image_path.name)
            width, height, boxes, stats = parse_voc(xml_path, set(classes))
            totals.update(stats)
            materialize_image(image_path, image_dir / image_path.name, image_storage)
            images.append({
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            })
            for box in boxes:
                original_classes[box["source_category_name"]] += 1
                annotations.append({
                    "id": next_annotation_id,
                    "image_id": image_id,
                    "category_id": 0,
                    "bbox": box["bbox"],
                    "area": box["area"],
                    "iscrowd": 0,
                    "source_category_name": box["source_category_name"],
                })
                next_annotation_id += 1

        actual = {path.name for path in image_dir.iterdir()}
        extras = sorted(actual - expected_images[split])
        if extras:
            raise RuntimeError(f"Stale image links in {image_dir}: {extras[:10]}")
        payload = {
            "info": {"description": "AIC steel defects, class-agnostic D-FINE split"},
            "licenses": [],
            "images": images,
            "annotations": annotations,
            # D-FINE uses labels directly when remap_mscoco_category=False.
            "categories": [{"id": 0, "name": "defect", "supercategory": "defect"}],
        }
        annotation_path = annotations_dir / f"instances_{split}.json"
        annotation_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        split_reports[split] = {
            "images": len(images),
            "empty_images": len(images) - len({item["image_id"] for item in annotations}),
            "annotations": len(annotations),
            "annotation_file": str(annotation_path),
        }

    report = {
        "source_manifest": str(manifest.resolve()),
        "output_dir": str(output_dir),
        "category": {"id": 0, "name": "defect"},
        "splits": split_reports,
        "source_class_annotations": dict(original_classes),
        "duplicates_removed": totals["duplicates"],
        "boxes_clipped_to_image": totals["clipped"],
        "invalid_boxes_dropped": totals["invalid"],
        "image_storage": image_storage,
    }
    (output_dir / "conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    args = parse_args()
    report = convert(args.manifest, args.classes, args.output_dir, args.image_storage)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
