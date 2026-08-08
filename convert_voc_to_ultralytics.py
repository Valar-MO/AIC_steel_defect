#!/usr/bin/env python3
"""Convert the fixed Pascal VOC split to Ultralytics detection format."""

from __future__ import annotations

import argparse
import csv
import json
import os
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import yaml


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=Path("splits/v1/split_manifest.csv"))
    p.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    p.add_argument(
        "--output-dir", type=Path,
        default=Path("/root/autodl-tmp/datasets/AIC_steel_defect_ultralytics"),
    )
    return p.parse_args()


def load_classes(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        names = yaml.safe_load(f).get("names")
    if not isinstance(names, list) or not names or len(set(names)) != len(names):
        raise ValueError(f"Invalid names list in {path}")
    return [str(x) for x in names]


def load_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"image_path", "xml_path", "group_id", "split"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest lacks required fields: {required}")
    if any(row["split"] not in {"train", "val"} for row in rows):
        raise ValueError("Manifest contains a split other than train/val")
    if len({row["image_path"] for row in rows}) != len(rows):
        raise ValueError("Manifest contains duplicate images")
    return rows


def parse_voc(xml_path: Path, class_to_id: dict[str, int]):
    root = ET.parse(xml_path).getroot()
    width = int(float(root.findtext("size/width", "0")))
    height = int(float(root.findtext("size/height", "0")))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in {xml_path}")

    boxes = []
    seen = set()
    duplicate_count = clipped_count = invalid_count = 0
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in class_to_id:
            raise ValueError(f"Unknown class {name!r} in {xml_path}")
        node = obj.find("bndbox")
        if node is None:
            invalid_count += 1
            continue
        try:
            raw = tuple(float(node.findtext(k, "nan")) for k in ("xmin", "ymin", "xmax", "ymax"))
        except ValueError:
            invalid_count += 1
            continue
        signature = (name, *raw)
        if signature in seen:
            duplicate_count += 1
            continue
        seen.add(signature)

        xmin, ymin, xmax, ymax = raw
        clean = (
            min(max(xmin, 0.0), float(width)),
            min(max(ymin, 0.0), float(height)),
            min(max(xmax, 0.0), float(width)),
            min(max(ymax, 0.0), float(height)),
        )
        clipped_count += int(clean != raw)
        xmin, ymin, xmax, ymax = clean
        if xmax <= xmin or ymax <= ymin:
            invalid_count += 1
            continue
        x = ((xmin + xmax) / 2.0) / width
        y = ((ymin + ymax) / 2.0) / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height
        boxes.append((class_to_id[name], x, y, w, h))
    return width, height, boxes, duplicate_count, clipped_count, invalid_count


def ensure_symlink(source: Path, destination: Path):
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        destination.unlink()
    elif destination.exists():
        raise FileExistsError(f"Refusing to replace non-symlink: {destination}")
    destination.symlink_to(source.resolve())


def main():
    args = parse_args()
    classes = load_classes(args.classes)
    class_to_id = {name: i for i, name in enumerate(classes)}
    rows = load_manifest(args.manifest)
    output = args.output_dir.resolve()
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    split_images = Counter()
    split_boxes = Counter()
    class_boxes = Counter({name: 0 for name in classes})
    empty_images = Counter()
    duplicates_removed = clipped_boxes = invalid_boxes = 0
    expected_files = {"train": set(), "val": set()}

    for row in rows:
        split = row["split"]
        image_path = Path(row["image_path"])
        xml_path = Path(row["xml_path"])
        if not image_path.is_file() or not xml_path.is_file():
            raise FileNotFoundError(f"Missing pair: {image_path}, {xml_path}")
        width, height, boxes, duplicates, clipped, invalid = parse_voc(xml_path, class_to_id)
        duplicates_removed += duplicates
        clipped_boxes += clipped
        invalid_boxes += invalid

        image_link = output / "images" / split / image_path.name
        label_path = output / "labels" / split / f"{image_path.stem}.txt"
        if image_link.name in expected_files[split]:
            raise ValueError(f"Duplicate basename in {split}: {image_link.name}")
        expected_files[split].add(image_link.name)
        ensure_symlink(image_path, image_link)
        label_path.write_text(
            "".join(f"{cid} {x:.10f} {y:.10f} {w:.10f} {h:.10f}\n" for cid, x, y, w, h in boxes),
            encoding="utf-8",
        )
        split_images[split] += 1
        split_boxes[split] += len(boxes)
        empty_images[split] += int(not boxes)
        for cid, *_ in boxes:
            class_boxes[classes[cid]] += 1

    # Refuse silent stale data from an older conversion.
    for split in ("train", "val"):
        actual_images = {p.name for p in (output / "images" / split).iterdir()}
        actual_labels = {p.name for p in (output / "labels" / split).glob("*.txt")}
        expected_labels = {f"{Path(name).stem}.txt" for name in expected_files[split]}
        extras = sorted((actual_images - expected_files[split]) | (actual_labels - expected_labels))
        if extras:
            raise RuntimeError(f"Stale files in output {split}: {extras[:10]}")

    dataset_yaml = {
        "path": str(output),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(classes)},
    }
    with (output / "steel.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_yaml, f, allow_unicode=True, sort_keys=False)

    report = {
        "source_manifest": str(args.manifest.resolve()),
        "output_dir": str(output),
        "classes": classes,
        "images": dict(split_images),
        "empty_images": dict(empty_images),
        "bboxes": dict(split_boxes),
        "class_bboxes": dict(class_boxes),
        "duplicates_removed": duplicates_removed,
        "boxes_clipped_to_image": clipped_boxes,
        "invalid_boxes_dropped": invalid_boxes,
        "image_storage": "symbolic_links",
    }
    (output / "conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
