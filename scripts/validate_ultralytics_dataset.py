#!/usr/bin/env python3
"""Independently validate the converted Ultralytics detection dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset-dir", type=Path,
        default=Path("/root/autodl-tmp/datasets/AIC_steel_defect_ultralytics"),
    )
    p.add_argument("--manifest", type=Path, default=Path("splits/v1/split_manifest.csv"))
    p.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    p.add_argument("--samples-per-split", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_yaml(path):
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_voc_boxes(xml_path: Path, class_to_id: dict[str, int]):
    root = ET.parse(xml_path).getroot()
    width = int(float(root.findtext("size/width")))
    height = int(float(root.findtext("size/height")))
    result, seen = [], set()
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        node = obj.find("bndbox")
        raw = tuple(float(node.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        signature = (name, *raw)
        if signature in seen:
            continue
        seen.add(signature)
        xmin, ymin, xmax, ymax = raw
        xmin, xmax = min(max(xmin, 0), width), min(max(xmax, 0), width)
        ymin, ymax = min(max(ymin, 0), height), min(max(ymax, 0), height)
        if xmax <= xmin or ymax <= ymin:
            continue
        result.append((class_to_id[name], xmin, ymin, xmax, ymax))
    return width, height, result


def read_yolo(label_path: Path, width: int, height: int, class_count: int, errors: list[str]):
    boxes = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        prefix = f"{label_path}:{line_number}"
        if len(parts) != 5:
            errors.append(f"{prefix}: expected 5 fields")
            continue
        try:
            cid = int(parts[0]); x, y, w, h = map(float, parts[1:])
        except ValueError:
            errors.append(f"{prefix}: non-numeric field")
            continue
        if not 0 <= cid < class_count:
            errors.append(f"{prefix}: class id {cid} outside range")
        if not all(math.isfinite(v) for v in (x, y, w, h)):
            errors.append(f"{prefix}: non-finite coordinate")
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            errors.append(f"{prefix}: normalized coordinates outside range")
        xmin, ymin = (x - w / 2) * width, (y - h / 2) * height
        xmax, ymax = (x + w / 2) * width, (y + h / 2) * height
        if xmin < -1e-4 or ymin < -1e-4 or xmax > width + 1e-4 or ymax > height + 1e-4:
            errors.append(f"{prefix}: reconstructed bbox outside image")
        boxes.append((cid, xmin, ymin, xmax, ymax))
    return boxes


def boxes_match(expected, actual, tolerance=1e-3):
    if len(expected) != len(actual):
        return False
    unmatched = list(actual)
    for box in expected:
        match = next((i for i, candidate in enumerate(unmatched)
                      if box[0] == candidate[0] and
                      all(abs(a - b) <= tolerance for a, b in zip(box[1:], candidate[1:]))), None)
        if match is None:
            return False
        unmatched.pop(match)
    return not unmatched


def draw_grid(records, destination: Path, rng: random.Random, sample_count: int):
    chosen = rng.sample(records, min(sample_count, len(records)))
    tile_w, tile_h, label_h, cols = 400, 300, 30, 4
    rows = math.ceil(len(chosen) / cols)
    canvas = Image.new("RGB", (cols * tile_w, rows * (tile_h + label_h)), "white")
    font = ImageFont.load_default()
    for i, record in enumerate(chosen):
        with Image.open(record["image"]) as source:
            image = source.convert("RGB")
            ow, oh = image.size
            thumb = ImageOps.contain(image, (tile_w, tile_h), Image.Resampling.BILINEAR)
        scale = min(tile_w / ow, tile_h / oh)
        draw = ImageDraw.Draw(thumb)
        for cid, xmin, ymin, xmax, ymax in record["boxes"]:
            draw.rectangle((xmin * scale, ymin * scale, xmax * scale, ymax * scale),
                           outline=(255, 40, 40), width=2)
            draw.text((xmin * scale + 2, ymin * scale + 2), str(cid), fill=(255, 255, 0), font=font)
        col, row = i % cols, i // cols
        x = col * tile_w + (tile_w - thumb.width) // 2
        y = row * (tile_h + label_h) + (tile_h - thumb.height) // 2
        canvas.paste(thumb, (x, y))
        ImageDraw.Draw(canvas).text((col * tile_w + 4, y + tile_h + 6),
                                    Path(record["image"]).name[:45], fill="black", font=font)
    canvas.save(destination, quality=90)


def main():
    args = parse_args()
    dataset = args.dataset_dir.resolve()
    classes = [str(x) for x in load_yaml(args.classes)["names"]]
    class_to_id = {name: i for i, name in enumerate(classes)}
    config = load_yaml(dataset / "steel.yaml")
    yaml_names = config.get("names", {})
    normalized_names = [yaml_names.get(i, yaml_names.get(str(i))) for i in range(len(classes))]
    errors = []
    if normalized_names != classes:
        errors.append("steel.yaml class mapping differs from configs/classes.yaml")
    if Path(config.get("path", "")).resolve() != dataset:
        errors.append("steel.yaml path does not point to dataset directory")

    with args.manifest.open(encoding="utf-8-sig", newline="") as f:
        manifest = list(csv.DictReader(f))
    expected_counts = Counter(row["split"] for row in manifest)
    manifest_by_split = {split: [r for r in manifest if r["split"] == split] for split in ("train", "val")}
    all_stems, split_stems = set(), {}
    clean_bbox_counts = Counter()
    label_bbox_counts = Counter()
    empty_counts = Counter()
    visual_records = {"train": [], "val": []}

    for split in ("train", "val"):
        image_dir, label_dir = dataset / "images" / split, dataset / "labels" / split
        images = sorted(p for p in image_dir.iterdir() if p.is_file())
        labels = sorted(label_dir.glob("*.txt"))
        if len(images) != expected_counts[split]:
            errors.append(f"{split}: expected {expected_counts[split]} images, found {len(images)}")
        if len(labels) != expected_counts[split]:
            errors.append(f"{split}: expected {expected_counts[split]} labels, found {len(labels)}")
        split_stems[split] = {p.stem for p in images}
        overlap = all_stems & split_stems[split]
        if overlap:
            errors.append(f"{split}: image stems overlap another split: {sorted(overlap)[:5]}")
        all_stems.update(split_stems[split])

        for row in manifest_by_split[split]:
            source_image = Path(row["image_path"])
            linked_image = image_dir / source_image.name
            label_path = label_dir / f"{source_image.stem}.txt"
            if not linked_image.is_symlink():
                errors.append(f"{linked_image}: expected symbolic link")
            elif linked_image.resolve() != source_image.resolve():
                errors.append(f"{linked_image}: points to wrong source")
            if not label_path.is_file():
                errors.append(f"Missing label: {label_path}")
                continue
            expected_w, expected_h, expected = clean_voc_boxes(Path(row["xml_path"]), class_to_id)
            try:
                with Image.open(linked_image) as image:
                    actual_w, actual_h = image.size
            except OSError as exc:
                errors.append(f"Unreadable image {linked_image}: {exc}")
                continue
            if (actual_w, actual_h) != (expected_w, expected_h):
                errors.append(f"Size mismatch for {linked_image}")
            actual = read_yolo(label_path, actual_w, actual_h, len(classes), errors)
            if not boxes_match(expected, actual):
                errors.append(f"BBox mismatch after round trip: {source_image.name}")
            clean_bbox_counts[split] += len(expected)
            label_bbox_counts[split] += len(actual)
            empty_counts[split] += int(not actual)
            visual_records[split].append({"image": linked_image, "boxes": actual})

    output = dataset / "validation"
    output.mkdir(exist_ok=True)
    rng = random.Random(args.seed)
    for split in ("train", "val"):
        draw_grid(visual_records[split], output / f"{split}_samples.jpg", rng, args.samples_per_split)
    report = {
        "status": "passed" if not errors else "failed",
        "classes": classes,
        "expected_images": dict(expected_counts),
        "clean_voc_bboxes": dict(clean_bbox_counts),
        "ultralytics_bboxes": dict(label_bbox_counts),
        "empty_images": dict(empty_counts),
        "errors": errors,
        "visualizations": [str(output / "train_samples.jpg"), str(output / "val_samples.jpg")],
    }
    (dataset / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
