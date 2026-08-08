#!/usr/bin/env python3
"""Build a GT-box crop classification dataset for defect re-classification.

This script is the first step of the detector/classifier decoupling pipeline:

1. read the fixed train/val split manifest;
2. parse Pascal VOC XML annotations;
3. crop each GT defect box with surrounding context;
4. save crops in ImageFolder layout: split/class_name/*.jpg;
5. write metadata.csv and dataset_report.json for auditability.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("splits/v1/split_manifest.csv"))
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    parser.add_argument(
        "--output",
        "--output-dir",
        dest="output",
        type=Path,
        default=Path("/root/autodl-tmp/datasets/AIC_steel_crop_cls_v1"),
    )
    parser.add_argument(
        "--expand",
        type=float,
        default=1.8,
        help="Crop side/width-height expansion ratio around each GT box.",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=128,
        help="Minimum crop side length in source-image pixels before resizing.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=384,
        help="Saved crop size. Crops are resized to image_size x image_size.",
    )
    parser.add_argument(
        "--rectangular",
        action="store_true",
        help="Use expanded rectangular boxes instead of default square crops.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality for saved crops.",
    )
    parser.add_argument(
        "--limit-per-split",
        type=int,
        help="Optional smoke-test limit on source images per split.",
    )
    parser.add_argument(
        "--preview-per-class",
        type=int,
        default=8,
        help="Number of crops per class/split to include in preview grids. Use 0 to disable.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_classes(path: Path) -> list[str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    names = payload.get("names")
    if not isinstance(names, list) or not names or len(names) != len(set(names)):
        raise ValueError(f"Invalid classes file: {path}")
    return [str(name) for name in names]


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image_path", "xml_path", "split"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest lacks required columns: {sorted(required)}")
    bad_splits = sorted({row["split"] for row in rows} - {"train", "val"})
    if bad_splits:
        raise ValueError(f"Unexpected split values in manifest: {bad_splits}")
    return rows


def parse_voc(xml_path: Path, class_to_id: dict[str, int]) -> tuple[int, int, list[dict]]:
    root = ET.parse(xml_path).getroot()
    width = int(float(root.findtext("size/width", "0")))
    height = int(float(root.findtext("size/height", "0")))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in {xml_path}")

    boxes: list[dict] = []
    seen = set()
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in class_to_id:
            raise ValueError(f"Unknown class {name!r} in {xml_path}")
        node = obj.find("bndbox")
        if node is None:
            continue
        try:
            raw = tuple(float(node.findtext(k, "nan")) for k in ("xmin", "ymin", "xmax", "ymax"))
        except ValueError:
            continue
        if not all(math.isfinite(v) for v in raw):
            continue
        signature = (name, *raw)
        if signature in seen:
            continue
        seen.add(signature)
        x1, y1, x2, y2 = raw
        x1 = min(max(x1, 0.0), float(width))
        y1 = min(max(y1, 0.0), float(height))
        x2 = min(max(x2, 0.0), float(width))
        y2 = min(max(y2, 0.0), float(height))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append(
            {
                "class_id": class_to_id[name],
                "class_name": name,
                "bbox_xyxy": (x1, y1, x2, y2),
            }
        )
    return width, height, boxes


def make_crop_box(
    box: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    expand: float,
    min_size: int,
    square: bool,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    box_w = x2 - x1
    box_h = y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    if square:
        side = max(box_w, box_h) * expand
        side = max(float(min_size), side)
        crop_w = crop_h = side
    else:
        crop_w = max(float(min_size), box_w * expand)
        crop_h = max(float(min_size), box_h * expand)

    crop_w = min(crop_w, float(image_width))
    crop_h = min(crop_h, float(image_height))
    left = cx - crop_w / 2.0
    top = cy - crop_h / 2.0
    left = min(max(left, 0.0), float(image_width) - crop_w)
    top = min(max(top, 0.0), float(image_height) - crop_h)
    right = left + crop_w
    bottom = top + crop_h
    return (
        int(math.floor(left)),
        int(math.floor(top)),
        int(math.ceil(right)),
        int(math.ceil(bottom)),
    )


def relative_box(
    box: tuple[float, float, float, float], crop: tuple[int, int, int, int]
) -> tuple[float, float, float, float]:
    return (box[0] - crop[0], box[1] - crop[1], box[2] - crop[0], box[3] - crop[1])


def ensure_clean_output(output: Path, overwrite: bool) -> None:
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output}. Pass --overwrite to replace it.")
        shutil.rmtree(output)
    output.mkdir(parents=True)


def save_crop(
    image: Image.Image,
    crop_box: tuple[int, int, int, int],
    output_path: Path,
    image_size: int,
    jpeg_quality: int,
) -> None:
    crop = image.crop(crop_box)
    crop = ImageOps.exif_transpose(crop)
    crop = crop.resize((image_size, image_size), Image.Resampling.BICUBIC)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop.convert("RGB").save(output_path, quality=jpeg_quality, optimize=True)


def build_preview_grid(
    samples: list[Path],
    output_path: Path,
    title: str,
    tile_size: int = 160,
    columns: int = 4,
) -> None:
    if not samples:
        return
    rows = math.ceil(len(samples) / columns)
    title_h = 28
    canvas = Image.new("RGB", (columns * tile_size, rows * tile_size + title_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill=(0, 0, 0))
    for index, path in enumerate(samples):
        try:
            img = Image.open(path).convert("RGB").resize((tile_size, tile_size), Image.Resampling.BICUBIC)
        except Exception:
            continue
        x = (index % columns) * tile_size
        y = title_h + (index // columns) * tile_size
        canvas.paste(img, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def main() -> None:
    args = parse_args()
    if args.expand <= 0:
        raise ValueError("--expand must be positive")
    if args.min_size <= 0 or args.image_size <= 0:
        raise ValueError("--min-size and --image-size must be positive")

    classes = load_classes(args.classes)
    class_to_id = {name: idx for idx, name in enumerate(classes)}
    rows = load_manifest(args.manifest)
    output = args.output.resolve()
    ensure_clean_output(output, args.overwrite)

    for split in ("train", "val"):
        for class_name in classes:
            (output / split / class_name).mkdir(parents=True, exist_ok=True)

    metadata_rows: list[dict] = []
    split_source_images = Counter()
    split_positive_images = Counter()
    split_crop_counts = Counter()
    split_class_counts: dict[str, Counter] = {"train": Counter(), "val": Counter()}
    skipped_empty = Counter()
    skipped_missing = 0
    crop_area_ratios = []
    preview_samples: dict[tuple[str, str], list[Path]] = defaultdict(list)
    per_split_seen = Counter()

    for row in rows:
        split = row["split"]
        if args.limit_per_split is not None and per_split_seen[split] >= args.limit_per_split:
            continue
        per_split_seen[split] += 1

        image_path = Path(row["image_path"])
        xml_path = Path(row["xml_path"])
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if not image_path.is_file() or not xml_path.is_file():
            skipped_missing += 1
            continue

        width, height, boxes = parse_voc(xml_path, class_to_id)
        split_source_images[split] += 1
        if not boxes:
            skipped_empty[split] += 1
            continue
        split_positive_images[split] += 1

        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            if image.width != width or image.height != height:
                width, height = image.width, image.height
            for obj_index, obj in enumerate(boxes):
                class_name = obj["class_name"]
                gt_box = obj["bbox_xyxy"]
                crop_box = make_crop_box(
                    gt_box,
                    width,
                    height,
                    args.expand,
                    args.min_size,
                    square=not args.rectangular,
                )
                rel = relative_box(gt_box, crop_box)
                crop_w = crop_box[2] - crop_box[0]
                crop_h = crop_box[3] - crop_box[1]
                crop_area_ratios.append((crop_w * crop_h) / float(width * height))
                crop_name = f"{image_path.stem}__obj{obj_index:03d}__{class_name}.jpg"
                crop_path = output / split / class_name / crop_name
                save_crop(image, crop_box, crop_path, args.image_size, args.jpeg_quality)

                if args.preview_per_class > 0:
                    key = (split, class_name)
                    if len(preview_samples[key]) < args.preview_per_class:
                        preview_samples[key].append(crop_path)

                split_crop_counts[split] += 1
                split_class_counts[split][class_name] += 1
                metadata_rows.append(
                    {
                        "split": split,
                        "class_id": obj["class_id"],
                        "class_name": class_name,
                        "crop_path": str(crop_path),
                        "source_image": str(image_path),
                        "source_xml": str(xml_path),
                        "source_width": width,
                        "source_height": height,
                        "gt_x1": round(gt_box[0], 4),
                        "gt_y1": round(gt_box[1], 4),
                        "gt_x2": round(gt_box[2], 4),
                        "gt_y2": round(gt_box[3], 4),
                        "crop_x1": crop_box[0],
                        "crop_y1": crop_box[1],
                        "crop_x2": crop_box[2],
                        "crop_y2": crop_box[3],
                        "relative_gt_x1": round(rel[0], 4),
                        "relative_gt_y1": round(rel[1], 4),
                        "relative_gt_x2": round(rel[2], 4),
                        "relative_gt_y2": round(rel[3], 4),
                    }
                )

    metadata_path = output / "metadata.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata_rows[0].keys()) if metadata_rows else [])
        if metadata_rows:
            writer.writeheader()
            writer.writerows(metadata_rows)

    for split in ("train", "val"):
        for class_name in classes:
            build_preview_grid(
                preview_samples[(split, class_name)],
                output / "previews" / f"{split}_{class_name}.jpg",
                f"{split}/{class_name}",
            )

    sorted_ratios = sorted(crop_area_ratios)
    quantiles = {}
    if sorted_ratios:
        for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
            quantiles[str(q)] = sorted_ratios[int((len(sorted_ratios) - 1) * q)]

    report = {
        "source_manifest": str(args.manifest.resolve()),
        "classes": classes,
        "output_dir": str(output),
        "layout": "torchvision.datasets.ImageFolder compatible: output/{train,val}/{class_name}/*.jpg",
        "crop_policy": {
            "expand": args.expand,
            "min_size": args.min_size,
            "image_size": args.image_size,
            "square": not args.rectangular,
            "jpeg_quality": args.jpeg_quality,
        },
        "source_images": dict(split_source_images),
        "positive_source_images": dict(split_positive_images),
        "empty_source_images_skipped": dict(skipped_empty),
        "missing_pairs_skipped": skipped_missing,
        "crop_counts": dict(split_crop_counts),
        "class_crop_counts": {
            split: {class_name: split_class_counts[split].get(class_name, 0) for class_name in classes}
            for split in ("train", "val")
        },
        "crop_area_ratio_quantiles": quantiles,
        "metadata": str(metadata_path),
    }
    (output / "dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
