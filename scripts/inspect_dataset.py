#!/usr/bin/env python3
"""Lightweight EDA for the AIC steel-defect Pascal VOC dataset."""

from __future__ import annotations

import argparse
import csv
import math
import random
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from PIL import Image, ImageDraw, ImageFont


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/root/autodl-tmp/datasets/AIC_steel_defect/train/train"),
        help="Directory containing paired image and Pascal VOC XML files.",
    )
    parser.add_argument(
        "--classes",
        type=Path,
        default=Path("configs/classes.yaml"),
        help="YAML file containing a top-level 'names' list.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/data_inspection"))
    parser.add_argument("--samples-per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_classes(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        names = yaml.safe_load(f).get("names")
    if not isinstance(names, list) or not names:
        raise ValueError(f"{path} must contain a non-empty 'names' list")
    return [str(name) for name in names]


def find_image(data_dir: Path, stem: str, xml_filename: str | None) -> Path | None:
    candidates = []
    if xml_filename:
        candidates.append(data_dir / Path(xml_filename).name)
    candidates.extend(data_dir / f"{stem}{suffix}" for suffix in IMAGE_SUFFIXES)
    for path in candidates:
        if path.is_file():
            return path
    return None


def parse_dataset(data_dir: Path, classes: list[str]):
    records: list[dict] = []
    issues: list[dict] = []
    image_paths: set[str] = set()
    known = set(classes)

    xml_paths = sorted(data_dir.glob("*.xml"))
    if not xml_paths:
        raise FileNotFoundError(f"No XML annotations found in {data_dir}")

    for xml_path in xml_paths:
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError as exc:
            issues.append({"file": str(xml_path), "type": "invalid_xml", "detail": str(exc)})
            continue

        image_path = find_image(data_dir, xml_path.stem, root.findtext("filename"))
        if image_path is None:
            issues.append({"file": str(xml_path), "type": "missing_image", "detail": ""})
            continue

        try:
            width = int(float(root.findtext("size/width", "0")))
            height = int(float(root.findtext("size/height", "0")))
        except ValueError:
            width = height = 0
        if width <= 0 or height <= 0:
            try:
                with Image.open(image_path) as image:
                    width, height = image.size
                issues.append({"file": str(xml_path), "type": "missing_xml_size", "detail": "used image size"})
            except OSError as exc:
                issues.append({"file": str(image_path), "type": "unreadable_image", "detail": str(exc)})
                continue

        image_paths.add(str(image_path))
        objects = root.findall("object")
        if not objects:
            issues.append({"file": str(xml_path), "type": "empty_annotation", "detail": ""})

        seen_boxes = set()
        for index, obj in enumerate(objects):
            class_name = (obj.findtext("name") or "").strip()
            box = obj.find("bndbox")
            if class_name not in known:
                issues.append({"file": str(xml_path), "type": "unknown_class", "detail": class_name})
                continue
            if box is None:
                issues.append({"file": str(xml_path), "type": "missing_bbox", "detail": f"object {index}"})
                continue
            try:
                xmin = float(box.findtext("xmin", "nan"))
                ymin = float(box.findtext("ymin", "nan"))
                xmax = float(box.findtext("xmax", "nan"))
                ymax = float(box.findtext("ymax", "nan"))
            except ValueError:
                issues.append({"file": str(xml_path), "type": "invalid_bbox", "detail": f"object {index}"})
                continue

            box_width, box_height = xmax - xmin, ymax - ymin
            if not all(math.isfinite(v) for v in (xmin, ymin, xmax, ymax)) or box_width <= 0 or box_height <= 0:
                issues.append({"file": str(xml_path), "type": "invalid_bbox", "detail": f"{class_name}: {xmin,ymin,xmax,ymax}"})
                continue
            if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
                issues.append({"file": str(xml_path), "type": "out_of_bounds_bbox", "detail": f"{class_name}: {xmin,ymin,xmax,ymax}"})

            signature = (class_name, xmin, ymin, xmax, ymax)
            if signature in seen_boxes:
                issues.append({"file": str(xml_path), "type": "duplicate_bbox", "detail": str(signature)})
            seen_boxes.add(signature)

            records.append(
                {
                    "image_path": str(image_path),
                    "xml_path": str(xml_path),
                    "class_name": class_name,
                    "image_width": width,
                    "image_height": height,
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "bbox_width": box_width,
                    "bbox_height": box_height,
                    "bbox_area_px": box_width * box_height,
                    "relative_area": box_width * box_height / (width * height),
                    "aspect_ratio": box_width / box_height,
                }
            )
    return records, issues, image_paths, len(xml_paths)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def make_summary(records: list[dict], classes: list[str]) -> list[dict]:
    by_class = defaultdict(list)
    for record in records:
        by_class[record["class_name"]].append(record)
    rows = []
    for name in classes:
        group = by_class[name]
        areas = [r["relative_area"] for r in group]
        ratios = [r["aspect_ratio"] for r in group]
        rows.append(
            {
                "class_name": name,
                "instance_count": len(group),
                "image_count": len({r["image_path"] for r in group}),
                "relative_area_p05": quantile(areas, 0.05),
                "relative_area_median": quantile(areas, 0.5),
                "relative_area_p95": quantile(areas, 0.95),
                "aspect_ratio_p05": quantile(ratios, 0.05),
                "aspect_ratio_median": quantile(ratios, 0.5),
                "aspect_ratio_p95": quantile(ratios, 0.95),
                "visual_features": "",
                "confusable_with": "",
                "notes": "",
            }
        )
    return rows


def plot_counts(summary: list[dict], output: Path) -> None:
    names = [row["class_name"] for row in summary]
    counts = [row["instance_count"] for row in summary]
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(names, counts, color="#4472C4")
    ax.bar_label(bars, padding=3)
    ax.set(title="Instances per class", ylabel="Bounding-box count")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_distributions(records: list[dict], classes: list[str], output_dir: Path) -> None:
    by_class = defaultdict(list)
    for record in records:
        by_class[record["class_name"]].append(record)
    for key, title, filename in (
        ("relative_area", "Relative bbox area by class (log scale)", "bbox_relative_area.png"),
        ("aspect_ratio", "BBox aspect ratio (width / height) by class", "bbox_aspect_ratio.png"),
    ):
        labels = [name for name in classes if by_class[name]]
        values = [[r[key] for r in by_class[name]] for name in labels]
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.boxplot(values, tick_labels=labels, showfliers=False)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)


def select_samples(records: list[dict], count: int, rng: random.Random) -> list[dict]:
    if len(records) <= count:
        return records
    ordered = sorted(records, key=lambda row: row["relative_area"])
    # Even ranks cover tiny, typical and large defects; jitter avoids always choosing exact ranks.
    selected = []
    width = len(ordered) / count
    for i in range(count):
        lo = int(i * width)
        hi = max(lo + 1, int((i + 1) * width))
        selected.append(ordered[rng.randrange(lo, min(hi, len(ordered)))])
    return selected


def crop_with_context(image: Image.Image, record: dict, context: float = 2.2):
    xmin, ymin, xmax, ymax = (record[k] for k in ("xmin", "ymin", "xmax", "ymax"))
    bw, bh = xmax - xmin, ymax - ymin
    side = max(bw, bh) * context
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    left, top = max(0, cx - side / 2), max(0, cy - side / 2)
    right, bottom = min(image.width, cx + side / 2), min(image.height, cy + side / 2)
    if right - left < side:
        left = max(0, right - side)
    if bottom - top < side:
        top = max(0, bottom - side)
    crop = image.crop((int(left), int(top), int(right), int(bottom))).convert("RGB")
    local_box = (xmin - left, ymin - top, xmax - left, ymax - top)
    return crop, local_box


def make_sample_grids(records: list[dict], classes: list[str], output_dir: Path, count: int, seed: int) -> None:
    by_class = defaultdict(list)
    for record in records:
        by_class[record["class_name"]].append(record)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    tile, label_h, columns = 280, 36, 5
    font = ImageFont.load_default()
    for class_name in classes:
        selected = select_samples(by_class[class_name], count, rng)
        if not selected:
            continue
        rows = math.ceil(len(selected) / columns)
        canvas = Image.new("RGB", (columns * tile, rows * (tile + label_h)), "white")
        for index, record in enumerate(selected):
            try:
                with Image.open(record["image_path"]) as source:
                    crop, box = crop_with_context(source, record)
            except OSError:
                continue
            scale = min(tile / crop.width, tile / crop.height)
            resized = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))))
            draw = ImageDraw.Draw(resized)
            draw.rectangle(tuple(v * scale for v in box), outline=(255, 50, 50), width=3)
            col, row = index % columns, index // columns
            x = col * tile + (tile - resized.width) // 2
            y = row * (tile + label_h) + (tile - resized.height) // 2
            canvas.paste(resized, (x, y))
            label = f"area={record['relative_area']:.4%}  w/h={record['aspect_ratio']:.2f}"
            ImageDraw.Draw(canvas).text((col * tile + 6, row * (tile + label_h) + tile + 8), label, fill="black", font=font)
        canvas.save(output_dir / f"{class_name}.jpg", quality=92)


def main() -> None:
    args = parse_args()
    classes = load_classes(args.classes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, issues, image_paths, annotation_count = parse_dataset(args.data_dir, classes)
    if not records:
        raise RuntimeError("No valid bounding boxes were parsed")

    summary = make_summary(records, classes)
    write_csv(args.output_dir / "instances.csv", records, list(records[0]))
    write_csv(args.output_dir / "class_summary.csv", summary, list(summary[0]))
    issue_fields = ["file", "type", "detail"]
    write_csv(args.output_dir / "quality_issues.csv", issues, issue_fields)
    plot_counts(summary, args.output_dir / "class_counts.png")
    plot_distributions(records, classes, args.output_dir)
    make_sample_grids(records, classes, args.output_dir / "samples", args.samples_per_class, args.seed)

    issue_counts = Counter(row["type"] for row in issues)
    print(f"Annotations: {annotation_count}")
    print(f"Images:      {len(image_paths)}")
    print(f"Instances:   {len(records)}")
    print(f"Classes:     {len(classes)}")
    print(f"Issues:      {len(issues)} {dict(issue_counts)}")
    print(f"Output:      {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
