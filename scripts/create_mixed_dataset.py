#!/usr/bin/env python3
"""Build a leakage-safe mixed whole-image and patch detection dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image


RARE_REPEATS = {"qilie": 5, "huashang": 4, "jiaza": 2}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=Path("splits/v1/split_manifest.csv"))
    p.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("/root/autodl-tmp/datasets/AIC_steel_defect_mixed"))
    p.add_argument("--patch-size", type=int, default=2048)
    p.add_argument("--overlap", type=int, default=512)
    p.add_argument("--min-visible", type=float, default=0.70)
    p.add_argument("--max-unlabeled-visible", type=float, default=0.10,
                   help="Skip a crop if an omitted object remains more visible than this")
    p.add_argument(
        "--max-box-fraction", type=float, default=0.98,
        help="Skip a patch if any retained box spans this fraction of patch width or height.",
    )
    p.add_argument("--background-ratio", type=float, default=0.25,
                   help="Random background patches / regular positive patches")
    p.add_argument("--rare-center-jitter", type=float, default=0.20)
    p.add_argument("--jpeg-quality", type=int, default=92)
    p.add_argument("--hard-negative-predictions", type=Path)
    p.add_argument("--hard-negative-score", type=float, default=0.50)
    p.add_argument("--hard-negative-max-per-image", type=int, default=3)
    p.add_argument("--hard-negative-max-iou", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-train-images", type=int)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_classes(path):
    with path.open(encoding="utf-8") as f:
        names = yaml.safe_load(f).get("names")
    if not isinstance(names, list) or not names or len(names) != len(set(names)):
        raise ValueError(f"Invalid classes file: {path}")
    return [str(x) for x in names]


def load_split(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or any(row["split"] not in {"train", "val"} for row in rows):
        raise ValueError("Invalid train/val manifest")
    if len({Path(row["image_path"]).name for row in rows}) != len(rows):
        raise ValueError("Source image basenames must be unique")
    return rows


def parse_voc(path, class_to_id):
    root = ET.parse(path).getroot()
    width = int(float(root.findtext("size/width", "0")))
    height = int(float(root.findtext("size/height", "0")))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions in {path}")
    boxes, seen = [], set()
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in class_to_id:
            raise ValueError(f"Unknown class {name!r} in {path}")
        node = obj.find("bndbox")
        if node is None:
            continue
        try:
            raw = tuple(float(node.findtext(k, "nan"))
                        for k in ("xmin", "ymin", "xmax", "ymax"))
        except ValueError:
            continue
        if not all(math.isfinite(v) for v in raw):
            continue
        signature = (name, *raw)
        if signature in seen:
            continue
        seen.add(signature)
        x1, y1, x2, y2 = raw
        clean = (max(0.0, min(x1, width)), max(0.0, min(y1, height)),
                 max(0.0, min(x2, width)), max(0.0, min(y2, height)))
        if clean[2] > clean[0] and clean[3] > clean[1]:
            boxes.append((class_to_id[name], clean))
    return width, height, boxes


def axis_starts(length, patch_size, stride):
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def windows(width, height, patch_size, overlap):
    stride = patch_size - overlap
    return [(x, y, min(x + patch_size, width), min(y + patch_size, height))
            for y in axis_starts(height, patch_size, stride)
            for x in axis_starts(width, patch_size, stride)]


def intersection(box, crop):
    x1, y1 = max(box[0], crop[0]), max(box[1], crop[1])
    x2, y2 = min(box[2], crop[2]), min(box[3], crop[3])
    if x2 <= x1 or y2 <= y1:
        return None, 0.0
    inter = (x2 - x1) * (y2 - y1)
    area = (box[2] - box[0]) * (box[3] - box[1])
    return (x1, y1, x2, y2), inter / area


def crop_labels(boxes, crop, min_visible, max_unlabeled_visible, max_box_fraction):
    kept, intersects_any, ambiguous = [], False, False
    crop_width, crop_height = crop[2] - crop[0], crop[3] - crop[1]
    for class_id, box in boxes:
        clipped, visibility = intersection(box, crop)
        intersects_any |= clipped is not None
        if clipped is not None and visibility >= min_visible:
            local = (clipped[0] - crop[0], clipped[1] - crop[1],
                     clipped[2] - crop[0], clipped[3] - crop[1])
            if ((local[2] - local[0]) / crop_width >= max_box_fraction or
                    (local[3] - local[1]) / crop_height >= max_box_fraction):
                ambiguous = True
            kept.append((class_id, local, visibility))
        elif visibility > max_unlabeled_visible:
            ambiguous = True
    return kept, intersects_any, ambiguous


def center_crop(box, width, height, size, rng, jitter):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    cx += rng.uniform(-jitter, jitter) * size
    cy += rng.uniform(-jitter, jitter) * size
    crop_w, crop_h = min(size, width), min(size, height)
    x1 = int(round(max(0, min(cx - crop_w / 2, width - crop_w))))
    y1 = int(round(max(0, min(cy - crop_h / 2, height - crop_h))))
    return (x1, y1, x1 + crop_w, y1 + crop_h)


def write_label(path, labels, width, height):
    lines = []
    for class_id, box, _ in labels:
        x1, y1, x2, y2 = box
        lines.append(f"{class_id} {((x1+x2)/2)/width:.10f} {((y1+y2)/2)/height:.10f} "
                     f"{(x2-x1)/width:.10f} {(y2-y1)/height:.10f}\n")
    path.write_text("".join(lines), encoding="utf-8")


def full_labels(boxes):
    return [(class_id, box, 1.0) for class_id, box in boxes]


def safe_link(source, destination):
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.symlink_to(source.resolve())


def box_iou(a, b):
    clipped, _ = intersection(a, b)
    if clipped is None:
        return 0.0
    inter = (clipped[2] - clipped[0]) * (clipped[3] - clipped[1])
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def load_hard_negatives(path, train_names, gt_by_name, score_threshold, max_iou, max_per_image):
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("images", []) if isinstance(payload, dict) else []
    output = {}
    for record in records:
        image_id = Path(record["image_id"]).name
        if image_id not in train_names:
            continue
        candidates = []
        gt_boxes = [box for _, box in gt_by_name[image_id]]
        for pred in record.get("predictions", []):
            score = float(pred["score"])
            box = tuple(float(v) for v in pred["bbox_xyxy"])
            if score >= score_threshold and max((box_iou(box, gt) for gt in gt_boxes), default=0) < max_iou:
                candidates.append((score, box))
        candidates.sort(reverse=True)
        output[image_id] = candidates[:max_per_image]
    return output


def main():
    args = parse_args()
    if args.patch_size <= 0 or not 0 <= args.overlap < args.patch_size:
        raise ValueError("Invalid patch size/overlap")
    if (not 0 < args.min_visible <= 1 or
            not 0 <= args.max_unlabeled_visible < args.min_visible or
            not 0 <= args.background_ratio <= 1):
        raise ValueError("Invalid visibility/background ratio")
    if not 0 < args.max_box_fraction < 1:
        raise ValueError("max_box_fraction must be in (0, 1)")
    if not 0 <= args.rare_center_jitter <= 0.5 or not 1 <= args.jpeg_quality <= 100:
        raise ValueError("Invalid jitter/JPEG quality")
    classes = load_classes(args.classes)
    class_to_id = {name: i for i, name in enumerate(classes)}
    rows = load_split(args.manifest)
    train_rows = [row for row in rows if row["split"] == "train"]
    if args.limit_train_images is not None:
        train_rows = train_rows[:args.limit_train_images]
    val_rows = [row for row in rows if row["split"] == "val"]
    output = args.output_dir.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output}; pass --overwrite")
        shutil.rmtree(output)
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)

    parsed = {}
    for row in train_rows + val_rows:
        parsed[Path(row["image_path"]).name] = parse_voc(Path(row["xml_path"]), class_to_id)
    train_names = {Path(row["image_path"]).name for row in train_rows}
    gt_by_name = {name: parsed[name][2] for name in train_names}
    hard_negatives = load_hard_negatives(
        args.hard_negative_predictions, train_names, gt_by_name,
        args.hard_negative_score, args.hard_negative_max_iou,
        args.hard_negative_max_per_image)

    rng = random.Random(args.seed)
    manifest_records = []
    counters, class_instances = Counter(), Counter({name: 0 for name in classes})

    def record_sample(source_path, source_split, sample_type, crop, labels, image_data=None):
        index = counters["all_samples"]
        if sample_type == "whole":
            filename = source_path.name
        else:
            filename = f"{source_path.stem}__{sample_type}__{index:07d}.jpg"
        image_path = output / "images" / source_split / filename
        label_path = output / "labels" / source_split / f"{Path(filename).stem}.txt"
        if sample_type == "whole":
            safe_link(source_path, image_path)
            width, height = parsed[source_path.name][:2]
        else:
            width, height = crop[2] - crop[0], crop[3] - crop[1]
            image_data.crop(crop).save(image_path, quality=args.jpeg_quality, subsampling=0)
        write_label(label_path, labels, width, height)
        counters["all_samples"] += 1
        counters[f"{source_split}_{sample_type}"] += 1
        counters[f"{source_split}_instances"] += len(labels)
        for class_id, _, _ in labels:
            class_instances[classes[class_id]] += 1
        manifest_records.append({
            "sample_path": str(image_path), "label_path": str(label_path),
            "source_image": str(source_path.resolve()), "source_image_id": source_path.name,
            "source_split": source_split, "sample_type": sample_type,
            "crop_xyxy": "" if crop is None else json.dumps(crop),
            "bbox_count": len(labels),
            "classes": "|".join(sorted({classes[x[0]] for x in labels})),
        })

    # Keep original train and val images. Validation remains full-image only so
    # its metrics stay directly comparable with the first baseline.
    for row in train_rows + val_rows:
        source = Path(row["image_path"])
        width, height, boxes = parsed[source.name]
        record_sample(source, row["split"], "whole", None, full_labels(boxes))

    regular_positive = []
    background_pool = []
    for row_index, row in enumerate(train_rows, start=1):
        source_path = Path(row["image_path"])
        width, height, boxes = parsed[source_path.name]
        with Image.open(source_path) as opened:
            image = opened.convert("RGB")
            for crop in windows(width, height, args.patch_size, args.overlap):
                labels, intersects_any, ambiguous = crop_labels(
                    boxes, crop, args.min_visible, args.max_unlabeled_visible,
                    args.max_box_fraction)
                if labels and not ambiguous:
                    regular_positive.append((source_path, crop, labels))
                elif not intersects_any:
                    background_pool.append((source_path, crop, []))

            for class_id, target_box in boxes:
                repeats = RARE_REPEATS.get(classes[class_id], 0)
                used = set()
                for _ in range(repeats):
                    accepted = None
                    for _attempt in range(20):
                        crop = center_crop(target_box, width, height, args.patch_size,
                                           rng, args.rare_center_jitter)
                        if crop in used:
                            continue
                        labels, _, ambiguous = crop_labels(
                            boxes, crop, args.min_visible, args.max_unlabeled_visible,
                            args.max_box_fraction)
                        target_visible = any(cid == class_id and intersection(target_box, crop)[1] >= args.min_visible
                                             for cid, _, _ in labels)
                        if target_visible and not ambiguous:
                            accepted = (crop, labels)
                            break
                    if accepted:
                        used.add(accepted[0])
                        record_sample(source_path, "train", "rare_center", accepted[0],
                                      accepted[1], image)

            for _score, predicted_box in hard_negatives.get(source_path.name, []):
                crop = center_crop(predicted_box, width, height, args.patch_size, rng, 0.05)
                labels, intersects_any, _ = crop_labels(
                    boxes, crop, args.min_visible, args.max_unlabeled_visible,
                    args.max_box_fraction)
                if not labels and not intersects_any:
                    record_sample(source_path, "train", "hard_negative", crop, [], image)
        if row_index % 250 == 0:
            print(f"Planned patches for {row_index}/{len(train_rows)} train images")

    # Write deterministic regular positives after planning, then a bounded
    # random subset of unambiguous background patches.
    positives_by_source = defaultdict(list)
    for source_path, crop, labels in regular_positive:
        positives_by_source[source_path].append((crop, labels))
    for source_path, samples in positives_by_source.items():
        with Image.open(source_path) as opened:
            image = opened.convert("RGB")
            for crop, labels in samples:
                record_sample(source_path, "train", "regular_positive", crop, labels, image)
    background_target = min(len(background_pool),
                            round(len(regular_positive) * args.background_ratio))
    backgrounds_by_source = defaultdict(list)
    for source_path, crop, labels in rng.sample(background_pool, background_target):
        backgrounds_by_source[source_path].append((crop, labels))
    for source_path, samples in backgrounds_by_source.items():
        with Image.open(source_path) as opened:
            image = opened.convert("RGB")
            for crop, labels in samples:
                record_sample(source_path, "train", "random_background", crop, labels, image)

    dataset_yaml = {"path": str(output), "train": "images/train", "val": "images/val",
                    "names": {i: name for i, name in enumerate(classes)}}
    with (output / "steel.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_yaml, f, allow_unicode=True, sort_keys=False)
    fields = list(manifest_records[0])
    with (output / "sample_manifest.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(manifest_records)
    report = {
        "source_manifest": str(args.manifest.resolve()), "output_dir": str(output),
        "seed": args.seed, "patch_size": args.patch_size, "overlap": args.overlap,
        "min_visible": args.min_visible, "background_ratio": args.background_ratio,
        "max_unlabeled_visible": args.max_unlabeled_visible,
        "max_box_fraction": args.max_box_fraction,
        "rare_repeats": RARE_REPEATS, "sample_counts": dict(counters),
        "class_instance_counts_all_samples": dict(class_instances),
        "regular_positive_planned": len(regular_positive),
        "unambiguous_background_candidates": len(background_pool),
        "random_background_selected": background_target,
        "hard_negative_source": (str(args.hard_negative_predictions.resolve())
                                 if args.hard_negative_predictions else None),
        "validation_policy": "original full val images only",
    }
    (output / "mixed_dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
