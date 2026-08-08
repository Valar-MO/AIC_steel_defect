#!/usr/bin/env python3
"""Add adaptive weak-class crops to the existing hierarchical COCO dataset."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


TARGET_GROUPS = {
    "qilie": "qilie",
    "huashang": "huashang",
    "yiwuyaru": "yiwuyaru",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-images",
        type=Path,
        default=Path("/root/autodl-tmp/datasets/AIC_steel_defect_mixed"),
    )
    parser.add_argument(
        "--base-coco",
        type=Path,
        default=Path(
            "/root/autodl-tmp/datasets/AIC_steel_defect_mixed_coco_hierarchical"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/root/autodl-tmp/datasets/AIC_steel_defect_gtaware_e1"),
    )
    parser.add_argument("--qilie-repeats", type=int, default=4)
    parser.add_argument("--huashang-repeats", type=int, default=4)
    parser.add_argument("--yiwuyaru-repeats", type=int, default=2)
    parser.add_argument("--min-side", type=int, default=512)
    parser.add_argument("--max-side", type=int, default=1536)
    parser.add_argument("--context-min", type=float, default=3.0)
    parser.add_argument("--context-max", type=float, default=6.0)
    parser.add_argument("--center-jitter", type=float, default=0.10)
    parser.add_argument("--min-visible", type=float, default=0.70)
    parser.add_argument("--max-omitted-visible", type=float, default=0.10)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-whole-images", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def intersection(box, crop):
    x1, y1 = max(box[0], crop[0]), max(box[1], crop[1])
    x2, y2 = min(box[0] + box[2], crop[0] + crop[2]), min(
        box[1] + box[3], crop[1] + crop[3]
    )
    if x2 <= x1 or y2 <= y1:
        return None, 0.0
    area = max(box[2] * box[3], 1e-12)
    return (x1, y1, x2, y2), ((x2 - x1) * (y2 - y1)) / area


def make_crop(target_box, image_width, image_height, scale, min_side, max_side,
              jitter, rng):
    x, y, width, height = target_box
    side = max(float(min_side), max(width, height) * scale)
    side = min(side, float(max_side), float(image_width), float(image_height))
    center_x = x + width / 2 + rng.uniform(-jitter, jitter) * side
    center_y = y + height / 2 + rng.uniform(-jitter, jitter) * side
    left = min(max(center_x - side / 2, 0.0), image_width - side)
    top = min(max(center_y - side / 2, 0.0), image_height - side)
    left, top = int(round(left)), int(round(top))
    side = int(round(side))
    return (left, top, min(side, image_width - left), min(side, image_height - top))


def labels_for_crop(annotations, crop, min_visible, max_omitted_visible):
    retained, ambiguous = [], False
    for annotation in annotations:
        clipped, visible = intersection(annotation["bbox"], crop)
        if clipped is not None and visible >= min_visible:
            retained.append({
                "source_annotation_id": int(annotation["id"]),
                "category_id": int(annotation["category_id"]),
                "bbox": [
                    clipped[0] - crop[0], clipped[1] - crop[1],
                    clipped[2] - clipped[0], clipped[3] - clipped[1],
                ],
                "visibility": visible,
            })
        elif visible > max_omitted_visible:
            ambiguous = True
    return retained, ambiguous


def validate_args(args):
    if min(args.qilie_repeats, args.huashang_repeats, args.yiwuyaru_repeats) < 0:
        raise ValueError("repeat counts must be non-negative")
    if not 0 < args.min_side <= args.max_side:
        raise ValueError("invalid crop side limits")
    if not 1 < args.context_min <= args.context_max:
        raise ValueError("invalid context range")
    if not 0 <= args.center_jitter <= 0.5:
        raise ValueError("center_jitter must be in [0, 0.5]")
    if not 0 < args.min_visible <= 1:
        raise ValueError("min_visible must be in (0, 1]")
    if not 0 <= args.max_omitted_visible < args.min_visible:
        raise ValueError("invalid max_omitted_visible")


def build(args):
    validate_args(args)
    base_images = args.base_images.resolve()
    base_coco = args.base_coco.resolve()
    output = args.output.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite")
        shutil.rmtree(output)
    (output / "annotations").mkdir(parents=True)
    (output / "gtaware" / "train").mkdir(parents=True)
    os.symlink(base_images, output / "base", target_is_directory=True)

    train = json.loads(
        (base_coco / "annotations" / "instances_train.json").read_text(encoding="utf-8")
    )
    val = json.loads(
        (base_coco / "annotations" / "instances_val.json").read_text(encoding="utf-8")
    )
    categories = {int(row["id"]): str(row["name"]) for row in train["categories"]}
    name_to_id = {name: category_id for category_id, name in categories.items()}
    missing = sorted(set(TARGET_GROUPS) - set(name_to_id))
    if missing:
        raise ValueError(f"target classes missing from COCO categories: {missing}")

    # Prefix all existing file names so one derived root can address both the
    # immutable base dataset and newly generated adaptive crops.
    for payload in (train, val):
        for image in payload["images"]:
            image["file_name"] = f"base/{image['file_name']}"
            image["gtaware_group"] = "base"

    annotations_by_image = defaultdict(list)
    for annotation in train["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)
    whole_images = [row for row in train["images"] if row.get("sample_type") == "whole"]
    if args.limit_whole_images is not None:
        whole_images = whole_images[:args.limit_whole_images]

    repeat_by_name = {
        "qilie": args.qilie_repeats,
        "huashang": args.huashang_repeats,
        "yiwuyaru": args.yiwuyaru_repeats,
    }
    rng = random.Random(args.seed)
    next_image_id = max(int(row["id"]) for row in train["images"]) + 1
    next_annotation_id = max(int(row["id"]) for row in train["annotations"]) + 1
    crop_counts, instance_counts, rejected = Counter(), Counter(), Counter()
    side_values, target_pixel_gains = [], []

    for whole_index, image_row in enumerate(whole_images, 1):
        source_path = output / image_row["file_name"]
        source_annotations = annotations_by_image[int(image_row["id"])]
        targets = [row for row in source_annotations if categories[int(row["category_id"])]
                   in repeat_by_name]
        if not targets:
            continue
        with Image.open(source_path) as opened:
            source = opened.convert("RGB")
            for target_index, target in enumerate(targets):
                class_name = categories[int(target["category_id"])]
                for repeat in range(repeat_by_name[class_name]):
                    accepted = None
                    for _attempt in range(30):
                        scale = rng.uniform(args.context_min, args.context_max)
                        crop = make_crop(
                            target["bbox"], source.width, source.height, scale,
                            args.min_side, args.max_side, args.center_jitter, rng,
                        )
                        labels, ambiguous = labels_for_crop(
                            source_annotations, crop, args.min_visible,
                            args.max_omitted_visible,
                        )
                        target_kept = any(
                            row["source_annotation_id"] == int(target["id"])
                            and row["visibility"] >= args.min_visible
                            for row in labels
                        )
                        if target_kept and not ambiguous:
                            accepted = crop, labels
                            break
                    if accepted is None:
                        rejected[class_name] += 1
                        continue
                    crop, labels = accepted
                    filename = (
                        f"{Path(image_row['source_image_id']).stem}__gtaware_"
                        f"{class_name}_{target_index:03d}_{repeat:02d}.jpg"
                    )
                    relative = Path("gtaware") / "train" / filename
                    destination = output / relative
                    source.crop((crop[0], crop[1], crop[0] + crop[2], crop[1] + crop[3])).save(
                        destination, quality=args.jpeg_quality, subsampling=0
                    )
                    group = TARGET_GROUPS[class_name]
                    train["images"].append({
                        "id": next_image_id,
                        "file_name": relative.as_posix(),
                        "width": crop[2],
                        "height": crop[3],
                        "source_image_id": image_row["source_image_id"],
                        "sample_type": "gtaware_adaptive",
                        "gtaware_group": group,
                        "gtaware_target_class": class_name,
                        "crop_xywh": list(crop),
                    })
                    for label in labels:
                        bbox = [float(value) for value in label["bbox"]]
                        train["annotations"].append({
                            "id": next_annotation_id,
                            "image_id": next_image_id,
                            "category_id": int(label["category_id"]),
                            "bbox": bbox,
                            "area": bbox[2] * bbox[3],
                            "iscrowd": 0,
                            "sample_type": "gtaware_adaptive",
                            "gtaware_target_class": class_name,
                        })
                        instance_counts[categories[int(label["category_id"])]] += 1
                        next_annotation_id += 1
                    crop_counts[class_name] += 1
                    side_values.append(crop[2])
                    target_pixel_gains.append(source.width / crop[2])
                    next_image_id += 1
        if whole_index % 250 == 0:
            print(f"processed whole images {whole_index}/{len(whole_images)}", flush=True)

    train["info"] = {"description": "AIC hierarchical GT-aware E1 train dataset"}
    val["info"] = {"description": "AIC hierarchical GT-aware E1 unchanged validation"}
    (output / "annotations" / "instances_train.json").write_text(
        json.dumps(train, ensure_ascii=False), encoding="utf-8"
    )
    (output / "annotations" / "instances_val.json").write_text(
        json.dumps(val, ensure_ascii=False), encoding="utf-8"
    )

    def quantile(values, fraction):
        if not values:
            return None
        ordered = sorted(values)
        return round(float(ordered[round((len(ordered) - 1) * fraction)]), 4)

    report = {
        "base_images": str(base_images),
        "base_coco": str(base_coco),
        "output": str(output),
        "policy": {
            "repeats": repeat_by_name,
            "side_range": [args.min_side, args.max_side],
            "context_range": [args.context_min, args.context_max],
            "center_jitter": args.center_jitter,
            "min_visible": args.min_visible,
            "max_omitted_visible": args.max_omitted_visible,
            "seed": args.seed,
        },
        "base_train_images": len(train["images"]) - sum(crop_counts.values()),
        "adaptive_crop_counts": dict(crop_counts),
        "adaptive_total": sum(crop_counts.values()),
        "adaptive_instance_counts": dict(instance_counts),
        "rejected_targets": dict(rejected),
        "crop_side_quantiles": {
            "min": quantile(side_values, 0), "p25": quantile(side_values, 0.25),
            "p50": quantile(side_values, 0.5), "p75": quantile(side_values, 0.75),
            "max": quantile(side_values, 1),
        },
        "relative_linear_resolution_gain_quantiles": {
            "min": quantile(target_pixel_gains, 0),
            "p50": quantile(target_pixel_gains, 0.5),
            "max": quantile(target_pixel_gains, 1),
        },
        "train_images_total": len(train["images"]),
        "train_annotations_total": len(train["annotations"]),
        "validation_policy": "unchanged full-image validation annotations",
    }
    (output / "dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main():
    report = build(parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
