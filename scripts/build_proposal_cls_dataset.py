#!/usr/bin/env python3
"""Build a proposal-crop classification dataset from detector predictions.

Compared with GT crops, proposal crops are cut from detector-predicted boxes,
then assigned labels by matching those boxes to VOC ground-truth boxes. This
reduces the train/inference distribution gap for second-stage reclassification.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-predictions", type=Path, required=True)
    parser.add_argument("--val-predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("splits/v1/split_manifest.csv"))
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    parser.add_argument(
        "--output",
        "--output-dir",
        dest="output",
        type=Path,
        default=Path("/root/autodl-tmp/datasets/AIC_steel_proposal_cls_v1"),
    )
    parser.add_argument("--positive-iou", type=float, default=0.50)
    parser.add_argument("--min-score", type=float, default=0.03)
    parser.add_argument(
        "--max-proposals-per-gt",
        type=int,
        default=3,
        help="Keep at most this many matched proposal crops for each GT object.",
    )
    parser.add_argument(
        "--max-proposals-per-image",
        type=int,
        default=80,
        help="Pre-filter top scoring predictions per image before matching.",
    )
    parser.add_argument(
        "--crop-expand",
        type=float,
        default=1.0,
        help="Expansion applied to proposal box before cropping. 1.0 means crop exact proposal.",
    )
    parser.add_argument("--min-crop-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--preview-per-class", type=int, default=8)
    parser.add_argument("--limit-images-per-split", type=int, help="Smoke-test limit.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_classes(path: Path) -> list[str]:
    names = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("names")
    if not isinstance(names, list) or not names or len(names) != len(set(names)):
        raise ValueError(f"Invalid classes file: {path}")
    return [str(name) for name in names]


def load_manifest(path: Path) -> dict[str, dict[str, dict]]:
    by_split: dict[str, dict[str, dict]] = {"train": {}, "val": {}}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            split = row["split"]
            if split not in by_split:
                continue
            image_id = Path(row["image_path"]).name
            by_split[split][image_id] = row
    return by_split


def parse_voc(xml_path: Path, class_to_id: dict[str, int]):
    root = ET.parse(xml_path).getroot()
    width = int(float(root.findtext("size/width", "0")))
    height = int(float(root.findtext("size/height", "0")))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in {xml_path}")
    boxes = []
    seen = set()
    for object_index, obj in enumerate(root.findall("object")):
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
        clean = (
            min(max(x1, 0.0), float(width)),
            min(max(y1, 0.0), float(height)),
            min(max(x2, 0.0), float(width)),
            min(max(y2, 0.0), float(height)),
        )
        if clean[2] <= clean[0] or clean[3] <= clean[1]:
            continue
        boxes.append(
            {
                "gt_index": len(boxes),
                "source_object_index": object_index,
                "class_id": class_to_id[name],
                "class_name": name,
                "bbox_xyxy": clean,
            }
        )
    return width, height, boxes


def box_iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def clean_box(box, width: int, height: int):
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return None
    clean = (
        min(max(x1, 0.0), float(width)),
        min(max(y1, 0.0), float(height)),
        min(max(x2, 0.0), float(width)),
        min(max(y2, 0.0), float(height)),
    )
    if clean[2] <= clean[0] or clean[3] <= clean[1]:
        return None
    return clean


def expanded_crop_box(box, width: int, height: int, expand: float, min_size: int):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    side = max(max(bw, bh) * expand, float(min_size))
    side = min(side, float(max(width, height)))
    crop_w = min(side, float(width))
    crop_h = min(side, float(height))
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    left = min(max(cx - crop_w / 2.0, 0.0), float(width) - crop_w)
    top = min(max(cy - crop_h / 2.0, 0.0), float(height) - crop_h)
    return (
        int(math.floor(left)),
        int(math.floor(top)),
        int(math.ceil(left + crop_w)),
        int(math.ceil(top + crop_h)),
    )


def load_prediction_records(path: Path, classes: list[str]) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "images" not in payload:
        raise ValueError(f"Expected raw prediction JSON with 'images': {path}")
    metadata_classes = payload.get("metadata", {}).get("classes")
    if metadata_classes is not None and list(metadata_classes) != classes:
        raise ValueError(f"Prediction classes differ from config in {path}")
    records = {}
    for record in payload["images"]:
        image_id = Path(str(record["image_id"])).name
        if image_id in records:
            raise ValueError(f"Duplicate image_id in {path}: {image_id}")
        records[image_id] = record
    return records


def ensure_output(output: Path, overwrite: bool):
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {output}; pass --overwrite")
        shutil.rmtree(output)
    output.mkdir(parents=True)


def save_crop(image: Image.Image, crop_box, output_path: Path, image_size: int, quality: int):
    crop = image.crop(crop_box).resize((image_size, image_size), Image.Resampling.BICUBIC)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop.convert("RGB").save(output_path, quality=quality, optimize=True)


def preview_grid(paths: list[Path], output_path: Path, title: str, tile: int = 160, columns: int = 4):
    if not paths:
        return
    rows = math.ceil(len(paths) / columns)
    title_h = 28
    canvas = Image.new("RGB", (columns * tile, rows * tile + title_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill=(0, 0, 0))
    for index, path in enumerate(paths):
        try:
            image = Image.open(path).convert("RGB").resize((tile, tile), Image.Resampling.BICUBIC)
        except Exception:
            continue
        x = (index % columns) * tile
        y = title_h + (index // columns) * tile
        canvas.paste(image, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def process_split(
    *,
    split: str,
    prediction_records: dict[str, dict],
    manifest_rows: dict[str, dict],
    output: Path,
    classes: list[str],
    class_to_id: dict[str, int],
    args: argparse.Namespace,
):
    metadata_rows = []
    stats = {
        "images_seen": 0,
        "images_with_gt": 0,
        "images_with_predictions": 0,
        "predictions_considered": 0,
        "positive_matches": 0,
        "saved_crops": 0,
        "gt_without_positive": 0,
    }
    class_counts = Counter({name: 0 for name in classes})
    matched_iou_values = []
    preview_samples: dict[str, list[Path]] = defaultdict(list)

    image_ids = sorted(set(prediction_records) & set(manifest_rows))
    if args.limit_images_per_split:
        image_ids = image_ids[: args.limit_images_per_split]
    missing_predictions = sorted(set(manifest_rows) - set(prediction_records))
    if missing_predictions and not args.limit_images_per_split:
        raise ValueError(
            f"{split}: predictions missing {len(missing_predictions)} manifest images; "
            f"examples: {missing_predictions[:5]}"
        )

    for image_id in image_ids:
        row = manifest_rows[image_id]
        record = prediction_records[image_id]
        image_path = Path(row["image_path"])
        xml_path = Path(row["xml_path"])
        if not image_path.is_file() or not xml_path.is_file():
            raise FileNotFoundError(f"Missing image/XML pair: {image_path}, {xml_path}")

        width, height, gt_boxes = parse_voc(xml_path, class_to_id)
        stats["images_seen"] += 1
        if gt_boxes:
            stats["images_with_gt"] += 1
        predictions = []
        for pred_index, pred in enumerate(record.get("predictions", [])):
            score = float(pred.get("score", 0.0))
            if score < args.min_score:
                continue
            box = clean_box(pred.get("bbox_xyxy", pred.get("bbox")), width, height)
            if box is None:
                continue
            predictions.append(
                {
                    "pred_index": pred_index,
                    "pred_class_id": int(pred.get("class_id", -1)),
                    "pred_class_name": str(pred.get("category_name", "")),
                    "score": score,
                    "bbox_xyxy": box,
                }
            )
        predictions.sort(key=lambda item: item["score"], reverse=True)
        predictions = predictions[: args.max_proposals_per_image]
        if predictions:
            stats["images_with_predictions"] += 1
        stats["predictions_considered"] += len(predictions)

        candidates_by_gt: dict[int, list[dict]] = defaultdict(list)
        for pred in predictions:
            best_gt, best_iou = None, 0.0
            for gt in gt_boxes:
                iou = box_iou(pred["bbox_xyxy"], gt["bbox_xyxy"])
                if iou > best_iou:
                    best_gt, best_iou = gt, iou
            if best_gt is None or best_iou < args.positive_iou:
                continue
            candidate = {**pred, "gt": best_gt, "iou": best_iou}
            candidates_by_gt[int(best_gt["gt_index"])].append(candidate)

        for gt in gt_boxes:
            candidates = sorted(
                candidates_by_gt.get(int(gt["gt_index"]), []),
                key=lambda item: (item["iou"], item["score"]),
                reverse=True,
            )[: args.max_proposals_per_gt]
            if not candidates:
                stats["gt_without_positive"] += 1
                continue
            with Image.open(image_path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                width, height = image.size
                for rank, candidate in enumerate(candidates):
                    label = candidate["gt"]["class_name"]
                    crop_box = expanded_crop_box(
                        candidate["bbox_xyxy"],
                        width,
                        height,
                        args.crop_expand,
                        args.min_crop_size,
                    )
                    filename = (
                        f"{Path(image_id).stem}__gt{gt['gt_index']:03d}"
                        f"__p{candidate['pred_index']:04d}__r{rank}__{label}.jpg"
                    )
                    crop_path = output / split / label / filename
                    save_crop(image, crop_box, crop_path, args.image_size, args.jpeg_quality)
                    if len(preview_samples[label]) < args.preview_per_class:
                        preview_samples[label].append(crop_path)
                    stats["positive_matches"] += 1
                    stats["saved_crops"] += 1
                    class_counts[label] += 1
                    matched_iou_values.append(candidate["iou"])
                    metadata_rows.append(
                        {
                            "split": split,
                            "class_id": candidate["gt"]["class_id"],
                            "class_name": label,
                            "crop_path": str(crop_path),
                            "image_id": image_id,
                            "source_image": str(image_path),
                            "source_xml": str(xml_path),
                            "gt_index": gt["gt_index"],
                            "pred_index": candidate["pred_index"],
                            "pred_class_id": candidate["pred_class_id"],
                            "pred_class_name": candidate["pred_class_name"],
                            "det_score": round(candidate["score"], 8),
                            "iou": round(candidate["iou"], 6),
                            "gt_x1": round(gt["bbox_xyxy"][0], 4),
                            "gt_y1": round(gt["bbox_xyxy"][1], 4),
                            "gt_x2": round(gt["bbox_xyxy"][2], 4),
                            "gt_y2": round(gt["bbox_xyxy"][3], 4),
                            "proposal_x1": round(candidate["bbox_xyxy"][0], 4),
                            "proposal_y1": round(candidate["bbox_xyxy"][1], 4),
                            "proposal_x2": round(candidate["bbox_xyxy"][2], 4),
                            "proposal_y2": round(candidate["bbox_xyxy"][3], 4),
                        }
                    )

    for class_name, paths in preview_samples.items():
        preview_grid(paths, output / "previews" / f"{split}_{class_name}.jpg", f"{split}/{class_name}")

    return metadata_rows, {
        **stats,
        "class_crop_counts": {name: class_counts[name] for name in classes},
        "matched_iou_mean": sum(matched_iou_values) / len(matched_iou_values)
        if matched_iou_values
        else 0.0,
        "matched_iou_min": min(matched_iou_values) if matched_iou_values else None,
    }


def main() -> None:
    args = parse_args()
    if not 0 <= args.positive_iou <= 1:
        raise ValueError("--positive-iou must be in [0, 1]")
    if not 0 <= args.min_score <= 1:
        raise ValueError("--min-score must be in [0, 1]")
    if args.max_proposals_per_gt <= 0 or args.max_proposals_per_image <= 0:
        raise ValueError("max proposal limits must be positive")
    classes = load_classes(args.classes)
    class_to_id = {name: idx for idx, name in enumerate(classes)}
    output = args.output.resolve()
    ensure_output(output, args.overwrite)
    for split in ("train", "val"):
        for class_name in classes:
            (output / split / class_name).mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.manifest)
    predictions = {
        "train": load_prediction_records(args.train_predictions, classes),
        "val": load_prediction_records(args.val_predictions, classes),
    }

    all_rows = []
    split_reports = {}
    for split in ("train", "val"):
        rows, report = process_split(
            split=split,
            prediction_records=predictions[split],
            manifest_rows=manifest[split],
            output=output,
            classes=classes,
            class_to_id=class_to_id,
            args=args,
        )
        all_rows.extend(rows)
        split_reports[split] = report

    metadata_path = output / "metadata.csv"
    if all_rows:
        with metadata_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

    report = {
        "output_dir": str(output),
        "train_predictions": str(args.train_predictions.resolve()),
        "val_predictions": str(args.val_predictions.resolve()),
        "source_manifest": str(args.manifest.resolve()),
        "classes": classes,
        "policy": {
            "positive_iou": args.positive_iou,
            "min_score": args.min_score,
            "max_proposals_per_gt": args.max_proposals_per_gt,
            "max_proposals_per_image": args.max_proposals_per_image,
            "crop_expand": args.crop_expand,
            "min_crop_size": args.min_crop_size,
            "image_size": args.image_size,
        },
        "splits": split_reports,
        "metadata": str(metadata_path),
    }
    (output / "dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
