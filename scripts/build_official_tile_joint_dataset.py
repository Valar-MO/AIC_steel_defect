#!/usr/bin/env python3
"""Build a source-aware D-FINE COCO dataset using the official tile geometry.

Train tiles are exactly the 2048/512 windows used at inference, including
empty tiles.  COCO image metadata records overlapping sampling pools so a
sampler can draw natural tiles, positive tiles, weak-positive tiles, and A0
hard-background tiles without duplicating crop files.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from PIL import Image
import yaml


WEAK = {"huashang", "qilie", "zonglie"}


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=Path("splits/v1/split_manifest.csv"))
    p.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--tile-size", type=int, default=2048)
    p.add_argument("--overlap", type=int, default=512)
    p.add_argument("--min-visible", type=float, default=.50)
    p.add_argument("--hard-tile-predictions", type=Path,
                   help="A0 train tile raw JSON; marks high-score pure-background tile views.")
    p.add_argument("--hard-score", type=float, default=.50)
    p.add_argument("--hard-max-iou", type=float, default=.10)
    p.add_argument("--jpeg-quality", type=int, default=92)
    p.add_argument("--limit-train-images", type=int,
                   help="Build only this many train images; intended for smoke tests.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_classes(path):
    names = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("names")
    if not isinstance(names, list) or len(names) != 9 or len(set(names)) != 9:
        raise ValueError("Expected nine unique class names")
    return [str(name) for name in names]


def read_voc(path, class_to_id):
    root = ET.parse(path).getroot(); width = int(root.findtext("size/width", "0")); height = int(root.findtext("size/height", "0"))
    boxes, seen = [], set()
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip(); node = obj.find("bndbox")
        if name not in class_to_id or node is None: continue
        raw = tuple(float(node.findtext(key, "nan")) for key in ("xmin", "ymin", "xmax", "ymax"))
        if not all(math.isfinite(value) for value in raw) or (name, raw) in seen: continue
        seen.add((name, raw)); x1, y1, x2, y2 = raw
        box = (max(0., x1), max(0., y1), min(float(width), x2), min(float(height), y2))
        if box[2] > box[0] and box[3] > box[1]: boxes.append((class_to_id[name], box))
    return width, height, boxes


def starts(length, size, stride):
    if length <= size: return [0]
    result = list(range(0, length - size + 1, stride)); last = length - size
    if result[-1] != last: result.append(last)
    return result


def windows(width, height, size, overlap):
    stride = size - overlap
    return [(x, y, min(x + size, width), min(y + size, height))
            for y in starts(height, size, stride) for x in starts(width, size, stride)]


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0., x2-x1) * max(0., y2-y1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union else 0.


def crop_labels(boxes, view, min_visible):
    labels = []
    for class_id, box in boxes:
        x1, y1, x2, y2 = max(box[0], view[0]), max(box[1], view[1]), min(box[2], view[2]), min(box[3], view[3])
        if x2 <= x1 or y2 <= y1: continue
        visible = (x2-x1)*(y2-y1) / ((box[2]-box[0])*(box[3]-box[1]))
        if visible >= min_visible:
            labels.append((class_id, (x1-view[0], y1-view[1], x2-view[0], y2-view[1])))
    return labels


def hard_views(path, train_names, gt_by_image, score, max_iou):
    if path is None: return set()
    payload = json.loads(path.read_text(encoding="utf-8")); result = set()
    for record in payload.get("images", []):
        image_id = Path(record["image_id"]).name
        if image_id not in train_names: continue
        for pred in record.get("predictions", []):
            if float(pred.get("score", 0.)) < score or "view_xyxy" not in pred: continue
            box = tuple(float(value) for value in pred["bbox_xyxy"])
            if max((iou(box, gt) for _, gt in gt_by_image[image_id]), default=0.) < max_iou:
                result.add((image_id, tuple(int(value) for value in pred["view_xyxy"])))
    return result


def main():
    opt = args()
    if opt.tile_size <= 0 or not 0 <= opt.overlap < opt.tile_size or not 0 < opt.min_visible <= 1: raise ValueError("Invalid tile parameters")
    names = load_classes(opt.classes); class_to_id = {name: index for index, name in enumerate(names)}
    rows = list(csv.DictReader(opt.manifest.open(encoding="utf-8-sig", newline="")))
    parsed = {Path(row["image_path"]).name: read_voc(Path(row["xml_path"]), class_to_id) for row in rows}
    train = [row for row in rows if row["split"] == "train"]; val = [row for row in rows if row["split"] == "val"]
    if opt.limit_train_images is not None:
        train = train[:opt.limit_train_images]
    train_names = {Path(row["image_path"]).name for row in train}
    gt_by_image = {name: parsed[name][2] for name in train_names}
    hard = hard_views(opt.hard_tile_predictions, train_names, gt_by_image, opt.hard_score, opt.hard_max_iou)
    output = opt.output_dir.resolve()
    if output.exists():
        if not opt.overwrite: raise FileExistsError(output)
        shutil.rmtree(output)
    coco, annotation_id, report = {"train": ([], []), "val": ([], [])}, 1, Counter()

    def add_image(split, source, width, height, **metadata):
        image_id = len(coco[split][0]) + 1
        coco[split][0].append({"id": image_id, "file_name": source.name, "source_path": str(source.resolve()),
                                "width": width, "height": height, "source_image_id": source.name, **metadata})
        report[f"{split}_{metadata['view_type']}"] += 1
        return image_id

    def add_labels(split, image_id, labels):
        nonlocal annotation_id
        for class_id, box in labels:
            x1, y1, x2, y2 = box
            coco[split][1].append({"id": annotation_id, "image_id": image_id, "category_id": class_id,
                                   "bbox": [x1, y1, x2-x1, y2-y1], "area": (x2-x1)*(y2-y1), "iscrowd": 0})
            annotation_id += 1

    for row in train + val:
        split, source = row["split"], Path(row["image_path"]); width, height, boxes = parsed[source.name]
        whole_id = add_image(split, source, width, height, view_type="whole", is_whole=True,
                             is_tile=False, is_positive=bool(boxes), is_weak_positive=any(names[c] in WEAK for c, _ in boxes),
                             is_hard_background=False, view_xyxy=[0, 0, width, height])
        add_labels(split, whole_id, boxes)
        if split != "train": continue
        for index, view in enumerate(windows(width, height, opt.tile_size, opt.overlap)):
            labels = crop_labels(boxes, view, opt.min_visible)
            weak = any(names[class_id] in WEAK for class_id, _ in labels)
            image_id = add_image(split, source, view[2]-view[0], view[3]-view[1], view_type="tile",
                                 is_whole=False, is_tile=True, is_positive=bool(labels), is_weak_positive=weak,
                                 is_hard_background=(source.name, view) in hard, view_xyxy=list(view))
            add_labels(split, image_id, labels)
    annotation_dir = output / "annotations"; annotation_dir.mkdir(parents=True)
    categories = [{"id": index, "name": name, "supercategory": "defect"} for index, name in enumerate(names)]
    for split, (images, annotations) in coco.items():
        payload = {"info": {"description": "AIC official-view joint training"}, "licenses": [], "images": images,
                   "annotations": annotations, "categories": categories}
        (annotation_dir / f"instances_{split}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        report[f"{split}_annotations"] = len(annotations); report[f"{split}_empty"] = len(images) - len({x["image_id"] for x in annotations})
    summary = {"tile_size": opt.tile_size, "overlap": opt.overlap, "min_visible": opt.min_visible,
               "hard_tile_predictions": str(opt.hard_tile_predictions) if opt.hard_tile_predictions else None,
               "hard_views": len(hard), "storage": "virtual crops decoded from source_path by OfficialViewCocoDetection",
               "counts": dict(report),
               "sampling_groups": ["whole", "natural_tile", "positive_tile", "weak_positive_tile", "hard_background_tile"]}
    (output / "dataset_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
