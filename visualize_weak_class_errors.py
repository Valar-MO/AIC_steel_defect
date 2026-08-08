#!/usr/bin/env python3
"""Render weak-class TP/FP/FN crop sheets and a GT-centric confusion matrix."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--diagnostics-dir", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-samples", type=int, default=24)
    p.add_argument("--thumb-size", type=int, default=320)
    return p.parse_args()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_box(value):
    return [float(v) for v in ast.literal_eval(value)]


def crop_context(image, box, scale=2.0, minimum=160):
    x1, y1, x2, y2 = box
    w, h = max(x2 - x1, 1), max(y2 - y1, 1)
    side_w, side_h = max(w * scale, minimum), max(h * scale, minimum)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left, top = max(0, cx - side_w / 2), max(0, cy - side_h / 2)
    right, bottom = min(image.width, cx + side_w / 2), min(image.height, cy + side_h / 2)
    return image.crop((left, top, right, bottom)), [x1-left, y1-top, x2-left, y2-top]


def make_tile(path, box, title, color, size):
    with Image.open(path) as source:
        image = source.convert("RGB")
    crop, local = crop_context(image, box)
    draw = ImageDraw.Draw(crop)
    width = max(2, round(min(crop.size) / 80))
    draw.rectangle(local, outline=color, width=width)
    crop.thumbnail((size, size - 36), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (size, size), "white")
    tile.paste(crop, ((size-crop.width)//2, 32+(size-32-crop.height)//2))
    ImageDraw.Draw(tile).text((6, 7), title[:55], fill=(15, 15, 15), font=ImageFont.load_default())
    return tile


def save_sheet(items, paths, output, size):
    if not items:
        return
    cols = 4
    rows = math.ceil(len(items) / cols)
    sheet = Image.new("RGB", (cols*size, rows*size), (225, 225, 225))
    for i, (image_id, box, title, color) in enumerate(items):
        path = paths.get(image_id)
        if not path or not Path(path).exists():
            continue
        tile = make_tile(path, box, title, color, size)
        sheet.paste(tile, ((i % cols)*size, (i // cols)*size))
    sheet.save(output, quality=92)


def main():
    a = args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((a.diagnostics_dir / "summary.json").read_text(encoding="utf-8"))
    targets = summary["metadata"]["target_classes"]
    pred_payload = json.loads(a.predictions.read_text(encoding="utf-8"))
    paths = {item["image_id"]: item["source_path"] for item in pred_payload["images"]}
    pred_rows = read_csv(a.diagnostics_dir / "target_prediction_outcomes.csv")
    gt_rows = read_csv(a.diagnostics_dir / "ground_truth_diagnostics.csv")

    for name in targets:
        class_dir = a.output_dir / name
        class_dir.mkdir(exist_ok=True)
        tp = [r for r in pred_rows if r["predicted_class"] == name and r["outcome"] == "TP"]
        fp = [r for r in pred_rows if r["predicted_class"] == name and r["outcome"] == "FP"]
        fn = [r for r in gt_rows if r["true_class"] == name and r["status"] != "TP"]
        tp.sort(key=lambda r: float(r["score"]), reverse=True)
        fp.sort(key=lambda r: float(r["score"]), reverse=True)
        fn.sort(key=lambda r: float(r["best_any_iou"] or 0), reverse=True)
        save_sheet([(r["image_id"], parse_box(r["bbox_xyxy"]),
                     f"TP score={float(r['score']):.3f}", (25, 190, 70))
                    for r in tp[:a.max_samples]], paths, class_dir / "TP.jpg", a.thumb_size)
        save_sheet([(r["image_id"], parse_box(r["bbox_xyxy"]),
                     f"FP {r['fp_reason']} score={float(r['score']):.3f} near={r['nearest_gt_class']}",
                     (230, 45, 45)) for r in fp[:a.max_samples]],
                   paths, class_dir / "FP.jpg", a.thumb_size)
        save_sheet([(r["image_id"], parse_box(r["gt_bbox_xyxy"]),
                     f"FN {r['status']} best={r['best_any_class']} IoU={float(r['best_any_iou']):.2f}",
                     (245, 150, 20)) for r in fn[:a.max_samples]],
                   paths, class_dir / "FN.jpg", a.thumb_size)

    labels = targets + sorted({r["confusion_label"] for r in gt_rows if r["confusion_label"] not in targets})
    matrix = np.zeros((len(targets), len(labels)), dtype=int)
    for r in gt_rows:
        if r["true_class"] in targets:
            matrix[targets.index(r["true_class"]), labels.index(r["confusion_label"])] += 1
    fig, ax = plt.subplots(figsize=(max(9, len(labels)*1.05), 5.5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
    ax.set_yticks(range(len(targets)), targets)
    ax.set_xlabel("Predicted / failure label")
    ax.set_ylabel("Ground-truth class")
    ax.set_title("Weak-class confusion matrix (IoU=0.5)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max()*0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(a.output_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    counts = {name: dict(Counter(r["status"] for r in gt_rows if r["true_class"] == name))
              for name in targets}
    (a.output_dir / "visualization_summary.json").write_text(
        json.dumps(counts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(a.output_dir), "counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
