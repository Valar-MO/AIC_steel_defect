#!/usr/bin/env python3
"""Run RT-DETR inference and save lossless raw predictions for submission building."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
DEFAULT_WEIGHTS = Path(
    "/root/autodl-tmp/runs/rtdetr/rtdetr_l_1024_baseline_v2/weights/best.pt"
)
DEFAULT_TEST_DIR = Path("/root/autodl-tmp/datasets/AIC_steel_defect/initial_test")
DEFAULT_OUTPUT_DIR = Path("/root/autodl-tmp/submissions/rtdetr_l_1024_baseline")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    p.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--visualize-count", type=int, default=20)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def load_classes(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        names = yaml.safe_load(f).get("names")
    if not isinstance(names, list) or len(names) != 9 or len(set(names)) != 9:
        raise ValueError(f"Expected 9 unique classes in {path}")
    return [str(x) for x in names]


def find_images(root: Path) -> list[Path]:
    images = sorted(p.resolve() for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"No test images under {root}")
    basenames = [p.name for p in images]
    if len(set(basenames)) != len(basenames):
        duplicates = [name for name, count in Counter(basenames).items() if count > 1]
        raise ValueError(f"Duplicate test image basenames: {duplicates[:10]}")
    return images


def normalize_model_names(names) -> list[str]:
    if isinstance(names, dict):
        return [str(names[i]) for i in range(len(names))]
    return [str(x) for x in names]


def main():
    args = parse_args()
    weights = args.weights.resolve()
    test_dir = args.test_dir.resolve()
    output_dir = args.output_dir.resolve()
    raw_path = output_dir / "raw_predictions.json"
    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    if not 0 <= args.conf <= 1 or args.imgsz <= 0 or args.max_det <= 0 or args.batch <= 0:
        raise ValueError("Invalid conf/imgsz/max-det/batch")
    classes = load_classes(args.classes)
    images = find_images(test_dir)
    config = {
        "weights": str(weights), "test_dir": str(test_dir), "image_count": len(images),
        "classes": classes, "imgsz": args.imgsz, "conf": args.conf,
        "max_det": args.max_det, "batch": args.batch, "device": args.device,
        "raw_predictions": str(raw_path),
    }
    print(json.dumps(config, ensure_ascii=False, indent=2))
    if args.dry_run:
        print("Dry run passed; inference was not started.")
        return
    if raw_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {raw_path}; pass --overwrite to replace it")
    output_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir = output_dir / "visualizations"
    visualization_dir.mkdir(exist_ok=True)

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from ultralytics import RTDETR

    model = RTDETR(str(weights))
    model_names = normalize_model_names(model.names)
    if model_names != classes:
        raise ValueError(f"Model classes differ from config: {model_names} != {classes}")

    started = time.time()
    records = []
    class_counts = Counter({name: 0 for name in classes})
    # A single Python list is treated by Ultralytics as one in-memory batch,
    # ignoring the CLI batch limit. Explicit chunks prevent all test images
    # from being moved to the GPU together.
    for offset in range(0, len(images), args.batch):
        chunk = images[offset : offset + args.batch]
        chunk_results = model.predict(
            source=[str(p) for p in chunk], imgsz=args.imgsz, conf=args.conf,
            max_det=args.max_det, device=args.device, verbose=False,
        )
        if len(chunk_results) != len(chunk):
            raise RuntimeError(f"Expected {len(chunk)} results, received {len(chunk_results)}")
        for local_index, (expected_path, result) in enumerate(zip(chunk, chunk_results, strict=True)):
            index = offset + local_index
            # Ultralytics assigns synthetic paths (image0.jpg, image1.jpg)
            # when the source is an in-memory Python list. Result order is
            # preserved, so bind each result to the corresponding chunk item.
            height, width = result.orig_shape
            predictions = []
            if result.boxes is not None:
                xyxy = result.boxes.xyxy.detach().cpu().tolist()
                scores = result.boxes.conf.detach().cpu().tolist()
                class_ids = result.boxes.cls.detach().cpu().to(dtype=int).tolist()
                for box, score, class_id in zip(xyxy, scores, class_ids, strict=True):
                    if not 0 <= class_id < len(classes):
                        raise RuntimeError(f"Invalid predicted class ID {class_id} for {expected_path.name}")
                    predictions.append({
                        "class_id": class_id,
                        "category_name": classes[class_id],
                        "bbox_xyxy": [float(v) for v in box],
                        "score": float(score),
                    })
                    class_counts[classes[class_id]] += 1
            predictions.sort(key=lambda item: item["score"], reverse=True)
            records.append({
                "image_id": expected_path.name,
                "source_path": str(expected_path),
                "width": int(width), "height": int(height),
                "predictions": predictions,
            })
            if index < args.visualize_count:
                result.save(filename=str(visualization_dir / expected_path.name))
        completed = min(offset + len(chunk), len(images))
        if completed % 50 < args.batch or completed == len(images):
            print(f"Processed {completed}/{len(images)} images")

    payload = {
        "metadata": {
            **config,
            "elapsed_seconds": round(time.time() - started, 3),
            "prediction_count": sum(class_counts.values()),
            "images_with_predictions": sum(bool(r["predictions"]) for r in records),
            "class_prediction_counts": dict(class_counts),
        },
        "images": records,
    }
    raw_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["metadata"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
