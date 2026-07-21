#!/usr/bin/env python3
"""Run the single-model Hierarchical D-FINE on whole images or tiles."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from PIL import Image

from predict_dfine_agnostic import (load_images, load_model, make_views,
                                    preprocess, touches_internal_border)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dfine-dir", type=Path, default=Path("/root/autodl-tmp/D-FINE"))
    parser.add_argument("--config", default=
        "configs/dfine/custom/objects365/dfine_hgnetv2_l_hierarchical_aic.yml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("splits/v1/split_manifest.csv"))
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--mode", choices=("whole", "tiles"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--tile-size", type=int, default=2048)
    parser.add_argument("--overlap", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--max-det-per-view", type=int, default=300)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-images", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")
    classes = (yaml.safe_load(args.classes.read_text(encoding="utf-8")) or {}).get("names")
    if not isinstance(classes, list) or len(classes) != 9:
        raise ValueError("Expected nine classes")
    paths = load_images(args.manifest, args.split, args.image_dir, args.limit_images)
    device = torch.device(args.device)
    model, postprocessor, checkpoint = load_model(
        args.dfine_dir, args.config, args.checkpoint, device)
    records, total_views, total_predictions = [], 0, 0
    started = time.time()
    for image_index, image_path in enumerate(paths, 1):
        with Image.open(image_path) as opened:
            source = opened.convert("RGB")
        width, height = source.size
        views = make_views(width, height, args.mode, args.tile_size, args.overlap)
        predictions = []
        for offset in range(0, len(views), args.batch):
            batch_views = views[offset:offset+args.batch]
            tensors = preprocess([source.crop(view) for view in batch_views], args.imgsz, device)
            sizes = torch.tensor([[view[2]-view[0], view[3]-view[1]] for view in batch_views],
                                 dtype=torch.float32, device=device)
            with torch.inference_mode(), torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                labels, boxes, scores = postprocessor(model(tensors), sizes)
            for view, view_labels, view_boxes, view_scores in zip(
                    batch_views, labels, boxes, scores, strict=True):
                order = view_scores.argsort(descending=True)[:args.max_det_per_view]
                for label, box, score in zip(view_labels[order].cpu().tolist(),
                                             view_boxes[order].cpu().tolist(),
                                             view_scores[order].cpu().tolist(), strict=True):
                    if score < args.conf: continue
                    if not 0 <= int(label) < len(classes):
                        raise ValueError(f"Invalid class id {label}")
                    local = list(map(float, box))
                    global_box = [local[0]+view[0], local[1]+view[1],
                                  local[2]+view[0], local[3]+view[1]]
                    predictions.append({
                        "class_id": int(label), "category_name": classes[int(label)],
                        "bbox_xyxy": global_box, "score": float(score),
                        "source": "tile" if args.mode == "tiles" else "whole",
                        "view_xyxy": list(view),
                        "touches_internal_border": args.mode == "tiles" and
                        touches_internal_border(local, view, (width, height)),
                    })
        total_views += len(views); total_predictions += len(predictions)
        records.append({"image_id":image_path.name,"source_path":str(image_path),
                        "width":width,"height":height,"predictions":predictions})
        if image_index % 20 == 0 or image_index == len(paths):
            print(f"Processed {image_index}/{len(paths)} images", flush=True)
    metadata = {
        "classes": classes, "architecture": "hierarchical_dfine",
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_epoch": checkpoint.get("epoch"),
        "mode": args.mode, "image_count": len(paths), "view_count": total_views,
        "imgsz": args.imgsz, "tile_size": args.tile_size if args.mode == "tiles" else None,
        "overlap": args.overlap if args.mode == "tiles" else None,
        "confidence": args.conf, "prediction_count": total_predictions,
        "elapsed_seconds": round(time.time()-started, 3), "score_mode": "defectness",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"metadata":metadata,"images":records},
                                     ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

