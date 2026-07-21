#!/usr/bin/env python3
"""Run class-agnostic D-FINE inference on whole images or sliding tiles."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dfine-dir", type=Path, default=Path("/root/autodl-tmp/D-FINE"))
    parser.add_argument(
        "--config", default="configs/dfine/custom/objects365/dfine_hgnetv2_l_obj2aic_defect.yml"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("splits/v1/split_manifest.csv"))
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


def load_images(
    manifest: Path, split: str, image_dir: Path | None, limit: int | None
) -> list[Path]:
    if image_dir:
        suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        paths = sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in suffixes)
    else:
        with manifest.open(encoding="utf-8-sig", newline="") as handle:
            paths = [Path(row["image_path"]) for row in csv.DictReader(handle) if row["split"] == split]
    if not paths:
        raise ValueError("No input images found")
    if len({path.name for path in paths}) != len(paths):
        raise ValueError("Image basenames must be unique")
    return paths[:limit] if limit else paths


def axis_starts(length: int, size: int, stride: int) -> list[int]:
    if length <= size:
        return [0]
    values = list(range(0, length - size + 1, stride))
    if values[-1] != length - size:
        values.append(length - size)
    return values


def make_views(width: int, height: int, mode: str, tile_size: int, overlap: int):
    if mode == "whole":
        return [(0, 0, width, height)]
    stride = tile_size - overlap
    return [
        (x, y, min(x + tile_size, width), min(y + tile_size, height))
        for y in axis_starts(height, tile_size, stride)
        for x in axis_starts(width, tile_size, stride)
    ]


def load_model(dfine_dir: Path, config: str, checkpoint_path: Path, device: torch.device):
    sys.path.insert(0, str(dfine_dir.resolve()))
    from src.core import YAMLConfig  # pylint: disable=import-outside-toplevel

    config_path = Path(config)
    if not config_path.is_absolute():
        config_path = dfine_dir / config_path
    cfg = YAMLConfig(str(config_path), resume=str(checkpoint_path))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    cfg.model.load_state_dict(state, strict=True)
    model = cfg.model.deploy().to(device).eval()
    postprocessor = cfg.postprocessor.deploy().to(device).eval()
    return model, postprocessor, checkpoint


def preprocess(images: list[Image.Image], size: int, device: torch.device) -> torch.Tensor:
    tensors = [
        pil_to_tensor(image.resize((size, size), Image.Resampling.BILINEAR)).float().div_(255.0)
        for image in images
    ]
    return torch.stack(tensors).to(device, non_blocking=True)


def touches_internal_border(box, view, image_size, tolerance: float = 2.0) -> bool:
    x1, y1, x2, y2 = box
    vx1, vy1, vx2, vy2 = view
    width, height = image_size
    return bool(
        (vx1 > 0 and x1 <= tolerance) or
        (vy1 > 0 and y1 <= tolerance) or
        (vx2 < width and x2 >= (vx2 - vx1) - tolerance) or
        (vy2 < height and y2 >= (vy2 - vy1) - tolerance)
    )


def main() -> None:
    args = parse_args()
    if args.imgsz <= 0 or args.batch <= 0 or args.max_det_per_view <= 0:
        raise ValueError("imgsz, batch, and max detections must be positive")
    if args.tile_size <= 0 or not 0 <= args.overlap < args.tile_size:
        raise ValueError("Invalid tile size/overlap")
    if not 0 <= args.conf <= 1:
        raise ValueError("conf must be in [0, 1]")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}; pass --overwrite")

    images = load_images(args.manifest, args.split, args.image_dir, args.limit_images)
    device = torch.device(args.device)
    model, postprocessor, checkpoint = load_model(
        args.dfine_dir, args.config, args.checkpoint, device
    )
    records, total_views, total_predictions = [], 0, 0
    started = time.time()
    for image_index, image_path in enumerate(images, 1):
        with Image.open(image_path) as opened:
            source = opened.convert("RGB")
        width, height = source.size
        views = make_views(width, height, args.mode, args.tile_size, args.overlap)
        predictions = []
        for offset in range(0, len(views), args.batch):
            batch_views = views[offset:offset + args.batch]
            crops = [source.crop(view) for view in batch_views]
            tensors = preprocess(crops, args.imgsz, device)
            sizes = torch.tensor(
                [[view[2] - view[0], view[3] - view[1]] for view in batch_views],
                dtype=torch.float32, device=device,
            )
            with torch.inference_mode(), torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                outputs = model(tensors)
                labels, boxes, scores = postprocessor(outputs, sizes)
            for view, labs, box_batch, score_batch in zip(
                batch_views, labels, boxes, scores, strict=True
            ):
                order = torch.argsort(score_batch, descending=True)[:args.max_det_per_view]
                for label, box, score in zip(
                    labs[order].cpu().tolist(), box_batch[order].cpu().tolist(),
                    score_batch[order].cpu().tolist(), strict=True,
                ):
                    if score < args.conf:
                        continue
                    if int(label) != 0:
                        raise ValueError(f"Expected class 0 from class-agnostic model, got {label}")
                    local_box = [float(value) for value in box]
                    global_box = [
                        local_box[0] + view[0], local_box[1] + view[1],
                        local_box[2] + view[0], local_box[3] + view[1],
                    ]
                    predictions.append({
                        "class_id": 0, "category_name": "defect",
                        "bbox_xyxy": global_box, "score": float(score),
                        "source": "tile" if args.mode == "tiles" else "whole",
                        "view_xyxy": list(view),
                        "touches_internal_border": args.mode == "tiles" and
                        touches_internal_border(local_box, view, (width, height)),
                    })
        total_views += len(views)
        total_predictions += len(predictions)
        records.append({
            "image_id": image_path.name, "source_path": str(image_path),
            "width": width, "height": height, "predictions": predictions,
        })
        if image_index % 20 == 0 or image_index == len(images):
            print(f"Processed {image_index}/{len(images)} images", flush=True)

    metadata = {
        "classes": ["defect"], "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"), "mode": args.mode,
        "image_count": len(images), "view_count": total_views, "imgsz": args.imgsz,
        "tile_size": args.tile_size if args.mode == "tiles" else None,
        "overlap": args.overlap if args.mode == "tiles" else None,
        "confidence": args.conf, "prediction_count": total_predictions,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"metadata": metadata, "images": records}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
