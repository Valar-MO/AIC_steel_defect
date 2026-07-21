#!/usr/bin/env python3
"""Assign nine classes and re-score class-agnostic D-FINE proposals."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import yaml
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

from dfine_crop_classifier import (DFineDualHeadClassifier, expanded_crop_box,
                                   image_to_tensor, metadata_features)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dinov2-weights", type=Path, default=Path("/root/autodl-tmp/weights/dinov2-base"))
    p.add_argument("--manifest", type=Path, default=Path("splits/v1/split_manifest.csv"))
    p.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--tight-expand", type=float, default=1.0)
    p.add_argument("--context-expand", type=float, default=1.8)
    p.add_argument("--min-crop-size", type=int, default=128)
    p.add_argument("--min-det-score", type=float, default=0.001,
                   help="Discard proposals below this score; every retained proposal is classified.")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--score-mode", choices=("det_obj", "det_obj_cls", "det_obj_softcls", "objectness", "keep"),
                   default="det_obj_softcls")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def fuse(det, obj, cls, mode):
    if mode == "det_obj": return det * obj
    if mode == "det_obj_cls": return det * obj * cls
    if mode == "det_obj_softcls": return det * obj * (0.5 + 0.5 * cls)
    if mode == "objectness": return obj
    if mode == "keep": return det
    raise ValueError(mode)


class ProposalDataset(Dataset):
    def __init__(self, payload, image_map, args):
        self.payload, self.args, self.items = payload, args, []
        for ri, record in enumerate(payload["images"]):
            image_id = Path(str(record["image_id"])).name
            source_path = record.get("source_path") or image_map.get(image_id)
            if not source_path or not Path(source_path).is_file():
                raise FileNotFoundError(f"Missing source image for {image_id}: {source_path}")
            for pi, pred in enumerate(record.get("predictions", [])):
                if float(pred.get("score", 0)) >= args.min_det_score:
                    self.items.append((ri, pi, Path(source_path)))

    def __len__(self): return len(self.items)

    def __getitem__(self, index):
        ri, pi, path = self.items[index]; record = self.payload["images"][ri]; pred = record["predictions"][pi]
        box = pred.get("bbox_xyxy", pred.get("bbox")); score = float(pred["score"])
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB"); width, height = image.size
            tight = image.crop(expanded_crop_box(box, width, height, self.args.tight_expand, self.args.min_crop_size))
            context = image.crop(expanded_crop_box(box, width, height, self.args.context_expand, self.args.min_crop_size))
        meta = metadata_features(score=score, box=box, width=width, height=height,
                                 source=str(pred.get("source", "unknown")),
                                 touches_internal_border=bool(pred.get("touches_internal_border", False)))
        return image_to_tensor(tight, self.args.image_size), image_to_tensor(context, self.args.image_size), meta, index


def main():
    args = parse_args()
    if args.output.exists() and not args.overwrite: raise FileExistsError(f"{args.output} exists; pass --overwrite")
    payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("images"), list):
        raise ValueError("Expected raw D-FINE JSON with top-level images")
    classes = (yaml.safe_load(args.classes.read_text(encoding="utf-8")) or {})["names"]
    image_map = {}
    if args.manifest.is_file():
        with args.manifest.open(encoding="utf-8-sig", newline="") as f:
            image_map = {Path(r["image_path"]).name: r["image_path"] for r in csv.DictReader(f)}
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint.get("classes") != classes: raise ValueError("Checkpoint/config class mismatch")
    config = checkpoint.get("config", {})
    architecture = str(config.get("architecture", "legacy"))
    model = DFineDualHeadClassifier(args.dinov2_weights, len(classes),
                                    int(config.get("metadata_hidden", 128)), 0.0, architecture,
                                    int(config.get("adapter_dim", 128)))
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model.to(device).eval(); dataset = ProposalDataset(payload, image_map, args)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)
    dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    results = {}
    with torch.inference_mode():
        for tight, context, meta, indices in loader:
            tight, context, meta = (x.to(device, non_blocking=True) for x in (tight, context, meta))
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=args.amp and device.type == "cuda"):
                out = model(tight, context, meta)
            obj = out["objectness_logits"].float().sigmoid().cpu(); probs = out["class_logits"].float().softmax(1).cpu()
            conf, label = probs.max(1)
            for i, o, c, lab in zip(indices.tolist(), obj.tolist(), conf.tolist(), label.tolist()):
                results[i] = (o, c, lab)
    for item_index, (ri, pi, _) in enumerate(dataset.items):
        pred = payload["images"][ri]["predictions"][pi]; obj, cls_conf, class_id = results[item_index]
        det = float(pred["score"]); pred["original_score"] = det
        pred["score"] = max(0.0, min(1.0, fuse(det, obj, cls_conf, args.score_mode)))
        pred["class_id"] = class_id; pred["category_name"] = classes[class_id]
        pred["crop_classifier"] = {"objectness": obj, "class_probability": cls_conf,
                                   "score_mode": args.score_mode}
    for record in payload["images"]:
        record["predictions"] = [
            pred for pred in record.get("predictions", [])
            if float(pred.get("original_score", pred.get("score", 0.0))) >= args.min_det_score
        ]
    payload.setdefault("metadata", {})["classes"] = classes
    payload["metadata"]["dfine_crop_classifier"] = {"checkpoint": str(args.checkpoint.resolve()),
        "architecture": architecture, "score_mode": args.score_mode, "processed_proposals": len(dataset)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["metadata"]["dfine_crop_classifier"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
