#!/usr/bin/env python3
"""Cache frozen V2.1 fused features for cheap proposal-head experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from torch.utils.data import DataLoader

from apply_dfine_crop_classifier import ProposalDataset
from dfine_crop_classifier import DFineCropDataset, DFineDualHeadClassifier


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path, help="Curated crop dataset root.")
    source.add_argument("--predictions", type=Path, help="Raw full proposal JSON.")
    p.add_argument("--split", choices=("train", "val"), default="train")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dinov2-weights", type=Path, default=Path("/root/autodl-tmp/weights/dinov2-base"))
    p.add_argument("--manifest", type=Path, default=Path("splits/v1/split_manifest.csv"))
    p.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--image-size", type=int, default=518)
    p.add_argument("--tight-expand", type=float, default=1.0)
    p.add_argument("--context-expand", type=float, default=1.8)
    p.add_argument("--min-crop-size", type=int, default=128)
    p.add_argument("--min-det-score", type=float, default=0.03)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")
    classes = (yaml.safe_load(args.classes.read_text(encoding="utf-8")) or {})["names"]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    architecture = str(config.get("architecture", "legacy"))
    model = DFineDualHeadClassifier(
        args.dinov2_weights, len(classes), int(config.get("metadata_hidden", 128)), 0.0,
        architecture, int(config.get("adapter_dim", 128)),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model.to(device).eval().requires_grad_(False)

    payload = None
    if args.data:
        dataset = DFineCropDataset(
            args.data, args.split, image_size=args.image_size, tight_expand=args.tight_expand,
            context_expand=args.context_expand, min_crop_size=args.min_crop_size, train=False,
        )
        det_scores = torch.tensor([float(r["det_score"]) for r in dataset.rows], dtype=torch.float32)
        locations = None
        relation_metadata = {
            "image_ids": [Path(row["image_id"]).name for row in dataset.rows],
            "boxes_xyxy": torch.tensor([[float(row[key]) for key in
                ("proposal_x1", "proposal_y1", "proposal_x2", "proposal_y2")]
                for row in dataset.rows], dtype=torch.float32),
            "image_sizes": torch.tensor([[int(row["image_width"]), int(row["image_height"])]
                for row in dataset.rows], dtype=torch.int32),
            "audit_outcomes": [row["audit_outcome"] for row in dataset.rows],
            "reference_gt_ids": [row.get("reference_gt_id", "") for row in dataset.rows],
        }
    else:
        payload = json.loads(args.predictions.read_text(encoding="utf-8"))
        image_map = {}
        if args.manifest.is_file():
            with args.manifest.open(encoding="utf-8-sig", newline="") as f:
                image_map = {Path(r["image_path"]).name: r["image_path"] for r in csv.DictReader(f)}
        proposal_args = SimpleNamespace(
            min_det_score=args.min_det_score, tight_expand=args.tight_expand,
            context_expand=args.context_expand, min_crop_size=args.min_crop_size,
            image_size=args.image_size,
        )
        dataset = ProposalDataset(payload, image_map, proposal_args)
        det_scores = torch.tensor([
            float(payload["images"][ri]["predictions"][pi]["score"])
            for ri, pi, _ in dataset.items
        ], dtype=torch.float32)
        locations = torch.tensor([(ri, pi) for ri, pi, _ in dataset.items], dtype=torch.int32)
        relation_metadata = None

    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                        pin_memory=True, persistent_workers=args.workers > 0)
    dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    features, objectness_logits, class_logits = [], [], []
    obj_targets, class_targets = [], []
    with torch.inference_mode():
        for step, batch in enumerate(loader, 1):
            if args.data:
                tight, context, meta, obj, cls = batch
                obj_targets.append(obj.to(torch.uint8)); class_targets.append(cls)
            else:
                tight, context, meta, _ = batch
            tight, context, meta = (x.to(device, non_blocking=True) for x in (tight, context, meta))
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                out = model(tight, context, meta)
            features.append(out["fused_features"].detach().to("cpu", torch.float16))
            objectness_logits.append(out["objectness_logits"].detach().to("cpu", torch.float16))
            class_logits.append(out["class_logits"].detach().to("cpu", torch.float16))
            if step == 1 or step % 25 == 0 or step == len(loader):
                print(f"cache step={step}/{len(loader)} samples={min(step * args.batch, len(dataset))}")

    cache = {
        "features": torch.cat(features),
        "baseline_objectness_logits": torch.cat(objectness_logits),
        "baseline_class_logits": torch.cat(class_logits),
        "detector_scores": det_scores,
        "classes": classes,
        "architecture": architecture,
        "checkpoint": str(args.checkpoint.resolve()),
        "source": str((args.data or args.predictions).resolve()),
        "split": args.split if args.data else "raw_predictions",
    }
    if args.data:
        cache["objectness_targets"] = torch.cat(obj_targets)
        cache["class_targets"] = torch.cat(class_targets)
        cache["relation_metadata"] = relation_metadata
    else:
        cache["proposal_locations"] = locations
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, args.output)
    print(json.dumps({"output": str(args.output), "samples": len(dataset),
                      "feature_shape": list(cache["features"].shape)}, indent=2))


if __name__ == "__main__":
    main()
