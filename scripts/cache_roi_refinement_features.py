#!/usr/bin/env python3
"""Create frozen P2/P3 ROI caches and raw candidate JSON for R2 refinement.

Run separately for train and val.  The cache is candidate-level, never a full
feature-map dump: it stores only candidates that can reach the .30 protocol
threshold after the bounded score residual.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch.nn import functional as F

from predict_dfine_agnostic import load_images, make_views, preprocess, touches_internal_border
from roi_refinement_common import (cxcywh_to_xyxy, extract_candidate_rois)


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dfine-dir", type=Path, default=Path("/root/autodl-tmp/D-FINE"))
    p.add_argument("--config", default="configs/dfine/custom/objects365/dfine_hgnetv2_l_hierarchical_aic.yml")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=Path("splits/v1/split_manifest.csv"))
    p.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    p.add_argument("--image-dir", type=Path)
    p.add_argument("--split", choices=("train", "val", "test"), required=True,
                   help="test requires --image-dir and never reads annotations")
    p.add_argument("--mode", choices=("whole", "tiles"), required=True)
    p.add_argument("--output-cache", type=Path, required=True)
    p.add_argument("--output-raw", type=Path, required=True)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--tile-size", type=int, default=2048)
    p.add_argument("--overlap", type=int, default=512)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--prefilter", type=float, default=.26)
    p.add_argument("--context-ratio", type=float, default=1.25)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit-images", type=int)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_model(dfine_dir: Path, config: str, checkpoint_path: Path, device: torch.device):
    sys.path.insert(0, str(dfine_dir.resolve()))
    from src.core import YAMLConfig  # pylint: disable=import-outside-toplevel

    config_path = Path(config)
    if not config_path.is_absolute(): config_path = dfine_dir / config_path
    cfg = YAMLConfig(str(config_path), resume=str(checkpoint_path))
    # Expose true stride-4 output. Encoder still receives only P3/P4/P5.
    cfg.yaml_cfg["HGNetv2"]["return_idx"] = [0, 1, 2, 3]
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Base checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return model.deploy().to(device).eval(), config_path


def cat(rows, shape, dtype):
    return torch.cat(rows, dim=0) if rows else torch.empty(shape, dtype=dtype)


def main() -> None:
    opt = args()
    if not 0 < opt.prefilter < .5: raise ValueError("--prefilter must be in (0,.5)")
    if opt.split == "test" and opt.image_dir is None:
        raise ValueError("--split test requires --image-dir")
    if opt.output_cache.exists() and not opt.overwrite: raise FileExistsError(opt.output_cache)
    if opt.output_raw.exists() and not opt.overwrite: raise FileExistsError(opt.output_raw)
    classes = (yaml.safe_load(opt.classes.read_text(encoding="utf-8")) or {}).get("names")
    if not isinstance(classes, list) or len(classes) != 9: raise ValueError("Expected nine classes")
    device = torch.device(opt.device)
    model, config_path = load_model(opt.dfine_dir, opt.config, opt.checkpoint, device)
    captured: list[torch.Tensor] = []
    handle = model.decoder.dec_score_head[-1].register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach()))
    p2_rows, p3_rows, query_rows, selection_rows, class_rows, box_rows = [], [], [], [], [], []
    image_rows, candidate_image_rows, query_index_rows = [], [], []
    source_rows, border_rows, context_rows, distance_rows, view_rows = [], [], [], [], []
    records, candidate_count, started = [], 0, time.time()
    try:
        image_paths = load_images(opt.manifest, opt.split, opt.image_dir, opt.limit_images)
        for image_number, image_path in enumerate(image_paths, 1):
            with Image.open(image_path) as opened: image = opened.convert("RGB")
            width, height = image.size
            views = make_views(width, height, opt.mode, opt.tile_size, opt.overlap)
            predictions, image_cache_index = [], len(image_rows)
            image_rows.append({"image_id": image_path.name, "source_path": str(image_path.resolve()),
                               "width": width, "height": height, "mode": opt.mode})
            for offset in range(0, len(views), opt.batch):
                batch_views = views[offset:offset + opt.batch]
                tensors = preprocess([image.crop(view) for view in batch_views], opt.imgsz, device)
                captured.clear()
                with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16,
                                                             enabled=device.type == "cuda"):
                    features = model.backbone(tensors)
                    if len(features) != 4: raise RuntimeError("Expected true P2/P3/P4/P5 backbone outputs")
                    outputs = model.decoder(model.encoder(features[1:]))
                    p2, p3 = features[:2]
                    selection_logits = outputs["pred_objectness_logits"].squeeze(-1)
                    selection = selection_logits.sigmoid()
                    class_logits = outputs["pred_class_logits"]
                    class_probability = F.softmax(class_logits, dim=-1)
                    boxes = outputs["pred_boxes"]
                if not captured: raise RuntimeError("Decoder score head did not expose query features")
                queries = captured[-1]
                active = selection >= opt.prefilter
                p2_roi, p3_roi, context_clipped = extract_candidate_rois(
                    p2, p3, boxes, active, opt.imgsz, opt.context_ratio)
                for b, view in enumerate(batch_views):
                    keep = torch.where(active[b])[0]
                    if not len(keep): continue
                    local_xyxy = cxcywh_to_xyxy(boxes[b, keep]).clamp(0, 1)
                    pixels = local_xyxy * local_xyxy.new_tensor([view[2]-view[0], view[3]-view[1],
                                                                  view[2]-view[0], view[3]-view[1]])
                    offset_xyxy = pixels + pixels.new_tensor([view[0], view[1], view[0], view[1]])
                    roi_start = sum(len(row) for row in p2_rows)
                    # Select the per-view portion from flattened ROIAlign output.
                    before = int(active[:b].sum().item()); after = before + len(keep)
                    p2_rows.append(p2_roi[before:after].cpu().to(torch.float16))
                    p3_rows.append(p3_roi[before:after].cpu().to(torch.float16))
                    query_rows.append(queries[b, keep].cpu().to(torch.float16))
                    selection_rows.append(selection_logits[b, keep].cpu().to(torch.float16))
                    class_rows.append(class_logits[b, keep].cpu().to(torch.float16))
                    box_rows.append(boxes[b, keep].cpu().to(torch.float16))
                    view_rows.append(torch.tensor(view, dtype=torch.int32).repeat(len(keep), 1))
                    candidate_image_rows.append(torch.full((len(keep),), image_cache_index, dtype=torch.int32))
                    query_index_rows.append(keep.cpu().to(torch.int16))
                    source_rows.append(torch.full((len(keep),), 1 if opt.mode == "tiles" else 0, dtype=torch.uint8))
                    local_pixels = pixels.cpu()
                    border = torch.tensor([touches_internal_border(row.tolist(), view, (width, height))
                                           for row in local_pixels], dtype=torch.bool)
                    border_rows.append(border)
                    context_rows.append(context_clipped[before:after].cpu())
                    # nearest distance to an internal tile edge, normalized by shorter view side
                    if opt.mode == "tiles":
                        left, top, right, bottom = view
                        distances = torch.stack((pixels[:, 0], pixels[:, 1],
                                                 (right-left)-pixels[:, 2], (bottom-top)-pixels[:, 3]), dim=1).amin(1)
                        distance_rows.append((distances / min(right-left, bottom-top)).clamp(0, 1).cpu().to(torch.float16))
                    else:
                        distance_rows.append(torch.ones(len(keep), dtype=torch.float16))
                    predicted = class_probability[b, keep].argmax(-1)
                    for local_i, query_i in enumerate(keep.cpu().tolist()):
                        cache_index = roi_start + local_i
                        predictions.append({
                            # A query index repeats for every tile view.  The cache row is
                            # unique within this source and therefore safe for later joins.
                            "candidate_id": f"{opt.mode}:{cache_index}", "roi_cache_index": cache_index,
                            "class_id": int(predicted[local_i]), "category_name": classes[int(predicted[local_i])],
                            "bbox_xyxy": [float(v) for v in offset_xyxy[local_i].cpu().tolist()],
                            "score": float(selection[b, query_i]), "selection_probability": float(selection[b, query_i]),
                            "source": "tile" if opt.mode == "tiles" else "whole", "view_xyxy": list(view),
                            "query_index": query_i, "touches_internal_border": bool(border[local_i]),
                        })
                    candidate_count += len(keep)
            records.append({"image_id": image_path.name, "source_path": str(image_path), "width": width,
                            "height": height, "predictions": predictions})
            if image_number % 10 == 0 or image_number == len(image_paths):
                print(f"cached {opt.mode} {image_number}/{len(image_paths)} images, {candidate_count} candidates", flush=True)
    finally:
        handle.remove()
    cache = {
        "version": 1, "architecture": "hierarchical_dfine_roi_refinement", "split": opt.split,
        "mode": opt.mode, "classes": classes, "checkpoint": str(opt.checkpoint.resolve()),
        "config": str(config_path.resolve()), "imgsz": opt.imgsz, "prefilter": opt.prefilter,
        "context_ratio": opt.context_ratio, "images": image_rows,
        "p2_roi": cat(p2_rows, (0, 128, 14, 14), torch.float16), "p3_roi": cat(p3_rows, (0, 512, 7, 7), torch.float16),
        "query": cat(query_rows, (0, 256), torch.float16), "selection_logit": cat(selection_rows, (0,), torch.float16),
        "class_logits": cat(class_rows, (0, 9), torch.float16), "boxes_cxcywh": cat(box_rows, (0, 4), torch.float16),
        "view_xyxy": cat(view_rows, (0, 4), torch.int32),
        "candidate_image_index": cat(candidate_image_rows, (0,), torch.int32), "query_index": cat(query_index_rows, (0,), torch.int16),
        "source_id": cat(source_rows, (0,), torch.uint8), "touches_internal_border": cat(border_rows, (0,), torch.bool),
        "context_clipped": cat(context_rows, (0,), torch.bool), "border_distance": cat(distance_rows, (0,), torch.float16),
        "elapsed_seconds": time.time() - started,
    }
    opt.output_cache.parent.mkdir(parents=True, exist_ok=True); torch.save(cache, opt.output_cache)
    raw = {"metadata": {"classes": classes, "architecture": "hierarchical_dfine_roi_cache", "split": opt.split,
                           "mode": opt.mode, "prefilter": opt.prefilter, "prediction_count": candidate_count}, "images": records}
    opt.output_raw.parent.mkdir(parents=True, exist_ok=True); opt.output_raw.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"cache": str(opt.output_cache.resolve()), "raw": str(opt.output_raw.resolve()),
                      "images": len(records), "candidates": candidate_count,
                      "gigabytes": round((opt.output_cache.stat().st_size / 2**30), 3)}, indent=2))


if __name__ == "__main__": main()
