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
from torch.nn import functional as F
from torchvision.ops import box_convert

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
    parser.add_argument(
        "--class-topk", type=int, default=1,
        help="Emit the top-k conditional classes for every retained query. "
             "All copies keep the same class-agnostic objectness score; "
             "class_probability is saved separately.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-images", type=int)
    parser.add_argument(
        "--candidate-cache", type=Path,
        help="Optionally save decoder-query candidates aligned to the raw "
             "prediction JSON for a later Query+DINO verifier pass.",
    )
    parser.add_argument(
        "--cache-min-selection", type=float, default=0.05,
        help="Selection floor for --candidate-cache. This must be no higher "
             "than any downstream Selection threshold.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 1 <= args.class_topk <= 9:
        raise ValueError("class-topk must be in [1, 9]")
    if not 0 <= args.cache_min_selection < 1:
        raise ValueError("--cache-min-selection must be in [0,1)")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")
    if (
        args.candidate_cache is not None
        and args.candidate_cache.exists()
        and not args.overwrite
    ):
        raise FileExistsError(
            f"{args.candidate_cache} exists; pass --overwrite"
        )
    classes = (yaml.safe_load(args.classes.read_text(encoding="utf-8")) or {}).get("names")
    if not isinstance(classes, list) or len(classes) != 9:
        raise ValueError("Expected nine classes")
    paths = load_images(args.manifest, args.split, args.image_dir, args.limit_images)
    device = torch.device(args.device)
    model, postprocessor, checkpoint = load_model(
        args.dfine_dir, args.config, args.checkpoint, device)
    del postprocessor
    captured_queries = []
    capture_handle = None
    cache_features, cache_boxes = [], []
    cache_selection_logits, cache_class_logits = [], []
    cache_candidate_image_index, cache_query_indices = [], []
    cache_images = []
    if args.candidate_cache is not None:
        score_head = model.decoder.dec_score_head[-1]

        def capture_query(_module, inputs):
            captured_queries.append(inputs[0].detach())

        capture_handle = score_head.register_forward_pre_hook(capture_query)
    records, total_views, total_predictions = [], 0, 0
    started = time.time()
    try:
        for image_index, image_path in enumerate(paths, 1):
            with Image.open(image_path) as opened:
                source = opened.convert("RGB")
            width, height = source.size
            cache_image_index = len(cache_images)
            if args.candidate_cache is not None:
                cache_images.append({
                    "image_id": image_path.name,
                    "source_path": str(image_path.resolve()),
                    "width": width,
                    "height": height,
                })
            views = make_views(width, height, args.mode, args.tile_size, args.overlap)
            predictions = []
            for offset in range(0, len(views), args.batch):
                batch_views = views[offset:offset+args.batch]
                tensors = preprocess([source.crop(view) for view in batch_views], args.imgsz, device)
                sizes = torch.tensor([[view[2]-view[0], view[3]-view[1]] for view in batch_views],
                                     dtype=torch.float32, device=device)
                captured_queries.clear()
                with torch.inference_mode(), torch.autocast(
                        device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    outputs = model(tensors)
                    raw_selection_logits = outputs["pred_objectness_logits"].squeeze(-1)
                    selection = raw_selection_logits.sigmoid()
                    defectness_logits = outputs.get("pred_defectness_logits")
                    defectness = (
                        defectness_logits.sigmoid().squeeze(-1)
                        if defectness_logits is not None else selection
                    )
                    quality_logits = outputs.get("pred_quality_logits")
                    quality = (
                        quality_logits.sigmoid().squeeze(-1)
                        if quality_logits is not None else torch.ones_like(selection)
                    )
                    objectness = (
                        selection * defectness * quality
                        if defectness_logits is not None else defectness * quality
                    )
                    class_probability = F.softmax(outputs["pred_class_logits"], dim=-1)
                    class_probabilities, class_ids = class_probability.topk(
                        args.class_topk, dim=-1
                    )
                    boxes = box_convert(outputs["pred_boxes"], in_fmt="cxcywh", out_fmt="xyxy")
                    boxes = boxes * sizes.repeat(1, 2).unsqueeze(1)
                if args.candidate_cache is not None and not captured_queries:
                    raise RuntimeError("Decoder query hook captured no features")
                query_features = (
                    captured_queries[-1] if args.candidate_cache is not None else None
                )
                for batch_index, (
                    view, view_scores, view_selection, view_defectness, view_quality,
                    view_class_ids, view_class_probabilities, view_boxes,
                ) in enumerate(zip(
                    batch_views, objectness, selection, defectness, quality,
                    class_ids, class_probabilities, boxes, strict=True,
                )):
                    cache_indices = {}
                    if args.candidate_cache is not None:
                        keep = torch.where(
                            view_selection >= args.cache_min_selection
                        )[0]
                        for query_index in keep.cpu().tolist():
                            local = list(map(
                                float, view_boxes[query_index].cpu().tolist()
                            ))
                            global_box = [
                                local[0] + view[0], local[1] + view[1],
                                local[2] + view[0], local[3] + view[1],
                            ]
                            candidate_index = len(cache_features)
                            cache_indices[query_index] = candidate_index
                            cache_features.append(
                                query_features[batch_index, query_index]
                                .cpu().to(torch.float16)
                            )
                            cache_boxes.append(torch.tensor(
                                global_box, dtype=torch.float32
                            ))
                            cache_selection_logits.append(
                                raw_selection_logits[batch_index, query_index]
                                .cpu().to(torch.float16)
                            )
                            cache_class_logits.append(
                                outputs["pred_class_logits"][batch_index, query_index]
                                .cpu().to(torch.float16)
                            )
                            cache_candidate_image_index.append(cache_image_index)
                            cache_query_indices.append(query_index)

                    query_order = view_scores.argsort(descending=True)[:args.max_det_per_view]
                    for query_index in query_order.cpu().tolist():
                        score = float(view_scores[query_index].item())
                        if score < args.conf:
                            continue
                        local = list(map(float, view_boxes[query_index].cpu().tolist()))
                        global_box = [local[0]+view[0], local[1]+view[1],
                                      local[2]+view[0], local[3]+view[1]]
                        ranked_ids = [int(value) for value in
                                      view_class_ids[query_index].cpu().tolist()]
                        ranked_probabilities = [float(value) for value in
                                                view_class_probabilities[query_index].cpu().tolist()]
                        if any(not 0 <= label < len(classes) for label in ranked_ids):
                            raise ValueError(f"Invalid class ids {ranked_ids}")
                        predictions.append({
                        # Keep the ordinary top-1 fields so existing merge/evaluation tools
                        # remain backward compatible. The compact ranked fields let the
                        # extreme-recall scanner expand Top-K without duplicating boxes on disk.
                        "class_id": ranked_ids[0],
                        "category_name": classes[ranked_ids[0]],
                        "bbox_xyxy": global_box, "score": score,
                        "objectness": score,
                        "selection_probability": float(
                            view_selection[query_index].item()
                        ),
                        "defectness_probability": float(
                            view_defectness[query_index].item()
                        ),
                        "quality_probability": float(
                            view_quality[query_index].item()
                        ),
                        "class_probability": ranked_probabilities[0],
                        "joint_score": score * ranked_probabilities[0],
                        "class_ids_ranked": ranked_ids,
                        "class_probabilities_ranked": ranked_probabilities,
                        "source": "tile" if args.mode == "tiles" else "whole",
                        "view_xyxy": list(view),
                        "touches_internal_border": args.mode == "tiles" and
                        touches_internal_border(local, view, (width, height)),
                        "dino_candidate_index": cache_indices.get(query_index),
                    })
            total_views += len(views); total_predictions += len(predictions)
            records.append({"image_id":image_path.name,"source_path":str(image_path),
                            "width":width,"height":height,"predictions":predictions})
            if image_index % 20 == 0 or image_index == len(paths):
                print(f"Processed {image_index}/{len(paths)} images", flush=True)
    finally:
        if capture_handle is not None:
            capture_handle.remove()
    metadata = {
        "classes": classes, "architecture": "hierarchical_dfine",
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_epoch": checkpoint.get("epoch"),
        "mode": args.mode, "image_count": len(paths), "view_count": total_views,
        "imgsz": args.imgsz, "tile_size": args.tile_size if args.mode == "tiles" else None,
        "overlap": args.overlap if args.mode == "tiles" else None,
        "confidence": args.conf, "prediction_count": total_predictions,
        "elapsed_seconds": round(time.time()-started, 3),
        "score_mode": (
            "selection_x_defectness_x_quality"
            if records and any(
                "selection_probability" in prediction
                and abs(
                    prediction["selection_probability"]
                    - prediction["defectness_probability"]
                ) > 1e-7
                for record in records
                for prediction in record["predictions"][:1]
            )
            else "defectness_x_quality"
            if records and any(
                "quality_probability" in prediction
                and prediction["quality_probability"] != 1.0
                for record in records
                for prediction in record["predictions"][:1]
            )
            else "defectness"
        ),
        "class_topk": args.class_topk,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"metadata":metadata,"images":records},
                                     ensure_ascii=False), encoding="utf-8")
    if args.candidate_cache is not None:
        feature_dim = (
            int(cache_features[0].numel()) if cache_features
            else int(model.decoder.dec_score_head[-1].class_weight.shape[1])
        )

        def stack(values, shape, dtype):
            return torch.stack(values) if values else torch.empty(shape, dtype=dtype)

        cache = {
            "version": 2,
            "architecture": "hierarchical_dfine_view_verifier_candidates",
            "split": args.split,
            "mode": args.mode,
            "classes": classes,
            "feature_dim": feature_dim,
            "checkpoint": str(args.checkpoint.resolve()),
            "config": str(args.config),
            "min_selection": args.cache_min_selection,
            "images": cache_images,
            "query_features": stack(
                cache_features, (0, feature_dim), torch.float16
            ),
            "boxes_xyxy": stack(cache_boxes, (0, 4), torch.float32),
            "selection_logits": stack(
                cache_selection_logits, (0,), torch.float16
            ),
            "class_logits": stack(
                cache_class_logits, (0, len(classes)), torch.float16
            ),
            "candidate_image_index": torch.tensor(
                cache_candidate_image_index, dtype=torch.int32
            ),
            "query_indices": torch.tensor(cache_query_indices, dtype=torch.int16),
            # Scoring does not consume labels. Keep schema compatibility with
            # VerifierCandidateDataset; final metrics use the VOC manifest.
            "max_iou": torch.zeros(len(cache_features), dtype=torch.float16),
            "nearest_gt_label": torch.full(
                (len(cache_features),), -1, dtype=torch.int16
            ),
            "elapsed_seconds": round(time.time() - started, 3),
        }
        args.candidate_cache.parent.mkdir(parents=True, exist_ok=True)
        if args.candidate_cache.exists():
            args.candidate_cache.unlink()
        torch.save(cache, args.candidate_cache)
        metadata["candidate_cache"] = str(args.candidate_cache.resolve())
        metadata["cached_candidates"] = len(cache_features)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
