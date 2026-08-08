#!/usr/bin/env python3
"""Apply an R2 ROI head to its immutable raw candidate cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from roi_refinement_common import ROIRefinementHead, apply_box_delta, cxcywh_to_xyxy, geometry_features


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--veto-threshold", type=float, default=.98)
    p.add_argument("--disable-class-residual", action="store_true")
    p.add_argument("--disable-box-refine", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    opt = p.parse_args()
    if opt.output.exists() and not opt.overwrite: raise FileExistsError(opt.output)
    cache = torch.load(opt.cache, map_location="cpu", weights_only=False)
    checkpoint = torch.load(opt.checkpoint, map_location="cpu", weights_only=False)
    if cache["classes"] != checkpoint["classes"]: raise ValueError("Cache/checkpoint class mismatch")
    device = torch.device(opt.device); head = ROIRefinementHead(**checkpoint["head"]).to(device).eval()
    head.load_state_dict(checkpoint["model"], strict=True)
    count = len(cache["query"]); veto, scores, class_ids, refined_boxes = [], [], [], []
    with torch.inference_mode():
        for start in range(0, count, opt.batch):
            end = min(start + opt.batch, count)
            rows = {key: value[start:end].to(device) for key, value in cache.items() if isinstance(value, torch.Tensor)}
            geom = geometry_features(rows["boxes_cxcywh"].float(), rows["source_id"], rows["touches_internal_border"],
                                     rows["context_clipped"], rows["border_distance"].float())
            output = head(rows["p2_roi"].float(), rows["p3_roi"].float(), rows["query"].float(), geom)
            veto.append(output["veto_logit"].sigmoid().cpu())
            scores.append((rows["selection_logit"].float() + output["score_delta"]).sigmoid().cpu())
            logits = rows["class_logits"].float() + (0 if opt.disable_class_residual else output["class_delta"])
            class_ids.append(logits.argmax(-1).cpu())
            boxes = rows["boxes_cxcywh"].float() if opt.disable_box_refine else apply_box_delta(rows["boxes_cxcywh"].float(), output["box_delta"])
            refined_boxes.append(boxes.cpu())
    veto, scores, class_ids, refined_boxes = map(torch.cat, (veto, scores, class_ids, refined_boxes))
    payload = json.loads(opt.raw.read_text(encoding="utf-8")); kept = dropped = 0
    for record in payload["images"]:
        revised = []
        for pred in record.get("predictions", []):
            index = int(pred["roi_cache_index"])
            if float(veto[index]) >= opt.veto_threshold:
                dropped += 1; continue
            view = pred["view_xyxy"]; view_w, view_h = view[2] - view[0], view[3] - view[1]
            box = cxcywh_to_xyxy(refined_boxes[index].unsqueeze(0)).squeeze(0).clamp(0, 1)
            box *= box.new_tensor([view_w, view_h, view_w, view_h]); box += box.new_tensor([view[0], view[1], view[0], view[1]])
            pred["bbox_xyxy"] = [float(value) for value in box.tolist()]
            pred["score"] = float(scores[index]); pred["selection_probability"] = float(scores[index])
            pred["class_id"] = int(class_ids[index]); pred["category_name"] = cache["classes"][int(class_ids[index])]
            pred["roi_refinement"] = {"veto_probability": float(veto[index]),
                                        "score_delta": float(torch.logit(scores[index]) - cache["selection_logit"][index]),
                                        "class_residual": not opt.disable_class_residual,
                                        "box_refined": not opt.disable_box_refine}
            revised.append(pred); kept += 1
        record["predictions"] = revised
    payload.setdefault("metadata", {})["roi_refinement"] = {"checkpoint": str(opt.checkpoint.resolve()),
        "cache": str(opt.cache.resolve()), "veto_threshold": opt.veto_threshold, "kept": kept, "veto_dropped": dropped,
        "score_mode": "selection_plus_bounded_roi_residual"}
    opt.output.parent.mkdir(parents=True, exist_ok=True); opt.output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["metadata"]["roi_refinement"], indent=2))


if __name__ == "__main__": main()
