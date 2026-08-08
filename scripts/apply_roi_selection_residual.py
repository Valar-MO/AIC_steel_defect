#!/usr/bin/env python3
"""Apply an R2b Selection-residual checkpoint to immutable ROI candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from roi_refinement_common import (ROISelectionResidualHead, ROIShapeSelectionResidualHead,
                                   geometry_features)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    opt = p.parse_args()
    if opt.output.exists() and not opt.overwrite: raise FileExistsError(opt.output)
    cache = torch.load(opt.cache, map_location="cpu", weights_only=False)
    checkpoint = torch.load(opt.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") not in {"roi_selection_residual", "roi_selection_residual_shape"}:
        raise ValueError("Not an ROI Selection-residual checkpoint")
    if cache["classes"] != checkpoint["classes"]: raise ValueError("Class mismatch")
    head_class = (ROIShapeSelectionResidualHead if checkpoint["architecture"] == "roi_selection_residual_shape"
                  else ROISelectionResidualHead)
    device = torch.device(opt.device); head = head_class(**checkpoint["head"]).to(device).eval()
    head.load_state_dict(checkpoint["model"], strict=True)
    scores, deltas = [], []
    with torch.inference_mode():
        for start in range(0, len(cache["query"]), opt.batch):
            end = min(start + opt.batch, len(cache["query"]))
            data = {key: value[start:end].to(device) for key, value in cache.items() if isinstance(value, torch.Tensor)}
            geometry = geometry_features(data["boxes_cxcywh"].float(), data["source_id"],
                                         data["touches_internal_border"], data["context_clipped"],
                                         data["border_distance"].float())
            delta = head(data["p2_roi"].float(), data["p3_roi"].float(), data["query"].float(), geometry)
            deltas.append(delta.cpu()); scores.append((data["selection_logit"].float() + delta).sigmoid().cpu())
    scores, deltas = torch.cat(scores), torch.cat(deltas)
    payload = json.loads(opt.raw.read_text(encoding="utf-8"))
    for record in payload["images"]:
        for prediction in record.get("predictions", []):
            index = int(prediction["roi_cache_index"])
            prediction["score"] = float(scores[index]); prediction["selection_probability"] = float(scores[index])
            prediction["roi_selection_residual"] = float(deltas[index])
    payload.setdefault("metadata", {})["roi_selection_residual"] = {
        "checkpoint": str(opt.checkpoint.resolve()), "cache": str(opt.cache.resolve()),
        "score_mode": "base_selection_plus_roi_residual",
    }
    opt.output.parent.mkdir(parents=True, exist_ok=True); opt.output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__": main()
