#!/usr/bin/env python3
"""Attach GT and final whole+tile/NMS roles to frozen ROI candidate caches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torchvision.ops import box_convert

from evaluate_predictions import box_iou, load_classes, load_ground_truth


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--whole-cache", type=Path, required=True)
    p.add_argument("--tile-cache", type=Path, required=True)
    p.add_argument("--whole-raw", type=Path, required=True)
    p.add_argument("--tile-raw", type=Path, required=True)
    p.add_argument("--fused", type=Path, required=True,
                   help="Base whole+tile output from the official .30/.80 merge.")
    p.add_argument("--manifest", type=Path, default=Path("splits/v1/split_manifest.csv"))
    p.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    p.add_argument("--split", choices=("train", "val"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_boxes(raw: dict) -> dict[int, tuple[str, int, list[float], str]]:
    rows = {}
    for image in raw["images"]:
        image_id = Path(image["image_id"]).name
        for prediction in image.get("predictions", []):
            index = prediction.get("roi_cache_index")
            if index is not None:
                rows[int(index)] = (
                    image_id,
                    int(prediction["class_id"]),
                    prediction["bbox_xyxy"],
                    str(prediction.get("candidate_id", "")),
                )
    return rows


def final_tp_ids(fused: dict, gt, classes: list[str]) -> set[str]:
    """Replicate class-wise score-ranked one-to-one matching at IoU .50."""
    by_class: dict[int, list[tuple[float, str, list[float], str | None]]] = {i: [] for i in range(len(classes))}
    for image in fused["images"]:
        image_id = Path(image["image_id"]).name
        for pred in image.get("predictions", []):
            by_class[int(pred["class_id"])].append((float(pred["score"]), image_id, pred["bbox_xyxy"], pred.get("candidate_id")))
    result = set()
    for class_id, rows in by_class.items():
        matched: dict[str, set[int]] = {}
        for _, image_id, box, candidate_id in sorted(rows, key=lambda row: row[0], reverse=True):
            best_index, best_iou = -1, -1.0
            for index, gt_box in enumerate(gt[class_id].get(image_id, [])):
                if index in matched.setdefault(image_id, set()): continue
                iou = box_iou(box, gt_box)
                if iou > best_iou: best_iou, best_index = iou, index
            if best_iou >= .5 and candidate_id:
                matched[image_id].add(best_index); result.add(candidate_id)
    return result


def main() -> None:
    opt = parse_args()
    if opt.output.exists() and not opt.overwrite: raise FileExistsError(opt.output)
    whole = torch.load(opt.whole_cache, map_location="cpu", weights_only=False)
    tiles = torch.load(opt.tile_cache, map_location="cpu", weights_only=False)
    if whole["classes"] != tiles["classes"] or whole["split"] != opt.split or tiles["split"] != opt.split:
        raise ValueError("Cache classes/split mismatch")
    classes = load_classes(opt.classes); class_to_id = {name: i for i, name in enumerate(classes)}
    gt, _, _, = load_ground_truth(opt.manifest, opt.split, class_to_id)
    raw_whole, raw_tiles, fused = read_json(opt.whole_raw), read_json(opt.tile_raw), read_json(opt.fused)
    base_tp = final_tp_ids(fused, gt, classes)
    final_ids = {str(p["candidate_id"]) for record in fused["images"] for p in record.get("predictions", []) if p.get("candidate_id")}
    sources = ((whole, candidate_boxes(raw_whole), "whole"), (tiles, candidate_boxes(raw_tiles), "tile"))
    output: dict[str, list[torch.Tensor]] = {}
    metadata = {"version": 1, "architecture": "hierarchical_dfine_roi_labeled_cache", "split": opt.split,
                "classes": classes, "whole_cache": str(opt.whole_cache.resolve()), "tile_cache": str(opt.tile_cache.resolve()),
                "fused": str(opt.fused.resolve())}
    gt_group_ids: dict[tuple[str, int, tuple[float, ...]], int] = {}
    for cache, boxes_by_index, mode in sources:
        rows = []
        for index in range(len(cache["query"])):
            if index not in boxes_by_index: raise RuntimeError(f"Missing raw candidate {mode}:{index}")
            image_id, predicted_class, box, candidate_id = boxes_by_index[index]
            candidates = [(label, gt_box) for label, image_boxes in gt.items() for gt_box in image_boxes.get(image_id, [])]
            best_iou, best_label, best_box, best_index = 0.0, -1, [0., 0., 0., 0.], -1
            for candidate_index, (label, gt_box) in enumerate(candidates):
                value = box_iou(box, gt_box)
                if value > best_iou:
                    best_iou, best_label, best_box, best_index = value, label, gt_box, candidate_index
            if not candidate_id:
                raise RuntimeError(f"Missing candidate id for {mode}:{index}")
            view = cache["view_xyxy"][index].float()
            view_w = (view[2] - view[0]).clamp_min(1); view_h = (view[3] - view[1]).clamp_min(1)
            target = torch.tensor(best_box, dtype=torch.float32)
            target[[0, 2]] = (target[[0, 2]] - view[0]) / view_w
            target[[1, 3]] = (target[[1, 3]] - view[1]) / view_h
            target = box_convert(target.unsqueeze(0), in_fmt="xyxy", out_fmt="cxcywh").squeeze(0).clamp(0, 1)
            group_key = (image_id, best_label, tuple(float(value) for value in best_box))
            if best_index < 0:
                group_id = -1
            else:
                group_id = gt_group_ids.setdefault(group_key, len(gt_group_ids))
            rows.append((best_iou, best_label, best_box, target.tolist(), group_id, candidate_id in base_tp,
                         candidate_id in final_ids, predicted_class == best_label))
        output.setdefault("max_iou", []).append(torch.tensor([r[0] for r in rows], dtype=torch.float16))
        output.setdefault("nearest_gt_label", []).append(torch.tensor([r[1] for r in rows], dtype=torch.int16))
        output.setdefault("nearest_gt_box", []).append(torch.tensor([r[2] for r in rows], dtype=torch.float32))
        output.setdefault("target_box_cxcywh", []).append(torch.tensor([r[3] for r in rows], dtype=torch.float16))
        output.setdefault("nearest_gt_group", []).append(torch.tensor([r[4] for r in rows], dtype=torch.int32))
        output.setdefault("is_base_tp", []).append(torch.tensor([r[5] for r in rows], dtype=torch.bool))
        output.setdefault("is_final_survivor", []).append(torch.tensor([r[6] for r in rows], dtype=torch.bool))
        output.setdefault("is_class_correct", []).append(torch.tensor([r[7] for r in rows], dtype=torch.bool))
        for key, value in cache.items():
            if isinstance(value, torch.Tensor) and key not in output:
                output[key] = []
            if key in output and isinstance(value, torch.Tensor) and key not in {"max_iou", "nearest_gt_label"}:
                output[key].append(value)
    merged = {key: torch.cat(value, 0) for key, value in output.items()}
    # A duplicate is foreground and class-correct but was not the final winning TP.
    merged["is_duplicate"] = ((merged["max_iou"] >= .5) & merged["is_class_correct"] & ~merged["is_base_tp"])
    merged["is_final_hard_bg"] = ((merged["max_iou"] < .1) & merged["is_final_survivor"])
    merged["metadata"] = metadata
    opt.output.parent.mkdir(parents=True, exist_ok=True); torch.save(merged, opt.output)
    print(json.dumps({"output": str(opt.output.resolve()), "candidates": len(merged["query"]),
                      "base_tp": int(merged["is_base_tp"].sum()), "duplicates": int(merged["is_duplicate"].sum()),
                      "final_hard_bg": int(merged["is_final_hard_bg"].sum()),
                      "background": int((merged["max_iou"] < .1).sum())}, indent=2))


if __name__ == "__main__": main()
