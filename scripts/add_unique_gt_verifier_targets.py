#!/usr/bin/env python3
"""Add one usable winner per GT to a fixed D-FINE candidate cache."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torchvision.ops import box_convert, box_iou


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--positive-iou", type=float, default=0.5)
    parser.add_argument("--background-iou", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    coco = json.loads(Path(cache["annotation_path"]).read_text(encoding="utf-8"))
    annotations = defaultdict(list)
    for annotation in coco.get("annotations", []):
        if not int(annotation.get("iscrowd", 0)):
            annotations[int(annotation["image_id"])].append(annotation)

    count = len(cache["selection_logits"])
    nearest_gt_id = torch.full((count,), -1, dtype=torch.int64)
    class_correct = torch.zeros(count, dtype=torch.bool)
    usable_winner = torch.zeros(count, dtype=torch.bool)
    predicted_label = cache["class_logits"].float().argmax(dim=1)
    candidate_image = cache["candidate_image_index"].long()
    max_difference = 0.0
    represented_gt, class_correct_gt = set(), set()

    for image_index, image in enumerate(cache["images"]):
        candidate_indices = torch.where(candidate_image == image_index)[0]
        image_annotations = annotations.get(int(image["coco_image_id"]), [])
        if not len(candidate_indices) or not image_annotations:
            continue
        target_xywh = torch.tensor(
            [annotation["bbox"] for annotation in image_annotations],
            dtype=torch.float32,
        )
        target_xyxy = box_convert(target_xywh, in_fmt="xywh", out_fmt="xyxy")
        overlaps = box_iou(cache["boxes_xyxy"][candidate_indices].float(), target_xyxy)
        best_iou, best_target = overlaps.max(dim=1)
        cached_iou = cache["max_iou"][candidate_indices].float()
        max_difference = max(
            max_difference, float((best_iou - cached_iou).abs().max())
        )
        target_labels = torch.tensor(
            [int(annotation["category_id"]) for annotation in image_annotations],
            dtype=torch.long,
        )
        target_ids = torch.tensor(
            [int(annotation["id"]) for annotation in image_annotations],
            dtype=torch.long,
        )
        matched_ids = target_ids[best_target]
        matched_labels = target_labels[best_target]
        nearest_gt_id[candidate_indices] = matched_ids
        correct = predicted_label[candidate_indices] == matched_labels
        class_correct[candidate_indices] = correct
        eligible = (best_iou >= args.positive_iou) & correct

        for target_position in torch.where(eligible)[0].tolist():
            represented_gt.add((image_index, int(matched_ids[target_position])))
        for target_id in torch.unique(matched_ids[eligible]).tolist():
            positions = torch.where(eligible & (matched_ids == target_id))[0]
            # Highest IoU is the usable winner; baseline score breaks exact ties.
            ranking = (
                best_iou[positions] * 100.0
                + cache["selection_logits"][candidate_indices[positions]].float().sigmoid()
            )
            winner_position = positions[int(ranking.argmax())]
            winner_index = candidate_indices[winner_position]
            usable_winner[winner_index] = True
            class_correct_gt.add((image_index, int(target_id)))

    if max_difference > 0.005:
        raise RuntimeError(f"Recomputed IoU differs from cache by {max_difference:.6f}")
    pure_background = cache["max_iou"].float() < args.background_iou
    duplicate_or_other = ~(usable_winner | pure_background)
    cache = dict(cache)
    cache.update({
        "version": int(cache.get("version", 1)) + 1,
        "target_policy": (
            "one highest-IoU class-correct winner per GT; "
            "pure background negative; all other candidates ignored"
        ),
        "nearest_gt_id": nearest_gt_id,
        "predicted_label": predicted_label.to(torch.int16),
        "class_correct": class_correct,
        "verifier_positive": usable_winner,
        "verifier_negative": pure_background,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()
    torch.save(cache, args.output)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "candidates": count,
        "unique_usable_positive_winners": int(usable_winner.sum()),
        "pure_background_negatives": int(pure_background.sum()),
        "ignored": int(duplicate_or_other.sum()),
        "maximum_represented_class_correct_gt": len(class_correct_gt),
        "max_iou_recompute_difference": max_difference,
    }, indent=2))


if __name__ == "__main__":
    main()
