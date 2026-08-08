#!/usr/bin/env python3
"""Attach final whole+tile counterfactual utility roles to frozen ROI caches.

The source cache row is the stable join key.  This avoids the historical
tile-view candidate-ID collision without rerunning frozen D-FINE extraction.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch


ROLE_TO_ID = {
    "essential_tp": 0,
    "replaceable_tp": 1,
    "duplicate_or_replaceable": 2,
    "wrong_class_high_iou": 3,
    "near_or_localization": 4,
    "final_harmful_background": 5,
    "potential_harmful_background": 6,
    "harmless_background": 7,
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--whole-cache", type=Path, required=True)
    p.add_argument("--tile-cache", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_lookup(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    lookup = {}
    for row in payload["candidates"]:
        key = (row["source"], int(row["roi_cache_index"]))
        if key in lookup:
            raise RuntimeError(f"Duplicate audit key: {key}")
        lookup[key] = row
    return payload["metadata"], lookup


def main():
    opt = parse_args()
    if opt.output.exists() and not opt.overwrite:
        raise FileExistsError(opt.output)
    audit_meta, lookup = load_lookup(opt.audit)
    whole = torch.load(opt.whole_cache, map_location="cpu", weights_only=False)
    tiles = torch.load(opt.tile_cache, map_location="cpu", weights_only=False)
    if whole["classes"] != tiles["classes"] or whole["split"] != tiles["split"]:
        raise ValueError("Cache class/split mismatch")

    merged = {}
    all_roles = []
    labels, gt_indices, gt_groups = [], [], []
    group_lookup = {}
    for source, cache in (("whole", whole), ("tile", tiles)):
        count = len(cache["query"])
        rows = []
        for index in range(count):
            try:
                rows.append(lookup[(source, index)])
            except KeyError as exc:
                raise RuntimeError(f"Audit is missing {source}:{index}") from exc
        roles = [row["utility_role"] for row in rows]
        all_roles.extend(roles)
        labels.append(torch.tensor([ROLE_TO_ID[role] for role in roles], dtype=torch.uint8))
        gt_indices.append(torch.tensor([row["nearest_gt_index"] for row in rows], dtype=torch.int16))
        group_values = []
        for row in rows:
            if row["nearest_gt_index"] < 0:
                group_values.append(-1)
                continue
            key = (row["image_id"], int(row["nearest_gt_label"]), int(row["nearest_gt_index"]))
            group_values.append(group_lookup.setdefault(key, len(group_lookup)))
        gt_groups.append(torch.tensor(group_values, dtype=torch.int32))
        for key, value in cache.items():
            if isinstance(value, torch.Tensor):
                merged.setdefault(key, []).append(value)

    output = {key: torch.cat(value, dim=0) for key, value in merged.items()}
    role_ids = torch.cat(labels)
    output["utility_role_id"] = role_ids
    output["utility_nearest_gt_index"] = torch.cat(gt_indices)
    output["utility_nearest_gt_group"] = torch.cat(gt_groups)
    output["is_essential_tp"] = role_ids == ROLE_TO_ID["essential_tp"]
    output["is_replaceable_tp"] = role_ids == ROLE_TO_ID["replaceable_tp"]
    output["is_duplicate_or_replaceable"] = role_ids == ROLE_TO_ID["duplicate_or_replaceable"]
    output["is_final_harmful_background"] = role_ids == ROLE_TO_ID["final_harmful_background"]
    output["is_potential_harmful_background"] = role_ids == ROLE_TO_ID["potential_harmful_background"]
    output["metadata"] = {
        "version": 1,
        "architecture": "hierarchical_dfine_counterfactual_utility_cache",
        "split": whole["split"], "classes": whole["classes"],
        "role_to_id": ROLE_TO_ID, "audit": str(opt.audit.resolve()),
        "whole_cache": str(opt.whole_cache.resolve()), "tile_cache": str(opt.tile_cache.resolve()),
        "audit_protocol": audit_meta,
    }
    opt.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, opt.output)
    print(json.dumps({"output": str(opt.output.resolve()), "candidates": len(role_ids),
                      "role_counts": dict(Counter(all_roles))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
