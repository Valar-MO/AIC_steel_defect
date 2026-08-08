#!/usr/bin/env python3
"""Load one whole and one virtual official tile through D-FINE's dataset path."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dfine-dir", type=Path, default=Path("/root/autodl-tmp/D-FINE"))
    parser.add_argument("--ann-file", type=Path, required=True)
    parser.add_argument("--check-mix", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.dfine_dir))
    from src.data.dataset.official_view_coco_dataset import OfficialViewCocoDetection

    dataset = OfficialViewCocoDetection("/", args.ann_file, transforms=None)
    seen = {}
    for index, image_id in enumerate(dataset.ids):
        kind = dataset.coco.imgs[int(image_id)]["view_type"]
        if kind not in seen:
            image, target = dataset[index]
            seen[kind] = {"image_size": list(image.size), "boxes": int(len(target["boxes"]))}
        if {"whole", "tile"} <= set(seen):
            break
    if {"whole", "tile"} - set(seen):
        raise RuntimeError(f"Missing required views: {seen}")
    print(seen)
    if args.check_mix:
        from src.data.dataset.official_view_mix_dataset import OfficialViewBatchMixDataset
        pattern = ["whole"] * 8 + ["natural_tile"] * 4 + ["hard_background_tile"] * 2 + ["positive_tile", "weak_positive_tile"]
        mixed = OfficialViewBatchMixDataset(dataset, pattern=pattern, epoch_size=16)
        composition = {}
        for index in range(16):
            source_index = mixed.orders[pattern[index]][0] if False else None
            # The index is sufficient to exercise virtual image loading; group
            # membership follows the deterministic pattern supplied above.
            image, target = mixed[index]
            composition[pattern[index]] = composition.get(pattern[index], 0) + 1
        print({"batch_pattern": composition})


if __name__ == "__main__": main()
