#!/usr/bin/env python3
"""Validate E1 grouped sampling and load real transformed D-FINE batches."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dfine-dir", type=Path, default=Path("/root/autodl-tmp/D-FINE"))
    parser.add_argument(
        "--config",
        default="configs/dfine/custom/objects365/dfine_hgnetv2_l_hierarchical_gtaware_e1.yml",
    )
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--expect-pattern", nargs="*", default=[], metavar="GROUP=COUNT",
        help="Optional exact group counts expected in one sampling pattern.",
    )
    args = parser.parse_args()
    sys.path.insert(0, str(args.dfine_dir.resolve()))
    from src.core import YAMLConfig

    config = Path(args.config)
    if not config.is_absolute():
        config = args.dfine_dir / config
    cfg = YAMLConfig(str(config))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg.yaml_cfg["train_dataloader"]["num_workers"] = args.workers
    loader = cfg.train_dataloader
    dataset = loader.dataset
    if not hasattr(dataset, "sampling_report"):
        raise TypeError(type(dataset))
    loader.set_epoch(0)

    source_groups = []
    for index in range(len(dataset.pattern)):
        group, _source_index = dataset.source_index(index)
        source_groups.append(group)
    pattern_counts = Counter(source_groups)
    expected = {}
    for item in args.expect_pattern:
        name, separator, raw_count = item.partition("=")
        if not separator or not name or int(raw_count) < 0:
            raise ValueError(f"Invalid --expect-pattern item: {item}")
        expected[name] = int(raw_count)
    if expected and dict(pattern_counts) != expected:
        raise RuntimeError({"actual": dict(pattern_counts), "expected": expected})

    loaded_images = loaded_targets = loaded_boxes = 0
    shapes = set()
    for step, (images, targets) in enumerate(loader, 1):
        loaded_images += len(images)
        loaded_targets += len(targets)
        loaded_boxes += sum(len(target["labels"]) for target in targets)
        shapes.add(tuple(images.shape[1:]))
        if not images.isfinite().all():
            raise RuntimeError("non-finite image tensor")
        for target in targets:
            boxes = target["boxes"]
            if len(boxes) and ((boxes < 0).any() or (boxes > 1).any()):
                raise RuntimeError("normalized boxes outside [0,1]")
        if step >= args.batches:
            break
    report = {
        "sampling": dataset.sampling_report(),
        "first_pattern": source_groups,
        "loaded_batches": args.batches,
        "workers": args.workers,
        "loaded_images": loaded_images,
        "loaded_targets": loaded_targets,
        "loaded_boxes": loaded_boxes,
        "tensor_shapes": [list(row) for row in sorted(shapes)],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
