#!/usr/bin/env python3
"""Smoke-test installed D-FINE config and one validation batch without training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dfine-dir", type=Path, default=Path("/root/autodl-tmp/D-FINE"))
    parser.add_argument(
        "--config", default="configs/dfine/custom/objects365/dfine_hgnetv2_l_obj2aic_defect.yml"
    )
    args = parser.parse_args()
    sys.path.insert(0, str(args.dfine_dir.resolve()))
    from src.core import YAMLConfig  # pylint: disable=import-outside-toplevel

    config = Path(args.config)
    if not config.is_absolute():
        config = args.dfine_dir / config
    cfg = YAMLConfig(str(config))
    images, targets = next(iter(cfg.val_dataloader))
    result = {
        "num_classes": cfg.yaml_cfg["num_classes"],
        "remap_mscoco_category": cfg.yaml_cfg["remap_mscoco_category"],
        "train_images": len(cfg.train_dataloader.dataset),
        "val_images": len(cfg.val_dataloader.dataset),
        "val_batch_shape": list(images.shape),
        "val_targets": len(targets),
        "categories": cfg.val_dataloader.dataset.categories,
    }
    if result["num_classes"] != 1 or result["remap_mscoco_category"] is not False:
        raise RuntimeError(f"Invalid class-agnostic settings: {result}")
    if result["val_batch_shape"][2:] != [1024, 1024]:
        raise RuntimeError(f"Unexpected validation resolution: {result}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
