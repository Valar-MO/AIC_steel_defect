#!/usr/bin/env python3
"""Install the tracked AIC configs into an existing official D-FINE checkout."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dfine-dir", type=Path, default=Path("/root/autodl-tmp/D-FINE"))
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


def copy_config(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    print(f"installed {destination}")


def main() -> None:
    args = parse_args()
    if not (args.dfine_dir / "train.py").is_file():
        raise FileNotFoundError(f"Not an official D-FINE checkout: {args.dfine_dir}")
    config_dir = args.project_dir / "configs" / "dfine"
    copy_config(
        config_dir / "aic_defect_detection.yml",
        args.dfine_dir / "configs" / "dataset" / "aic_defect_detection.yml",
    )
    copy_config(
        config_dir / "dfine_hgnetv2_l_obj2aic_defect.yml",
        args.dfine_dir / "configs" / "dfine" / "custom" / "objects365"
        / "dfine_hgnetv2_l_obj2aic_defect.yml",
    )
    installed = (
        args.dfine_dir / "configs" / "dfine" / "custom" / "objects365"
        / "dfine_hgnetv2_l_obj2aic_defect.yml"
    )
    text = installed.read_text(encoding="utf-8")
    required = (
        "num_classes: 1", "size: [1024, 1024]", "total_batch_size: 4",
        "eval_spatial_size: [1024, 1024]",
    )
    # num_classes lives in the included dataset config, so validate both files together.
    dataset_text = (
        args.dfine_dir / "configs" / "dataset" / "aic_defect_detection.yml"
    ).read_text(encoding="utf-8")
    combined = text + "\n" + dataset_text
    missing = [value for value in required if value not in combined]
    if missing:
        raise RuntimeError(f"Installed config validation failed: {missing}")
    print("config: configs/dfine/custom/objects365/dfine_hgnetv2_l_obj2aic_defect.yml")


if __name__ == "__main__":
    main()
