#!/usr/bin/env python3
"""Measure resolved D-FINE DataLoader startup and steady-state batch latency."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dfine-dir", type=Path, default=Path("/root/autodl-tmp/D-FINE"))
    parser.add_argument("--config", default=
        "configs/dfine/custom/objects365/dfine_hgnetv2_l_hierarchical_peft_aic.yml")
    parser.add_argument("--batches", type=int, default=12)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--prefetch-factor", type=int)
    parser.add_argument("--pin-memory", choices=("true", "false"))
    args = parser.parse_args()
    sys.path.insert(0, str(args.dfine_dir.resolve()))
    from src.core import YAMLConfig

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = args.dfine_dir / config_path
    cfg = YAMLConfig(str(config_path))
    if args.workers is not None:
        cfg.yaml_cfg["train_dataloader"]["num_workers"] = args.workers
    if args.prefetch_factor is not None:
        cfg.yaml_cfg["train_dataloader"]["prefetch_factor"] = args.prefetch_factor
    if args.pin_memory is not None:
        cfg.yaml_cfg["train_dataloader"]["pin_memory"] = args.pin_memory == "true"
    cfg.train_dataloader.set_epoch(0)
    loader = cfg.train_dataloader
    iterator = iter(loader)
    times, shapes = [], []
    for _ in range(args.batches):
        started = time.perf_counter()
        images, targets = next(iterator)
        times.append(time.perf_counter() - started)
        shapes.append(list(images.shape))
        del images, targets
    steady = times[1:] if len(times) > 1 else times
    report = {
        "batch_size": loader.batch_size,
        "num_workers": loader.num_workers,
        "prefetch_factor": loader.prefetch_factor,
        "pin_memory": loader.pin_memory,
        "batches": args.batches,
        "first_batch_seconds": times[0],
        "steady_batch_seconds_mean": sum(steady) / len(steady),
        "steady_batch_seconds_max": max(steady),
        "batch_shapes": shapes,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
