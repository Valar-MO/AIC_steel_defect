#!/usr/bin/env python3
"""Train an Ultralytics RT-DETR model with explicit, reproducible settings."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml


DEFAULT_WEIGHTS = Path("/root/autodl-tmp/weights/rtdetr-l.pt")
DEFAULT_DATA = Path("/root/autodl-tmp/datasets/AIC_steel_defect_ultralytics/steel.yaml")
DEFAULT_PROJECT = Path("/root/autodl-tmp/runs/rtdetr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", default="rtdetr_l_1024_sanity")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0", help="CUDA device, e.g. 0; use cpu only for debugging.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--cache", action="store_true", help="Cache images; disabled by default.")
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow an existing run directory.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print settings without training.")
    return parser.parse_args()


def indexed_names(value) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, dict):
        normalized = {int(k): str(v) for k, v in value.items()}
        expected = list(range(len(normalized)))
        if sorted(normalized) != expected:
            raise ValueError(f"Class IDs must be contiguous from 0, got {sorted(normalized)}")
        return [normalized[i] for i in expected]
    raise ValueError("Dataset YAML 'names' must be a list or mapping")


def resolve_split(dataset_yaml: Path, config: dict, key: str) -> Path:
    root_value = config.get("path", dataset_yaml.parent)
    root = Path(root_value)
    if not root.is_absolute():
        root = (dataset_yaml.parent / root).resolve()
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Dataset YAML lacks a valid '{key}' path")
    path = Path(value)
    return path if path.is_absolute() else root / path


def count_images(path: Path) -> int:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if path.is_dir():
        return sum(p.is_file() and p.suffix.lower() in suffixes for p in path.iterdir())
    if path.is_file() and path.suffix.lower() == ".txt":
        return sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())
    raise FileNotFoundError(f"Dataset split not found: {path}")


def preflight(args: argparse.Namespace) -> dict:
    weights = args.weights.expanduser().resolve()
    data = args.data.expanduser().resolve()
    project = args.project.expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not data.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data}")
    if args.imgsz <= 0 or args.epochs <= 0 or args.batch <= 0 or args.workers < 0:
        raise ValueError("imgsz, epochs and batch must be positive; workers cannot be negative")

    with data.open(encoding="utf-8") as f:
        dataset_config = yaml.safe_load(f)
    if not isinstance(dataset_config, dict):
        raise ValueError(f"Invalid dataset YAML: {data}")
    names = indexed_names(dataset_config.get("names"))
    if len(names) != 9:
        raise ValueError(f"Expected 9 classes, found {len(names)}: {names}")
    train_path = resolve_split(data, dataset_config, "train")
    val_path = resolve_split(data, dataset_config, "val")
    train_images, val_images = count_images(train_path), count_images(val_path)
    if (train_images, val_images) != (2560, 640):
        raise ValueError(
            f"Expected fixed split 2560/640, found {train_images}/{val_images}. "
            "Refusing to train on an unexpected split."
        )

    run_dir = project / args.name
    if run_dir.exists() and not args.exist_ok:
        raise FileExistsError(
            f"Run directory already exists: {run_dir}. Choose another --name or pass --exist-ok."
        )
    project.mkdir(parents=True, exist_ok=True)
    return {
        "weights": str(weights),
        "data": str(data),
        "project": str(project),
        "run_dir": str(run_dir),
        "name": args.name,
        "classes": names,
        "train_images": train_images,
        "val_images": val_images,
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "seed": args.seed,
        "patience": args.patience,
        "cache": args.cache,
        "amp": not args.no_amp,
    }


def check_runtime(device: str) -> dict:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed in the active environment") from exc
    info = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
    }
    if device != "cpu":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA training requested, but torch.cuda.is_available() is False")
        try:
            index = int(str(device).split(",")[0])
        except ValueError as exc:
            raise ValueError(f"Unsupported --device value for preflight: {device}") from exc
        info["gpu"] = torch.cuda.get_device_name(index)
        info["gpu_memory_gib"] = round(torch.cuda.get_device_properties(index).total_memory / 2**30, 2)
    return info


def main() -> None:
    args = parse_args()
    config = preflight(args)
    print("Training configuration:")
    print(json.dumps(config, ensure_ascii=False, indent=2))
    if args.dry_run:
        print("Dry run passed; training was not started.")
        return

    # Must be set before importing torch so the CUDA allocator sees it.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    runtime = check_runtime(args.device)
    print("Runtime:")
    print(json.dumps(runtime, ensure_ascii=False, indent=2))
    try:
        from ultralytics import RTDETR
        import ultralytics
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Install it in the aic-steel environment before training."
        ) from exc
    print(f"Ultralytics: {ultralytics.__version__}")

    model = RTDETR(config["weights"])
    model.train(
        data=config["data"],
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        patience=args.patience,
        project=config["project"],
        name=args.name,
        seed=args.seed,
        deterministic=True,
        amp=not args.no_amp,
        cache=args.cache,
        plots=True,
        exist_ok=args.exist_ok,
        verbose=True,
    )


if __name__ == "__main__":
    main()
