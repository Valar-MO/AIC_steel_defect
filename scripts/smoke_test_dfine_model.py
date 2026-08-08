#!/usr/bin/env python3
"""Load D-FINE-L pretrained weights and run one GPU forward batch, without training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dfine-dir", type=Path, default=Path("/root/autodl-tmp/D-FINE"))
    parser.add_argument(
        "--config", default="configs/dfine/custom/objects365/dfine_hgnetv2_l_obj2aic_defect.yml"
    )
    parser.add_argument(
        "--weights", type=Path,
        default=Path("/root/autodl-tmp/weights/dfine/dfine_l_obj2coco_e25.pth"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--backward", action="store_true",
        help="Also run one training loss/backward pass without an optimizer step.",
    )
    parser.add_argument("--batch", type=int, default=4, help="Train/val batch for memory testing.")
    parser.add_argument(
        "--optimizer-step", action="store_true",
        help="Run AdamW/scaler/EMA update so optimizer states are included in peak memory.",
    )
    args = parser.parse_args()
    sys.path.insert(0, str(args.dfine_dir.resolve()))
    from src.core import YAMLConfig  # pylint: disable=import-outside-toplevel

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = args.dfine_dir / config_path
    cfg = YAMLConfig(str(config_path), tuning=str(args.weights))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg.yaml_cfg["train_dataloader"]["total_batch_size"] = args.batch
    cfg.yaml_cfg["val_dataloader"]["total_batch_size"] = min(args.batch, 8)
    model = cfg.model
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    pretrained = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    current = model.state_dict()
    matched = {
        name: value for name, value in pretrained.items()
        if name in current and value.shape == current[name].shape
    }
    mismatched = sorted(
        name for name, value in pretrained.items()
        if name in current and value.shape != current[name].shape
    )
    result = model.load_state_dict(matched, strict=False)
    device = torch.device(args.device)
    model = model.to(device).eval()
    images, targets = next(iter(cfg.val_dataloader))
    images = images.to(device, non_blocking=True)
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
    ):
        outputs = model(images)
    report = {
        "weights": str(args.weights.resolve()),
        "matched_parameter_tensors": len(matched),
        "mismatched_parameter_tensors": mismatched,
        "missing_after_load": len(result.missing_keys),
        "unexpected_after_load": len(result.unexpected_keys),
        "input_shape": list(images.shape),
        "output_keys": sorted(outputs),
        "pred_logits_shape": list(outputs["pred_logits"].shape),
        "pred_boxes_shape": list(outputs["pred_boxes"].shape),
        "peak_gpu_memory_gib": round(
            torch.cuda.max_memory_allocated(device) / 1024 ** 3, 3
        ) if device.type == "cuda" else None,
    }
    if outputs["pred_logits"].shape[-1] != 1:
        raise RuntimeError(f"Expected one-class logits: {report}")
    if args.backward or args.optimizer_step:
        del outputs, images
        model.train()
        criterion = cfg.criterion.to(device).train()
        train_images, train_targets = next(iter(cfg.train_dataloader))
        train_images = train_images.to(device, non_blocking=True)
        train_targets = [
            {key: value.to(device) if isinstance(value, torch.Tensor) else value
             for key, value in target.items()}
            for target in train_targets
        ]
        optimizer = cfg.optimizer if args.optimizer_step else None
        ema = cfg.ema.to(device) if args.optimizer_step else None
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        model.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            train_outputs = model(train_images, targets=train_targets)
        # Official training computes the criterion in fp32.
        with torch.autocast(device_type=device.type, enabled=False):
            losses = criterion(
                train_outputs, train_targets, epoch=0, step=0,
                global_step=0, epoch_step=len(cfg.train_dataloader),
            )
            total_loss = sum(losses.values())
        if args.optimizer_step:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model)
        else:
            total_loss.backward()
        report["backward"] = {
            "input_shape": list(train_images.shape),
            "loss": float(total_loss.detach().cpu()),
            "loss_keys": sorted(losses),
            "finite": bool(torch.isfinite(total_loss)),
            "optimizer_step": args.optimizer_step,
            "allocated_after_step_gib": round(
                torch.cuda.memory_allocated(device) / 1024 ** 3, 3
            ) if device.type == "cuda" else None,
            "peak_gpu_memory_gib": round(
                torch.cuda.max_memory_allocated(device) / 1024 ** 3, 3
            ) if device.type == "cuda" else None,
        }
        if not report["backward"]["finite"]:
            raise RuntimeError(f"Non-finite smoke loss: {report}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
