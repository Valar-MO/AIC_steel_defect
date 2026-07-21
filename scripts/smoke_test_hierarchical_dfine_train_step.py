#!/usr/bin/env python3
"""Run a real 1024px Hierarchical D-FINE forward/backward without updating weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dfine-dir", type=Path, default=Path("/root/autodl-tmp/D-FINE"))
    parser.add_argument("--config", default=
        "configs/dfine/custom/objects365/dfine_hgnetv2_l_hierarchical_aic.yml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap-output", type=Path,
                        help="Optionally save the migrated epoch-zero model for inference smoke tests.")
    args = parser.parse_args()
    sys.path.insert(0, str(args.dfine_dir.resolve()))
    from src.core import YAMLConfig

    config_path = Path(args.config)
    if not config_path.is_absolute(): config_path = args.dfine_dir/config_path
    cfg = YAMLConfig(str(config_path), tuning=str(args.checkpoint))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg.yaml_cfg["train_dataloader"]["total_batch_size"] = args.batch
    cfg.yaml_cfg["train_dataloader"]["num_workers"] = 0
    model = cfg.model
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    pretrained = payload["ema"]["module"] if "ema" in payload else payload["model"]
    current = model.state_dict()
    matched = {name:value for name,value in pretrained.items()
               if name in current and value.shape == current[name].shape}
    load = model.load_state_dict(matched, strict=False)
    if args.bootstrap_output:
        args.bootstrap_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "epoch": 0,
                    "source_checkpoint": str(args.checkpoint)}, args.bootstrap_output)
    transferred_objectness = [name for name in matched if
        name.startswith("decoder.dec_score_head") and
        (name.endswith(".weight") or name.endswith(".bias"))]
    device = torch.device(args.device)
    model = model.to(device).train(); criterion = cfg.criterion.to(device).train()
    for images, targets in cfg.train_dataloader:
        if sum(len(target["labels"]) for target in targets) > 0:
            break
    else:
        raise RuntimeError("No positive training batch found")
    images = images.to(device, non_blocking=True)
    targets = [{key:value.to(device) if isinstance(value, torch.Tensor) else value
                for key,value in target.items()} for target in targets]
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    with torch.autocast(device_type=device.type, dtype=torch.float16,
                        enabled=device.type == "cuda"):
        outputs = model(images, targets=targets)
    with torch.autocast(device_type=device.type, enabled=False):
        losses = criterion(outputs, targets, epoch=0, step=0, global_step=0,
                           epoch_step=len(cfg.train_dataloader))
        total = sum(losses.values())
    total.backward()
    class_gradient = sum(
        float(parameter.grad.float().norm()) for name, parameter in model.named_parameters()
        if "class_weight" in name and parameter.grad is not None
    )
    objectness_gradient = sum(
        float(parameter.grad.float().norm()) for name, parameter in model.named_parameters()
        if "dec_score_head" in name and name.endswith("weight")
        and "class_weight" not in name and parameter.grad is not None
    )
    report = {
        "input_shape": list(images.shape), "output_keys": sorted(outputs),
        "objectness_shape": list(outputs["pred_objectness_logits"].shape),
        "class_shape": list(outputs["pred_class_logits"].shape),
        "box_shape": list(outputs["pred_boxes"].shape),
        "loss": float(total.detach()), "losses": {key:float(value.detach()) for key,value in losses.items()},
        "finite": bool(torch.isfinite(total)), "matched_checkpoint_tensors": len(matched),
        "missing_new_tensors": len(load.missing_keys),
        "transferred_objectness_tensors": len(transferred_objectness),
        "class_gradient_norm": class_gradient, "objectness_gradient_norm": objectness_gradient,
        "peak_gpu_memory_gib": round(torch.cuda.max_memory_allocated(device)/1024**3, 3)
        if device.type == "cuda" else None,
    }
    if not report["finite"] or class_gradient <= 0 or objectness_gradient <= 0:
        raise RuntimeError(report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
