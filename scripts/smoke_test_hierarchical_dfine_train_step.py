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
    parser.add_argument("--expect-peft", action="store_true",
                        help="Require frozen backbone/encoder and nonzero adapter gradients.")
    parser.add_argument("--expect-objectness-only", action="store_true",
                        help="Require gradients and trainable parameters only in decoder objectness heads.")
    parser.add_argument("--expect-decoupled-objectness", action="store_true",
                        help="Require gradients only in nonlinear defectness/quality branches.")
    parser.add_argument("--expect-selection-gated", action="store_true",
                        help="Require final-layer Selection/Defectness/Quality gradients.")
    args = parser.parse_args()
    sys.path.insert(0, str(args.dfine_dir.resolve()))
    from src.core import YAMLConfig

    config_path = Path(args.config)
    if not config_path.is_absolute(): config_path = args.dfine_dir/config_path
    cfg = YAMLConfig(str(config_path), tuning=str(args.checkpoint))
    configured_train_batch_size = cfg.yaml_cfg["train_dataloader"]["total_batch_size"]
    configured_train_workers = cfg.yaml_cfg["train_dataloader"]["num_workers"]
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
    adapter_gradient = sum(
        float(parameter.grad.float().norm()) for name, parameter in model.named_parameters()
        if "adaptformer" in name and parameter.grad is not None
    )
    defectness_residual_gradient = sum(
        float(parameter.grad.float().norm()) for name, parameter in model.named_parameters()
        if "defectness_mlp" in name and parameter.grad is not None
    )
    quality_gradient = sum(
        float(parameter.grad.float().norm()) for name, parameter in model.named_parameters()
        if "quality_mlp" in name and parameter.grad is not None
    )
    selection_gradient = sum(
        float(parameter.grad.float().norm()) for name, parameter in model.named_parameters()
        if "selection_mlp" in name and parameter.grad is not None
    )
    trainable_names = [name for name, parameter in model.named_parameters()
                       if parameter.requires_grad]
    forbidden_peft_names = [name for name in trainable_names
                            if name.startswith(("backbone.", "encoder."))]
    peft_report = model.peft_parameter_report() \
        if hasattr(model, "peft_parameter_report") else None
    objectness_report = model.objectness_parameter_report() \
        if hasattr(model, "objectness_parameter_report") else None
    decoupled_report = model.decoupled_objectness_parameter_report() \
        if hasattr(model, "decoupled_objectness_parameter_report") else None
    selection_gated_report = model.selection_gated_parameter_report() \
        if hasattr(model, "selection_gated_parameter_report") else None
    optimizer = cfg.optimizer
    optimizer_parameter_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    trainable_parameter_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    optimizer_groups = [{
        "lr": float(group["lr"]),
        "weight_decay": float(group.get("weight_decay", 0.0)),
        "parameters": sum(parameter.numel() for parameter in group["params"]),
    } for group in optimizer.param_groups]
    ema = cfg.ema
    transform_cfg = cfg.yaml_cfg["train_dataloader"]["dataset"]
    while "transforms" not in transform_cfg and "dataset" in transform_cfg:
        transform_cfg = transform_cfg["dataset"]
    report = {
        "input_shape": list(images.shape), "output_keys": sorted(outputs),
        "objectness_shape": list(outputs["pred_objectness_logits"].shape),
        "quality_shape": list(outputs["pred_quality_logits"].shape)
        if "pred_quality_logits" in outputs else None,
        "defectness_gate_shape": list(outputs["pred_defectness_logits"].shape)
        if "pred_defectness_logits" in outputs else None,
        "class_shape": list(outputs["pred_class_logits"].shape),
        "box_shape": list(outputs["pred_boxes"].shape),
        "loss": float(total.detach()), "losses": {key:float(value.detach()) for key,value in losses.items()},
        "finite": bool(torch.isfinite(total)), "matched_checkpoint_tensors": len(matched),
        "missing_new_tensors": len(load.missing_keys),
        "transferred_objectness_tensors": len(transferred_objectness),
        "class_gradient_norm": class_gradient, "objectness_gradient_norm": objectness_gradient,
        "adapter_gradient_norm": adapter_gradient,
        "defectness_residual_gradient_norm": defectness_residual_gradient,
        "quality_gradient_norm": quality_gradient,
        "selection_gradient_norm": selection_gradient,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()
                                    if parameter.requires_grad),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "forbidden_peft_trainable_names": forbidden_peft_names,
        "peft_report": peft_report,
        "objectness_report": objectness_report,
        "decoupled_objectness_report": decoupled_report,
        "selection_gated_report": selection_gated_report,
        "optimizer_groups": optimizer_groups,
        "optimizer_covers_trainable_exactly": optimizer_parameter_ids == trainable_parameter_ids,
        "resolved_training_config": {
            "epochs": cfg.epochs,
            "configured_batch_size": configured_train_batch_size,
            "configured_steps_per_epoch": len(cfg.train_dataloader.dataset) // configured_train_batch_size,
            "configured_train_workers": configured_train_workers,
            "smoke_batch_size": cfg.train_dataloader.batch_size,
            "smoke_steps_per_epoch": len(cfg.train_dataloader),
            "val_batch_size": cfg.val_dataloader.batch_size,
            "val_workers": cfg.val_dataloader.num_workers,
            "augmentation_stop_epoch": transform_cfg["transforms"]["policy"]["epoch"],
            "solver_stage_stop_epoch": cfg.yaml_cfg["train_dataloader"]["collate_fn"]["stop_epoch"],
            "checkpoint_freq": cfg.checkpoint_freq,
            "print_freq": cfg.print_freq,
            "ema_decay": float(ema.decay) if ema is not None else None,
            "ema_warmups": int(ema.warmups) if ema is not None else None,
            "output_dir": str(cfg.output_dir),
        },
        "peak_gpu_memory_gib": round(torch.cuda.max_memory_allocated(device)/1024**3, 3)
        if device.type == "cuda" else None,
    }
    if (not report["finite"] or objectness_gradient <= 0 or
            (not args.expect_objectness_only
             and not args.expect_decoupled_objectness
             and not args.expect_selection_gated
             and class_gradient <= 0)):
        raise RuntimeError(report)
    if args.expect_peft and (
            adapter_gradient <= 0 or forbidden_peft_names or peft_report is None
            or optimizer_parameter_ids != trainable_parameter_ids):
        raise RuntimeError(report)
    if args.expect_objectness_only:
        invalid = [name for name in trainable_names if (
            not name.startswith("decoder.dec_score_head.")
            or "class_weight" in name or "class_bias" in name
            or not name.endswith((".weight", ".bias"))
        )]
        native_losses = {
            "loss_selection_native_positive",
            "loss_selection_native_duplicate_rank",
            "loss_selection_native_hard_background_rank",
        }
        legacy_losses = {"loss_hard_background_rank", "loss_duplicate_rank"}
        has_expected_losses = (
            legacy_losses.issubset(losses) or native_losses.issubset(losses)
        )
        if (class_gradient != 0 or adapter_gradient != 0 or invalid
                or objectness_report is None
                or objectness_report["trainable_parameters"] not in {257, 1542}
                or optimizer_parameter_ids != trainable_parameter_ids
                or not has_expected_losses):
            raise RuntimeError(report | {"invalid_objectness_trainable_names": invalid})
    if args.expect_decoupled_objectness:
        allowed = (
            "defectness_norm", "defectness_mlp", "defectness_residual_scale",
            "quality_norm", "quality_mlp",
        )
        invalid = [
            name for name in trainable_names
            if not name.startswith("decoder.dec_score_head.")
            or not any(token in name for token in allowed)
        ]
        if (
            class_gradient != 0 or adapter_gradient != 0 or invalid
            or defectness_residual_gradient <= 0 or quality_gradient <= 0
            or decoupled_report is None
            or optimizer_parameter_ids != trainable_parameter_ids
            or "pred_quality_logits" not in outputs
            or "loss_quality" not in losses
            or "loss_quality_duplicate_rank" not in losses
        ):
            raise RuntimeError(report | {"invalid_decoupled_trainable_names": invalid})
    if args.expect_selection_gated:
        allowed = (
            "selection_norm", "selection_mlp", "selection_residual_scale",
            "defectness_norm", "defectness_mlp",
            "quality_norm", "quality_mlp",
        )
        final_head = len(model.decoder.dec_score_head) - 1
        prefix = f"decoder.dec_score_head.{final_head}."
        invalid = [
            name for name in trainable_names
            if not name.startswith(prefix)
            or not any(token in name for token in allowed)
        ]
        required_losses = {
            "loss_selection", "loss_defectness_gate", "loss_quality",
            "loss_selection_hard_background_rank",
            "loss_selection_duplicate_rank",
        }
        if (
            class_gradient != 0 or adapter_gradient != 0 or invalid
            or selection_gradient <= 0
            or defectness_residual_gradient <= 0 or quality_gradient <= 0
            or selection_gated_report is None
            or optimizer_parameter_ids != trainable_parameter_ids
            or "pred_defectness_logits" not in outputs
            or "pred_quality_logits" not in outputs
            or not required_losses.issubset(losses)
        ):
            raise RuntimeError(report | {
                "invalid_selection_gated_trainable_names": invalid,
                "missing_selection_gated_losses": sorted(required_losses - losses.keys()),
            })
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
