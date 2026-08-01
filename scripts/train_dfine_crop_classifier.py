#!/usr/bin/env python3
"""Train DINOv2-Base shared-backbone objectness + nine-class crop heads."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from collections import Counter
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from dfine_crop_classifier import DFineCropDataset, DFineDualHeadClassifier, META_FEATURE_NAMES


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    p.add_argument("--dinov2-weights", type=Path, default=Path("/root/autodl-tmp/weights/dinov2-base"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", type=Path,
                   help="Resume model, optimizer, scheduler, scaler, epoch and early-stopping state.")
    p.add_argument("--init-checkpoint", type=Path,
                   help="Initialize compatible backbone/metadata weights without resuming heads or optimizer.")
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--tight-expand", type=float, default=1.0)
    p.add_argument("--context-expand", type=float, default=1.8)
    p.add_argument("--min-crop-size", type=int, default=128)
    p.add_argument("--batch", type=int, default=16, help="Physical batch; two crops are encoded per sample.")
    p.add_argument("--accumulate", type=int, default=2, help="Effective batch=batch*accumulate.")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--unfreeze-last-blocks", type=int, default=4)
    p.add_argument("--metadata-hidden", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--architecture", choices=("legacy", "token_fusion_v2", "residual_token_v21"),
                   default="token_fusion_v2")
    p.add_argument("--adapter-dim", type=int, default=128)
    p.add_argument("--adapter-only-epochs", type=int, default=3)
    p.add_argument("--head-finetune-epochs", type=int, default=3,
                   help="Additional epochs with adapter+heads before backbone unfreezing.")
    p.add_argument("--consistency-weight", type=float, default=0.15)
    p.add_argument("--eval-initialized", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--lambda-objectness", type=float, default=1.0)
    p.add_argument("--lambda-class", type=float, default=1.0)
    p.add_argument("--class-weight-power", type=float, default=0.5)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--workers-persistent", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--balanced-sampler", action=argparse.BooleanOptionalAction, default=False,
        help=("Balance all (objectness,class) groups. Disabled by default because equalizing "
              "background plus nine positive classes would make batches about 90%% positive; "
              "class-weighted CE already handles positive-class imbalance."),
    )
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-samples", type=int)
    p.add_argument("--max-val-samples", type=int)
    p.add_argument("--max-steps", type=int, help="Smoke test only.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def save_checkpoint_atomic(checkpoint: dict, path: Path) -> None:
    """Avoid leaving a half-written last.pt if the machine powers off while saving."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def metrics(confusion):
    c = confusion.double(); tp = c.diag(); support = c.sum(1); predicted = c.sum(0)
    precision = tp / predicted.clamp_min(1); recall = tp / support.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    valid = support > 0
    return float(f1[valid].mean()) if valid.any() else 0.0


def configure_training_stage(model, epoch: int, args) -> str:
    """Progressively expose old parameters so the zero-init adapter cannot erase the baseline."""
    if args.architecture != "residual_token_v21":
        model.set_trainable_blocks(args.unfreeze_last_blocks)
        return "joint"
    adapter_end = args.adapter_only_epochs
    head_end = adapter_end + args.head_finetune_epochs
    model.spatial_adapter.requires_grad_(True)
    train_old_heads = epoch > adapter_end
    for module in (model.metadata_encoder, model.objectness, model.classifier):
        module.requires_grad_(train_old_heads)
    model.set_trainable_blocks(args.unfreeze_last_blocks if epoch > head_end else 0)
    if epoch <= adapter_end:
        return "adapter_only"
    if epoch <= head_end:
        return "adapter_heads"
    return "joint_low_lr"


@torch.inference_mode()
def evaluate(model, loader, device, obj_loss_fn, cls_loss_fn, args, dtype, num_classes):
    model.eval(); total_loss = total = 0; tp = fp = fn = 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for tight, context, meta, obj, cls in loader:
        tight, context, meta = (x.to(device, non_blocking=True) for x in (tight, context, meta))
        obj, cls = obj.to(device), cls.to(device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=args.amp and device.type == "cuda"):
            out = model(tight, context, meta)
            obj_loss = obj_loss_fn(out["objectness_logits"].float(), obj)
            positive = cls >= 0
            cls_loss = cls_loss_fn(out["class_logits"][positive].float(), cls[positive]) if positive.any() else out["class_logits"].sum() * 0
            loss = args.lambda_objectness * obj_loss + args.lambda_class * cls_loss
        pred_obj = out["objectness_logits"] >= 0; true_obj = obj.bool()
        tp += int((pred_obj & true_obj).sum()); fp += int((pred_obj & ~true_obj).sum()); fn += int((~pred_obj & true_obj).sum())
        if positive.any():
            pred_cls = out["class_logits"][positive].argmax(1).cpu(); true_cls = cls[positive].cpu()
            for t, p in zip(true_cls.tolist(), pred_cls.tolist()): confusion[t, p] += 1
        total_loss += float(loss) * obj.numel(); total += obj.numel()
    obj_p = tp / max(1, tp + fp); obj_r = tp / max(1, tp + fn)
    obj_f1 = 2 * obj_p * obj_r / max(1e-12, obj_p + obj_r); cls_macro_f1 = metrics(confusion)
    return {"loss": total_loss / max(1, total), "objectness_precision": obj_p,
            "objectness_recall": obj_r, "objectness_f1": obj_f1,
            "class_macro_f1": cls_macro_f1,
            "selection_score": 0.5 * obj_f1 + 0.5 * cls_macro_f1}, confusion


def main():
    args = args_parser()
    if args.batch < 1 or args.accumulate < 1:
        raise ValueError("batch and accumulate must be positive")
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    classes = (yaml.safe_load(args.classes.read_text(encoding="utf-8")) or {})["names"]
    if len(classes) != 9: raise ValueError("Expected nine classes")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be used together")
    if args.resume and args.init_checkpoint:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    if args.resume and not args.resume.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
    if args.output.exists() and not args.resume:
        if not args.overwrite: raise FileExistsError(f"{args.output} exists; pass --overwrite")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    ds_kw = dict(image_size=args.image_size, tight_expand=args.tight_expand,
                 context_expand=args.context_expand, min_crop_size=args.min_crop_size, seed=args.seed)
    train_ds = DFineCropDataset(args.data, "train", train=True, max_samples=args.max_train_samples, **ds_kw)
    val_ds = DFineCropDataset(args.data, "val", train=False, max_samples=args.max_val_samples, **ds_kw)
    sampler = None
    if args.balanced_sampler:
        keys = list(zip(train_ds.obj_targets, train_ds.class_targets))
        counts = Counter(keys); weights = [1 / counts[k] for k in keys]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    common = dict(batch_size=args.batch, num_workers=args.workers, pin_memory=True,
                  persistent_workers=args.workers_persistent and args.workers > 0)
    train_loader = DataLoader(train_ds, shuffle=sampler is None, sampler=sampler, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = DFineDualHeadClassifier(args.dinov2_weights, len(classes), args.metadata_hidden,
                                    args.dropout, args.architecture, args.adapter_dim).to(device)
    model.set_trainable_blocks(args.unfreeze_last_blocks)
    initialization_metrics = None
    if args.init_checkpoint:
        if not args.init_checkpoint.is_file():
            raise FileNotFoundError(f"Initialization checkpoint not found: {args.init_checkpoint}")
        initialization = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if initialization.get("classes") != classes:
            raise ValueError("Initialization checkpoint classes differ from --classes")
        current_state = model.state_dict()
        transferable = {k: v for k, v in initialization["model"].items()
                        if k in current_state and current_state[k].shape == v.shape}
        incompatible = model.load_state_dict(transferable, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(f"Unexpected initialization keys: {incompatible.unexpected_keys}")
        loaded_parameters = sum(v.numel() for v in transferable.values())
        print(f"Initialized {len(transferable)} tensors ({loaded_parameters} parameters) "
              f"from {args.init_checkpoint}; skipped {len(incompatible.missing_keys)} new/incompatible tensors")
        initialization_metrics = initialization.get("metrics", {})
    backbone = [p for n, p in model.named_parameters() if n.startswith("backbone.") and p.requires_grad]
    adapter = [p for n, p in model.named_parameters() if n.startswith("spatial_adapter.")]
    heads = [p for n, p in model.named_parameters()
             if not n.startswith(("backbone.", "spatial_adapter.")) and p.requires_grad]
    param_groups = []
    if adapter:
        param_groups.append({"params": adapter, "lr": args.lr, "group_name": "adapter"})
    if heads:
        head_lr = args.head_lr if args.architecture == "residual_token_v21" else args.lr
        param_groups.append({"params": heads, "lr": head_lr, "group_name": "heads"})
    if backbone:
        param_groups.append({"params": backbone, "lr": args.backbone_lr, "group_name": "backbone"})
    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)
    updates_per_epoch = math.ceil(len(train_loader) / args.accumulate); total_updates = args.epochs * updates_per_epoch
    warmup = int(args.warmup_epochs * updates_per_epoch)
    def lr_factor(step):
        if step < warmup: return max(1e-3, (step + 1) / max(1, warmup))
        progress = (step - warmup) / max(1, total_updates - warmup)
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2
    scheduler = LambdaLR(optimizer, lr_factor)
    pos, neg = sum(train_ds.obj_targets), len(train_ds) - sum(train_ds.obj_targets)
    obj_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / max(1, pos), device=device))
    class_counts = Counter(c for c in train_ds.class_targets if c >= 0); largest = max(class_counts.values())
    class_weights = torch.tensor([(largest / max(1, class_counts[i])) ** args.class_weight_power for i in range(9)], device=device)
    class_weights /= class_weights.mean()
    cls_loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda" and dtype == torch.float16)
    config = {**vars(args), "classes": classes, "metadata_features": META_FEATURE_NAMES,
              "train_samples": len(train_ds), "val_samples": len(val_ds),
              "effective_batch": args.batch * args.accumulate,
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    best, stale, global_step, history, start_epoch = -1.0, 0, 0, [], 1
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resume.get("classes") != classes:
            raise ValueError("Resume checkpoint classes differ from --classes")
        required = {"optimizer", "scheduler", "epoch"}
        missing = required - set(resume)
        if missing:
            raise ValueError(
                f"Checkpoint lacks full resume state {sorted(missing)}. "
                "It may be from the pre-resume script; start a new run or use it only as model initialization."
            )
        model.load_state_dict(resume["model"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        if resume.get("scaler") and scaler.is_enabled():
            scaler.load_state_dict(resume["scaler"])
        start_epoch = int(resume["epoch"]) + 1
        global_step = int(resume.get("global_step", resume.get("metrics", {}).get("updates", 0)))
        best = float(resume.get("best_score", resume.get("metrics", {}).get("val_selection_score", -1.0)))
        stale = int(resume.get("stale", 0))
        history_path = args.output / "history.json"
        if history_path.is_file():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        if resume.get("python_rng_state") is not None:
            random.setstate(resume["python_rng_state"])
        if resume.get("torch_rng_state") is not None:
            torch.set_rng_state(resume["torch_rng_state"])
        if torch.cuda.is_available() and resume.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(resume["cuda_rng_state"])
        print(f"Resumed {args.resume}: completed_epoch={start_epoch - 1} "
              f"global_step={global_step} best={best:.6f} stale={stale}")
    (args.output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(config, ensure_ascii=False, indent=2))
    if (not args.resume and args.init_checkpoint and args.eval_initialized
            and args.architecture == "residual_token_v21"):
        initial_start = time.time()
        initial_val, initial_confusion = evaluate(
            model, val_loader, device, obj_loss_fn, cls_loss_fn, args, dtype, 9)
        initial_record = {
            "epoch": 0, "stage": "initialized_legacy_equivalent", "updates": 0,
            "train_loss": None,
            **{f"val_{k}": v for k, v in initial_val.items()},
            "seconds": round(time.time() - initial_start, 1),
            "peak_cuda_gib": (round(torch.cuda.max_memory_allocated(device) / 2**30, 3)
                              if device.type == "cuda" else 0.0),
        }
        expected = initialization_metrics.get("val_selection_score") if initialization_metrics else None
        if expected is not None and abs(initial_val["selection_score"] - float(expected)) > 1e-6:
            raise RuntimeError(
                f"Residual epoch0 is not legacy-equivalent: got {initial_val['selection_score']:.9f}, "
                f"expected {float(expected):.9f}"
            )
        best = initial_val["selection_score"]
        history = [initial_record]
        (args.output / "initial_metrics.json").write_text(
            json.dumps(initial_record, indent=2), encoding="utf-8")
        (args.output / "best_metrics.json").write_text(
            json.dumps(initial_record, indent=2), encoding="utf-8")
        (args.output / "best_confusion.json").write_text(
            json.dumps(initial_confusion.tolist()), encoding="utf-8")
        initial_checkpoint = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "classes": classes, "config": config,
            "model_type": "dfine_dinov2_residual_token_v21", "epoch": 0,
            "global_step": 0, "best_score": best, "stale": 0,
            "metrics": initial_record, "python_rng_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        save_checkpoint_atomic(initial_checkpoint, args.output / "best.pt")
        (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print("EPOCH0_LEGACY_EQUIVALENCE_OK", json.dumps(initial_record))
    if start_epoch > args.epochs:
        print(f"Nothing to do: checkpoint completed epoch {start_epoch - 1}, target is {args.epochs}.")
        return
    for epoch in range(start_epoch, args.epochs + 1):
        stage = configure_training_stage(model, epoch, args)
        trainable_now = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"epoch={epoch} stage={stage} trainable_parameters={trainable_now}")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train(); optimizer.zero_grad(set_to_none=True); start = time.time(); running = 0
        for step, (tight, context, meta, obj, cls) in enumerate(train_loader, 1):
            tight, context, meta = (x.to(device, non_blocking=True) for x in (tight, context, meta))
            obj, cls = obj.to(device), cls.to(device)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=args.amp and device.type == "cuda"):
                out = model(tight, context, meta); positive = cls >= 0
                obj_loss = obj_loss_fn(out["objectness_logits"].float(), obj)
                cls_loss = cls_loss_fn(out["class_logits"][positive].float(), cls[positive]) if positive.any() else out["class_logits"].sum() * 0
                consistency = out["class_logits"].sum() * 0
                if "base_objectness_logits" in out and args.consistency_weight > 0:
                    obj_consistency = F.smooth_l1_loss(
                        out["objectness_logits"].float(), out["base_objectness_logits"].detach().float())
                    if positive.any():
                        cls_consistency = F.kl_div(
                            F.log_softmax(out["class_logits"][positive].float(), dim=1),
                            F.softmax(out["base_class_logits"][positive].detach().float(), dim=1),
                            reduction="batchmean")
                    else:
                        cls_consistency = consistency
                    consistency = obj_consistency + cls_consistency
                loss = (args.lambda_objectness * obj_loss + args.lambda_class * cls_loss
                        + args.consistency_weight * consistency) / args.accumulate
            scaler.scale(loss).backward() if scaler.is_enabled() else loss.backward()
            update = step % args.accumulate == 0 or step == len(train_loader)
            if update:
                if scaler.is_enabled(): scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                if scaler.is_enabled(): scaler.step(optimizer); scaler.update()
                else: optimizer.step()
                optimizer.zero_grad(set_to_none=True); scheduler.step(); global_step += 1
            detached_loss = float(loss.detach()) * args.accumulate
            running += detached_loss
            if step == 1 or step % 20 == 0: print(f"epoch={epoch} step={step}/{len(train_loader)} loss={detached_loss:.5f}")
            if args.max_steps and global_step >= args.max_steps: break
        val, confusion = evaluate(model, val_loader, device, obj_loss_fn, cls_loss_fn, args, dtype, 9)
        record = {"epoch": epoch, "stage": stage, "updates": global_step,
                  "train_loss": running / max(1, step),
                  **{f"val_{k}": v for k, v in val.items()}, "seconds": round(time.time()-start, 1),
                  "peak_cuda_gib": (round(torch.cuda.max_memory_allocated(device) / 2**30, 3)
                                    if device.type == "cuda" else 0.0)}
        history.append(record); print(json.dumps(record))
        improved = val["selection_score"] > best + 1e-4
        if improved:
            best, stale = val["selection_score"], 0
            (args.output / "best_metrics.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            (args.output / "best_confusion.json").write_text(json.dumps(confusion.tolist()), encoding="utf-8")
        else: stale += 1
        checkpoint = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "classes": classes, "config": config,
            "model_type": ({"token_fusion_v2": "dfine_dinov2_token_fusion_v2",
                            "residual_token_v21": "dfine_dinov2_residual_token_v21"}
                           .get(args.architecture, "dfine_dinov2_dual_head_metadata")), "epoch": epoch,
            "global_step": global_step, "best_score": best, "stale": stale,
            "metrics": record, "python_rng_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        save_checkpoint_atomic(checkpoint, args.output / "last.pt")
        if improved:
            save_checkpoint_atomic(checkpoint, args.output / "best.pt")
        (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if args.max_steps and global_step >= args.max_steps: break
        if args.patience and stale >= args.patience: print("Early stopping"); break


if __name__ == "__main__":
    main()
