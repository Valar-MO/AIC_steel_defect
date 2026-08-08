#!/usr/bin/env python3
"""Score every fixed validation candidate with a trained verifier checkpoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dfine_dinov2_teacher import VerifierCandidateDataset, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("baseline", "query", "dino", "query_dino"), required=True
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dinov2-weights",
        type=Path,
        default=Path("/root/autodl-tmp/weights/dinov2-base"),
    )
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument(
        "--save-features",
        action="store_true",
        help="Also cache the verifier's fused pre-output features as float16.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    if args.mode == "baseline":
        scores = cache["selection_logits"].float()
        payload = {
            "version": 1,
            "mode": "baseline",
            "cache": str(args.cache.resolve()),
            "scores": scores,
            "elapsed_seconds": 0.0,
            "peak_cuda_gib": 0.0,
        }
    else:
        if args.checkpoint is None or not args.checkpoint.is_file():
            raise FileNotFoundError("A verifier --checkpoint is required")
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        config = state["config"]
        if state["mode"] != args.mode:
            raise ValueError(f"Checkpoint mode={state['mode']} but requested {args.mode}")
        model = build_model(
            args.mode,
            query_dim=int(cache["feature_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            dropout=float(config["dropout"]),
            dinov2_weights=args.dinov2_weights,
        )
        model.load_state_dict(state["model"], strict=True)
        device = torch.device(args.device)
        model.to(device).eval()
        dataset = VerifierCandidateDataset(
            args.cache,
            need_crops=args.mode != "query",
            image_size=int(config["image_size"]),
            tight_expand=float(config["tight_expand"]),
            context_expand=float(config["context_expand"]),
            min_crop_size=int(config["min_crop_size"]),
            train=False,
        )
        loader_kwargs = {
            "dataset": dataset,
            "batch_size": args.batch,
            "shuffle": False,
            "num_workers": args.workers,
            "pin_memory": True,
            "persistent_workers": args.workers > 0,
        }
        if args.workers > 0:
            loader_kwargs["prefetch_factor"] = 1
        loader = DataLoader(**loader_kwargs)
        dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
        score_rows, index_rows, feature_rows = [], [], []
        started = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            for step, batch in enumerate(loader, 1):
                batch = {
                    key: value.to(device, non_blocking=True)
                    if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
                with torch.autocast(
                    device_type=device.type,
                    dtype=dtype,
                    enabled=device.type == "cuda",
                ):
                    output = model(**batch)
                score_rows.append(output["logit"].float().cpu())
                index_rows.append(batch["candidate_index"].long().cpu())
                if args.save_features:
                    feature_rows.append(output["feature"].to(torch.float16).cpu())
                if step % 50 == 0 or step == len(loader):
                    print(f"score mode={args.mode} step={step}/{len(loader)}", flush=True)
        indices = torch.cat(index_rows)
        if not torch.equal(indices, torch.arange(len(indices))):
            raise RuntimeError("Scoring loader changed fixed candidate order")
        payload = {
            "version": 1,
            "mode": args.mode,
            "cache": str(args.cache.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "scores": torch.cat(score_rows),
            "elapsed_seconds": time.time() - started,
            "peak_cuda_gib": (
                torch.cuda.max_memory_allocated(device) / 2**30
                if device.type == "cuda" else 0.0
            ),
        }
        if args.save_features:
            payload["features"] = torch.cat(feature_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()
    torch.save(payload, args.output)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "mode": args.mode,
        "candidates": len(payload["scores"]),
        "elapsed_seconds": round(payload["elapsed_seconds"], 3),
        "peak_cuda_gib": round(payload["peak_cuda_gib"], 3),
    }, indent=2))


if __name__ == "__main__":
    main()
