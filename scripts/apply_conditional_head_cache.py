#!/usr/bin/env python3
"""Apply a cached-feature conditional verifier head to a full proposal JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from train_conditional_verifier_head import ConditionalVerifierHead


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if cache["classes"] != checkpoint["classes"]: raise ValueError("Cache/checkpoint classes differ")
    model = ConditionalVerifierHead(checkpoint["feature_dim"], len(checkpoint["classes"]),
                                    checkpoint["hidden"], 0.0)
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model.to(device).eval()
    features = cache["features"]
    probabilities, labels = [], []
    with torch.inference_mode():
        for start in range(0, len(features), args.batch):
            logits = model(features[start:start + args.batch].float().to(device))
            probability, label = logits.sigmoid().max(dim=1)
            probabilities.append(probability.cpu()); labels.append(label.cpu())
    probabilities, labels = torch.cat(probabilities), torch.cat(labels)
    payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    keep_by_record = {}
    for index, (ri, pi) in enumerate(cache["proposal_locations"].tolist()):
        keep_by_record.setdefault(ri, set()).add(pi)
        pred = payload["images"][ri]["predictions"][pi]
        detector_score = float(pred.get("original_score", pred["score"]))
        probability = float(probabilities[index]); label = int(labels[index])
        pred["score"] = max(0.0, min(1.0, detector_score * probability))
        pred["class_id"] = label; pred["category_name"] = checkpoint["classes"][label]
        pred["conditional_verifier"] = {"probability": probability,
                                         "variant": checkpoint["variant"]}
    for ri, record in enumerate(payload["images"]):
        keep = keep_by_record.get(ri, set())
        record["predictions"] = [pred for pi, pred in enumerate(record.get("predictions", [])) if pi in keep]
    payload.setdefault("metadata", {})["classes"] = checkpoint["classes"]
    payload["metadata"]["conditional_verifier"] = {
        "checkpoint": str(args.checkpoint.resolve()), "variant": checkpoint["variant"],
        "processed_proposals": len(features), "score_mode": "det_conditional",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["metadata"]["conditional_verifier"], indent=2))


if __name__ == "__main__":
    main()
