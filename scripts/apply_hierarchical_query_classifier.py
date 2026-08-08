#!/usr/bin/env python3
"""Replace only class labels in cached whole-image D-FINE predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from hierarchical_query_classifier import HierarchicalQueryClassifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=16384)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if cache["classes"] != checkpoint["classes"]:
        raise ValueError("Cache and classifier classes differ")
    required = {"prediction_features", "prediction_locations"}
    missing = required - cache.keys()
    if missing:
        raise ValueError(f"Cache missing val prediction fields: {sorted(missing)}")
    model = HierarchicalQueryClassifier(
        checkpoint["feature_dim"], len(checkpoint["classes"])
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    features = cache["prediction_features"]
    labels = []
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(features), args.batch):
            logits = model(features[start:start + args.batch].float().to(device))
            probability = F.softmax(logits, dim=-1)
            confidence, label = probability.max(dim=-1)
            labels.append(label.cpu())
            probabilities.append(confidence.cpu())
    labels = torch.cat(labels) if labels else torch.empty(0, dtype=torch.long)
    probabilities = torch.cat(probabilities) if probabilities else torch.empty(0)
    payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    locations = cache["prediction_locations"]
    if len(locations) != len(labels):
        raise ValueError(f"Location/prediction mismatch: {len(locations)} != {len(labels)}")
    changed = 0
    classes = checkpoint["classes"]
    for index, (record_index, prediction_index) in enumerate(locations.tolist()):
        prediction = payload["images"][record_index]["predictions"][prediction_index]
        label = int(labels[index])
        changed += int(label != int(prediction["class_id"]))
        prediction["class_id"] = label
        prediction["category_name"] = classes[label]
        prediction["class_probability"] = float(probabilities[index])
        prediction["query_classifier"] = {
            "variant": checkpoint["variant"],
            "seed": checkpoint["seed"],
            "epoch": checkpoint["epoch"],
        }
        # Deliberately preserve objectness, score and bbox.  Detection metric
        # changes therefore come only from replacing the conditional class.
    payload.setdefault("metadata", {})["query_classifier"] = {
        "checkpoint": str(args.checkpoint.resolve()),
        "variant": checkpoint["variant"],
        "seed": checkpoint["seed"],
        "epoch": checkpoint["epoch"],
        "changed_labels": changed,
        "processed_predictions": len(labels),
        "score_mode": "unchanged_objectness",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["metadata"]["query_classifier"], indent=2))


if __name__ == "__main__":
    main()
