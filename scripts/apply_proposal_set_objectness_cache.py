#!/usr/bin/env python3
"""Apply one-to-one proposal-set objectness to raw cached proposals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from proposal_relation import grouped_indices, load_raw_relation_metadata, pad_group_batch
from proposal_set_objectness import ProposalSetObjectness, load_frozen_relation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-images", type=int, default=16)
    parser.add_argument("--max-proposals", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = load_raw_relation_metadata(cache, args.predictions)
    # Recreate the frozen relation checkpoint from the copy embedded in the V2 state dict.
    relation_stub = {
        "feature_dim": checkpoint["feature_dim"], "auxiliary_dim": checkpoint["auxiliary_dim"],
        "classes": checkpoint["classes"], "config": checkpoint["relation_config"],
        "model": {key.removeprefix("relation."): value for key, value in checkpoint["model"].items()
                  if key.startswith("relation.")},
    }
    config = checkpoint["config"]
    model = ProposalSetObjectness(load_frozen_relation(relation_stub),
                                  config["decision_hidden"], 0.0,
                                  config.get("max_objectness_delta", 1.0))
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model.to(device).eval()
    features = cache["features"].float().to(device)
    auxiliary = metadata["auxiliary"].to(device)
    baseline_objectness = cache["baseline_objectness_logits"].float().to(device)
    baseline_classes = cache["baseline_class_logits"].float().to(device)
    baseline_score = cache["detector_scores"] * baseline_objectness.cpu().sigmoid()
    groups = grouped_indices(metadata["image_ids"])
    selected = []
    for group in groups:
        group.sort(key=lambda index: float(baseline_score[index]), reverse=True)
        selected.append(group[:args.max_proposals])
    selected.sort(key=len)
    final_objectness = baseline_objectness.clone()
    final_classes = baseline_classes.clone()
    suppression = torch.zeros_like(baseline_objectness)
    gated_rescue = torch.zeros_like(baseline_objectness)
    with torch.inference_mode():
        for start in range(0, len(selected), args.batch_images):
            group_batch = selected[start:start+args.batch_images]
            batch = pad_group_batch(group_batch, features, auxiliary, baseline_objectness, baseline_classes)
            output = model(batch["features"], batch["auxiliary"], batch["baseline_objectness"],
                           batch["baseline_classes"], batch["padding_mask"])
            for row, group in enumerate(group_batch):
                length = len(group)
                final_objectness[group] = output["objectness_logits"][row, :length]
                final_classes[group] = output["class_logits"][row, :length]
                suppression[group] = output["suppression"][row, :length]
                gated_rescue[group] = (output["rescue_gate"] * output["rescue"])[row, :length]
    probabilities = final_objectness.sigmoid().cpu(); labels = final_classes.argmax(dim=-1).cpu()
    suppression = suppression.cpu(); gated_rescue = gated_rescue.cpu()
    payload = metadata["payload"]; keep_by_record = {}
    for index, (record_index, prediction_index) in enumerate(cache["proposal_locations"].tolist()):
        keep_by_record.setdefault(record_index, set()).add(prediction_index)
        pred = payload["images"][record_index]["predictions"][prediction_index]
        detector_score = float(pred.get("original_score", pred["score"]))
        probability, label = float(probabilities[index]), int(labels[index])
        pred["score"] = max(0.0, min(1.0, detector_score * probability))
        pred["class_id"] = label; pred["category_name"] = checkpoint["classes"][label]
        pred["proposal_set_objectness"] = {
            "objectness": probability, "suppression": float(suppression[index]),
            "gated_rescue": float(gated_rescue[index]), "epoch": checkpoint["epoch"],
        }
    for record_index, record in enumerate(payload["images"]):
        keep = keep_by_record.get(record_index, set())
        record["predictions"] = [pred for index, pred in enumerate(record.get("predictions", [])) if index in keep]
    payload.setdefault("metadata", {})["classes"] = checkpoint["classes"]
    payload["metadata"]["proposal_set_objectness"] = {
        "checkpoint": str(args.checkpoint.resolve()), "epoch": checkpoint["epoch"],
        "processed_proposals": len(features), "max_proposals_per_image": args.max_proposals,
        "score_mode": "det_obj_set_suppress_rescue",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["metadata"]["proposal_set_objectness"], indent=2))


if __name__ == "__main__":
    main()
