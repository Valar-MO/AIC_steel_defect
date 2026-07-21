#!/usr/bin/env python3
"""Apply a proposal relation checkpoint to a raw cached proposal JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from proposal_relation import (ProposalRelationTransformer, grouped_indices,
                               load_raw_relation_metadata, pad_group_batch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-images", type=int, default=16)
    parser.add_argument("--max-proposals", type=int, default=256)
    parser.add_argument("--residual-mode", choices=("full", "objectness", "class"), default="full",
                        help="Enable both residuals or isolate one branch without retraining.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = load_raw_relation_metadata(cache, args.predictions)
    config = checkpoint["config"]
    model = ProposalRelationTransformer(
        checkpoint["feature_dim"], checkpoint["auxiliary_dim"], len(checkpoint["classes"]),
        config["dimension"], config["heads"], config["layers"], 0.0,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model.to(device).eval()
    features = cache["features"].float().to(device)
    auxiliary = metadata["auxiliary"].to(device)
    baseline_objectness = cache["baseline_objectness_logits"].float().to(device)
    baseline_classes = cache["baseline_class_logits"].float().to(device)
    groups = grouped_indices(metadata["image_ids"])
    # Attention is quadratic.  Keep the highest baseline proposals relational and leave
    # any exceptionally long low-score tail exactly at the V2.1 baseline.
    selected_groups = []
    baseline_score = cache["detector_scores"] * cache["baseline_objectness_logits"].float().sigmoid()
    for group in groups:
        group.sort(key=lambda index: float(baseline_score[index]), reverse=True)
        selected_groups.append(group[:args.max_proposals])
    selected_groups.sort(key=len)
    output_objectness = baseline_objectness.clone(); output_classes = baseline_classes.clone()
    with torch.inference_mode():
        for start in range(0, len(selected_groups), args.batch_images):
            group_batch = selected_groups[start:start + args.batch_images]
            batch = pad_group_batch(
                group_batch, features, auxiliary, baseline_objectness, baseline_classes,
            )
            output = model(batch["features"], batch["auxiliary"], batch["baseline_objectness"],
                           batch["baseline_classes"], batch["padding_mask"])
            if args.residual_mode == "class":
                output["objectness_logits"] = batch["baseline_objectness"]
            elif args.residual_mode == "objectness":
                output["class_logits"] = batch["baseline_classes"]
            for row, group in enumerate(group_batch):
                output_objectness[group] = output["objectness_logits"][row, :len(group)]
                output_classes[group] = output["class_logits"][row, :len(group)]
    probabilities = output_objectness.sigmoid().cpu()
    labels = output_classes.argmax(dim=1).cpu()
    payload = metadata["payload"]
    keep_by_record = {}
    for index, (record_index, prediction_index) in enumerate(cache["proposal_locations"].tolist()):
        keep_by_record.setdefault(record_index, set()).add(prediction_index)
        pred = payload["images"][record_index]["predictions"][prediction_index]
        detector_score = float(pred.get("original_score", pred["score"]))
        probability, label = float(probabilities[index]), int(labels[index])
        pred["score"] = max(0.0, min(1.0, detector_score * probability))
        pred["class_id"] = label; pred["category_name"] = checkpoint["classes"][label]
        pred["proposal_relation"] = {"objectness": probability, "epoch": checkpoint["epoch"]}
    for record_index, record in enumerate(payload["images"]):
        keep = keep_by_record.get(record_index, set())
        record["predictions"] = [pred for index, pred in enumerate(record.get("predictions", [])) if index in keep]
    payload.setdefault("metadata", {})["classes"] = checkpoint["classes"]
    payload["metadata"]["proposal_relation"] = {
        "checkpoint": str(args.checkpoint.resolve()), "epoch": checkpoint["epoch"],
        "processed_proposals": len(features), "max_proposals_per_image": args.max_proposals,
        "residual_mode": args.residual_mode,
        "score_mode": f"det_obj_relation_{args.residual_mode}_residual",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["metadata"]["proposal_relation"], indent=2))


if __name__ == "__main__":
    main()
