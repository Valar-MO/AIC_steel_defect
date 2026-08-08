#!/usr/bin/env python3
"""Gate aligned raw D-FINE predictions with cached Query+DINO scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--verifier-scores", type=Path, required=True)
    parser.add_argument("--gate-threshold", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.gate_threshold <= 1:
        raise ValueError("--gate-threshold must be in [0,1]")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")
    payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    score_payload = torch.load(
        args.verifier_scores, map_location="cpu", weights_only=False
    )
    probabilities = score_payload["scores"].double().sigmoid()
    before = after = missing = 0
    records = []
    for record in payload["images"]:
        predictions = []
        for prediction in record.get("predictions", []):
            before += 1
            candidate_index = prediction.get("dino_candidate_index")
            selection = float(prediction.get(
                "selection_probability", prediction["score"]
            ))
            if candidate_index is None:
                # Candidates below the cache floor cannot pass a downstream
                # Selection threshold above that floor. Preserve them so the
                # ordinary merge threshold remains the single source of truth.
                missing += 1
                predictions.append(prediction)
                continue
            candidate_index = int(candidate_index)
            if not 0 <= candidate_index < len(probabilities):
                raise IndexError(f"Invalid candidate index {candidate_index}")
            verifier = float(probabilities[candidate_index])
            if verifier < args.gate_threshold:
                continue
            item = dict(prediction)
            item["score"] = selection
            item["objectness"] = selection
            item["dino_defectness_probability"] = verifier
            predictions.append(item)
        after += len(predictions)
        updated = dict(record)
        updated["predictions"] = predictions
        records.append(updated)
    metadata = dict(payload.get("metadata", {}))
    metadata.update({
        "dino_gate_threshold": args.gate_threshold,
        "dino_verifier_scores": str(args.verifier_scores.resolve()),
        "score_mode": "selection_after_dino_binary_gate",
        "predictions_before_gate": before,
        "predictions_after_gate": after,
        "predictions_without_cached_verifier": missing,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"metadata": metadata, "images": records}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
