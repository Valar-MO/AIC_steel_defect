#!/usr/bin/env python3
"""Apply an R1 Selection residual ranker to cached D-FINE predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hierarchical_query_classifier import SelectionResidualRanker


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=16384)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--line-protect-classes",
        nargs="*",
        default=(),
        help="Class names whose tile line-like mid-score candidates get a residual floor.",
    )
    parser.add_argument("--line-protect-source", default="tile")
    parser.add_argument("--line-protect-min-aspect", type=float, default=5.0)
    parser.add_argument("--line-protect-min-base-score", type=float, default=0.30)
    parser.add_argument("--line-protect-residual-floor", type=float, default=-0.10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def logit(scores: torch.Tensor) -> torch.Tensor:
    scores = scores.float().clamp(1e-5, 1 - 1e-5)
    return torch.logit(scores)


def normalized_prediction_view(cache: dict) -> dict:
    if "prediction_features" in cache:
        return {
            "features": cache["prediction_features"],
            "scores": cache["prediction_scores"].float(),
            "location_mode": "locations",
        }
    if "query_features" in cache:
        return {
            "features": cache["query_features"],
            "scores": cache["selection_logits"].float().sigmoid(),
            "location_mode": "candidate_index",
        }
    raise ValueError("Unsupported candidate cache schema")


def aspect_ratio(box) -> float:
    x1, y1, x2, y2 = [float(value) for value in box]
    width = max(1e-6, x2 - x1)
    height = max(1e-6, y2 - y1)
    return max(width / height, height / width)


def main():
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if cache["classes"] != checkpoint["classes"]:
        raise ValueError("Cache and ranker classes differ")
    view = normalized_prediction_view(cache)

    model = SelectionResidualRanker(
        checkpoint["feature_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        residual_scale=checkpoint["residual_scale"],
        dropout=checkpoint.get("dropout", 0.0),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    features = view["features"]
    base_scores = view["scores"]
    base_logits = logit(base_scores)
    final_scores = []
    residuals = []
    with torch.inference_mode():
        for start in range(0, len(features), args.batch):
            end = start + args.batch
            logits = model(features[start:end].float().to(device), base_logits[start:end].to(device))
            score = logits.sigmoid().cpu()
            final_scores.append(score)
            residuals.append((logit(score) - base_logits[start:end]).cpu())
    final_scores = torch.cat(final_scores) if final_scores else torch.empty(0)
    residuals = torch.cat(residuals) if residuals else torch.empty(0)

    payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    changed_order_fields = 0
    skipped = 0
    protected = 0
    protected_by_class = {}
    protected_residual_lift = 0.0
    protect_classes = set(args.line_protect_classes)
    if view["location_mode"] == "locations":
        locations = cache["prediction_locations"]
        if len(locations) != len(final_scores):
            raise ValueError(
                f"Location/prediction mismatch: {len(locations)} != {len(final_scores)}"
            )
        iterator = (
            (int(cache_index), payload["images"][int(record_index)]["predictions"][int(prediction_index)])
            for cache_index, (record_index, prediction_index) in enumerate(locations.tolist())
        )
    else:
        rows = []
        for record in payload["images"]:
            for prediction in record.get("predictions", []):
                cache_index = prediction.get("dino_candidate_index")
                if cache_index is None:
                    skipped += 1
                    continue
                rows.append((int(cache_index), prediction))
        iterator = rows
    for index, prediction in iterator:
        if index < 0 or index >= len(final_scores):
            skipped += 1
            continue
        old = float(prediction["score"])
        residual = float(residuals[index])
        protected_this = False
        if protect_classes:
            class_name = str(prediction.get("category_name"))
            base_score = float(base_scores[index])
            source = str(prediction.get("source"))
            if (
                class_name in protect_classes
                and source == args.line_protect_source
                and base_score >= args.line_protect_min_base_score
                and aspect_ratio(prediction["bbox_xyxy"]) >= args.line_protect_min_aspect
                and residual < args.line_protect_residual_floor
            ):
                protected_this = True
                protected += 1
                protected_by_class[class_name] = protected_by_class.get(class_name, 0) + 1
                protected_residual_lift += args.line_protect_residual_floor - residual
                residual = args.line_protect_residual_floor
        new = float((base_logits[index] + residual).sigmoid())
        prediction["score"] = new
        prediction["selection_probability"] = new
        prediction["selection_ranker"] = {
            "base_score": old,
            "ranker_base_selection": float(base_scores[index]),
            "logit_residual": residual,
            "line_protected": protected_this,
        }
        changed_order_fields += int(abs(old - new) > 1e-8)

    metadata = payload.setdefault("metadata", {})
    metadata["selection_ranker"] = {
        "checkpoint": str(args.checkpoint.resolve()),
        "epoch": checkpoint["epoch"],
        "residual_scale": checkpoint["residual_scale"],
        "processed_predictions": int(len(final_scores)),
        "changed_scores": int(changed_order_fields),
        "skipped_predictions": int(skipped),
        "line_protection": {
            "classes": sorted(protect_classes),
            "source": args.line_protect_source,
            "min_aspect": args.line_protect_min_aspect,
            "min_base_score": args.line_protect_min_base_score,
            "residual_floor": args.line_protect_residual_floor,
            "protected_predictions": int(protected),
            "protected_by_class": protected_by_class,
            "mean_residual_lift": (
                protected_residual_lift / protected if protected else 0.0
            ),
        },
        "score_mode": "selection_plus_residual",
        "base_score_min": float(base_scores.min()) if len(base_scores) else None,
        "base_score_max": float(base_scores.max()) if len(base_scores) else None,
        "final_score_min": float(final_scores.min()) if len(final_scores) else None,
        "final_score_max": float(final_scores.max()) if len(final_scores) else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata["selection_ranker"], indent=2))


if __name__ == "__main__":
    main()
