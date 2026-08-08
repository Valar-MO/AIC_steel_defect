#!/usr/bin/env python3
"""Exercise cached-query training and label-only application on synthetic data."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


def main():
    project = Path(__file__).parents[1]
    classes = [f"class_{index}" for index in range(9)]
    generator = torch.Generator().manual_seed(7)
    feature_dim = 16
    weight = torch.randn(9, feature_dim, generator=generator)
    bias = torch.randn(9, generator=generator)

    def cache(samples_per_class, include_predictions=False):
        targets = torch.arange(9).repeat_interleave(samples_per_class)
        centers = torch.nn.functional.normalize(weight, dim=1)
        features = centers[targets] + 0.25 * torch.randn(
            len(targets), feature_dim, generator=generator
        )
        payload = {
            "classes": classes,
            "feature_dim": feature_dim,
            "matched_features": features.half(),
            "matched_class_targets": targets,
            "detector_class_weight": weight,
            "detector_class_bias": bias,
        }
        if include_predictions:
            payload["prediction_features"] = features.half()
            payload["prediction_locations"] = torch.tensor(
                [(0, index) for index in range(len(features))], dtype=torch.long
            )
        return payload

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        train_path, val_path = root / "train.pt", root / "val.pt"
        torch.save(cache(30), train_path)
        val = cache(5, include_predictions=True)
        torch.save(val, val_path)
        heads = root / "heads"
        subprocess.run([
            sys.executable, str(project / "scripts" / "train_hierarchical_query_classifier.py"),
            "--train-cache", str(train_path), "--val-cache", str(val_path),
            "--output", str(heads), "--variants", "H0", "H1", "H2",
            "--seeds", "42", "--batch", "64", "--epochs", "2", "--patience", "2",
            "--device", "cpu", "--overwrite",
        ], check=True)
        report = json.loads((heads / "classification_report.json").read_text())
        assert len(report["runs"]) == 3

        predictions = {
            "metadata": {"classes": classes},
            "images": [{
                "image_id": "synthetic.jpg",
                "predictions": [{
                    "class_id": 0, "category_name": classes[0],
                    "bbox_xyxy": [0.0, 0.0, 1.0, 1.0], "score": 0.25,
                } for _ in range(len(val["prediction_features"]))],
            }],
        }
        prediction_path = root / "baseline.json"
        prediction_path.write_text(json.dumps(predictions), encoding="utf-8")
        output_path = root / "applied.json"
        subprocess.run([
            sys.executable, str(project / "scripts" / "apply_hierarchical_query_classifier.py"),
            "--cache", str(val_path), "--predictions", str(prediction_path),
            "--checkpoint", str(heads / "H2_seed42.pt"),
            "--output", str(output_path), "--device", "cpu", "--overwrite",
        ], check=True)
        applied = json.loads(output_path.read_text())
        scores = [row["score"] for row in applied["images"][0]["predictions"]]
        assert scores == [0.25] * len(scores)
        assert applied["metadata"]["query_classifier"]["score_mode"] == \
            "unchanged_objectness"
    print("HIERARCHICAL_QUERY_ABLATION_SMOKE_OK")


if __name__ == "__main__":
    main()
