#!/usr/bin/env python3
"""Unit checks for the Selection × Defectness × Quality architecture."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


def main():
    project = Path(__file__).resolve().parents[1]
    dfine = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "/root/autodl-tmp/D-FINE"
    )
    sys.path.insert(0, str(dfine.resolve()))
    from src.zoo.dfine.hierarchical_decoder import HierarchicalScoreHead
    from src.zoo.dfine.hierarchical_postprocessor import (
        HierarchicalDFINEPostProcessor,
    )

    torch.manual_seed(42)
    head = HierarchicalScoreHead(
        256, 9, objectness_head_type="selection_gated",
        objectness_dropout=0.0, quality_prior_probability=0.99,
    ).eval()
    features = torch.randn(2, 12, 256)
    base = torch.nn.functional.linear(features, head.weight, head.bias)
    logits = head(features)
    selection = logits[..., :1]
    defectness = head.defectness_history[-1]
    quality = head.quality_history[-1]
    prior = torch.full_like(defectness, 0.99)
    assert torch.equal(selection, base)
    assert torch.allclose(defectness.sigmoid(), prior, atol=1e-6)
    assert torch.allclose(quality.sigmoid(), prior, atol=1e-6)
    fused = selection.sigmoid() * defectness.sigmoid() * quality.sigmoid()
    assert torch.equal(
        selection.sigmoid().argsort(dim=1),
        fused.argsort(dim=1),
    )

    postprocessor = HierarchicalDFINEPostProcessor(
        score_mode="selection_gated"
    )
    outputs = {
        "pred_objectness_logits": selection[:1, :2],
        "pred_defectness_logits": defectness[:1, :2],
        "pred_quality_logits": quality[:1, :2],
        "pred_class_logits": logits[:1, :2, 1:],
        "pred_boxes": torch.tensor([[
            [0.5, 0.5, 0.2, 0.2],
            [0.25, 0.25, 0.1, 0.1],
        ]]),
    }
    result = postprocessor(outputs, torch.tensor([[100.0, 100.0]]))[0]
    expected = (
        selection[:1, :2].sigmoid().squeeze(-1)
        * defectness[:1, :2].sigmoid().squeeze(-1)
        * quality[:1, :2].sigmoid().squeeze(-1)
    )
    assert torch.allclose(
        result["scores"].sort().values, expected.flatten().sort().values
    )

    report = {
        "selection_equals_pretrained_base": True,
        "gate_prior": float(defectness.sigmoid().mean().detach()),
        "initial_fused_scale": float(
            (defectness.sigmoid() * quality.sigmoid()).mean().detach()
        ),
        "initial_ranking_equivalent": True,
        "postprocessor_exact_product": True,
        "project": str(project),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
