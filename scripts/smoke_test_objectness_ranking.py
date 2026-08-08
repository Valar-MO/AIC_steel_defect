#!/usr/bin/env python3
"""Validate ranking losses and the strict objectness-only trainable set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dfine-dir", type=Path, default=Path("/root/autodl-tmp/D-FINE"))
    parser.add_argument("--config", default=
        "configs/dfine/custom/objects365/dfine_hgnetv2_l_hierarchical_objectness_rank.yml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.dfine_dir.resolve()))
    from src.core import YAMLConfig
    from src.zoo.dfine.hierarchical_criterion import HierarchicalDFINECriterion
    from src.zoo.dfine.hierarchical_matcher import HierarchicalHungarianMatcher

    matcher = HierarchicalHungarianMatcher(
        {"cost_objectness": 2, "cost_bbox": 5, "cost_giou": 2}
    )
    criterion = HierarchicalDFINECriterion(
        matcher,
        {"loss_defectness": 0.25, "loss_hard_background_rank": 1.0,
         "loss_duplicate_rank": 1.0},
        ["hierarchical"], num_classes=9,
        hard_background_topk=2, hard_background_max_iou=0.1,
        duplicate_topk=2, duplicate_iou=0.5, rank_positive_min_iou=0.5,
    )
    logits = torch.tensor([[[0.0], [2.0], [1.0], [-2.0]]], requires_grad=True)
    outputs = {
        "pred_objectness_logits": logits,
        "pred_logits": logits,
        "pred_class_logits": torch.zeros(1, 4, 9),
        "pred_boxes": torch.tensor([[[.5, .5, .2, .2], [.1, .1, .1, .1],
                                     [.5, .5, .2, .2], [.9, .9, .05, .05]]]),
    }
    targets = [{"labels": torch.tensor([0]),
                "boxes": torch.tensor([[.5, .5, .2, .2]])}]
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    losses = criterion.loss_hierarchical(outputs, targets, indices, 1)
    assert losses["loss_hard_background_rank"] > 0
    assert losses["loss_duplicate_rank"] > 0
    (losses["loss_hard_background_rank"] + losses["loss_duplicate_rank"]).backward()
    assert logits.grad[0, 0, 0] < 0
    assert logits.grad[0, 1, 0] > 0
    assert logits.grad[0, 2, 0] > 0

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = args.dfine_dir / config_path
    cfg = YAMLConfig(str(config_path), tuning=str(args.checkpoint))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model
    report = model.objectness_parameter_report()
    forbidden = [name for name in report["trainable_names"] if any(token in name for token in (
        "class_weight", "class_bias", "bbox_head", "backbone.", "encoder.", "adaptformer"
    ))]
    expected = [name for name in report["trainable_names"]
                if name.endswith((".weight", ".bias")) and "dec_score_head" in name]
    if forbidden or len(expected) != 12 or report["trainable_parameters"] != 1542:
        raise RuntimeError({"report": report, "forbidden": forbidden, "expected": expected})
    optimizer_ids = {id(parameter) for group in cfg.optimizer.param_groups
                     for parameter in group["params"]}
    trainable_ids = {id(parameter) for parameter in model.parameters()
                     if parameter.requires_grad}
    if optimizer_ids != trainable_ids:
        raise RuntimeError("Optimizer does not exactly cover trainable parameters")
    print(json.dumps({
        "synthetic_losses": {key: float(value.detach()) for key, value in losses.items()},
        "positive_gradient": float(logits.grad[0, 0, 0]),
        "hard_background_gradient": float(logits.grad[0, 1, 0]),
        "duplicate_gradient": float(logits.grad[0, 2, 0]),
        "parameter_report": report,
        "optimizer_covers_trainable_exactly": True,
    }, indent=2))


if __name__ == "__main__":
    main()
