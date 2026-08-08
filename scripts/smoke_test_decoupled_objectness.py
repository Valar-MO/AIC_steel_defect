#!/usr/bin/env python3
"""Validate dual-objectness initialization, IoU masks, gradients and score fusion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dfine-dir", type=Path, default=Path("/root/autodl-tmp/D-FINE"))
    parser.add_argument("--config", default=
        "configs/dfine/custom/objects365/"
        "dfine_hgnetv2_l_hierarchical_objectness_dual_v1.yml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.dfine_dir.resolve()))

    from src.core import YAMLConfig
    from src.zoo.dfine.hierarchical_criterion import HierarchicalDFINECriterion
    from src.zoo.dfine.hierarchical_decoder import HierarchicalScoreHead
    from src.zoo.dfine.hierarchical_matcher import HierarchicalHungarianMatcher
    from src.zoo.dfine.hierarchical_postprocessor import HierarchicalDFINEPostProcessor

    torch.manual_seed(7)
    head = HierarchicalScoreHead(
        256, 9, objectness_head_type="dual_mlp",
        objectness_hidden_dim=128, objectness_dropout=0.0,
        defectness_residual_sign=-1.0, quality_prior_probability=0.99,
    )
    features = torch.randn(2, 4, 256)
    baseline = torch.nn.functional.linear(features, head.weight, head.bias)
    output = head(features)
    initial_defectness_error = (output[..., :1] - baseline).abs().max()
    initial_quality = head.quality_history[-1].sigmoid()
    if initial_defectness_error != 0:
        raise AssertionError("Zero residual must exactly preserve baseline defectness")
    output[..., :1].sum().backward()
    residual_final_gradient = head.defectness_mlp[-1].weight.grad.norm()
    if residual_final_gradient <= 0:
        raise AssertionError("Zero-initialized residual projection has no gradient")

    matcher = HierarchicalHungarianMatcher(
        {"cost_objectness": 2, "cost_bbox": 5, "cost_giou": 2}
    )
    criterion = HierarchicalDFINECriterion(
        matcher,
        {
            "loss_defectness": 1.0,
            "loss_quality": 1.0,
            "loss_quality_duplicate_rank": 0.25,
        },
        ["hierarchical"],
        num_classes=9,
        decoupled_objectness=True,
        defect_positive_iou=0.5,
        defect_background_iou=0.1,
        quality_duplicate_topk=3,
        quality_duplicate_iou=0.5,
        quality_duplicate_rank_margin=0.25,
    )
    defect_logits = torch.zeros(1, 4, 1, requires_grad=True)
    quality_logits = torch.zeros(1, 4, 1, requires_grad=True)
    outputs = {
        "pred_objectness_logits": defect_logits,
        "pred_quality_logits": quality_logits,
        "quality_duplicate_ranking": True,
        "pred_logits": defect_logits,
        "pred_class_logits": torch.zeros(1, 4, 9),
        # q0 positive IoU=1; q1 duplicate positive; q2 IoU=0.25 ignore; q3 background.
        "pred_boxes": torch.tensor([[
            [.50, .50, .20, .20],
            [.50, .50, .22, .22],
            [.50, .50, .40, .40],
            [.90, .90, .05, .05],
        ]]),
    }
    targets = [{
        "labels": torch.tensor([0]),
        "boxes": torch.tensor([[.50, .50, .20, .20]]),
    }]
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    losses = criterion.loss_hierarchical(outputs, targets, indices, 1)
    defect_gradient = torch.autograd.grad(
        losses["loss_defectness"], defect_logits, retain_graph=True
    )[0].squeeze()
    quality_gradient = torch.autograd.grad(
        losses["loss_quality"], quality_logits, retain_graph=True
    )[0].squeeze()
    duplicate_gradient = torch.autograd.grad(
        losses["loss_quality_duplicate_rank"], quality_logits
    )[0].squeeze()
    if not (
        defect_gradient[0] < 0 and defect_gradient[1] < 0
        and defect_gradient[2] == 0 and defect_gradient[3] > 0
    ):
        raise AssertionError(f"Unexpected defect gradients: {defect_gradient}")
    if not (
        quality_gradient[0] < 0 and quality_gradient[1] < 0
        and quality_gradient[2] == 0 and quality_gradient[3] == 0
    ):
        raise AssertionError(f"Unexpected quality gradients: {quality_gradient}")
    if not (duplicate_gradient[0] < 0 and duplicate_gradient[1] > 0):
        raise AssertionError(f"Unexpected duplicate gradients: {duplicate_gradient}")

    postprocessor = HierarchicalDFINEPostProcessor(score_mode="decoupled")
    post_outputs = {
        "pred_objectness_logits": torch.tensor([[[0.0], [2.0]]]),
        "pred_quality_logits": torch.tensor([[[0.0], [-2.0]]]),
        "pred_class_logits": torch.tensor([[
            [8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]]),
        "pred_boxes": torch.tensor([[
            [.5, .5, .2, .2], [.5, .5, .2, .2],
        ]]),
    }
    processed = postprocessor(post_outputs, torch.tensor([[100.0, 100.0]]))[0]
    expected_scores = (
        post_outputs["pred_objectness_logits"].sigmoid().squeeze(-1)
        * post_outputs["pred_quality_logits"].sigmoid().squeeze(-1)
    )[0]
    if not torch.allclose(processed["scores"], expected_scores):
        raise AssertionError("Postprocessor did not use defectness * quality")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = args.dfine_dir / config_path
    cfg = YAMLConfig(str(config_path), tuning=str(args.checkpoint))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model
    report = model.decoupled_objectness_parameter_report()
    optimizer_ids = {
        id(parameter)
        for group in cfg.optimizer.param_groups
        for parameter in group["params"]
    }
    trainable_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    if optimizer_ids != trainable_ids:
        raise RuntimeError("Optimizer does not exactly cover trainable parameters")
    forbidden = [
        name for name in report["trainable_names"]
        if any(token in name for token in (
            "class_weight", "class_bias", "bbox_head", "backbone.",
            "encoder.", "adaptformer",
        ))
    ]
    if forbidden:
        raise RuntimeError({"forbidden": forbidden, "report": report})

    print(json.dumps({
        "initial_defectness_max_error": float(initial_defectness_error),
        "initial_quality_probability": float(initial_quality.mean()),
        "residual_final_gradient_norm": float(residual_final_gradient),
        "losses": {name: float(value.detach()) for name, value in losses.items()},
        "defect_gradient": defect_gradient.tolist(),
        "quality_gradient": quality_gradient.tolist(),
        "quality_duplicate_gradient": duplicate_gradient.tolist(),
        "postprocessor_scores": processed["scores"].tolist(),
        "parameter_report": report,
        "optimizer_covers_trainable_exactly": True,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
