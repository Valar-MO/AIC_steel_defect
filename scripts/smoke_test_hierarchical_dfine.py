#!/usr/bin/env python3
"""Validate registration, factorized losses, postprocessing and checkpoint transfer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dfine-dir", type=Path, default=Path("/root/autodl-tmp/D-FINE"))
    parser.add_argument("--config", default=
        "configs/dfine/custom/objects365/dfine_hgnetv2_l_hierarchical_aic.yml")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.dfine_dir.resolve()))
    from src.core import YAMLConfig
    from src.zoo.dfine.hierarchical_decoder import HierarchicalScoreHead
    from src.zoo.dfine.hierarchical_matcher import HierarchicalHungarianMatcher
    from src.zoo.dfine.hierarchical_criterion import HierarchicalDFINECriterion
    from src.zoo.dfine.hierarchical_postprocessor import HierarchicalDFINEPostProcessor

    head = HierarchicalScoreHead(16, 9)
    old_weight, old_bias = torch.randn(1, 16), torch.randn(1)
    head.load_state_dict({"weight": old_weight, "bias": old_bias}, strict=False)
    torch.testing.assert_close(head.weight, old_weight)
    assert head(torch.randn(2, 5, 16)).shape == (2, 5, 10)

    matcher = HierarchicalHungarianMatcher(
        {"cost_objectness": 2, "cost_bbox": 5, "cost_giou": 2})
    outputs = {
        "pred_objectness_logits": torch.tensor(
            [[[4.0], [-4.0], [0.0]]], requires_grad=True),
        "pred_class_logits": torch.randn(1, 3, 9, requires_grad=True),
        "pred_boxes": torch.tensor([[[.5,.5,.2,.2], [.1,.1,.1,.1], [.8,.8,.1,.1]]]),
    }
    outputs["pred_logits"] = outputs["pred_objectness_logits"]
    targets = [{"labels": torch.tensor([7]), "boxes": torch.tensor([[.5,.5,.2,.2]])}]
    indices = matcher(outputs, targets)["indices"]
    assert indices[0][0].tolist() == [0]
    criterion = HierarchicalDFINECriterion(
        matcher, {"loss_defectness":1,"loss_class":1}, ["hierarchical"], num_classes=9)
    losses = criterion.loss_hierarchical(outputs, targets, indices, 1)
    assert set(losses) == {"loss_defectness", "loss_class"}
    sum(losses.values()).backward()

    post = HierarchicalDFINEPostProcessor(num_classes=9, num_top_queries=3)
    result = post(outputs, torch.tensor([[100., 200.]]))[0]
    assert result["labels"].shape == result["scores"].shape == (3,)

    config_path = Path(args.config)
    if not config_path.is_absolute(): config_path = args.dfine_dir/config_path
    config = YAMLConfig(str(config_path))
    config.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = config.model
    assert model.decoder.__class__.__name__ == "HierarchicalDFINETransformer"
    assert config.criterion.__class__.__name__ == "HierarchicalDFINECriterion"
    assert config.postprocessor.__class__.__name__ == "HierarchicalDFINEPostProcessor"
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        old = payload["ema"]["module"] if "ema" in payload else payload["model"]
        current = model.state_dict(); transferable = []
        for index in range(len(model.decoder.dec_score_head)):
            key = f"decoder.dec_score_head.{index}.weight"
            if key in old and key in current and old[key].shape == current[key].shape:
                transferable.append(key)
        if not transferable:
            raise AssertionError("Legacy one-class score heads do not transfer to defectness")
        print(f"TRANSFERABLE_OBJECTNESS_HEADS={len(transferable)}")
    print("HIERARCHICAL_DFINE_SMOKE_OK")


if __name__ == "__main__":
    main()
