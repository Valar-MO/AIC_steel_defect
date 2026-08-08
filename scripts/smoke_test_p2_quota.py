#!/usr/bin/env python3
"""Validate P2 quota registration, checkpoint migration and one forward pass."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dfine-dir", type=Path, default=Path("/root/autodl-tmp/D-FINE"))
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    sys.path.insert(0, str(args.dfine_dir.resolve()))
    from src.core import YAMLConfig  # pylint: disable=import-outside-toplevel

    config = Path(args.config)
    if not config.is_absolute(): config = args.dfine_dir / config
    cfg = YAMLConfig(str(config), resume=str(args.checkpoint))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    model.load_state_dict(state, strict=True)
    device = torch.device(args.device)
    model.to(device).eval()
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16,
                                                 enabled=device.type == "cuda"):
        output = model(torch.zeros((1, 3, 1024, 1024), device=device))
    quota = model.decoder.p2_query_quota
    selected = model.decoder._last_p2_query_indices
    assert quota == 50, quota
    assert selected is not None and selected.shape == (1, quota), selected.shape if selected is not None else None
    assert output["pred_boxes"].shape[:2] == (1, 300), output["pred_boxes"].shape
    assert output["pred_objectness_logits"].shape[:2] == (1, 300)
    print({"p2_quota": quota, "selected_shape": list(selected.shape),
           "pred_boxes": list(output["pred_boxes"].shape)})


if __name__ == "__main__": main()
