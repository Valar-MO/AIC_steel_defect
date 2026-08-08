#!/usr/bin/env python3
"""One-batch interface test for the DINOv2 dual-head crop classifier."""

import argparse
from pathlib import Path

import torch

from dfine_crop_classifier import DFineDualHeadClassifier, META_DIM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("/root/autodl-tmp/weights/dinov2-base"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--architecture", choices=("legacy", "token_fusion_v2", "residual_token_v21"),
                        default="token_fusion_v2")
    parser.add_argument("--adapter-dim", type=int, default=128)
    parser.add_argument("--checkpoint", type=Path,
                        help="Optionally verify strict checkpoint loading and infer its architecture.")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu") if args.checkpoint else None
    config = checkpoint.get("config", {}) if checkpoint else {}
    architecture = str(config.get("architecture", args.architecture if not checkpoint else "legacy"))
    model = DFineDualHeadClassifier(args.weights, 9, int(config.get("metadata_hidden", 128)),
                                    0.0, architecture, int(config.get("adapter_dim", args.adapter_dim)))
    if checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().to(device)
    image = torch.randn(1, 3, args.image_size, args.image_size, device=device)
    metadata = torch.zeros(1, META_DIM, device=device)
    with torch.inference_mode():
        output = model(image, image, metadata)
    assert output["objectness_logits"].shape == (1,)
    assert output["class_logits"].shape == (1, 9)
    print("SMOKE_OK", architecture, tuple(output["objectness_logits"].shape),
          tuple(output["class_logits"].shape), sum(p.numel() for p in model.parameters()))


if __name__ == "__main__":
    main()
