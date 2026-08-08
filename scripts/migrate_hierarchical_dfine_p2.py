#!/usr/bin/env python3
"""Migrate a trained 3-level Hierarchical D-FINE checkpoint to P2-P5."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dfine-dir", type=Path, default=Path("/root/autodl-tmp/D-FINE"))
    parser.add_argument("--config", default=
                        "configs/dfine/custom/objects365/"
                        "dfine_hgnetv2_l_hierarchical_p2_e2.yml")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--new-level-attention-bias", type=float, default=-4.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def expand_cross_attention(
    key: str,
    source: torch.Tensor,
    target: torch.Tensor,
    heads: int,
    old_points: int,
    new_points: int,
    prefix_points: int,
    new_level_attention_bias: float,
) -> torch.Tensor | None:
    if "cross_attn.sampling_offsets.weight" in key:
        old = source.reshape(heads, old_points, 2, source.shape[-1])
        new = target.clone().reshape(heads, new_points, 2, target.shape[-1])
        new[:, prefix_points:prefix_points + old_points].copy_(old)
        return new.reshape_as(target)
    if "cross_attn.sampling_offsets.bias" in key:
        old = source.reshape(heads, old_points, 2)
        new = target.clone().reshape(heads, new_points, 2)
        new[:, prefix_points:prefix_points + old_points].copy_(old)
        return new.reshape_as(target)
    if "cross_attn.attention_weights.weight" in key:
        old = source.reshape(heads, old_points, source.shape[-1])
        new = target.clone().reshape(heads, new_points, target.shape[-1])
        new[:, prefix_points:prefix_points + old_points].copy_(old)
        return new.reshape_as(target)
    if "cross_attn.attention_weights.bias" in key:
        old = source.reshape(heads, old_points)
        new = target.clone().reshape(heads, new_points)
        new[:, :prefix_points].fill_(new_level_attention_bias)
        new[:, prefix_points:prefix_points + old_points].copy_(old)
        return new.reshape_as(target)
    return None


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")
    sys.path.insert(0, str(args.dfine_dir.resolve()))
    from src.core import YAMLConfig  # pylint: disable=import-outside-toplevel

    config = Path(args.config)
    if not config.is_absolute():
        config = args.dfine_dir / config
    cfg = YAMLConfig(str(config))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model
    target_state = model.state_dict()
    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    source_state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]

    attention = model.decoder.decoder.layers[0].cross_attn
    heads = int(attention.num_heads)
    new_layout = list(attention.num_points_list)
    old_layout = [3, 6, 3]
    if new_layout[1:] != old_layout:
        raise ValueError({"expected_old_suffix": old_layout, "new_layout": new_layout})
    prefix_points = new_layout[0]
    old_points, new_points = sum(old_layout), sum(new_layout)

    migrated = {key: value.clone() for key, value in target_state.items()}
    exact, expanded, initialized, incompatible = [], [], [], []
    for key, target in target_state.items():
        source = source_state.get(key)
        # The quota selector starts from the trained P3-P5 encoder proposal
        # score head, not from a random local-objectness calibration.
        if source is None and key.startswith("decoder.p2_query_score_head."):
            source = source_state.get(key.replace("decoder.p2_query_score_head.",
                                                  "decoder.enc_score_head."))
        if source is None:
            initialized.append(key)
            continue
        if source.shape == target.shape:
            migrated[key] = source.clone()
            exact.append(key)
            continue
        value = expand_cross_attention(
            key, source, target, heads, old_points, new_points, prefix_points,
            args.new_level_attention_bias,
        )
        if value is None:
            incompatible.append({
                "key": key,
                "source_shape": list(source.shape),
                "target_shape": list(target.shape),
            })
        else:
            migrated[key] = value
            expanded.append(key)

    allowed_incompatible = {"decoder.anchors", "decoder.valid_mask"}
    unexpected = [
        row for row in incompatible
        if row["key"] not in allowed_incompatible
        and not row["key"].endswith("cross_attn.num_points_scale")
    ]
    if unexpected:
        raise RuntimeError({"unexpected_incompatible": unexpected})
    old_only = sorted(set(source_state) - set(target_state))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": migrated,
        "ema": {"module": migrated},
        "epoch": checkpoint.get("epoch"),
        "p2_migration": {
            "source": str(args.source),
            "new_level_attention_bias": args.new_level_attention_bias,
        },
    }, args.output)
    report = {
        "source": str(args.source.resolve()),
        "output": str(args.output.resolve()),
        "exact_tensor_count": len(exact),
        "expanded_attention_tensors": expanded,
        "new_initialized_tensors": initialized,
        "incompatible_regenerated_buffers": incompatible,
        "old_only_tensors": old_only,
        "old_points": old_layout,
        "new_points": new_layout,
        "query_select_start_level": model.decoder.query_select_start_level,
        "anchor_grid_size": model.decoder.anchor_grid_size,
    }
    report_path = args.output.with_suffix(".migration.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
