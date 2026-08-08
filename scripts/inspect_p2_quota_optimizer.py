#!/usr/bin/env python3
"""Print non-zero optimizer groups for the P2 quota gate configuration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dfine-dir", type=Path, default=Path("/root/autodl-tmp/D-FINE"))
    p.add_argument("--config", required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.dfine_dir.resolve()))
    from src.core import YAMLConfig  # pylint: disable=import-outside-toplevel

    config = Path(args.config)
    if not config.is_absolute(): config = args.dfine_dir / config
    cfg = YAMLConfig(str(config)); cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model, optimizer = cfg.model, cfg.optimizer
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    for index, group in enumerate(optimizer.param_groups):
        lr = float(group["lr"])
        if lr == 0: continue
        group_names = sorted(names[id(parameter)] for parameter in group["params"])
        print({"group": index, "lr": lr, "count": len(group_names), "names": group_names})


if __name__ == "__main__": main()
