#!/usr/bin/env python3
"""Install tracked Hierarchical D-FINE modules/configs into an official checkout."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


MODULES = (
    "hierarchical_decoder.py", "hierarchical_matcher.py",
    "hierarchical_criterion.py", "hierarchical_postprocessor.py",
)
IMPORTS = (
    "from .hierarchical_decoder import HierarchicalDFINETransformer",
    "from .hierarchical_matcher import HierarchicalHungarianMatcher",
    "from .hierarchical_criterion import HierarchicalDFINECriterion",
    "from .hierarchical_postprocessor import HierarchicalDFINEPostProcessor",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dfine-dir", type=Path, default=Path("/root/autodl-tmp/D-FINE"))
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    package = args.dfine_dir / "src" / "zoo" / "dfine"
    if not (package / "dfine_decoder.py").is_file():
        raise FileNotFoundError(f"Not an official D-FINE checkout: {args.dfine_dir}")
    source = args.project_dir / "integrations" / "dfine_hierarchical"
    for name in MODULES:
        shutil.copy2(source / name, package / name)
        print(f"installed {package/name}")
    init_path = package / "__init__.py"
    text = init_path.read_text(encoding="utf-8")
    missing = [line for line in IMPORTS if line not in text]
    if missing:
        init_path.write_text(text.rstrip()+"\n"+"\n".join(missing)+"\n", encoding="utf-8")
    config_source = args.project_dir / "configs" / "dfine"
    config_targets = {
        "aic_hierarchical_detection.yml": args.dfine_dir/"configs"/"dataset"/"aic_hierarchical_detection.yml",
        "dfine_hgnetv2_l_hierarchical_aic.yml": args.dfine_dir/"configs"/"dfine"/"custom"/"objects365"/"dfine_hgnetv2_l_hierarchical_aic.yml",
    }
    for name, target in config_targets.items():
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(config_source/name, target)
        print(f"installed {target}")
    print("config: configs/dfine/custom/objects365/dfine_hgnetv2_l_hierarchical_aic.yml")


if __name__ == "__main__":
    main()

