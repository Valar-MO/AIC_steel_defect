#!/usr/bin/env python3
"""Install tracked Hierarchical D-FINE modules/configs into an official checkout."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


MODULES = (
    "hierarchical_decoder.py", "hierarchical_matcher.py",
    "hierarchical_criterion.py", "hierarchical_postprocessor.py",
    "hierarchical_peft_model.py",
    "p2_hybrid_encoder.py",
)
IMPORTS = (
    "from .hierarchical_decoder import HierarchicalDFINETransformer",
    "from .hierarchical_matcher import HierarchicalHungarianMatcher",
    "from .hierarchical_criterion import HierarchicalDFINECriterion",
    "from .hierarchical_postprocessor import HierarchicalDFINEPostProcessor",
    "from .hierarchical_peft_model import (HierarchicalPEFTDFINE, "
    "HierarchicalObjectnessRankingDFINE, "
    "HierarchicalDecoupledObjectnessDFINE, "
    "HierarchicalSelectionGatedDFINE)",
    "from .p2_hybrid_encoder import P2HybridEncoder",
)
DATASET_MODULES = ("gt_aware_mix_dataset.py", "official_view_mix_dataset.py", "official_view_coco_dataset.py")
DATASET_IMPORTS = (
    "from .gt_aware_mix_dataset import GTAwareBatchMixDataset",
    "from .official_view_mix_dataset import OfficialViewBatchMixDataset",
    "from .official_view_coco_dataset import OfficialViewCocoDetection",
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
    dataset_package = args.dfine_dir / "src" / "data" / "dataset"
    for name in DATASET_MODULES:
        shutil.copy2(source / name, dataset_package / name)
        print(f"installed {dataset_package/name}")
    dataset_init = dataset_package / "__init__.py"
    dataset_text = dataset_init.read_text(encoding="utf-8")
    dataset_missing = [line for line in DATASET_IMPORTS if line not in dataset_text]
    if dataset_missing:
        dataset_init.write_text(
            dataset_text.rstrip()+"\n"+"\n".join(dataset_missing)+"\n",
            encoding="utf-8",
        )
    config_source = args.project_dir / "configs" / "dfine"
    config_targets = {
        "aic_hierarchical_detection.yml": args.dfine_dir/"configs"/"dataset"/"aic_hierarchical_detection.yml",
        "aic_hierarchical_joint_e1_detection.yml": args.dfine_dir/"configs"/"dataset"/"aic_hierarchical_joint_e1_detection.yml",
        "aic_hierarchical_joint_e1_fresh_detection.yml": args.dfine_dir/"configs"/"dataset"/"aic_hierarchical_joint_e1_fresh_detection.yml",
        "aic_hierarchical_gtaware_detection.yml": args.dfine_dir/"configs"/"dataset"/"aic_hierarchical_gtaware_detection.yml",
        "aic_hierarchical_gtaware_mild_detection.yml": args.dfine_dir/"configs"/"dataset"/"aic_hierarchical_gtaware_mild_detection.yml",
        "dfine_hgnetv2_l_hierarchical_aic.yml": args.dfine_dir/"configs"/"dfine"/"custom"/"objects365"/"dfine_hgnetv2_l_hierarchical_aic.yml",
        "dfine_hgnetv2_l_hierarchical_joint_e1.yml": args.dfine_dir/"configs"/"dfine"/"custom"/"objects365"/"dfine_hgnetv2_l_hierarchical_joint_e1.yml",
        "dfine_hgnetv2_l_hierarchical_joint_e1_fresh.yml": args.dfine_dir/"configs"/"dfine"/"custom"/"objects365"/"dfine_hgnetv2_l_hierarchical_joint_e1_fresh.yml",
        "dfine_hgnetv2_l_hierarchical_joint_s1a.yml": args.dfine_dir/"configs"/"dfine"/"custom"/"objects365"/"dfine_hgnetv2_l_hierarchical_joint_s1a.yml",
        "dfine_hgnetv2_l_hierarchical_peft_aic.yml": args.dfine_dir/"configs"/"dfine"/"custom"/"objects365"/"dfine_hgnetv2_l_hierarchical_peft_aic.yml",
        "dfine_hgnetv2_l_hierarchical_gtaware_e1.yml": args.dfine_dir/"configs"/"dfine"/"custom"/"objects365"/"dfine_hgnetv2_l_hierarchical_gtaware_e1.yml",
        "dfine_hgnetv2_l_hierarchical_gtaware_e1_1.yml": args.dfine_dir/"configs"/"dfine"/"custom"/"objects365"/"dfine_hgnetv2_l_hierarchical_gtaware_e1_1.yml",
        "dfine_hgnetv2_l_hierarchical_p2_e2.yml": args.dfine_dir/"configs"/"dfine"/"custom"/"objects365"/"dfine_hgnetv2_l_hierarchical_p2_e2.yml",
        "dfine_hgnetv2_l_hierarchical_p2_quota_e1.yml": args.dfine_dir/"configs"/"dfine"/"custom"/"objects365"/"dfine_hgnetv2_l_hierarchical_p2_quota_e1.yml",
        "dfine_hgnetv2_l_hierarchical_objectness_rank.yml": args.dfine_dir/"configs"/"dfine"/"custom"/"objects365"/"dfine_hgnetv2_l_hierarchical_objectness_rank.yml",
        "dfine_hgnetv2_l_hierarchical_selection_native_s1a.yml": args.dfine_dir/"configs"/"dfine"/"custom"/"objects365"/"dfine_hgnetv2_l_hierarchical_selection_native_s1a.yml",
        "dfine_hgnetv2_l_hierarchical_selection_native_s1b.yml": args.dfine_dir/"configs"/"dfine"/"custom"/"objects365"/"dfine_hgnetv2_l_hierarchical_selection_native_s1b.yml",
        "dfine_hgnetv2_l_hierarchical_objectness_dual_v1.yml": args.dfine_dir/"configs"/"dfine"/"custom"/"objects365"/"dfine_hgnetv2_l_hierarchical_objectness_dual_v1.yml",
        "dfine_hgnetv2_l_hierarchical_selection_gated_v1.yml": args.dfine_dir/"configs"/"dfine"/"custom"/"objects365"/"dfine_hgnetv2_l_hierarchical_selection_gated_v1.yml",
    }
    for name, target in config_targets.items():
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(config_source/name, target)
        print(f"installed {target}")
    print("config: configs/dfine/custom/objects365/dfine_hgnetv2_l_hierarchical_aic.yml")


if __name__ == "__main__":
    main()
