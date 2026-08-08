import importlib.util
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_metadata_features_are_finite_and_fixed_width():
    pytest.importorskip("torch")
    module = load_module("dfine_crop_classifier_test", "scripts/dfine_crop_classifier.py")
    features = module.metadata_features(
        score=0.8, box=[100, 50, 300, 150], width=1000, height=500,
        source="tile", touches_internal_border=True,
    )
    assert features.shape == (module.META_DIM,)
    assert all(math.isfinite(float(value)) for value in features)
    assert float(features[-2]) == 1.0
    assert float(features[-1]) == 1.0


def test_audit_selection_ignores_ambiguous_and_caps_duplicates():
    module = load_module("build_dfine_crop_dataset_test", "scripts/build_dfine_crop_dataset.py")
    class Args:
        min_score = 0.001
        max_positive_per_gt = 2
        train_background_ratio = 1.0
        max_train_background_per_image = 10
        max_val_background_per_image = 10
        seed = 42
    base = {"image_id": "a.jpg", "pred_index": "0", "score": "0.9", "best_iou": "0.8",
            "reference_gt_id": "1", "reference_class": "c0"}
    rows = [
        {**base, "outcome": "tp"},
        {**base, "pred_index": "1", "score": "0.8", "outcome": "duplicate"},
        {**base, "pred_index": "2", "score": "0.7", "outcome": "duplicate"},
        {**base, "pred_index": "3", "outcome": "near_miss"},
        {**base, "pred_index": "4", "reference_gt_id": "", "reference_class": "", "outcome": "background"},
        {**base, "pred_index": "5", "reference_gt_id": "", "reference_class": "", "outcome": "background"},
        {**base, "pred_index": "6", "reference_gt_id": "", "reference_class": "", "outcome": "background"},
    ]
    selected, stats = module.select_rows(rows, "train", ["c0"], Args())
    assert stats["positive"] == 2
    assert stats["background"] == 2
    assert stats["ignored_near_miss"] == 1
    assert stats["ignored_excess_duplicate"] == 1
    assert {row["outcome"] for row in selected} == {"tp", "duplicate", "background"}
