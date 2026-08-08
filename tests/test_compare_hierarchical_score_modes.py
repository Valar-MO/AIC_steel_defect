import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from compare_hierarchical_score_modes import (  # noqa: E402
    Prediction,
    match_and_decompose,
    score_prediction,
)


def prediction(index, image, class_id, name, box, objectness=0.8, confidence=0.5):
    return Prediction(
        prediction_id=index, image_id=image, class_id=class_id,
        class_name=name, box=box, objectness=objectness,
        class_confidence=confidence,
        score=objectness,
    )


def test_score_modes_only_change_score_formula():
    raw = {"score": 0.7, "objectness": 0.8, "class_probability": 0.25}
    assert score_prediction(raw, "objectness") == 0.8
    assert score_prediction(raw, "joint") == 0.2
    assert score_prediction({"score": 0.7, "objectness": 0.8}, "objectness") == 0.8


def test_fp_decomposition_is_mutually_exclusive_and_reconciles():
    gt = defaultdict(lambda: defaultdict(list))
    gt[0]["a.jpg"] = [(0.0, 0.0, 10.0, 10.0)]
    gt[1]["a.jpg"] = [(20.0, 0.0, 30.0, 10.0)]
    predictions = [
        prediction(0, "a.jpg", 0, "c0", (0.0, 0.0, 10.0, 10.0), 0.9),  # TP
        prediction(1, "a.jpg", 0, "c0", (0.0, 0.0, 10.0, 10.0), 0.8),  # duplicate
        prediction(2, "a.jpg", 0, "c0", (20.0, 0.0, 30.0, 10.0), 0.7),  # wrong class
        prediction(3, "a.jpg", 0, "c0", (7.0, 0.0, 17.0, 10.0), 0.6),  # localization
        prediction(4, "a.jpg", 0, "c0", (40.0, 0.0, 50.0, 10.0), 0.5),  # background
    ]
    summary, rows = match_and_decompose(
        predictions, gt, ["c0", "c1"], score_threshold=0.1,
        match_iou=0.5, background_iou=0.1,
    )
    assert summary["tp"] == 1
    assert summary["fp"] == 4
    assert summary["prediction_count"] == 5
    assert {row["outcome"] for row in rows} == {
        "tp", "duplicate", "wrong_class", "localization", "background"
    }
    assert {name: value["count"] for name, value in summary["fp_buckets"].items()} == {
        "background": 1, "duplicate": 1, "wrong_class": 1, "localization": 1,
    }
