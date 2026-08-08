import importlib.util
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "hierarchical_query_classifier.py"
SPEC = importlib.util.spec_from_file_location("hierarchical_query_classifier", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_balanced_weights_give_each_class_equal_mass():
    targets = torch.tensor([0, 0, 0, 1, 2, 2])
    weights = module.balanced_sample_weights(targets, 3)
    mass = torch.stack([weights[targets == index].sum() for index in range(3)])
    assert torch.allclose(mass, torch.full_like(mass, 1 / 3))


def test_margin_loss_is_zero_when_true_class_has_enough_margin():
    logits = torch.tensor([[2.0, 0.5, -1.0], [0.0, -0.5, 1.0]])
    targets = torch.tensor([0, 2])
    assert module.hardest_class_margin_loss(logits, targets, margin=0.2).item() == 0.0


def test_margin_loss_penalizes_hardest_wrong_class():
    logits = torch.tensor([[0.4, 0.5, -1.0]])
    targets = torch.tensor([0])
    loss = module.hardest_class_margin_loss(logits, targets, margin=0.2)
    assert torch.isclose(loss, torch.tensor(0.3))


def test_classifier_initialization_matches_detector_linear_head():
    model = module.HierarchicalQueryClassifier(4, 3)
    weight = torch.randn(3, 4)
    bias = torch.randn(3)
    model.initialize_from_detector(weight, bias)
    features = torch.randn(5, 4)
    expected = torch.nn.functional.linear(features, weight, bias)
    assert torch.allclose(model(features), expected)


def test_metrics_capture_named_confusions():
    classes = ["a", "b", "c"]
    targets = torch.tensor([0, 0, 1, 2])
    logits = torch.tensor([
        [3.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 2.0],
    ])
    metrics = module.classification_metrics(logits, targets, classes)
    assert metrics["accuracy"] == 0.75
    assert {"true": "a", "predicted": "b", "count": 1} in metrics["top_confusions"]


def test_selection_ranker_starts_as_identity_residual():
    model = module.SelectionResidualRanker(4, hidden_dim=8, residual_scale=0.25, dropout=0.0)
    features = torch.randn(6, 4)
    base_logits = torch.randn(6)
    assert torch.allclose(model(features, base_logits), base_logits)


def test_selection_ranking_loss_penalizes_bad_ordering():
    logits = torch.tensor([0.0, 1.0], requires_grad=True)
    targets = torch.tensor([1.0, 0.0])
    tiers = torch.tensor([4, 0])
    weights = torch.ones(2)
    loss, parts = module.selection_ranking_loss(
        logits, logits.detach(), targets, tiers, weights, margin=0.2,
        pairwise_weight=1.0, max_pairs=8
    )
    assert loss.item() > parts["bce"].item()


def test_selection_preserve_loss_penalizes_lowering_positive_too_much():
    final = torch.tensor([0.0, 0.0], requires_grad=True)
    base = torch.tensor([0.2, 0.2])
    targets = torch.tensor([1.0, 0.0])
    tiers = torch.tensor([4, 0])
    weights = torch.ones(2)
    loss, parts = module.selection_ranking_loss(
        final, base, targets, tiers, weights, pairwise_weight=0.0,
        preserve_weight=1.0, preserve_margin=0.05
    )
    assert parts["preserve"].item() > 0
    assert loss.item() > parts["bce"].item()


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"QUERY_CLASSIFIER_TESTS_OK count={len(tests)}")
