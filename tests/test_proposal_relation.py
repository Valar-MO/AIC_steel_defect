import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location("proposal_relation_test", ROOT / "scripts" / "proposal_relation.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_zero_initialization_exactly_preserves_baseline():
    torch = pytest.importorskip("torch")
    module = load_module()
    model = module.ProposalRelationTransformer(feature_dim=16, auxiliary_dim=19,
                                                dimension=32, heads=4, layers=2, dropout=0.0)
    features = torch.randn(2, 5, 16)
    auxiliary = torch.randn(2, 5, 19)
    auxiliary[..., :4] = torch.tensor([0.1, 0.2, 0.5, 0.7])
    objectness = torch.randn(2, 5)
    classes = torch.randn(2, 5, 9)
    output = model(features, auxiliary, objectness, classes)
    torch.testing.assert_close(output["objectness_logits"], objectness)
    torch.testing.assert_close(output["class_logits"], classes)


def test_pairwise_geometry_detects_overlap():
    torch = pytest.importorskip("torch")
    module = load_module()
    boxes = torch.tensor([[[0.0, 0.0, 0.5, 0.5], [0.25, 0.25, 0.75, 0.75],
                           [0.75, 0.75, 1.0, 1.0]]])
    relation = module.pairwise_geometry(boxes)
    assert relation.shape == (1, 3, 3, 7)
    assert float(relation[0, 0, 1, 0]) > 0
    assert float(relation[0, 0, 2, 0]) == 0

