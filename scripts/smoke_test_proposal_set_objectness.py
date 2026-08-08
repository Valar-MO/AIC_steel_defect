#!/usr/bin/env python3
"""Dependency-free zero-residual smoke test for proposal-set V2."""

import torch

from proposal_relation import ProposalRelationTransformer
from proposal_set_objectness import ProposalSetObjectness


relation = ProposalRelationTransformer(16, 19, 9, 32, 4, 2, 0.0)
model = ProposalSetObjectness(relation, decision_hidden=16, dropout=0.0)
features = torch.randn(2, 5, 16)
auxiliary = torch.randn(2, 5, 19)
auxiliary[:, :, :4] = torch.tensor([0.1, 0.2, 0.5, 0.7])
objectness = torch.randn(2, 5)
classes = torch.randn(2, 5, 9)
output = model(features, auxiliary, objectness, classes)
assert torch.equal(output["objectness_logits"], objectness)
assert torch.equal(output["class_logits"], classes)
output["objectness_logits"].sum().backward()
assert any(parameter.grad is not None for parameter in model.suppression.parameters())
assert any(parameter.grad is not None for parameter in model.rescue.parameters())
print("SET_OBJECTNESS_ZERO_INIT_FORWARD_BACKWARD_OK")

