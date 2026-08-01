#!/usr/bin/env python3
"""Dependency-free smoke test for the proposal relation module."""

import torch

from proposal_relation import ProposalRelationTransformer, pairwise_geometry


model = ProposalRelationTransformer(16, 19, 9, 32, 4, 2, 0.0)
features = torch.randn(2, 5, 16)
auxiliary = torch.randn(2, 5, 19)
auxiliary[:, :, :4] = torch.tensor([0.1, 0.2, 0.5, 0.7])
objectness = torch.randn(2, 5)
classes = torch.randn(2, 5, 9)
output = model(features, auxiliary, objectness, classes)
assert torch.equal(output["objectness_logits"], objectness)
assert torch.equal(output["class_logits"], classes)
output["objectness_logits"].sum().backward()
assert pairwise_geometry(auxiliary[:, :, :4]).shape == (2, 5, 5, 7)
print("ZERO_INIT_FORWARD_BACKWARD_OK")
