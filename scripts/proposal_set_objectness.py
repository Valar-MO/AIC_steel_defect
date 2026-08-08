#!/usr/bin/env python3
"""One-to-one proposal-set objectness on top of a frozen relation classifier."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from proposal_relation import ProposalRelationTransformer


class ZeroResidualBranch(nn.Module):
    """A small decision branch whose final layer starts at exact zero."""

    def __init__(self, input_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, value):
        # softplus(x)-softplus(0) is exactly zero at initialization and has a
        # healthy gradient. Sign-specialization is enforced by the branch loss.
        return F.softplus(self.net(value).squeeze(-1)) - math.log(2.0)


class ProposalSetObjectness(nn.Module):
    """Frozen category relation + asymmetric suppress/rescue objectness residuals."""

    EVIDENCE_DIM = 6

    def __init__(self, relation: ProposalRelationTransformer, decision_hidden=128, dropout=0.1,
                 max_objectness_delta=1.0):
        super().__init__()
        self.relation = relation.requires_grad_(False)
        self.max_objectness_delta = float(max_objectness_delta)
        dimension = relation.objectness_residual.in_features
        input_dim = dimension + self.EVIDENCE_DIM
        self.suppression = ZeroResidualBranch(input_dim, decision_hidden, dropout)
        self.rescue = ZeroResidualBranch(input_dim, decision_hidden, dropout)
        self.rescue_gate = nn.Sequential(
            nn.LayerNorm(self.EVIDENCE_DIM), nn.Linear(self.EVIDENCE_DIM, 32), nn.GELU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen epoch-8 relation features must be deterministic and must not
        # drift through dropout while the new objectness heads are trained.
        self.relation.eval()
        return self

    @staticmethod
    def decision_evidence(class_logits, baseline_objectness, auxiliary):
        probabilities = class_logits.softmax(dim=-1)
        top = probabilities.topk(2, dim=-1).values
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
        entropy = entropy / math.log(probabilities.shape[-1])
        return torch.stack((
            top[..., 0], top[..., 0] - top[..., 1], 1.0 - entropy,
            baseline_objectness.sigmoid(), auxiliary[..., 8], auxiliary[..., 6],
        ), dim=-1)

    def forward(self, features, auxiliary, baseline_objectness, baseline_classes, padding_mask=None):
        with torch.no_grad():
            tokens = self.relation.encode(features, auxiliary, padding_mask)
            category_delta = self.relation.class_residual(tokens)
            corrected_classes = baseline_classes + category_delta
        evidence = self.decision_evidence(corrected_classes, baseline_objectness, auxiliary)
        decision = torch.cat((tokens, evidence), dim=-1)
        suppression = self.suppression(decision)
        rescue = self.rescue(decision)
        gate = self.rescue_gate(evidence).squeeze(-1)
        raw_delta = -suppression + gate * rescue
        objectness_delta = self.max_objectness_delta * torch.tanh(
            raw_delta / self.max_objectness_delta
        )
        objectness = baseline_objectness + objectness_delta
        return {
            "objectness_logits": objectness,
            "class_logits": corrected_classes,
            "suppression": suppression,
            "rescue": rescue,
            "rescue_gate": gate,
            "objectness_delta": objectness_delta,
        }


def load_frozen_relation(checkpoint: dict) -> ProposalRelationTransformer:
    config = checkpoint["config"]
    model = ProposalRelationTransformer(
        checkpoint["feature_dim"], checkpoint["auxiliary_dim"], len(checkpoint["classes"]),
        config["dimension"], config["heads"], config["layers"], config.get("dropout", 0.1),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model
