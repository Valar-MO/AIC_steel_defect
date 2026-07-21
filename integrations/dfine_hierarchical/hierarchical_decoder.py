"""Hierarchical D-FINE decoder: class-agnostic defectness plus conditional classes."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from ...core import register
from .dfine_decoder import DFINETransformer


class HierarchicalScoreHead(nn.Module):
    """Keep the legacy one-class parameters and add a separate conditional class head.

    ``weight`` and ``bias`` deliberately retain ``nn.Linear(., 1)`` state-dict names,
    so a class-agnostic D-FINE checkpoint loads directly into the defectness branch.
    """

    def __init__(self, input_dim: int, num_classes: int, prior_probability: float = 0.01):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1, input_dim))
        self.bias = nn.Parameter(torch.empty(1))
        self.class_weight = nn.Parameter(torch.empty(num_classes, input_dim))
        self.class_bias = nn.Parameter(torch.zeros(num_classes))
        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.bias, math.log(prior_probability / (1.0 - prior_probability)))
        nn.init.xavier_uniform_(self.class_weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        defectness = F.linear(features, self.weight, self.bias)
        conditional_classes = F.linear(features, self.class_weight, self.class_bias)
        return torch.cat((defectness, conditional_classes), dim=-1)


@register()
class HierarchicalDFINETransformer(DFINETransformer):
    """Official D-FINE decoder with factorized defectness and nine-class logits."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dimensions = [head.in_features for head in self.dec_score_head]
        self.dec_score_head = nn.ModuleList([
            HierarchicalScoreHead(dimension, self.num_classes) for dimension in dimensions
        ])

    def _split_prediction(self, prediction: dict) -> dict:
        logits = prediction.get("pred_logits")
        if logits is not None:
            if logits.shape[-1] == self.num_classes + 1:
                prediction["pred_objectness_logits"] = logits[..., :1]
                prediction["pred_class_logits"] = logits[..., 1:]
                # D-FINE localization losses use pred_logits only as a quality
                # signal. Point it at defectness to avoid class leakage.
                prediction["pred_logits"] = logits[..., :1]
            elif logits.shape[-1] == 1:
                prediction["pred_objectness_logits"] = logits
        teacher = prediction.get("teacher_logits")
        if teacher is not None and teacher.shape[-1] == self.num_classes + 1:
            prediction["teacher_logits"] = teacher[..., :1]
        return prediction

    def forward(self, feats, targets=None):
        output = super().forward(feats, targets)
        self._split_prediction(output)
        for key in ("aux_outputs", "enc_aux_outputs", "dn_outputs"):
            for prediction in output.get(key, []):
                self._split_prediction(prediction)
        for key in ("pre_outputs", "dn_pre_outputs"):
            if key in output:
                self._split_prediction(output[key])
        return output

