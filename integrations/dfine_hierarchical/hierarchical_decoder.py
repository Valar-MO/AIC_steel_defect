"""Hierarchical D-FINE decoder: class-agnostic defectness plus conditional classes."""

from __future__ import annotations

import math
import types

import torch
from torch import nn
from torch.nn import functional as F

from ...core import register
from .dfine_decoder import DFINETransformer


class AdaptFormerAdapter(nn.Module):
    """Parallel bottleneck adapter for a frozen decoder FFN.

    The up projection is zero-initialized, so enabling adapters preserves the
    exact pretrained detector output at step zero.  Only the lightweight
    residual branch needs to learn a task-specific correction.
    """

    def __init__(self, hidden_dim: int, bottleneck_dim: int = 64,
                 dropout: float = 0.0, initial_scale: float = 0.1,
                 learnable_scale: bool = True):
        super().__init__()
        if hidden_dim <= 0 or bottleneck_dim <= 0:
            raise ValueError("Adapter dimensions must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("Adapter dropout must be in [0, 1)")
        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, hidden_dim)
        scale = torch.tensor(float(initial_scale), dtype=torch.float32)
        if learnable_scale:
            self.scale = nn.Parameter(scale)
        else:
            self.register_buffer("scale", scale)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.up(self.dropout(self.activation(self.down(features))))
        return residual * self.scale


def _adaptformer_forward_ffn(layer: nn.Module, target: torch.Tensor) -> torch.Tensor:
    """Keep the official FFN intact and add the registered parallel adapter."""
    base = layer.linear2(layer.dropout3(layer.activation(layer.linear1(target))))
    return base + layer.adaptformer(target)


class HierarchicalScoreHead(nn.Module):
    """Keep the legacy one-class parameters and add a separate conditional class head.

    ``weight`` and ``bias`` deliberately retain ``nn.Linear(., 1)`` state-dict names,
    so a class-agnostic D-FINE checkpoint loads directly into the defectness branch.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        prior_probability: float = 0.01,
        objectness_head_type: str = "linear",
        objectness_hidden_dim: int = 128,
        objectness_dropout: float = 0.1,
        defectness_residual_sign: float = -1.0,
        quality_prior_probability: float = 0.99,
    ):
        super().__init__()
        if objectness_head_type not in {"linear", "dual_mlp", "selection_gated"}:
            raise ValueError(f"Unsupported objectness_head_type: {objectness_head_type}")
        if objectness_hidden_dim <= 0:
            raise ValueError("objectness_hidden_dim must be positive")
        if not 0 <= objectness_dropout < 1:
            raise ValueError("objectness_dropout must be in [0, 1)")
        if defectness_residual_sign not in {-1.0, 1.0}:
            raise ValueError("defectness_residual_sign must be -1 or +1")
        if not 0 < quality_prior_probability < 1:
            raise ValueError("quality_prior_probability must be in (0, 1)")
        self.in_features = int(input_dim)
        self.out_features = int(num_classes + 1)
        self.objectness_head_type = objectness_head_type
        self.defectness_residual_sign = float(defectness_residual_sign)
        self.weight = nn.Parameter(torch.empty(1, input_dim))
        self.bias = nn.Parameter(torch.empty(1))
        self.class_weight = nn.Parameter(torch.empty(num_classes, input_dim))
        self.class_bias = nn.Parameter(torch.zeros(num_classes))
        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.bias, math.log(prior_probability / (1.0 - prior_probability)))
        nn.init.xavier_uniform_(self.class_weight)
        self._quality_history: list[torch.Tensor] = []
        self._defectness_history: list[torch.Tensor] = []

        if self.objectness_head_type in {"dual_mlp", "selection_gated"}:
            self.defectness_norm = nn.LayerNorm(input_dim)
            self.defectness_mlp = nn.Sequential(
                nn.Linear(input_dim, objectness_hidden_dim),
                nn.GELU(),
                nn.Dropout(objectness_dropout),
                nn.Linear(objectness_hidden_dim, 1),
            )
            self.quality_norm = nn.LayerNorm(input_dim)
            self.quality_mlp = nn.Sequential(
                nn.Linear(input_dim, objectness_hidden_dim),
                nn.GELU(),
                nn.Dropout(objectness_dropout),
                nn.Linear(objectness_hidden_dim, 1),
            )
            nn.init.xavier_uniform_(self.defectness_mlp[0].weight)
            nn.init.zeros_(self.defectness_mlp[0].bias)
            nn.init.zeros_(self.defectness_mlp[-1].weight)
            if self.objectness_head_type == "dual_mlp":
                # Preserve the old dual-head experiment exactly.
                nn.init.zeros_(self.defectness_mlp[-1].bias)
                self.defectness_residual_scale = nn.Parameter(torch.tensor(1.0))
            else:
                # Defectness is an independent multiplicative gate in the
                # three-way model. A high constant prior preserves the
                # pretrained ranking while still providing usable gradients.
                nn.init.constant_(
                    self.defectness_mlp[-1].bias,
                    math.log(quality_prior_probability /
                             (1.0 - quality_prior_probability)),
                )
            nn.init.xavier_uniform_(self.quality_mlp[0].weight)
            nn.init.zeros_(self.quality_mlp[0].bias)
            nn.init.zeros_(self.quality_mlp[-1].weight)
            nn.init.constant_(
                self.quality_mlp[-1].bias,
                math.log(quality_prior_probability / (1.0 - quality_prior_probability)),
            )
        if self.objectness_head_type == "selection_gated":
            self.selection_norm = nn.LayerNorm(input_dim)
            self.selection_mlp = nn.Sequential(
                nn.Linear(input_dim, objectness_hidden_dim),
                nn.GELU(),
                nn.Dropout(objectness_dropout),
                nn.Linear(objectness_hidden_dim, 1),
            )
            self.selection_residual_scale = nn.Parameter(torch.tensor(1.0))
            nn.init.xavier_uniform_(self.selection_mlp[0].weight)
            nn.init.zeros_(self.selection_mlp[0].bias)
            nn.init.zeros_(self.selection_mlp[-1].weight)
            nn.init.zeros_(self.selection_mlp[-1].bias)

    @property
    def has_decoupled_objectness(self) -> bool:
        return self.objectness_head_type == "dual_mlp"

    @property
    def has_selection_gates(self) -> bool:
        return self.objectness_head_type == "selection_gated"

    def reset_runtime_cache(self) -> None:
        self._quality_history = []
        self._defectness_history = []

    @property
    def quality_history(self) -> tuple[torch.Tensor, ...]:
        return tuple(self._quality_history)

    @property
    def defectness_history(self) -> tuple[torch.Tensor, ...]:
        return tuple(self._defectness_history)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        base_logit = F.linear(features, self.weight, self.bias)
        primary_logit = base_logit
        if self.has_decoupled_objectness:
            residual = self.defectness_mlp(self.defectness_norm(features))
            primary_logit = base_logit + (
                self.defectness_residual_sign
                * self.defectness_residual_scale
                * residual
            )
            quality = self.quality_mlp(self.quality_norm(features))
            self._quality_history.append(quality)
        elif self.has_selection_gates:
            selection_residual = self.selection_mlp(
                self.selection_norm(features)
            )
            primary_logit = (
                base_logit
                + self.selection_residual_scale * selection_residual
            )
            defectness = self.defectness_mlp(self.defectness_norm(features))
            quality = self.quality_mlp(self.quality_norm(features))
            self._defectness_history.append(defectness)
            self._quality_history.append(quality)
        conditional_classes = F.linear(features, self.class_weight, self.class_bias)
        return torch.cat((primary_logit, conditional_classes), dim=-1)


@register()
class HierarchicalDFINETransformer(DFINETransformer):
    """Official D-FINE decoder with factorized defectness and nine-class logits."""

    def __init__(self, *args, adapter_dim=0, adapter_dropout=0.0,
                 adapter_scale=0.1, adapter_learnable_scale=True,
                 adapter_layers=None, query_select_start_level=0,
                 anchor_grid_size=0.05,
                 p2_query_quota=0,
                 objectness_head_type="linear",
                 objectness_hidden_dim=128,
                 objectness_dropout=0.1,
                 defectness_residual_sign=-1.0,
                 quality_prior_probability=0.99,
                 **kwargs):
        self.query_select_start_level = int(query_select_start_level)
        self.anchor_grid_size = float(anchor_grid_size)
        self.p2_query_quota = int(p2_query_quota)
        self._query_select_skip_tokens = 0
        self._p2_query_token_count = 0
        self._last_p2_query_indices = None
        super().__init__(*args, **kwargs)
        if not 0 <= self.query_select_start_level < self.num_levels:
            raise ValueError(
                "query_select_start_level must be in [0, num_levels)"
            )
        if self.anchor_grid_size <= 0:
            raise ValueError("anchor_grid_size must be positive")
        if not 0 <= self.p2_query_quota < self.num_queries:
            raise ValueError("p2_query_quota must be in [0, num_queries)")
        dimensions = [head.in_features for head in self.dec_score_head]
        self.objectness_head_type = str(objectness_head_type)
        self.dec_score_head = nn.ModuleList([
            HierarchicalScoreHead(
                dimension,
                self.num_classes,
                objectness_head_type=self.objectness_head_type,
                objectness_hidden_dim=int(objectness_hidden_dim),
                objectness_dropout=float(objectness_dropout),
                defectness_residual_sign=float(defectness_residual_sign),
                quality_prior_probability=float(quality_prior_probability),
            )
            for dimension in dimensions
        ])
        if self.p2_query_quota:
            # P2 memory already contains projected local texture plus encoded
            # P3 context.  Keep this selector separate from the established
            # P3-P5 encoder selector so its quota cannot perturb old top-k.
            self.p2_query_score_head = nn.Linear(self.hidden_dim, 1)
            with torch.no_grad():
                self.p2_query_score_head.weight.copy_(self.enc_score_head.weight)
                self.p2_query_score_head.bias.copy_(self.enc_score_head.bias)
        self.adapter_dim = int(adapter_dim)
        if adapter_layers is None:
            adapter_layers = list(range(len(self.decoder.layers)))
        adapter_layers = {int(index) for index in adapter_layers}
        invalid = sorted(index for index in adapter_layers
                         if not 0 <= index < len(self.decoder.layers))
        if invalid:
            raise ValueError(f"Invalid decoder adapter layers: {invalid}")
        if self.adapter_dim < 0:
            raise ValueError("adapter_dim must be non-negative")
        if self.adapter_dim:
            for index, layer in enumerate(self.decoder.layers):
                if index not in adapter_layers:
                    continue
                layer.adaptformer = AdaptFormerAdapter(
                    hidden_dim=layer.linear2.out_features,
                    bottleneck_dim=self.adapter_dim,
                    dropout=float(adapter_dropout),
                    initial_scale=float(adapter_scale),
                    learnable_scale=bool(adapter_learnable_scale),
                )
                # Preserve all official layer/state-dict names.  The bound method
                # changes behavior only when the registered adapter exists.
                layer.forward_ffn = types.MethodType(_adaptformer_forward_ffn, layer)

    def adapter_modules(self):
        return [layer.adaptformer for layer in self.decoder.layers
                if hasattr(layer, "adaptformer")]

    def _generate_anchors(
        self, spatial_shapes=None, grid_size=None, dtype=torch.float32, device="cpu"
    ):
        """Allow the four-level model to retain the old P3-P5 anchor priors."""
        if grid_size is None:
            grid_size = self.anchor_grid_size
        return super()._generate_anchors(
            spatial_shapes=spatial_shapes,
            grid_size=grid_size,
            dtype=dtype,
            device=device,
        )

    def _get_decoder_input(
        self, memory, spatial_shapes, denoising_logits=None, denoising_bbox_unact=None
    ):
        self._query_select_skip_tokens = sum(
            int(height) * int(width)
            for height, width in spatial_shapes[: self.query_select_start_level]
        )
        self._p2_query_token_count = (
            int(spatial_shapes[0][0]) * int(spatial_shapes[0][1])
            if self.p2_query_quota else 0
        )
        try:
            return super()._get_decoder_input(
                memory,
                spatial_shapes,
                denoising_logits,
                denoising_bbox_unact,
            )
        finally:
            self._query_select_skip_tokens = 0
            self._p2_query_token_count = 0

    def _select_topk(self, memory, outputs_logits, outputs_anchors_unact, topk):
        """Optionally initialize queries from coarse levels while retaining P2 memory."""
        if self.p2_query_quota:
            if self._query_select_skip_tokens:
                raise RuntimeError("P2 quota requires query_select_start_level=0")
            p2_tokens = self._p2_query_token_count
            if not 0 < p2_tokens <= memory.shape[1]:
                raise RuntimeError("Missing P2 token extent for quota selection")
            if p2_tokens < self.p2_query_quota:
                raise RuntimeError("P2 feature map has fewer tokens than its query quota")
            coarse_quota = topk - self.p2_query_quota
            p2_logits = self.p2_query_score_head(memory[:, :p2_tokens])
            coarse_logits = outputs_logits[:, p2_tokens:]
            p2_indices = torch.topk(p2_logits.squeeze(-1), self.p2_query_quota, dim=-1).indices
            coarse_indices = torch.topk(coarse_logits.squeeze(-1), coarse_quota, dim=-1).indices + p2_tokens
            topk_ind = torch.cat((p2_indices, coarse_indices), dim=1)
            selector_logits = torch.cat((p2_logits, coarse_logits), dim=1)
            self._last_p2_query_indices = p2_indices.detach()
            topk_anchors = outputs_anchors_unact.gather(
                1, topk_ind.unsqueeze(-1).expand(-1, -1, outputs_anchors_unact.shape[-1])
            )
            topk_memory = memory.gather(
                1, topk_ind.unsqueeze(-1).expand(-1, -1, memory.shape[-1])
            )
            topk_logits = (selector_logits.gather(
                1, topk_ind.unsqueeze(-1).expand(-1, -1, selector_logits.shape[-1])
            ) if self.training else None)
            return topk_memory, topk_logits, topk_anchors
        skip = self._query_select_skip_tokens
        if skip:
            memory = memory[:, skip:]
            outputs_logits = outputs_logits[:, skip:]
            outputs_anchors_unact = outputs_anchors_unact[:, skip:]
        return super()._select_topk(
            memory, outputs_logits, outputs_anchors_unact, topk
        )

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

    def _reset_gate_caches(self) -> None:
        for head in self.dec_score_head:
            if isinstance(head, HierarchicalScoreHead):
                head.reset_runtime_cache()

    @staticmethod
    def _align_quality(
        quality: torch.Tensor, prediction: dict, take_prefix: bool = False
    ) -> torch.Tensor:
        query_count = prediction["pred_boxes"].shape[1]
        if quality.shape[1] < query_count:
            raise RuntimeError(
                f"Quality cache has {quality.shape[1]} queries, "
                f"prediction has {query_count}"
            )
        return quality[:, :query_count] if take_prefix else quality[:, -query_count:]

    def _attach_gate_predictions(self, output: dict) -> None:
        histories = {
            index: head.quality_history
            for index, head in enumerate(self.dec_score_head)
            if isinstance(head, HierarchicalScoreHead) and head.quality_history
        }
        if not histories:
            return

        last_index = max(histories)
        output["pred_quality_logits"] = self._align_quality(
            histories[last_index][-1], output
        )
        defectness_histories = {
            index: head.defectness_history
            for index, head in enumerate(self.dec_score_head)
            if isinstance(head, HierarchicalScoreHead) and head.defectness_history
        }
        if defectness_histories:
            output["pred_defectness_logits"] = self._align_quality(
                defectness_histories[last_index][-1], output
            )
            output["selection_gated_training"] = True
            # The final decoder layer is the only trainable selection-gated
            # head. Do not manufacture auxiliary gate losses for frozen heads.
            return
        # Duplicate competition is a final-selection objective. Applying its
        # per-GT hard mining independently to every auxiliary layer is both
        # semantically unnecessary and prohibitively synchronization-heavy.
        output["quality_duplicate_ranking"] = True
        for index, prediction in enumerate(output.get("aux_outputs", [])):
            if index in histories:
                prediction["pred_quality_logits"] = self._align_quality(
                    histories[index][-1], prediction
                )
        if "pre_outputs" in output and 0 in histories:
            output["pre_outputs"]["pred_quality_logits"] = self._align_quality(
                histories[0][0], output["pre_outputs"]
            )

        # Denoising queries occupy the prefix and ordinary one-to-one queries
        # occupy the suffix of the same score-head call.
        for index, prediction in enumerate(output.get("dn_outputs", [])):
            if index in histories:
                prediction["pred_quality_logits"] = self._align_quality(
                    histories[index][-1], prediction, take_prefix=True
                )
        if "dn_pre_outputs" in output and 0 in histories:
            output["dn_pre_outputs"]["pred_quality_logits"] = self._align_quality(
                histories[0][0], output["dn_pre_outputs"], take_prefix=True
            )

    def forward(self, feats, targets=None):
        self._reset_gate_caches()
        try:
            output = super().forward(feats, targets)
            self._attach_gate_predictions(output)
        finally:
            # The output dictionary owns the graph tensors after attachment.
            # Keeping them on modules would retain graphs and break EMA deepcopy.
            self._reset_gate_caches()
        self._split_prediction(output)
        for key in ("aux_outputs", "enc_aux_outputs", "dn_outputs"):
            for prediction in output.get(key, []):
                self._split_prediction(prediction)
        for key in ("pre_outputs", "dn_pre_outputs"):
            if key in output:
                self._split_prediction(output[key])
        return output
