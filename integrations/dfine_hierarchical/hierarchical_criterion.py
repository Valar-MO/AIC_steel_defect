"""Factorized defectness/localization/category losses for Hierarchical D-FINE."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from ...core import register
from .box_ops import box_cxcywh_to_xyxy, box_iou
from .dfine_criterion import DFINECriterion


@register()
class HierarchicalDFINECriterion(DFINECriterion):
    __share__ = ["num_classes"]
    __inject__ = ["matcher"]

    def __init__(self, matcher, weight_dict, losses, alpha=0.75, gamma=2.0,
                 num_classes=9, reg_max=32, boxes_weight_format=None,
                 share_matched_indices=False, class_weights=None,
                 class_label_smoothing=0.05, class_focal_gamma=1.0,
                 objectness_class_weights=None,
                 objectness_class_weight_power=1.0,
                 objectness_class_weight_min=0.0,
                 objectness_class_weight_max=None,
                 class_priors=None,
                 class_logit_adjustment_tau=0.0,
                 hard_background_topk=0,
                 hard_background_max_iou=0.1,
                 hard_background_rank_margin=0.5,
                 duplicate_topk=0,
                 duplicate_iou=0.5,
                 duplicate_rank_margin=0.5,
                 rank_positive_min_iou=0.5,
                 decoupled_objectness=False,
                 selection_gated_objectness=False,
                 defect_positive_iou=0.5,
                 defect_background_iou=0.1,
                 quality_duplicate_topk=0,
                 quality_duplicate_iou=0.5,
                 quality_duplicate_rank_margin=0.25,
                 selection_native_ranking=False,
                 selection_native_hard_background_topk=0,
                 selection_native_background_iou=0.1,
                 selection_native_positive_iou=0.5,
                 selection_native_duplicate_iou=0.5,
                 selection_native_duplicate_box_iou=0.5,
                 selection_native_rank_margin=0.25):
        super().__init__(matcher, weight_dict, losses, alpha, gamma, num_classes,
                         reg_max, boxes_weight_format, share_matched_indices)
        if class_weights is None:
            class_weights = [1.0] * num_classes
        weights = torch.as_tensor(class_weights, dtype=torch.float32)
        if weights.numel() != num_classes:
            raise ValueError(f"Expected {num_classes} class weights, got {weights.numel()}")
        self.register_buffer("conditional_class_weights", weights / weights.mean())
        self.class_label_smoothing = class_label_smoothing
        self.class_focal_gamma = class_focal_gamma
        if objectness_class_weights is None:
            objectness_class_weights = [1.0] * num_classes
        objectness_weights = torch.as_tensor(objectness_class_weights, dtype=torch.float32)
        if objectness_weights.numel() != num_classes or torch.any(objectness_weights <= 0):
            raise ValueError("objectness_class_weights must contain one positive value per class")
        if objectness_class_weight_power < 0:
            raise ValueError("objectness_class_weight_power must be non-negative")
        objectness_weights = (objectness_weights / objectness_weights.mean()).pow(
            float(objectness_class_weight_power)
        )
        objectness_weights = objectness_weights / objectness_weights.mean()
        maximum = float("inf") if objectness_class_weight_max is None \
            else float(objectness_class_weight_max)
        if objectness_class_weight_min < 0 or maximum < objectness_class_weight_min:
            raise ValueError("Invalid objectness class-weight clamp")
        objectness_weights = objectness_weights.clamp(
            min=float(objectness_class_weight_min), max=maximum
        )
        self.register_buffer("objectness_class_weights", objectness_weights)

        if class_logit_adjustment_tau < 0:
            raise ValueError("class_logit_adjustment_tau must be non-negative")
        if class_priors is None:
            class_priors = [1.0 / num_classes] * num_classes
        priors = torch.as_tensor(class_priors, dtype=torch.float32)
        if priors.numel() != num_classes or torch.any(priors <= 0):
            raise ValueError("class_priors must contain one positive value per class")
        priors = priors / priors.sum()
        self.register_buffer("conditional_class_log_priors", priors.clamp_min(1e-12).log())
        self.class_logit_adjustment_tau = float(class_logit_adjustment_tau)
        self.hard_background_topk = int(hard_background_topk)
        self.hard_background_max_iou = float(hard_background_max_iou)
        self.hard_background_rank_margin = float(hard_background_rank_margin)
        self.duplicate_topk = int(duplicate_topk)
        self.duplicate_iou = float(duplicate_iou)
        self.duplicate_rank_margin = float(duplicate_rank_margin)
        self.rank_positive_min_iou = float(rank_positive_min_iou)
        self.decoupled_objectness = bool(decoupled_objectness)
        self.selection_gated_objectness = bool(selection_gated_objectness)
        self.defect_positive_iou = float(defect_positive_iou)
        self.defect_background_iou = float(defect_background_iou)
        self.quality_duplicate_topk = int(quality_duplicate_topk)
        self.quality_duplicate_iou = float(quality_duplicate_iou)
        self.quality_duplicate_rank_margin = float(quality_duplicate_rank_margin)
        self.selection_native_ranking = bool(selection_native_ranking)
        self.selection_native_hard_background_topk = int(
            selection_native_hard_background_topk
        )
        self.selection_native_background_iou = float(selection_native_background_iou)
        self.selection_native_positive_iou = float(selection_native_positive_iou)
        self.selection_native_duplicate_iou = float(selection_native_duplicate_iou)
        self.selection_native_duplicate_box_iou = float(
            selection_native_duplicate_box_iou
        )
        self.selection_native_rank_margin = float(selection_native_rank_margin)
        if self.hard_background_topk < 0 or self.duplicate_topk < 0:
            raise ValueError("ranking top-k values must be non-negative")
        if not 0 <= self.hard_background_max_iou < self.duplicate_iou <= 1:
            raise ValueError(
                "Require 0 <= hard_background_max_iou < duplicate_iou <= 1"
            )
        if not 0 <= self.rank_positive_min_iou <= 1:
            raise ValueError("rank_positive_min_iou must be in [0, 1]")
        if self.hard_background_rank_margin < 0 or self.duplicate_rank_margin < 0:
            raise ValueError("ranking margins must be non-negative")
        if not 0 <= self.defect_background_iou < self.defect_positive_iou <= 1:
            raise ValueError(
                "Require 0 <= defect_background_iou < defect_positive_iou <= 1"
            )
        if self.quality_duplicate_topk < 0:
            raise ValueError("quality_duplicate_topk must be non-negative")
        if not self.defect_positive_iou <= self.quality_duplicate_iou <= 1:
            raise ValueError(
                "quality_duplicate_iou must be at least defect_positive_iou"
            )
        if self.quality_duplicate_rank_margin < 0:
            raise ValueError("quality_duplicate_rank_margin must be non-negative")
        if self.selection_native_hard_background_topk < 0:
            raise ValueError("selection_native_hard_background_topk must be non-negative")
        if not 0 <= self.selection_native_background_iou < self.selection_native_positive_iou <= 1:
            raise ValueError(
                "Require 0 <= selection_native_background_iou < "
                "selection_native_positive_iou <= 1"
            )
        if not self.selection_native_positive_iou <= self.selection_native_duplicate_iou <= 1:
            raise ValueError(
                "selection_native_duplicate_iou must be at least "
                "selection_native_positive_iou"
            )
        if not 0 <= self.selection_native_duplicate_box_iou <= 1:
            raise ValueError("selection_native_duplicate_box_iou must be in [0,1]")
        if self.selection_native_rank_margin < 0:
            raise ValueError("selection_native_rank_margin must be non-negative")

    @staticmethod
    def _zero_loss(reference):
        return reference.sum() * 0.0

    def _ranking_losses(self, outputs, targets, indices):
        """Rank one-to-one winners above hard background and duplicate queries.

        Query selection is discrete and detached. Gradients flow only through
        the selected objectness logits, which lets an objectness-head-only
        experiment keep decoder features, boxes and conditional classes frozen.
        """
        objectness = outputs.get(
            "pred_objectness_logits", outputs["pred_logits"]
        ).float().squeeze(-1)
        zero = self._zero_loss(objectness)
        if ("pred_boxes" not in outputs or outputs.get("is_dn") is not None or
                not (self.hard_background_topk or self.duplicate_topk)):
            return {
                "loss_hard_background_rank": zero,
                "loss_duplicate_rank": zero,
            }

        predicted_boxes = box_cxcywh_to_xyxy(outputs["pred_boxes"]).detach()
        background_losses, duplicate_losses = [], []
        for batch_index, ((matched_queries, matched_targets), target) in enumerate(
                zip(indices, targets)):
            matched_queries = matched_queries.to(objectness.device)
            matched_targets = matched_targets.to(objectness.device)
            target_boxes = target.get("boxes")
            if target_boxes is None or target_boxes.numel() == 0 or not matched_queries.numel():
                continue
            target_boxes = box_cxcywh_to_xyxy(target_boxes).detach()
            overlaps = box_iou(predicted_boxes[batch_index], target_boxes)[0].detach()
            unmatched = torch.ones(
                objectness.shape[1], dtype=torch.bool, device=objectness.device
            )
            unmatched[matched_queries] = False
            winner_iou = overlaps[matched_queries, matched_targets]
            valid_winners = winner_iou >= self.rank_positive_min_iou

            if self.hard_background_topk and valid_winners.any():
                background_mask = unmatched & (
                    overlaps.max(dim=1).values < self.hard_background_max_iou
                )
                background_indices = background_mask.nonzero(as_tuple=False).flatten()
                if background_indices.numel():
                    count = min(self.hard_background_topk, background_indices.numel())
                    hard_order = objectness[batch_index, background_indices].detach().topk(
                        count
                    ).indices
                    hard_indices = background_indices[hard_order]
                    positive_logits = objectness[
                        batch_index, matched_queries[valid_winners]
                    ]
                    negative_logits = objectness[batch_index, hard_indices]
                    background_losses.append(F.softplus(
                        self.hard_background_rank_margin
                        - positive_logits[:, None] + negative_logits[None, :]
                    ).mean())

            if self.duplicate_topk:
                per_image_duplicates = []
                for query_index, target_index, quality in zip(
                        matched_queries, matched_targets, winner_iou):
                    if quality < max(self.rank_positive_min_iou, self.duplicate_iou):
                        continue
                    duplicate_mask = unmatched & (
                        overlaps[:, target_index] >= self.duplicate_iou
                    )
                    duplicate_indices = duplicate_mask.nonzero(
                        as_tuple=False
                    ).flatten()
                    if not duplicate_indices.numel():
                        continue
                    count = min(self.duplicate_topk, duplicate_indices.numel())
                    hard_order = objectness[
                        batch_index, duplicate_indices
                    ].detach().topk(count).indices
                    duplicate_indices = duplicate_indices[hard_order]
                    per_image_duplicates.append(F.softplus(
                        self.duplicate_rank_margin
                        - objectness[batch_index, query_index]
                        + objectness[batch_index, duplicate_indices]
                    ).mean())
                if per_image_duplicates:
                    duplicate_losses.append(torch.stack(per_image_duplicates).mean())

        return {
            "loss_hard_background_rank": torch.stack(background_losses).mean()
            if background_losses else zero,
            "loss_duplicate_rank": torch.stack(duplicate_losses).mean()
            if duplicate_losses else zero,
        }

    def _selection_native_ranking_losses(self, outputs, targets):
        """Final-layer utility ranking for Selection logits.

        Gradients flow only through ``pred_objectness_logits``. Boxes and class
        predictions are detached and used solely to define final-utility winners,
        duplicates, and hard backgrounds.
        """
        selection = outputs.get(
            "pred_objectness_logits", outputs["pred_logits"]
        ).float().squeeze(-1)
        zero = self._zero_loss(selection)
        if (
            not self.selection_native_ranking
            or "pred_boxes" not in outputs
            or "pred_class_logits" not in outputs
            or outputs.get("is_dn") is not None
        ):
            return {
                "loss_selection_native_positive": zero,
                "loss_selection_native_duplicate_rank": zero,
                "loss_selection_native_hard_background_rank": zero,
            }

        predicted_boxes = box_cxcywh_to_xyxy(outputs["pred_boxes"]).detach()
        predicted_labels = outputs["pred_class_logits"].detach().argmax(dim=-1)
        positive_losses, duplicate_losses, background_losses = [], [], []
        for batch_index, target in enumerate(targets):
            target_boxes = target.get("boxes")
            target_labels = target.get("labels")
            if target_boxes is None or target_boxes.numel() == 0:
                continue
            target_boxes = box_cxcywh_to_xyxy(target_boxes).detach()
            target_labels = target_labels.to(selection.device)
            overlaps = box_iou(predicted_boxes[batch_index], target_boxes)[0].detach()
            same_class = predicted_labels[batch_index][:, None] == target_labels[None, :]
            valid = same_class & (overlaps >= self.selection_native_positive_iou)
            max_any_iou = overlaps.max(dim=1).values
            winner_indices, winner_targets, winner_utilities = [], [], []
            for target_index in range(target_boxes.shape[0]):
                candidates = valid[:, target_index].nonzero(as_tuple=False).flatten()
                if not candidates.numel():
                    continue
                utilities = overlaps[candidates, target_index]
                best = candidates[utilities.argmax()]
                winner_indices.append(best)
                winner_targets.append(target_index)
                winner_utilities.append(overlaps[best, target_index])
            if not winner_indices:
                continue
            winner_indices = torch.stack(winner_indices).long()
            winner_targets = torch.as_tensor(
                winner_targets, dtype=torch.long, device=selection.device
            )
            winner_utilities = torch.stack(winner_utilities).to(selection.dtype)
            winner_logits = selection[batch_index, winner_indices]
            positive_losses.append(F.binary_cross_entropy_with_logits(
                winner_logits, winner_utilities, reduction="mean"
            ))

            for winner, target_index in zip(winner_indices, winner_targets):
                duplicate_mask = valid[:, target_index].clone()
                duplicate_mask[winner] = False
                if self.selection_native_duplicate_box_iou > 0:
                    winner_box = predicted_boxes[batch_index, winner].unsqueeze(0)
                    candidate_overlap = box_iou(
                        predicted_boxes[batch_index], winner_box
                    )[0].squeeze(1).detach()
                    duplicate_mask &= (
                        candidate_overlap >= self.selection_native_duplicate_box_iou
                    )
                duplicate_indices = duplicate_mask.nonzero(as_tuple=False).flatten()
                if not duplicate_indices.numel():
                    continue
                duplicate_losses.append(F.softplus(
                    self.selection_native_rank_margin
                    - selection[batch_index, winner]
                    + selection[batch_index, duplicate_indices]
                ).mean())

            if self.selection_native_hard_background_topk:
                background_mask = max_any_iou < self.selection_native_background_iou
                background_indices = background_mask.nonzero(as_tuple=False).flatten()
                if background_indices.numel():
                    count = min(
                        self.selection_native_hard_background_topk,
                        background_indices.numel(),
                    )
                    hard_order = selection[
                        batch_index, background_indices
                    ].detach().topk(count).indices
                    hard_indices = background_indices[hard_order]
                    background_losses.append(F.softplus(
                        self.selection_native_rank_margin
                        - winner_logits[:, None]
                        + selection[batch_index, hard_indices][None, :]
                    ).mean())

        return {
            "loss_selection_native_positive": torch.stack(positive_losses).mean()
            if positive_losses else zero,
            "loss_selection_native_duplicate_rank": torch.stack(duplicate_losses).mean()
            if duplicate_losses else zero,
            "loss_selection_native_hard_background_rank": torch.stack(background_losses).mean()
            if background_losses else zero,
        }

    def _decoupled_iou_targets(self, outputs, targets):
        """Return detached max-IoU targets using one batched broadcast."""
        predicted_boxes = box_cxcywh_to_xyxy(outputs["pred_boxes"]).detach()
        target_counts = [
            int(target["boxes"].shape[0])
            if target.get("boxes") is not None else 0
            for target in targets
        ]
        max_targets = max(target_counts, default=0)
        if max_targets == 0:
            max_iou = predicted_boxes.new_zeros(predicted_boxes.shape[:2])
            empty_overlaps = predicted_boxes.new_zeros(
                (predicted_boxes.shape[0], predicted_boxes.shape[1], 0)
            )
            empty_valid = torch.zeros(
                (predicted_boxes.shape[0], 0),
                dtype=torch.bool,
                device=predicted_boxes.device,
            )
            return max_iou, empty_overlaps, empty_valid

        padded_targets = predicted_boxes.new_zeros(
            (predicted_boxes.shape[0], max_targets, 4)
        )
        valid_targets = torch.zeros(
            (predicted_boxes.shape[0], max_targets),
            dtype=torch.bool,
            device=predicted_boxes.device,
        )
        for batch_index, (target, count) in enumerate(zip(targets, target_counts)):
            if not count:
                continue
            target_boxes = target.get("boxes")
            padded_targets[batch_index, :count] = box_cxcywh_to_xyxy(
                target_boxes
            ).detach()
            valid_targets[batch_index, :count] = True

        pred_lt = predicted_boxes[..., :2].unsqueeze(2)
        pred_rb = predicted_boxes[..., 2:].unsqueeze(2)
        target_lt = padded_targets[..., :2].unsqueeze(1)
        target_rb = padded_targets[..., 2:].unsqueeze(1)
        intersection_wh = (
            torch.minimum(pred_rb, target_rb) - torch.maximum(pred_lt, target_lt)
        ).clamp_min(0)
        intersection = intersection_wh.prod(dim=-1)
        pred_area = (
            predicted_boxes[..., 2:] - predicted_boxes[..., :2]
        ).clamp_min(0).prod(dim=-1).unsqueeze(-1)
        target_area = (
            padded_targets[..., 2:] - padded_targets[..., :2]
        ).clamp_min(0).prod(dim=-1).unsqueeze(1)
        union = pred_area + target_area - intersection
        overlaps = intersection / union.clamp_min(1e-9)
        overlaps = overlaps.masked_fill(~valid_targets.unsqueeze(1), 0.0)
        max_iou = overlaps.max(dim=-1).values
        return max_iou, overlaps, valid_targets

    def _quality_duplicate_ranking_loss(
        self, quality_logits, overlaps, valid_targets
    ):
        zero = self._zero_loss(quality_logits)
        if (
            not self.quality_duplicate_topk
            or overlaps.shape[-1] == 0
            or overlaps.shape[1] < 2
        ):
            return zero
        quality = quality_logits.float().squeeze(-1)
        batch_size, num_queries, num_targets = overlaps.shape
        duplicate_count = min(self.quality_duplicate_topk, num_queries - 1)

        # Each target's highest-IoU query is the quality winner. Other queries
        # above the duplicate IoU threshold compete by their current quality
        # score, matching the former loop's hard-negative selection.
        winner_indices = overlaps.argmax(dim=1)
        candidate_mask = (
            overlaps >= self.quality_duplicate_iou
        ) & valid_targets.unsqueeze(1)
        candidate_mask.scatter_(
            1, winner_indices.unsqueeze(1), False
        )
        expanded_quality = quality.unsqueeze(-1).expand(
            batch_size, num_queries, num_targets
        )
        selection_scores = expanded_quality.detach().masked_fill(
            ~candidate_mask, torch.finfo(expanded_quality.dtype).min
        )
        duplicate_indices = selection_scores.topk(
            duplicate_count, dim=1
        ).indices
        selected_valid = candidate_mask.gather(1, duplicate_indices)
        duplicate_quality = expanded_quality.gather(1, duplicate_indices)
        winner_quality = quality.gather(1, winner_indices)
        pair_losses = F.softplus(
            self.quality_duplicate_rank_margin
            - winner_quality.unsqueeze(1)
            + duplicate_quality
        ) * selected_valid.to(quality.dtype)

        # Preserve the former reduction: mean duplicates per GT, then mean
        # valid GTs per image, then mean images containing a valid pair.
        pairs_per_target = selected_valid.sum(dim=1)
        valid_target_pairs = pairs_per_target > 0
        loss_per_target = pair_losses.sum(dim=1) / pairs_per_target.clamp_min(1)
        targets_per_image = valid_target_pairs.sum(dim=1)
        loss_per_image = (
            loss_per_target * valid_target_pairs.to(quality.dtype)
        ).sum(dim=1) / targets_per_image.clamp_min(1)
        valid_images = targets_per_image > 0
        if not valid_images.any():
            return zero
        return loss_per_image[valid_images].mean()

    def _decoupled_objectness_losses(self, outputs, targets):
        defectness = outputs["pred_objectness_logits"].float().squeeze(-1)
        quality = outputs["pred_quality_logits"].float().squeeze(-1)
        max_iou, overlaps, valid_targets = self._decoupled_iou_targets(
            outputs, targets
        )
        max_iou = max_iou.to(defectness.dtype)

        positive = max_iou >= self.defect_positive_iou
        background = max_iou < self.defect_background_iou
        valid = positive | background
        defect_target = positive.to(defectness.dtype)
        probability = defectness.sigmoid().detach()
        target_probability = (
            probability * defect_target + (1.0 - probability) * (1.0 - defect_target)
        )
        alpha_weight = (
            self.alpha * defect_target + (1.0 - self.alpha) * (1.0 - defect_target)
        )
        focal_weight = alpha_weight * (1.0 - target_probability).pow(self.gamma)
        defect_bce = F.binary_cross_entropy_with_logits(
            defectness, defect_target, reduction="none"
        )
        positive_count = positive.sum().clamp_min(1).to(defectness.dtype)
        defectness_loss = (
            defect_bce * focal_weight * valid.to(defectness.dtype)
        ).sum() / positive_count

        if positive.any():
            quality_loss = F.binary_cross_entropy_with_logits(
                quality[positive], max_iou[positive], reduction="sum"
            ) / positive_count
        else:
            quality_loss = self._zero_loss(quality)
        quality_duplicate_loss = (
            self._quality_duplicate_ranking_loss(
                outputs["pred_quality_logits"], overlaps, valid_targets
            )
            if outputs.get("quality_duplicate_ranking", False)
            else self._zero_loss(outputs["pred_quality_logits"])
        )
        return {
            "loss_defectness": defectness_loss,
            "loss_quality": quality_loss,
            "loss_quality_duplicate_rank": quality_duplicate_loss,
        }

    def _selection_loss(self, outputs, targets, indices, num_boxes):
        """Preserve the pretrained one-to-one soft-IoU scoring objective."""
        selection = outputs["pred_objectness_logits"].float()
        matched = self._get_src_permutation_idx(indices)
        target = torch.zeros_like(selection)
        if matched[0].numel():
            predicted_boxes = outputs["pred_boxes"][matched]
            target_boxes = torch.cat([
                item["boxes"][target_indices]
                for item, (_, target_indices) in zip(targets, indices)
            ])
            iou = torch.diag(box_iou(
                box_cxcywh_to_xyxy(predicted_boxes),
                box_cxcywh_to_xyxy(target_boxes),
            )[0]).detach().to(selection.dtype)
            target[matched] = iou.unsqueeze(-1)
        probability = selection.sigmoid().detach()
        foreground = target > 0
        weight = (
            self.alpha * probability.pow(self.gamma) * (~foreground)
            + target
        )
        return F.binary_cross_entropy_with_logits(
            selection, target, weight=weight, reduction="sum"
        ) / num_boxes

    def _selection_ranking_losses(
        self, outputs, targets, indices, overlaps, valid_targets
    ):
        """Vectorized winner-vs-background and winner-vs-duplicate ranking."""
        selection = outputs["pred_objectness_logits"].float().squeeze(-1)
        zero = self._zero_loss(selection)
        batch_size, num_queries = selection.shape
        num_targets = overlaps.shape[-1]
        if num_targets == 0:
            return {
                "loss_selection_hard_background_rank": zero,
                "loss_selection_duplicate_rank": zero,
            }

        winner_mask = torch.zeros_like(selection, dtype=torch.bool)
        winner_for_target = torch.full(
            (batch_size, num_targets), -1, dtype=torch.long,
            device=selection.device,
        )
        for batch_index, (query_indices, target_indices) in enumerate(indices):
            query_indices = query_indices.to(selection.device)
            target_indices = target_indices.to(selection.device)
            winner_mask[batch_index, query_indices] = True
            winner_for_target[batch_index, target_indices] = query_indices

        max_iou = overlaps.max(dim=-1).values
        hard_background_loss = zero
        if self.hard_background_topk:
            background_mask = (~winner_mask) & (
                max_iou < self.hard_background_max_iou
            )
            count = min(self.hard_background_topk, num_queries)
            background_scores = selection.detach().masked_fill(
                ~background_mask, torch.finfo(selection.dtype).min
            )
            background_indices = background_scores.topk(count, dim=1).indices
            selected_background_valid = background_mask.gather(
                1, background_indices
            )
            background_logits = selection.gather(1, background_indices)
            positive_valid = winner_mask & (
                max_iou >= self.rank_positive_min_iou
            )
            pair_valid = (
                positive_valid.unsqueeze(-1)
                & selected_background_valid.unsqueeze(1)
            )
            if pair_valid.any():
                pair_loss = F.softplus(
                    self.hard_background_rank_margin
                    - selection.unsqueeze(-1)
                    + background_logits.unsqueeze(1)
                )
                hard_background_loss = pair_loss[pair_valid].mean()

        duplicate_loss = zero
        if self.duplicate_topk and num_queries > 1:
            safe_winner = winner_for_target.clamp_min(0)
            winner_valid = (
                (winner_for_target >= 0) & valid_targets
            )
            winner_iou = overlaps.gather(
                1, safe_winner.unsqueeze(1)
            ).squeeze(1)
            winner_valid &= (
                winner_iou >= max(
                    self.rank_positive_min_iou, self.duplicate_iou
                )
            )
            duplicate_mask = (
                (overlaps >= self.duplicate_iou)
                & valid_targets.unsqueeze(1)
                & (~winner_mask).unsqueeze(-1)
            )
            duplicate_count = min(self.duplicate_topk, num_queries - 1)
            expanded_selection = selection.unsqueeze(-1).expand_as(overlaps)
            duplicate_order = expanded_selection.detach().masked_fill(
                ~duplicate_mask, torch.finfo(selection.dtype).min
            ).topk(duplicate_count, dim=1).indices
            duplicate_valid = duplicate_mask.gather(1, duplicate_order)
            duplicate_logits = expanded_selection.gather(1, duplicate_order)
            winner_logits = selection.gather(1, safe_winner)
            pair_valid = duplicate_valid & winner_valid.unsqueeze(1)
            if pair_valid.any():
                pair_loss = F.softplus(
                    self.duplicate_rank_margin
                    - winner_logits.unsqueeze(1)
                    + duplicate_logits
                )
                duplicate_loss = pair_loss[pair_valid].mean()

        return {
            "loss_selection_hard_background_rank": hard_background_loss,
            "loss_selection_duplicate_rank": duplicate_loss,
        }

    def _selection_gated_losses(
        self, outputs, targets, indices, num_boxes
    ):
        """Train three independent semantics without contaminating their labels."""
        defectness = outputs["pred_defectness_logits"].float().squeeze(-1)
        quality = outputs["pred_quality_logits"].float().squeeze(-1)
        max_iou, overlaps, valid_targets = self._decoupled_iou_targets(
            outputs, targets
        )
        max_iou = max_iou.to(defectness.dtype)
        positive = max_iou >= self.defect_positive_iou
        background = max_iou < self.defect_background_iou

        defect_target = positive.to(defectness.dtype)
        probability = defectness.sigmoid().detach()
        target_probability = (
            probability * defect_target
            + (1.0 - probability) * (1.0 - defect_target)
        )
        alpha_weight = (
            self.alpha * defect_target
            + (1.0 - self.alpha) * (1.0 - defect_target)
        )
        focal_weight = alpha_weight * (
            1.0 - target_probability
        ).pow(self.gamma)
        defect_bce = F.binary_cross_entropy_with_logits(
            defectness, defect_target, reduction="none"
        ) * focal_weight
        positive_loss = (
            defect_bce[positive].mean()
            if positive.any() else self._zero_loss(defectness)
        )
        background_loss = (
            defect_bce[background].mean()
            if background.any() else self._zero_loss(defectness)
        )
        defectness_loss = positive_loss + background_loss

        quality_loss = (
            F.binary_cross_entropy_with_logits(
                quality[positive], max_iou[positive], reduction="mean"
            )
            if positive.any() else self._zero_loss(quality)
        )
        ranking = self._selection_ranking_losses(
            outputs, targets, indices, overlaps, valid_targets
        )
        return {
            "loss_selection": self._selection_loss(
                outputs, targets, indices, num_boxes
            ),
            "loss_defectness_gate": defectness_loss,
            "loss_quality": quality_loss,
            **ranking,
        }

    def loss_hierarchical(self, outputs, targets, indices, num_boxes, **kwargs):
        objectness = outputs.get("pred_objectness_logits", outputs["pred_logits"]).float()
        matched = self._get_src_permutation_idx(indices)
        matched_labels = torch.cat([
            target["labels"][target_indices]
            for target, (_, target_indices) in zip(targets, indices)
        ]) if matched[0].numel() else torch.empty(
            0, dtype=torch.long, device=objectness.device
        )
        use_decoupled = (
            self.decoupled_objectness and
            outputs.get("pred_quality_logits") is not None and
            outputs.get("is_dn") is None
        )
        use_selection_gated = (
            self.selection_gated_objectness
            and outputs.get("selection_gated_training", False)
            and outputs.get("pred_defectness_logits") is not None
            and outputs.get("pred_quality_logits") is not None
            and outputs.get("is_dn") is None
        )
        if use_selection_gated:
            objectness_losses = self._selection_gated_losses(
                outputs, targets, indices, num_boxes
            )
        elif use_decoupled:
            objectness_losses = self._decoupled_objectness_losses(outputs, targets)
        else:
            objectness_target = torch.zeros_like(objectness)
            if matched[0].numel():
                predicted_boxes = outputs["pred_boxes"][matched]
                target_boxes = torch.cat([
                    target["boxes"][target_indices]
                    for target, (_, target_indices) in zip(targets, indices)
                ])
                iou = torch.diag(box_iou(
                    box_cxcywh_to_xyxy(predicted_boxes),
                    box_cxcywh_to_xyxy(target_boxes)
                )[0]).detach().to(objectness.dtype)
                objectness_target[matched] = iou.unsqueeze(-1)
            probability = objectness.sigmoid().detach()
            foreground = objectness_target > 0
            weight = (
                self.alpha * probability.pow(self.gamma) * (~foreground)
                + objectness_target
            )
            if matched[0].numel():
                positive_class_weight = self.objectness_class_weights[matched_labels]
                weight[matched] = (
                    weight[matched] * positive_class_weight.unsqueeze(-1)
                )
            defectness_loss = F.binary_cross_entropy_with_logits(
                objectness, objectness_target, weight=weight, reduction="sum"
            ) / num_boxes
            objectness_losses = {"loss_defectness": defectness_loss}

        class_logits = outputs.get("pred_class_logits")
        if class_logits is None or not matched[0].numel():
            class_loss = objectness.sum() * 0
        else:
            positive_logits = class_logits[matched].float()
            adjusted_logits = positive_logits + (
                self.class_logit_adjustment_tau * self.conditional_class_log_priors
            )
            per_item = F.cross_entropy(
                adjusted_logits, matched_labels, weight=self.conditional_class_weights,
                label_smoothing=self.class_label_smoothing, reduction="none",
            )
            if self.class_focal_gamma > 0:
                true_probability = adjusted_logits.softmax(-1).gather(
                    1, matched_labels[:, None]
                ).squeeze(1)
                per_item = per_item * (1.0 - true_probability).pow(self.class_focal_gamma)
            class_loss = per_item.sum() / num_boxes
        ranking_losses = (
            {} if (use_decoupled or use_selection_gated)
            else self._ranking_losses(outputs, targets, indices)
        )
        native_ranking_losses = (
            self._selection_native_ranking_losses(outputs, targets)
            if (
                self.selection_native_ranking
                and outputs.get("is_dn") is None
                and "aux_outputs" in outputs
            )
            else {}
        )
        return {
            **objectness_losses,
            "loss_class": class_loss,
            **ranking_losses,
            **native_ranking_losses,
        }

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        if loss == "hierarchical":
            return self.loss_hierarchical(outputs, targets, indices, num_boxes, **kwargs)
        return super().get_loss(loss, outputs, targets, indices, num_boxes, **kwargs)
