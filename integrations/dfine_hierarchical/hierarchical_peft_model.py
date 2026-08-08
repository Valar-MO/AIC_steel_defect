"""PEFT wrapper that protects D-FINE localization and trains only small corrections."""

from __future__ import annotations

from torch import nn

from ...core import register
from .dfine import DFINE
from .hierarchical_decoder import AdaptFormerAdapter


@register()
class HierarchicalPEFTDFINE(DFINE):
    """D-FINE with an explicit, auditable parameter-efficient trainable set.

    Freezing happens during model construction, before ``YAMLConfig`` creates
    optimizer groups, so frozen tensors are never passed to AdamW.
    """

    __inject__ = ["backbone", "encoder", "decoder"]

    def __init__(self, backbone: nn.Module, encoder: nn.Module, decoder: nn.Module,
                 train_decoder_norm=True, train_decoder_bias=False,
                 train_bbox_head=True, train_encoder_objectness=True):
        super().__init__(backbone, encoder, decoder)
        self.peft_options = {
            "train_decoder_norm": bool(train_decoder_norm),
            "train_decoder_bias": bool(train_decoder_bias),
            "train_bbox_head": bool(train_bbox_head),
            "train_encoder_objectness": bool(train_encoder_objectness),
        }
        self._configure_peft_parameters()

    @staticmethod
    def _unfreeze(module):
        if module is not None:
            module.requires_grad_(True)

    def _configure_peft_parameters(self):
        self.requires_grad_(False)
        adapters = [module for module in self.decoder.modules()
                    if isinstance(module, AdaptFormerAdapter)]
        if not adapters:
            raise ValueError("HierarchicalPEFTDFINE requires decoder adapters")
        for adapter in adapters:
            adapter.requires_grad_(True)

        # Both defectness and conditional class parameters live in these heads.
        self._unfreeze(getattr(self.decoder, "dec_score_head", None))
        if self.peft_options["train_encoder_objectness"]:
            self._unfreeze(getattr(self.decoder, "enc_score_head", None))
        if self.peft_options["train_bbox_head"]:
            self._unfreeze(getattr(self.decoder, "pre_bbox_head", None))
            self._unfreeze(getattr(self.decoder, "dec_bbox_head", None))

        if self.peft_options["train_decoder_norm"]:
            for module in self.decoder.modules():
                if isinstance(module, nn.LayerNorm):
                    module.requires_grad_(True)
        if self.peft_options["train_decoder_bias"]:
            for name, parameter in self.decoder.named_parameters():
                if name.endswith(".bias"):
                    parameter.requires_grad_(True)

    def peft_parameter_report(self):
        trainable = [(name, parameter.numel()) for name, parameter in self.named_parameters()
                     if parameter.requires_grad]
        frozen = sum(parameter.numel() for parameter in self.parameters()
                     if not parameter.requires_grad)
        groups = {
            "adapter": sum(count for name, count in trainable if "adaptformer" in name),
            "score_heads": sum(count for name, count in trainable
                               if "score_head" in name),
            "bbox_heads": sum(count for name, count in trainable
                              if "bbox_head" in name),
            "normalization_and_other": sum(count for name, count in trainable
                                            if not any(token in name for token in (
                                                "adaptformer", "score_head", "bbox_head"))),
        }
        return {
            "trainable_parameters": sum(count for _, count in trainable),
            "frozen_parameters": frozen,
            "trainable_fraction": sum(count for _, count in trainable) /
                                  max(1, sum(parameter.numel()
                                             for parameter in self.parameters())),
            "groups": groups,
            "trainable_names": [name for name, _ in trainable],
        }


@register()
class HierarchicalObjectnessRankingDFINE(DFINE):
    """Freeze the detector and train only decoder objectness linear heads."""

    __inject__ = ["backbone", "encoder", "decoder"]

    def __init__(self, backbone: nn.Module, encoder: nn.Module, decoder: nn.Module,
                 train_all_decoder_objectness_heads=True):
        super().__init__(backbone, encoder, decoder)
        self.train_all_decoder_objectness_heads = bool(
            train_all_decoder_objectness_heads
        )
        self._configure_objectness_parameters()

    def _configure_objectness_parameters(self):
        self.requires_grad_(False)
        heads = getattr(self.decoder, "dec_score_head", None)
        if heads is None or not len(heads):
            raise ValueError("Decoder does not expose dec_score_head")
        selected = heads if self.train_all_decoder_objectness_heads else heads[-1:]
        for head in selected:
            if not hasattr(head, "weight") or not hasattr(head, "bias"):
                raise ValueError("Hierarchical score head lacks objectness weight/bias")
            head.weight.requires_grad_(True)
            head.bias.requires_grad_(True)
        forbidden = [
            name for name, parameter in self.named_parameters()
            if parameter.requires_grad and (
                "class_weight" in name or "class_bias" in name
                or "bbox_head" in name or name.startswith(("backbone.", "encoder."))
            )
        ]
        if forbidden:
            raise RuntimeError(f"Forbidden trainable parameters: {forbidden}")

    def objectness_parameter_report(self):
        trainable = [
            (name, parameter.numel()) for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]
        total = sum(parameter.numel() for parameter in self.parameters())
        return {
            "trainable_parameters": sum(count for _, count in trainable),
            "total_parameters": total,
            "trainable_fraction": sum(count for _, count in trainable) / max(1, total),
            "trainable_names": [name for name, _ in trainable],
        }


@register()
class HierarchicalDecoupledObjectnessDFINE(DFINE):
    """Train nonlinear defectness/quality heads while freezing detector paths."""

    __inject__ = ["backbone", "encoder", "decoder"]

    def __init__(self, backbone: nn.Module, encoder: nn.Module, decoder: nn.Module,
                 train_all_decoder_objectness_heads=True):
        super().__init__(backbone, encoder, decoder)
        self.train_all_decoder_objectness_heads = bool(
            train_all_decoder_objectness_heads
        )
        self._configure_decoupled_objectness_parameters()

    def _configure_decoupled_objectness_parameters(self):
        self.requires_grad_(False)
        heads = getattr(self.decoder, "dec_score_head", None)
        if heads is None or not len(heads):
            raise ValueError("Decoder does not expose dec_score_head")
        selected = heads if self.train_all_decoder_objectness_heads else heads[-1:]
        modules = (
            "defectness_norm", "defectness_mlp", "quality_norm", "quality_mlp"
        )
        for head in selected:
            missing = [name for name in modules if not hasattr(head, name)]
            if missing or not hasattr(head, "defectness_residual_scale"):
                raise ValueError(
                    f"Score head is not dual-objectness capable; missing {missing}"
                )
            for name in modules:
                getattr(head, name).requires_grad_(True)
            head.defectness_residual_scale.requires_grad_(True)

        trainable = [
            name for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]
        allowed = (
            "defectness_norm", "defectness_mlp",
            "defectness_residual_scale", "quality_norm", "quality_mlp",
        )
        forbidden = [
            name for name in trainable
            if not name.startswith("decoder.dec_score_head.")
            or not any(token in name for token in allowed)
        ]
        if forbidden:
            raise RuntimeError(f"Forbidden trainable parameters: {forbidden}")

    def decoupled_objectness_parameter_report(self):
        trainable = [
            (name, parameter.numel()) for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]
        total = sum(parameter.numel() for parameter in self.parameters())
        return {
            "trainable_parameters": sum(count for _, count in trainable),
            "total_parameters": total,
            "trainable_fraction": sum(count for _, count in trainable) / max(1, total),
            "defectness_parameters": sum(
                count for name, count in trainable if "defectness_" in name
            ),
            "quality_parameters": sum(
                count for name, count in trainable if "quality_" in name
            ),
            "trainable_names": [name for name, _ in trainable],
        }


@register()
class HierarchicalSelectionGatedDFINE(DFINE):
    """Train only the final query Selection, Defectness and Quality branches."""

    __inject__ = ["backbone", "encoder", "decoder"]

    def __init__(self, backbone: nn.Module, encoder: nn.Module, decoder: nn.Module):
        super().__init__(backbone, encoder, decoder)
        self._configure_selection_gated_parameters()

    def _configure_selection_gated_parameters(self):
        self.requires_grad_(False)
        heads = getattr(self.decoder, "dec_score_head", None)
        if heads is None or not len(heads):
            raise ValueError("Decoder does not expose dec_score_head")
        head = heads[-1]
        modules = (
            "selection_norm", "selection_mlp",
            "defectness_norm", "defectness_mlp",
            "quality_norm", "quality_mlp",
        )
        missing = [name for name in modules if not hasattr(head, name)]
        if missing or not hasattr(head, "selection_residual_scale"):
            raise ValueError(
                f"Score head is not selection-gated capable; missing {missing}"
            )
        for name in modules:
            getattr(head, name).requires_grad_(True)
        head.selection_residual_scale.requires_grad_(True)

        trainable = [
            name for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]
        allowed = (
            "selection_norm", "selection_mlp", "selection_residual_scale",
            "defectness_norm", "defectness_mlp",
            "quality_norm", "quality_mlp",
        )
        forbidden = [
            name for name in trainable
            if not name.startswith(
                f"decoder.dec_score_head.{len(heads) - 1}."
            ) or not any(token in name for token in allowed)
        ]
        if forbidden:
            raise RuntimeError(f"Forbidden trainable parameters: {forbidden}")

    def selection_gated_parameter_report(self):
        trainable = [
            (name, parameter.numel()) for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]
        total = sum(parameter.numel() for parameter in self.parameters())
        return {
            "trainable_parameters": sum(count for _, count in trainable),
            "total_parameters": total,
            "trainable_fraction": sum(count for _, count in trainable) / max(1, total),
            "selection_parameters": sum(
                count for name, count in trainable if "selection_" in name
            ),
            "defectness_parameters": sum(
                count for name, count in trainable if "defectness_" in name
            ),
            "quality_parameters": sum(
                count for name, count in trainable if "quality_" in name
            ),
            "trainable_names": [name for name, _ in trainable],
        }
