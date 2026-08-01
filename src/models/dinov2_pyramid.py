"""DINOv2 backbone with a simple three-level feature pyramid.

The module converts ViT patch tokens into feature maps suitable for a
multi-scale detector. It intentionally contains no detector-specific code so
the backbone can be validated independently before wiring it into RT-DETR.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn


class ConvNormAct(nn.Sequential):
    """Convolution followed by GroupNorm and GELU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        transpose: bool = False,
    ) -> None:
        conv_cls = nn.ConvTranspose2d if transpose else nn.Conv2d
        convolution = conv_cls(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        groups = 32 if out_channels % 32 == 0 else 1
        super().__init__(convolution, nn.GroupNorm(groups, out_channels), nn.GELU())


class Dinov2FeaturePyramid(nn.Module):
    """Extract and reshape DINOv2 tokens into three feature-map levels.

    Args:
        weights: Local Hugging Face model directory.
        out_channels: Channel count of every output level.
        selected_layers: Transformer hidden-state indices to fuse. Index zero
            is the patch embedding output, so 3/6/9/12 select transformer depth.
        freeze_backbone: Disable backbone gradients and keep it in eval mode.
        backbone: Optional compatible module, primarily for unit tests.
    """

    def __init__(
        self,
        weights: str | Path,
        *,
        out_channels: int = 256,
        selected_layers: Sequence[int] = (3, 6, 9, 12),
        freeze_backbone: bool = True,
        backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if not selected_layers:
            raise ValueError("selected_layers cannot be empty")
        if min(selected_layers) < 1:
            raise ValueError("selected_layers must refer to transformer blocks (>= 1)")
        if out_channels <= 0:
            raise ValueError("out_channels must be positive")

        if backbone is None:
            from transformers import Dinov2Model

            backbone = Dinov2Model.from_pretrained(
                str(weights), local_files_only=True
            )
        self.backbone = backbone
        self.selected_layers = tuple(int(index) for index in selected_layers)
        self.freeze_backbone = bool(freeze_backbone)

        config = self.backbone.config
        self.hidden_size = int(config.hidden_size)
        patch_size = config.patch_size
        self.patch_size = int(patch_size[0] if isinstance(patch_size, (list, tuple)) else patch_size)
        num_hidden_layers = int(config.num_hidden_layers)
        if max(self.selected_layers) > num_hidden_layers:
            raise ValueError(
                f"selected layer {max(self.selected_layers)} exceeds backbone depth "
                f"{num_hidden_layers}"
            )

        self.layer_norms = nn.ModuleList(
            nn.LayerNorm(self.hidden_size) for _ in self.selected_layers
        )
        fused_channels = self.hidden_size * len(self.selected_layers)
        self.fuse = ConvNormAct(
            fused_channels, out_channels, kernel_size=1, padding=0
        )
        self.p3 = ConvNormAct(
            out_channels,
            out_channels,
            kernel_size=2,
            stride=2,
            padding=0,
            transpose=True,
        )
        self.p4 = ConvNormAct(out_channels, out_channels)
        self.p5 = ConvNormAct(out_channels, out_channels, stride=2)

        self.set_backbone_trainable(not self.freeze_backbone)

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Freeze or unfreeze every DINOv2 parameter."""
        self.freeze_backbone = not trainable
        self.backbone.requires_grad_(trainable)
        if self.freeze_backbone:
            self.backbone.eval()

    def unfreeze_last_blocks(self, count: int) -> None:
        """Train only the final ``count`` transformer blocks and final norm."""
        layers = self.backbone.encoder.layer
        if not 0 <= count <= len(layers):
            raise ValueError(f"count must be in [0, {len(layers)}], got {count}")
        self.backbone.requires_grad_(False)
        if count == 0:
            self.freeze_backbone = True
            self.backbone.eval()
            return
        self.freeze_backbone = False
        for layer in layers[-count:]:
            layer.requires_grad_(True)
        if hasattr(self.backbone, "layernorm"):
            self.backbone.layernorm.requires_grad_(True)

    def train(self, mode: bool = True) -> "Dinov2FeaturePyramid":
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _tokens_to_map(self, tokens: Tensor, height: int, width: int) -> Tensor:
        grid_h, grid_w = height // self.patch_size, width // self.patch_size
        patch_tokens = tokens[:, 1:, :]
        expected_tokens = grid_h * grid_w
        if patch_tokens.shape[1] != expected_tokens:
            raise RuntimeError(
                f"Expected {expected_tokens} patch tokens for a {grid_h}x{grid_w} "
                f"grid, received {patch_tokens.shape[1]}"
            )
        return patch_tokens.transpose(1, 2).reshape(
            tokens.shape[0], self.hidden_size, grid_h, grid_w
        )

    def forward(self, pixel_values: Tensor) -> OrderedDict[str, Tensor]:
        if pixel_values.ndim != 4 or pixel_values.shape[1] != 3:
            raise ValueError("pixel_values must have shape [batch, 3, height, width]")
        height, width = pixel_values.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"Input height and width must be divisible by patch size {self.patch_size}; "
                f"received {height}x{width}"
            )

        grad_context = torch.no_grad() if self.freeze_backbone else nullcontext()
        with grad_context:
            outputs = self.backbone(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = outputs.hidden_states
        maps = []
        for norm, index in zip(self.layer_norms, self.selected_layers):
            normalized = norm(hidden_states[index])
            maps.append(self._tokens_to_map(normalized, height, width))

        fused = self.fuse(torch.cat(maps, dim=1))
        return OrderedDict(
            p3=self.p3(fused),
            p4=self.p4(fused),
            p5=self.p5(fused),
        )
