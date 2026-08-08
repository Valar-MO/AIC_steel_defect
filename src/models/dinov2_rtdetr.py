"""RT-DETR detector using a local DINOv2 feature-pyramid backbone."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from .dinov2_pyramid import Dinov2FeaturePyramid


class Dinov2RTDetrBackbone(nn.Module):
    """Adapt :class:`Dinov2FeaturePyramid` to the RT-DETR backbone API."""

    def __init__(self, pyramid: Dinov2FeaturePyramid, out_channels: int) -> None:
        super().__init__()
        self.model = pyramid
        self.intermediate_channel_sizes = [out_channels, out_channels, out_channels]

    def forward(self, pixel_values: Tensor, pixel_mask: Tensor):
        feature_maps = self.model(pixel_values).values()
        output = []
        for feature_map in feature_maps:
            mask = nn.functional.interpolate(
                pixel_mask[:, None].float(),
                size=feature_map.shape[-2:],
                mode="nearest",
            )[:, 0].to(torch.bool)
            output.append((feature_map, mask))
        return output


def _make_input_projections(channels: int, hidden_dim: int, eps: float) -> nn.ModuleList:
    return nn.ModuleList(
        nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim, eps=eps),
        )
        for _ in range(3)
    )


def build_dinov2_rtdetr(
    dinov2_weights: str | Path,
    *,
    num_classes: int = 9,
    out_channels: int = 256,
    selected_layers: tuple[int, ...] = (3, 6, 9, 12),
    freeze_backbone: bool = True,
    rtdetr_weights: str | Path | None = None,
):
    """Build an RT-DETR detector and replace its convolutional backbone.

    When ``rtdetr_weights`` is provided, the encoder and decoder are initialized
    from that local Hugging Face checkpoint. The original classification layer
    is resized to ``num_classes``. With no checkpoint, the detector portion is
    randomly initialized and is intended only for integration tests.
    """
    from transformers import RTDetrConfig, RTDetrForObjectDetection

    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    pyramid = Dinov2FeaturePyramid(
        dinov2_weights,
        out_channels=out_channels,
        selected_layers=selected_layers,
        freeze_backbone=freeze_backbone,
    )

    if rtdetr_weights is None:
        config = RTDetrConfig(
            num_labels=num_classes,
            encoder_in_channels=[out_channels] * 3,
            decoder_in_channels=[256, 256, 256],
            feat_strides=[7, 14, 28],
            num_feature_levels=3,
        )
        detector = RTDetrForObjectDetection(config)
    else:
        checkpoint = Path(rtdetr_weights)
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"RT-DETR checkpoint not found: {checkpoint}")
        detector = RTDetrForObjectDetection.from_pretrained(
            str(checkpoint),
            local_files_only=True,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        detector.config.encoder_in_channels = [out_channels] * 3
        detector.config.decoder_in_channels = [256, 256, 256]
        detector.config.feat_strides = [7, 14, 28]
        detector.config.num_feature_levels = 3

    detector.model.backbone = Dinov2RTDetrBackbone(pyramid, out_channels)
    detector.model.encoder_input_proj = _make_input_projections(
        out_channels,
        detector.config.encoder_hidden_dim,
        detector.config.batch_norm_eps,
    )
    return detector
