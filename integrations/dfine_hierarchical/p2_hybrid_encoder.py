"""A minimally invasive stride-4 extension for the official HybridEncoder."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ...core import register
from .hybrid_encoder import HybridEncoder


@register()
class P2HybridEncoder(HybridEncoder):
    """Keep the trained P3-P5 neck intact and add a separate P2 output.

    The inherited encoder is constructed with the original three inputs, so all
    of its state-dict names and semantics remain unchanged.  ``forward`` accepts
    backbone features P2-P5, runs P3-P5 through that original path, then fuses
    the raw stride-4 P2 feature with upsampled encoded P3 context.
    """

    __share__ = ["eval_spatial_size"]

    def __init__(
        self,
        p2_in_channels: int = 128,
        p2_stride: int = 4,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if p2_in_channels <= 0 or p2_stride <= 0:
            raise ValueError("p2_in_channels and p2_stride must be positive")
        hidden_dim = self.hidden_dim
        self.p2_input_proj = nn.Sequential(
            nn.Conv2d(p2_in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.p2_fuse = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        )
        nn.init.xavier_uniform_(self.p2_input_proj[0].weight)
        nn.init.xavier_uniform_(self.p2_fuse[0].weight)
        self.out_channels = [hidden_dim, *self.out_channels]
        self.out_strides = [int(p2_stride), *self.out_strides]

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(feats) != 4:
            raise ValueError(f"P2HybridEncoder expects P2-P5, got {len(feats)} features")
        p2, p3, p4, p5 = feats
        encoded = super().forward([p3, p4, p5])
        p3_context = F.interpolate(
            encoded[0], size=p2.shape[-2:], mode="bilinear", align_corners=False
        )
        p2_out = self.p2_fuse(self.p2_input_proj(p2) + p3_context)
        return [p2_out, *encoded]
