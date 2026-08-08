"""Real P2-P6 FPN and local-dominant global context fusion."""
from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from .backbone import group_norm


class ConvGNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                                   group_norm(out_channels), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def _sample_global_context(global_map: torch.Tensor, view_xyxy: torch.Tensor, image_sizes: torch.Tensor,
                           output_size: tuple[int, int]) -> torch.Tensor:
    """Sample global features at the original-image coordinates of each local crop.

    Inputs use deterministic square resizing.  Consequently a crop's source
    coordinates map linearly to the global feature map without padded-image
    ambiguity.
    """
    batch, _, map_h, map_w = global_map.shape
    output_h, output_w = output_size
    device, dtype = global_map.device, global_map.dtype
    y = (torch.arange(output_h, device=device, dtype=dtype) + .5) / output_h
    x = (torch.arange(output_w, device=device, dtype=dtype) + .5) / output_w
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    grids = []
    for index in range(batch):
        x1, y1, x2, y2 = view_xyxy[index].to(dtype)
        image_w, image_h = image_sizes[index, 0].to(dtype), image_sizes[index, 1].to(dtype)
        source_x, source_y = x1 + xx * (x2 - x1), y1 + yy * (y2 - y1)
        # grid_sample coordinates refer to feature cell centres.
        norm_x = ((source_x / image_w) * map_w + .5) / map_w * 2 - 1
        norm_y = ((source_y / image_h) * map_h + .5) / map_h * 2 - 1
        grids.append(torch.stack((norm_x, norm_y), dim=-1))
    return F.grid_sample(global_map, torch.stack(grids), mode="bilinear", align_corners=False)


class GlobalLocalFPN(nn.Module):
    """FPN where P2 remains local and only P3/P4 receive gated global context."""

    def __init__(self, in_channels: dict[str, int], out_channels: int = 256):
        super().__init__()
        self.lateral = nn.ModuleDict({name: ConvGNAct(channels, out_channels, 1) for name, channels in in_channels.items()})
        self.output = nn.ModuleDict({name: ConvGNAct(out_channels, out_channels) for name in ("P2", "P3", "P4", "P5")})
        self.p6 = ConvGNAct(out_channels, out_channels, stride=2)
        self.context_projection = nn.ModuleDict({name: ConvGNAct(out_channels, out_channels, 1) for name in ("P3", "P4")})
        self.context_gate = nn.ModuleDict({name: nn.Conv2d(out_channels * 2, out_channels, 1) for name in ("P3", "P4")})
        for gate in self.context_gate.values():
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, -2.0)  # start local-dominant, not global-free.

    def _single_fpn(self, features: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
        lateral = {name: self.lateral[name](tensor) for name, tensor in features.items()}
        for high, low in (("P5", "P4"), ("P4", "P3"), ("P3", "P2")):
            lateral[low] = lateral[low] + F.interpolate(lateral[high], size=lateral[low].shape[-2:], mode="nearest")
        result = OrderedDict((name, self.output[name](lateral[name])) for name in ("P2", "P3", "P4", "P5"))
        result["P6"] = self.p6(result["P5"])
        return result

    def forward(self, local_features: OrderedDict[str, torch.Tensor], global_features: OrderedDict[str, torch.Tensor],
                view_xyxy: torch.Tensor, image_sizes: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        local = self._single_fpn(local_features)
        global_fpn = self._single_fpn(global_features)
        for name in ("P3", "P4"):
            context = _sample_global_context(global_fpn[name], view_xyxy, image_sizes, local[name].shape[-2:])
            context = self.context_projection[name](context)
            gate = self.context_gate[name](torch.cat((local[name], context), dim=1)).sigmoid()
            local[name] = local[name] + gate * context
        return local
