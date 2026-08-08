"""Shared ResNet-50 + deformable adapters backbone.

The detector interacts only with ``forward_features``.  InternImage-S can
replace this module later without changing the FPN, RPN, or ROI interfaces.
"""
from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.ops import DeformConv2d


class FrozenBatchNorm2d(nn.Module):
    """Frozen affine/statistics BN compatible with ImageNet ResNet weights."""

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.weight * torch.rsqrt(self.running_var + self.eps)
        bias = self.bias - self.running_mean * scale
        return x * scale.reshape(1, -1, 1, 1) + bias.reshape(1, -1, 1, 1)

    @classmethod
    def convert(cls, module: nn.Module) -> nn.Module:
        for name, child in list(module.named_children()):
            if isinstance(child, nn.BatchNorm2d):
                frozen = cls(child.num_features, child.eps)
                frozen.weight.copy_(child.weight.detach())
                frozen.bias.copy_(child.bias.detach())
                frozen.running_mean.copy_(child.running_mean.detach())
                frozen.running_var.copy_(child.running_var.detach())
                setattr(module, name, frozen)
            else:
                cls.convert(child)
        return module


def group_norm(channels: int) -> nn.GroupNorm:
    groups = min(32, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class DeformableAdapter(nn.Module):
    """Residual DCN refinement with zero offsets at initialization."""

    def __init__(self, channels: int):
        super().__init__()
        self.offset_mask = nn.Conv2d(channels, 27, 3, padding=1)
        self.dcn = DeformConv2d(channels, channels, 3, padding=1, bias=False)
        self.norm = group_norm(channels)
        self.act = nn.GELU()
        nn.init.zeros_(self.offset_mask.weight)
        nn.init.zeros_(self.offset_mask.bias)
        nn.init.zeros_(self.dcn.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset_mask = self.offset_mask(x)
        offset, mask = offset_mask[:, :18], offset_mask[:, 18:].sigmoid()
        return x + self.act(self.norm(self.dcn(x, offset, mask)))


class ResNet50DCNBackbone(nn.Module):
    out_channels = OrderedDict(P2=256, P3=512, P4=1024, P5=2048)

    def __init__(self, pretrained: bool = True, train_backbone: bool = True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        base = resnet50(weights=weights)
        FrozenBatchNorm2d.convert(base)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = base.layer1, base.layer2, base.layer3, base.layer4
        self.adapters = nn.ModuleDict({"P2": DeformableAdapter(256), "P3": DeformableAdapter(512),
                                       "P4": DeformableAdapter(1024), "P5": DeformableAdapter(2048)})
        if not train_backbone:
            for parameter in self.parameters():
                parameter.requires_grad_(False)

    def forward_features(self, image: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        x = self.stem(image)
        p2 = self.adapters["P2"](self.layer1(x))
        p3 = self.adapters["P3"](self.layer2(p2))
        p4 = self.adapters["P4"](self.layer3(p3))
        p5 = self.adapters["P5"](self.layer4(p4))
        return OrderedDict(P2=p2, P3=p3, P4=p4, P5=p5)
