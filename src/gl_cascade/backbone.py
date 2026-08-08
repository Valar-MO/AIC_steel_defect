"""Shared ResNet-50 + deformable adapters backbone.

The detector interacts only with ``forward_features``.  InternImage-S can
replace this module later without changing the FPN, RPN, or ROI interfaces.
"""
from __future__ import annotations

from collections import OrderedDict
import sys
from pathlib import Path

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


class InternImageSBackbone(nn.Module):
    """Official InternImage-S DCNv3 backbone exposing real P2-P5 features.

    The official repository remains external because it owns the compiled
    DCNv3 operator.  This adapter only imports its published architecture and
    checkpoint; all GL-Cascade modules stay in this repository.
    """

    out_channels = OrderedDict(P2=80, P3=160, P4=320, P5=640)

    def __init__(self, internimage_root: str | Path, checkpoint_path: str | Path | None = None,
                 with_checkpointing: bool = False):
        super().__init__()
        root = Path(internimage_root).expanduser().resolve()
        classification_root = root / "classification"
        if not (classification_root / "models" / "intern_image.py").is_file():
            raise FileNotFoundError(f"Official InternImage classification source not found under {classification_root}")
        if str(classification_root) not in sys.path:
            sys.path.insert(0, str(classification_root))
        from models.intern_image import InternImage  # Imported only after DCNv3 is installed.
        self.model = InternImage(core_op="DCNv3", channels=80, depths=[4, 4, 21, 4], groups=[5, 10, 20, 40],
                                 num_classes=0, mlp_ratio=4., drop_path_rate=.3, norm_layer="LN", layer_scale=1.,
                                 offset_scale=1., post_norm=True, with_cp=with_checkpointing)
        if checkpoint_path is not None:
            checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
            state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
            state = {key.removeprefix("module."): value for key, value in state.items()}
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            forbidden_missing = [key for key in missing if not key.startswith("head.")]
            forbidden_unexpected = [key for key in unexpected if not key.startswith("head.")]
            if forbidden_missing or forbidden_unexpected:
                raise RuntimeError(f"InternImage-S checkpoint mismatch; missing={forbidden_missing[:8]}, unexpected={forbidden_unexpected[:8]}")

    def forward_features(self, image: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        # Official sequence outputs are channels-last and emitted before each
        # stage downsample: stride 4, 8, 16, 32 respectively.
        outputs = self.model.forward_features_seq_out(image)
        return OrderedDict((name, tensor.permute(0, 3, 1, 2).contiguous())
                           for name, tensor in zip(("P2", "P3", "P4", "P5"), outputs))


def build_backbone(name: str, *, pretrained: bool, internimage_root: str | Path | None = None,
                   internimage_checkpoint: str | Path | None = None, with_checkpointing: bool = False) -> nn.Module:
    if name == "r50_dcn":
        return ResNet50DCNBackbone(pretrained=pretrained)
    if name == "internimage_s":
        if internimage_root is None:
            raise ValueError("internimage_s requires --internimage-root")
        if pretrained and internimage_checkpoint is None:
            raise ValueError("internimage_s training requires the official --internimage-checkpoint")
        return InternImageSBackbone(internimage_root, internimage_checkpoint if pretrained else None, with_checkpointing)
    raise ValueError(f"Unsupported backbone {name!r}")
