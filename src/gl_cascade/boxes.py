"""Box operations used by the GL-Cascade training and inference paths."""
from __future__ import annotations

import torch


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1, area2 = box_area(boxes1), box_area(boxes2)
    top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    inter = (bottom_right - top_left).clamp(min=0).prod(dim=-1)
    return inter / (area1[:, None] + area2[None, :] - inter).clamp(min=1e-6)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    iou = box_iou(boxes1, boxes2)
    top_left = torch.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    enclosing = (bottom_right - top_left).clamp(min=0).prod(dim=-1).clamp(min=1e-6)
    union = box_area(boxes1)[:, None] + box_area(boxes2)[None, :] - (
        (torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:]) -
         torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])).clamp(min=0).prod(dim=-1))
    return iou - (enclosing - union) / enclosing


def clip_boxes(boxes: torch.Tensor, height: int, width: int) -> torch.Tensor:
    boxes = boxes.clone()
    boxes[..., 0::2].clamp_(min=0, max=width)
    boxes[..., 1::2].clamp_(min=0, max=height)
    return boxes


def remove_small_boxes(boxes: torch.Tensor, min_size: float) -> torch.Tensor:
    return ((boxes[:, 2] - boxes[:, 0]) >= min_size) & ((boxes[:, 3] - boxes[:, 1]) >= min_size)


def encode_boxes(reference_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    widths = (reference_boxes[:, 2] - reference_boxes[:, 0]).clamp(min=1e-6)
    heights = (reference_boxes[:, 3] - reference_boxes[:, 1]).clamp(min=1e-6)
    ctr_x, ctr_y = reference_boxes[:, 0] + .5 * widths, reference_boxes[:, 1] + .5 * heights
    target_widths = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(min=1e-6)
    target_heights = (target_boxes[:, 3] - target_boxes[:, 1]).clamp(min=1e-6)
    target_ctr_x, target_ctr_y = target_boxes[:, 0] + .5 * target_widths, target_boxes[:, 1] + .5 * target_heights
    return torch.stack(((target_ctr_x - ctr_x) / widths, (target_ctr_y - ctr_y) / heights,
                        torch.log(target_widths / widths), torch.log(target_heights / heights)), dim=1)


def decode_boxes(reference_boxes: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    widths = (reference_boxes[:, 2] - reference_boxes[:, 0]).clamp(min=1e-6)
    heights = (reference_boxes[:, 3] - reference_boxes[:, 1]).clamp(min=1e-6)
    ctr_x, ctr_y = reference_boxes[:, 0] + .5 * widths, reference_boxes[:, 1] + .5 * heights
    dx, dy, dw, dh = deltas.unbind(dim=1)
    dw, dh = dw.clamp(max=math_log(1000.0 / 16)), dh.clamp(max=math_log(1000.0 / 16))
    pred_ctr_x, pred_ctr_y = dx * widths + ctr_x, dy * heights + ctr_y
    pred_w, pred_h = torch.exp(dw) * widths, torch.exp(dh) * heights
    return torch.stack((pred_ctr_x - .5 * pred_w, pred_ctr_y - .5 * pred_h,
                        pred_ctr_x + .5 * pred_w, pred_ctr_y + .5 * pred_h), dim=1)


def math_log(value: float) -> float:
    # Keeping this here avoids a Python ``math`` import in hot tensor code.
    return float(torch.log(torch.tensor(value)).item())
