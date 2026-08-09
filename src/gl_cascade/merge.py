"""Map local predictions to original images and run final class-wise Soft-NMS."""
from __future__ import annotations

from collections import defaultdict

import torch

from .postprocess import classwise_soft_nms


def map_tile_to_original(detection: dict[str, torch.Tensor], view_xyxy: torch.Tensor, local_size: int,
                         image_size: torch.Tensor, internal_border_penalty: float = .5) -> dict[str, torch.Tensor]:
    """Restore a local-tile detection to original coordinates.

    The penalty only changes ranking, preserving the separately calibrated
    detection score.  It applies to a box touching a tile edge that is not an
    outer edge of the source image.
    """
    boxes = detection["boxes"].clone()
    x1, y1, x2, y2 = view_xyxy.to(boxes)
    source_w, source_h = image_size.to(boxes)
    scale_x, scale_y = (x2 - x1) / local_size, (y2 - y1) / local_size
    boxes[:, 0::2] = boxes[:, 0::2] * scale_x + x1
    boxes[:, 1::2] = boxes[:, 1::2] * scale_y + y1
    ranking, scores = detection["ranking_logits"].clone(), detection["scores"].clone()
    margin = 2.0
    touches_left = (detection["boxes"][:, 0] <= margin) & (x1 > 0)
    touches_top = (detection["boxes"][:, 1] <= margin) & (y1 > 0)
    touches_right = (detection["boxes"][:, 2] >= local_size - margin) & (x2 < source_w)
    touches_bottom = (detection["boxes"][:, 3] >= local_size - margin) & (y2 < source_h)
    penalize = touches_left | touches_top | touches_right | touches_bottom
    penalty_log = float(torch.log(torch.tensor(internal_border_penalty)))
    ranking[penalize] += penalty_log
    scores[penalize] *= internal_border_penalty
    return {"boxes": boxes, "labels": detection["labels"], "scores": scores,
            "class_scores": detection["class_scores"], "ranking_logits": ranking}


def merge_tile_detections(tile_detections: list[dict[str, torch.Tensor]], soft_nms_iou: float = .80) -> dict[str, torch.Tensor]:
    if not tile_detections:
        raise ValueError("tile_detections may not be empty")
    exemplar = tile_detections[0]
    nonempty = [item for item in tile_detections if len(item["boxes"])]
    if not nonempty:
        return {"boxes": exemplar["boxes"].new_zeros((0, 4)), "labels": exemplar["labels"].new_zeros((0,), dtype=torch.long),
                "scores": exemplar["scores"].new_zeros((0,)), "class_scores": exemplar["class_scores"].new_zeros((0,)),
                "ranking_logits": exemplar["ranking_logits"].new_zeros((0,))}
    return classwise_soft_nms(torch.cat([item["boxes"] for item in nonempty]), torch.cat([item["labels"] for item in nonempty]),
                              torch.cat([item["scores"] for item in nonempty]), torch.cat([item["ranking_logits"] for item in nonempty]),
                              iou_threshold=soft_nms_iou,
                              class_scores=torch.cat([item["class_scores"] for item in nonempty]))


def group_and_merge(records: list[tuple[str, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]], local_size: int,
                    border_penalty: float = .5, soft_nms_iou: float = .80) -> dict[str, dict[str, torch.Tensor]]:
    grouped: dict[str, list[dict[str, torch.Tensor]]] = defaultdict(list)
    for image_id, detection, view, image_size in records:
        grouped[image_id].append(map_tile_to_original(detection, view, local_size, image_size, border_penalty))
    return {image_id: merge_tile_detections(items, soft_nms_iou) for image_id, items in grouped.items()}
