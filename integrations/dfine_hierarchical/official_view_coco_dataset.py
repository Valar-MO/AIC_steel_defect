"""COCO detection dataset that crops an official tile view at load time."""
from __future__ import annotations

from PIL import Image
import torch

from ...core import register
from .._misc import convert_to_tv_tensor
from .coco_dataset import CocoDetection


@register()
class OfficialViewCocoDetection(CocoDetection):
    """Read ``source_path`` and crop ``view_xyxy`` stored in COCO image metadata.

    Keeping virtual crops avoids materializing roughly 30GB of JPEG tiles while
    preserving exactly the view geometry used by formal inference.
    """
    def load_item(self, idx):
        image_id = self.ids[idx]
        info = self.coco.loadImgs(image_id)[0]
        source_path = info.get("source_path")
        if not source_path:
            raise ValueError("OfficialViewCocoDetection requires source_path metadata")
        with Image.open(source_path) as opened:
            image = opened.convert("RGB")
        view = info.get("view_xyxy")
        if view is not None:
            image = image.crop(tuple(int(value) for value in view))
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id))
        target = {"image_id": image_id, "image_path": source_path, "annotations": annotations}
        if self.remap_mscoco_category:
            image, target = self.prepare(image, target, category2label=self.category2label)
        else:
            image, target = self.prepare(image, target)
        target["idx"] = torch.tensor([idx])
        if "boxes" in target:
            target["boxes"] = convert_to_tv_tensor(target["boxes"], key="boxes", spatial_size=image.size[::-1])
        if "masks" in target:
            target["masks"] = convert_to_tv_tensor(target["masks"], key="masks")
        return image, target
