import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.convert_mixed_yolo_to_coco_agnostic import convert


class ConvertMixedTest(unittest.TestCase):
    def test_reuses_images_and_preserves_sample_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "mixed"
            rows = []
            for split, sample_type in (("train", "rare_center"), ("val", "whole")):
                image = root / "images" / split / f"{split}.jpg"
                label = root / "labels" / split / f"{split}.txt"
                image.parent.mkdir(parents=True)
                label.parent.mkdir(parents=True)
                Image.new("RGB", (100, 80)).save(image)
                label.write_text("3 0.5 0.5 0.4 0.5\n", encoding="utf-8")
                rows.append({
                    "sample_path": str(image), "label_path": str(label),
                    "source_image_id": f"source-{split}.jpg", "source_split": split,
                    "sample_type": sample_type, "crop_xyxy": "[1, 2, 101, 82]"
                    if split == "train" else "", "bbox_count": "1", "classes": "jiaza",
                })
            manifest = root / "sample_manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)

            output = Path(tmp) / "coco"
            report = convert(root, output)
            train = json.loads((output / "annotations" / "instances_train.json").read_text())
            self.assertEqual(train["categories"][0]["id"], 0)
            self.assertEqual(train["images"][0]["sample_type"], "rare_center")
            self.assertEqual(train["images"][0]["file_name"], "images/train/train.jpg")
            self.assertEqual(train["annotations"][0]["category_id"], 0)
            self.assertEqual(train["annotations"][0]["source_class_id"], 3)
            self.assertEqual(train["annotations"][0]["bbox"], [30.0, 20.0, 40.0, 40.0])
            self.assertEqual(report["splits"]["val"]["images"], 1)

    @unittest.skipIf(not hasattr(Path, "symlink_to"), "symlinks unavailable")
    def test_keeps_lexical_path_for_symlinked_whole_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "source.jpg"
            Image.new("RGB", (32, 32)).save(source)
            root = base / "mixed"
            image = root / "images" / "val" / "whole.jpg"
            label = root / "labels" / "val" / "whole.txt"
            image.parent.mkdir(parents=True); label.parent.mkdir(parents=True)
            try:
                image.symlink_to(source)
            except OSError:
                self.skipTest("symlink creation not permitted")
            label.write_text("", encoding="utf-8")
            manifest = root / "sample_manifest.csv"
            row = {
                "sample_path": str(image), "label_path": str(label),
                "source_image_id": "source.jpg", "source_split": "val",
                "sample_type": "whole", "crop_xyxy": "", "bbox_count": "0", "classes": "",
            }
            # Add one training sample because both splits are required.
            train_image = root / "images" / "train" / "train.jpg"
            train_label = root / "labels" / "train" / "train.txt"
            train_image.parent.mkdir(parents=True); train_label.parent.mkdir(parents=True)
            Image.new("RGB", (32, 32)).save(train_image); train_label.write_text("", encoding="utf-8")
            train_row = dict(row, sample_path=str(train_image), label_path=str(train_label),
                             source_image_id="train.jpg", source_split="train")
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row)); writer.writeheader()
                writer.writerows([train_row, row])
            output = base / "coco"
            convert(root, output)
            val = json.loads((output / "annotations" / "instances_val.json").read_text())
            self.assertEqual(val["images"][0]["file_name"], "images/val/whole.jpg")


if __name__ == "__main__":
    unittest.main()
