import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.convert_voc_to_coco_agnostic import convert


XML = """<annotation><size><width>100</width><height>80</height></size>
<object><name>jieba</name><bndbox><xmin>-2</xmin><ymin>10</ymin><xmax>40</xmax><ymax>50</ymax></bndbox></object>
<object><name>zonglie</name><bndbox><xmin>50</xmin><ymin>20</ymin><xmax>90</xmax><ymax>70</ymax></bndbox></object>
</annotation>"""


class ConvertAgnosticTest(unittest.TestCase):
    def test_conversion_uses_zero_category_and_preserves_source_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            classes = root / "classes.yaml"
            classes.write_text("names: [jieba, zonglie]\n", encoding="utf-8")
            rows = []
            for split in ("train", "val"):
                image = root / f"{split}.jpg"
                image.write_bytes(b"placeholder")
                xml = root / f"{split}.xml"
                xml.write_text(XML, encoding="utf-8")
                rows.append({"image_path": str(image), "xml_path": str(xml), "split": split})
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["image_path", "xml_path", "split"])
                writer.writeheader()
                writer.writerows(rows)

            output = root / "output"
            report = convert(manifest, classes, output, image_storage="copy")
            payload = json.loads((output / "annotations" / "instances_train.json").read_text())

            self.assertEqual(payload["categories"], [
                {"id": 0, "name": "defect", "supercategory": "defect"}
            ])
            self.assertEqual({item["category_id"] for item in payload["annotations"]}, {0})
            self.assertEqual(
                {item["source_category_name"] for item in payload["annotations"]},
                {"jieba", "zonglie"},
            )
            self.assertEqual(payload["annotations"][0]["bbox"], [0.0, 10.0, 40.0, 40.0])
            self.assertEqual(report["boxes_clipped_to_image"], 2)
            self.assertEqual(
                (output / "images" / "train" / "train.jpg").read_bytes(), b"placeholder"
            )


if __name__ == "__main__":
    unittest.main()
