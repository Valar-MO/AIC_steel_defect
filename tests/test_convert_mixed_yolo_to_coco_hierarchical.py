import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.convert_mixed_yolo_to_coco_hierarchical import convert


class HierarchicalCocoConversionTest(unittest.TestCase):
    def test_preserves_nine_class_ids_and_reuses_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mixed = root / "mixed"
            rows = []
            for split, class_id in (("train", 2), ("val", 8)):
                image = mixed / "images" / split / f"{split}.jpg"
                label = mixed / "labels" / split / f"{split}.txt"
                image.parent.mkdir(parents=True); label.parent.mkdir(parents=True)
                Image.new("RGB", (100, 80)).save(image)
                label.write_text(f"{class_id} 0.5 0.5 0.4 0.5\n", encoding="utf-8")
                rows.append({"sample_path":str(image),"label_path":str(label),
                             "source_image_id":f"source-{split}.jpg","source_split":split,
                             "sample_type":"whole","crop_xyxy":"","classes":"x"})
            with (mixed/"sample_manifest.csv").open("w",encoding="utf-8",newline="") as handle:
                writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            classes = root/"classes.yaml"
            classes.write_text("names: [a,b,c,d,e,f,g,h,i]\n",encoding="utf-8")
            output=root/"coco"
            report=convert(mixed,classes,output)
            train=json.loads((output/"annotations"/"instances_train.json").read_text())
            val=json.loads((output/"annotations"/"instances_val.json").read_text())
            self.assertEqual(train["annotations"][0]["category_id"],2)
            self.assertEqual(val["annotations"][0]["category_id"],8)
            self.assertEqual(len(train["categories"]),9)
            self.assertEqual(train["images"][0]["file_name"],"images/train/train.jpg")
            self.assertEqual(report["splits"]["train"]["class_counts"],{"c":1})


if __name__ == "__main__":
    unittest.main()
