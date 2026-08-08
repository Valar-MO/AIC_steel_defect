import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.build_gtaware_dfine_dataset import (  # noqa: E402
    intersection,
    labels_for_crop,
    make_crop,
)


class FixedRng:
    @staticmethod
    def uniform(low, high):
        return (low + high) / 2


def test_make_crop_stays_inside_image_and_keeps_minimum_side():
    crop = make_crop([5, 5, 20, 10], 4096, 3000, 3.0, 512, 1536, 0.1, FixedRng())
    assert crop == (0, 0, 512, 512)


def test_make_crop_caps_large_targets_and_image_boundaries():
    crop = make_crop([3500, 2600, 500, 300], 4096, 3000, 6.0, 512, 1536, 0, FixedRng())
    assert crop[2:] == (1536, 1536)
    assert crop[0] + crop[2] <= 4096
    assert crop[1] + crop[3] <= 3000


def test_visibility_and_ambiguous_omitted_annotation_policy():
    annotations = [
        {"id": 1, "category_id": 2, "bbox": [100, 100, 100, 100]},
        {"id": 2, "category_id": 0, "bbox": [450, 100, 100, 100]},
    ]
    labels, ambiguous = labels_for_crop(annotations, (0, 0, 500, 500), 0.7, 0.1)
    assert [row["source_annotation_id"] for row in labels] == [1]
    assert ambiguous  # Half of annotation 2 would otherwise become unlabeled texture.


def test_intersection_reports_source_box_visibility():
    clipped, visible = intersection([100, 100, 200, 100], (150, 50, 100, 200))
    assert clipped == (150, 100, 250, 200)
    assert visible == 0.5


if __name__ == "__main__":
    test_make_crop_stays_inside_image_and_keeps_minimum_side()
    test_make_crop_caps_large_targets_and_image_boundaries()
    test_visibility_and_ambiguous_omitted_annotation_policy()
    test_intersection_reports_source_box_visibility()
    print("GT_AWARE_DATASET_TESTS_OK count=4")
