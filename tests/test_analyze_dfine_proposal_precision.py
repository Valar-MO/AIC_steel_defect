import unittest

from scripts.analyze_dfine_proposal_precision import classify_predictions, summarize_rows


class ProposalPrecisionTest(unittest.TestCase):
    def test_one_to_one_outcomes(self):
        gt = {"a.jpg": [
            {"gt_id": 0, "class_name": "jieba", "size_bucket": "tiny", "box": (0, 0, 10, 10)},
            {"gt_id": 1, "class_name": "zonglie", "size_bucket": "small", "box": (20, 20, 30, 30)},
        ]}
        records = {"a.jpg": [
            {"pred_index": 0, "score": 0.9, "box": (0, 0, 10, 10), "source": "tile", "touches_internal_border": False},
            {"pred_index": 1, "score": 0.8, "box": (0, 0, 10, 10), "source": "tile", "touches_internal_border": False},
            {"pred_index": 2, "score": 0.7, "box": (20, 20, 27, 27), "source": "whole", "touches_internal_border": False},
            {"pred_index": 3, "score": 0.6, "box": (20, 20, 24, 24), "source": "whole", "touches_internal_border": False},
            {"pred_index": 4, "score": 0.5, "box": (50, 50, 60, 60), "source": "whole", "touches_internal_border": False},
        ]}
        rows, matched = classify_predictions(records, gt, 0.5, 0.3, 0.1)
        self.assertEqual([row["outcome"] for row in rows], [
            "tp", "duplicate", "near_miss", "weak_overlap", "background"
        ])
        self.assertEqual(matched, {0})
        summary = summarize_rows(rows, gt_count=2, image_count=1, label="threshold", value=0)
        self.assertEqual(summary["tp"], 1)
        self.assertEqual(summary["strict_precision"], 0.2)
        self.assertEqual(summary["recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
