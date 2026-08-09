import unittest
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


try:
    import torch
    from src.gl_cascade.merge import map_tile_to_original
    from src.gl_cascade.postprocess import classwise_soft_nms
    from src.gl_cascade.roi import CascadeROIHeads
    from src.gl_cascade.rpn import ATSSAssigner, AnchorGenerator
except ImportError:  # pragma: no cover - keeps CPU-only utility environments usable.
    torch = None


@unittest.skipIf(torch is None, "GL-Cascade tests require torch and torchvision")
class GLCascadeSupervisionTest(unittest.TestCase):
    def test_soft_nms_decays_selection_but_preserves_class_score(self):
        output = classwise_soft_nms(
            torch.tensor([[0.0, 0.0, 100.0, 100.0], [1.0, 1.0, 101.0, 101.0]]),
            torch.tensor([0, 0]), torch.tensor([0.9, 0.8]), torch.tensor([3.0, 2.0]),
        )
        self.assertEqual(len(output["boxes"]), 2)
        self.assertAlmostEqual(float(output["class_scores"][1]), 0.8, places=5)
        self.assertLess(float(output["scores"][1]), 0.3)

    def test_internal_border_penalty_changes_selection_score(self):
        detection = {"boxes": torch.tensor([[0.0, 40.0, 80.0, 120.0]]), "labels": torch.tensor([0]),
                     "scores": torch.tensor([0.8]), "class_scores": torch.tensor([0.8]),
                     "ranking_logits": torch.tensor([2.0])}
        output = map_tile_to_original(detection, torch.tensor([512.0, 0.0, 2560.0, 2048.0]), 1536,
                                      torch.tensor([4096.0, 4096.0]), internal_border_penalty=.5)
        self.assertAlmostEqual(float(output["scores"][0]), 0.4, places=5)
        self.assertAlmostEqual(float(output["class_scores"][0]), 0.8, places=5)

    def test_atss_fallback_assigns_substride_ground_truth(self):
        features = {name: torch.zeros(1, 256, height, width) for name, height, width in
                    (("P2", 64, 64), ("P3", 32, 32), ("P4", 16, 16), ("P5", 8, 8), ("P6", 4, 4))}
        anchors = AnchorGenerator()(features)
        _, positive, _ = ATSSAssigner()(anchors, {"boxes": torch.tensor([[3.8, 3.8, 4.2, 4.2]]),
                                                   "visible_fraction": torch.ones(1), "boundary_weight": torch.ones(1),
                                                   "ignore_boxes": torch.zeros((0, 4))})
        self.assertGreaterEqual(int(positive.sum()), 1)

    def test_roi_training_injects_gt_when_rpn_has_no_proposals(self):
        features = {name: torch.zeros(1, 256, height, width) for name, height, width in
                    (("P2", 32, 32), ("P3", 16, 16), ("P4", 8, 8), ("P5", 4, 4), ("P6", 2, 2))}
        target = {"boxes": torch.tensor([[20.0, 24.0, 60.0, 44.0]]), "labels": torch.tensor([2]),
                  "visible_fraction": torch.ones(1), "boundary_weight": torch.ones(1),
                  "ignore_boxes": torch.zeros((0, 4))}
        head = CascadeROIHeads(num_classes=9).train()
        output = head(features, [torch.zeros((0, 4))], (128, 128), [target])
        self.assertEqual(len(output["boxes"][0]), 1)
        self.assertTrue(torch.isfinite(sum(output["losses"].values())))


if __name__ == "__main__":
    unittest.main()
