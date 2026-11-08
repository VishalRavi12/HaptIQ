import unittest

from src.ghostgrip.ai.features import tracking_quality


def centered_hand():
    # Synthetic geometry for score checks, not anatomical model validation.
    points = [(0.5, 0.5)] * 21
    points[5], points[17] = (0.45, 0.5), (0.55, 0.5)
    return points


class TestQuality(unittest.TestCase):
    def test_centered_large_hand_scores_high(self):
        self.assertEqual(tracking_quality(centered_hand(), 640, 480), 1.0)

    def test_edge_and_small_hand_reduce_score(self):
        points = centered_hand()
        points[8] = (0.01, 0.5)
        self.assertLess(tracking_quality(points, 640, 480), 0.3)
        points = centered_hand()
        points[5], points[17] = (0.49, 0.5), (0.51, 0.5)
        self.assertLess(tracking_quality(points, 640, 480), 0.3)

    def test_missing_invalid_and_degenerate_are_zero(self):
        for points in (None, [], [(0.5, 0.5)] * 21):
            self.assertEqual(tracking_quality(points, 640, 480), 0)
        for value in (-0.1, float('nan'), float('inf')):
            points = centered_hand()
            points[8] = (value, 0.5)
            self.assertEqual(tracking_quality(points, 640, 480), 0)

    def test_translation_within_center_does_not_penalize_motion(self):
        points = centered_hand()
        moved = [(x + 0.1, y + 0.1) for x, y in points]
        self.assertEqual(tracking_quality(points, 640, 480),
                         tracking_quality(moved, 640, 480))
