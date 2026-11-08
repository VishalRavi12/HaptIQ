"""Synthetic checks: no webcam or teammate modules required."""
import math
import unittest

from src.ghostgrip.ai.features import calibrated_closure, normalized_pinch_gap


class TestPinchFeatures(unittest.TestCase):
    def test_gap_is_invariant_to_image_scale(self):
        points = ((0, 0), (3, 4), (0, 0), (10, 0))
        self.assertAlmostEqual(normalized_pinch_gap(*points), 0.5)
        scaled = [tuple(4 * v for v in p) for p in points]
        self.assertAlmostEqual(normalized_pinch_gap(*scaled), 0.5)

    def test_zero_and_tiny_palm_do_not_divide_by_zero(self):
        for width in (0, 1e-12):
            gap = normalized_pinch_gap((0, 0), (1, 0), (0, 0), (width, 0))
            self.assertTrue(math.isfinite(gap))
            self.assertGreaterEqual(calibrated_closure(gap, 1.0, 0.1), 0)
            self.assertLessEqual(calibrated_closure(gap, 1.0, 0.1), 1)

    def test_open_half_closed_and_clamping(self):
        for gap, expected in ((1.0, 0), (0.6, 0.5), (0.2, 1), (2.0, 0), (0, 1)):
            self.assertAlmostEqual(calibrated_closure(gap, 1.0, 0.2), expected)

    def test_invalid_calibration_is_rejected(self):
        for opened, closed in ((0.2, 0.2), (0.2, 0.2001), (0.2001, 0.2)):
            with self.assertRaises(ValueError):
                calibrated_closure(0.2, opened, closed)

    def test_nonfinite_input_is_rejected(self):
        for value in (float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                normalized_pinch_gap((value, 0), (0, 0), (0, 0), (1, 0))
            with self.assertRaises(ValueError):
                calibrated_closure(value, 1, 0)


if __name__ == '__main__':
    unittest.main()
