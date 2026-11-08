from dataclasses import asdict
import unittest

from contracts import HandStateV2
from src.ghostgrip.vision.pipeline import PerceptionPipeline


def hand(gap=0.1, center=0.5):
    points = [(center, 0.5)] * 21
    points[5], points[17] = (center - 0.05, 0.5), (center + 0.05, 0.5)
    points[4], points[8] = (center - gap / 2, 0.5), (center + gap / 2, 0.5)
    return points


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = PerceptionPipeline({'d_open': 1.0, 'd_closed': 0.1})

    def test_shared_type_and_fields(self):
        state = self.pipeline.update(1000, hand(center=0.7), 640, 480)
        self.assertIsInstance(state, HandStateV2)
        self.assertEqual(len(asdict(state)), 8)
        self.assertFalse(hasattr(state, 'reason'))
        self.assertTrue(state.tracking_ok)
        self.assertAlmostEqual(state.pointer_xy[0], 0.7)  # No extra mirroring.
        self.assertAlmostEqual(state.closure_q, 0.0)
        self.assertEqual(state.tracking_quality, 1.0)

    def test_pinch_maps_to_one(self):
        state = self.pipeline.update(1000, hand(gap=0.01), 640, 480)
        self.assertAlmostEqual(state.closure_q, 1.0)

    def test_loss_holds_state_and_clears_all_dynamic_fields(self):
        previous = self.pipeline.update(1000, hand(gap=0.01), 640, 480)
        lost = self.pipeline.update(1033, None, 640, 480)
        self.assertFalse(lost.tracking_ok)
        self.assertEqual(lost.pointer_xy, previous.pointer_xy)
        self.assertEqual(lost.closure_q, previous.closure_q)
        self.assertEqual((lost.velocity_xy, lost.dq_dt, lost.stable_score, lost.tracking_quality),
                         ((0, 0), 0, 0, 0))
        resumed = self.pipeline.update(1066, hand(center=0.7), 640, 480)
        self.assertTrue(resumed.tracking_ok)
        self.assertEqual(resumed.velocity_xy, (0, 0))

    def test_jump_and_timestamp_rejection_zero_quality(self):
        self.pipeline.update(1000, hand(center=0.2), 640, 480)
        jumped = self.pipeline.update(1033, hand(center=0.8), 640, 480)
        self.assertFalse(jumped.tracking_ok)
        self.assertEqual(jumped.tracking_quality, 0)
        self.assertEqual(self.pipeline.reason, 'jump rejected')
        stale = self.pipeline.update(999, hand(center=0.2), 640, 480)
        self.assertFalse(stale.tracking_ok)
        self.assertEqual(stale.tracking_quality, 0)

    def test_stream_includes_missing_observations(self):
        observations = [(1000, None, 640, 480), (1033, hand(), 640, 480)]
        states = list(self.pipeline.stream(observations))
        self.assertEqual([s.tracking_ok for s in states], [False, True])
        self.assertEqual(states[0].pointer_xy, (0.5, 0.5))
        self.assertTrue(all(isinstance(s, HandStateV2) for s in states))

    def test_invalid_geometry_and_timestamp(self):
        points = hand()
        points[8] = (float('nan'), 0.5)
        self.assertFalse(self.pipeline.update(1000, points, 640, 480).tracking_ok)
        for timestamp in (1.5, -1, True):
            with self.assertRaises(ValueError):
                self.pipeline.update(timestamp, hand(), 640, 480)
