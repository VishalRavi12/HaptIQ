import math
import unittest

from src.ghostgrip.ai.features import MotionFeatures


class TestMotion(unittest.TestCase):
    def test_smoothing_and_derivative_direction(self):
        motion = MotionFeatures()
        motion.update(0, (0.5, 0.5), 0)
        state = motion.update(50, (0.55, 0.5), 1)
        self.assertTrue(0.5 < state['pointer_xy'][0] < 0.55)
        self.assertGreater(state['velocity_xy'][0], 0)
        self.assertGreater(state['dq_dt'], 0)
        # Filtered speeds take a short time to follow a direction reversal.
        for t in range(100, 451, 50):
            state = motion.update(t, (0.5, 0.5), 0)
        self.assertLess(state['velocity_xy'][0], 0)
        self.assertLess(state['dq_dt'], 0)

    def test_jump_rejected_then_normal_tracking_recovers(self):
        motion = MotionFeatures()
        motion.update(0, (0.2, 0.2), 0)
        state = motion.update(30, (0.9, 0.9), 1)
        self.assertFalse(state['tracking_ok'])
        self.assertEqual(state['pointer_xy'], (0.2, 0.2))
        self.assertTrue(motion.update(60, (0.21, 0.2), 0)['tracking_ok'])

    def test_loss_holds_pointer_and_reacquisition_resets_speed(self):
        motion = MotionFeatures()
        motion.update(0, (0.2, 0.2), 0)
        lost = motion.update(30)
        self.assertFalse(lost['tracking_ok'])
        self.assertEqual(lost['pointer_xy'], (0.2, 0.2))
        self.assertEqual(motion.update(60, (0.8, 0.8), 1)['velocity_xy'], (0, 0))

    def test_duplicate_and_reversed_time(self):
        for timestamp in (100, 90):
            motion = MotionFeatures()
            motion.update(100, (0.5, 0.5), 0)
            state = motion.update(timestamp, (0.6, 0.5), 1)
            self.assertFalse(state['tracking_ok'])
            self.assertEqual(state['dq_dt'], 0)
            self.assertEqual(motion.update(130, (0.6, 0.5), 1)['dq_dt'], 0)

    def test_variable_dt_and_long_gap(self):
        motion = MotionFeatures()
        for timestamp in (0, 10, 45, 125, 200):
            state = motion.update(timestamp, (0.5 + timestamp / 10000, 0.5), 0.5)
            self.assertTrue(all(math.isfinite(v) for v in state['velocity_xy']))
        self.assertEqual(motion.update(1000, (0.9, 0.9), 1)['velocity_xy'], (0, 0))

    def test_stationary_stability_and_invalid_sample(self):
        motion = MotionFeatures()
        for timestamp in (0, 30, 60):
            state = motion.update(timestamp, (0.5, 0.5), 0)
        self.assertEqual(state['stable_score'], 1)
        self.assertFalse(motion.update(90, (float('nan'), 0), 0)['tracking_ok'])

    def test_smoothing_is_consistent_across_frame_rates(self):
        outputs = []
        for step in (10, 20):
            motion = MotionFeatures()
            motion.update(0, (0.5, 0.5), 0)
            for t in range(step, 201, step):
                state = motion.update(t, (0.5, 0.5), 1)
            outputs.append(state['closure_q'])
        self.assertAlmostEqual(*outputs)

    def test_stationary_jitter_is_attenuated(self):
        motion = MotionFeatures()
        motion.update(0, (0.5, 0.5), 0.5)
        positions, speeds, closure_speeds = [], [], []
        for i in range(1, 121):
            jitter = 0.004 if i % 2 else -0.004
            state = motion.update(i * 33, (0.5 + jitter, 0.5), 0.5 + jitter * 5)
            if i > 30:
                positions.append(state['pointer_xy'][0])
                speeds.append(abs(state['velocity_xy'][0]))
                closure_speeds.append(abs(state['dq_dt']))
        self.assertLess(max(positions) - min(positions), 0.002)
        self.assertLess(max(speeds), 0.01)
        self.assertLess(max(closure_speeds), 0.05)

    def test_sustained_movement_is_not_hidden(self):
        motion = MotionFeatures()
        for i in range(61):
            state = motion.update(i * 33, (0.2 + i * 0.005, 0.5), 0.2 + i * 0.01)
        self.assertGreater(state['velocity_xy'][0], 0.1)
        self.assertGreater(state['dq_dt'], 0.2)
        self.assertGreater(state['pointer_xy'][0], 0.47)
