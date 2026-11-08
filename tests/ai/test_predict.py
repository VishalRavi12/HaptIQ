"""Tests for src/ghostgrip/ai/predict.py. No camera or hardware needed."""
import math
from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))   # ghostgrip package
sys.path.insert(0, str(REPO_ROOT))           # shared contracts.py

from contracts import HandStateV2, Prediction  # noqa: E402
from ghostgrip.ai import predict as P  # noqa: E402
from ghostgrip.ai.predict import ErrorTracker, KalmanPredictor, LOST_SIGMA, evaluate  # noqa: E402
from ghostgrip.ai.synthetic import SCENARIOS, make_scenario  # noqa: E402


def hand(t_ms, x=0.5, y=0.5, ok=True, vel=(0.0, 0.0)):
    return HandStateV2(t_ms=t_ms, pointer_xy=(x, y), closure_q=0.0, dq_dt=0.0, velocity_xy=vel,
                       stable_score=1.0 if ok else 0.0, tracking_ok=ok,
                       tracking_quality=0.9 if ok else 0.0)


def feed(predictor, samples):
    last = None
    for s in samples:
        last = predictor.update(s)
    return last


def ramp(rate_x=0.4, frames=40, step_ms=33, x0=0.2):
    """Pointer moving right at rate_x (normalised units per second)."""
    return [hand(i * step_ms, x0 + rate_x * i * step_ms / 1000.0) for i in range(frames)]


class TestFilter(unittest.TestCase):
    def test_a_stationary_hand_is_predicted_to_stay(self):
        p = feed(KalmanPredictor(), [hand(i * 33, 0.4, 0.6) for i in range(40)])
        self.assertAlmostEqual(p.future_pointer_xy[0], 0.4, delta=1e-3)
        self.assertAlmostEqual(p.future_pointer_xy[1], 0.6, delta=1e-3)
        self.assertLess(max(p.sigma_xy), 0.05)

    def test_constant_velocity_is_extrapolated(self):
        predictor = KalmanPredictor(horizon_ms=120)
        samples = ramp(rate_x=0.4)
        p = feed(predictor, samples)
        expected = samples[-1].pointer_xy[0] + 0.4 * 0.120
        self.assertAlmostEqual(p.future_pointer_xy[0], expected, delta=0.003)
        self.assertAlmostEqual(predictor.velocity_xy[0], 0.4, delta=0.02)
        self.assertEqual((p.t_query_ms, p.horizon_ms), (samples[-1].t_ms, 120))

    def test_uncertainty_is_high_at_first_and_shrinks(self):
        predictor = KalmanPredictor()
        first = predictor.update(hand(0))
        self.assertGreater(max(first.sigma_xy), 0.1)
        later = feed(predictor, [hand(i * 33) for i in range(1, 30)])
        self.assertLess(max(later.sigma_xy), 0.05)
        self.assertLess(max(later.sigma_xy), max(first.sigma_xy) / 3)

    def test_uncertainty_grows_with_the_horizon(self):
        data = ramp()
        near = feed(KalmanPredictor(horizon_ms=50), data)
        far = feed(KalmanPredictor(horizon_ms=300), data)
        self.assertGreater(far.sigma_xy[0], near.sigma_xy[0])

    def test_predictions_are_clamped_to_the_frame(self):
        right = feed(KalmanPredictor(horizon_ms=300), ramp(rate_x=1.0, x0=0.5, frames=15))
        self.assertEqual(right.future_pointer_xy[0], 1.0)
        left = feed(KalmanPredictor(horizon_ms=300), ramp(rate_x=-1.0, x0=0.5, frames=15))
        self.assertEqual(left.future_pointer_xy[0], 0.0)

    def test_irregular_time_steps_are_handled(self):
        steps = [20, 60, 33, 45, 25, 33, 70, 28, 33, 40] * 6
        t, samples = 0, []
        for s in steps:
            t += s
            samples.append(hand(t, 0.2 + 0.25 * t / 1000.0))
        p = feed(KalmanPredictor(horizon_ms=100), samples)
        expected = samples[-1].pointer_xy[0] + 0.25 * 0.100
        self.assertAlmostEqual(p.future_pointer_xy[0], expected, delta=0.005)
        self.assertTrue(all(math.isfinite(v) for v in (*p.future_pointer_xy, *p.sigma_xy)))

    def test_results_are_deterministic(self):
        a = [KalmanPredictor().update(s) for s in ramp()]
        b = [KalmanPredictor().update(s) for s in ramp()]
        self.assertEqual(a, b)


class TestFailureHandling(unittest.TestCase):
    def test_tracking_loss_returns_maximum_uncertainty_and_holds_the_pointer(self):
        predictor = KalmanPredictor()
        feed(predictor, ramp())
        lost = predictor.update(hand(5000, 0.7, 0.3, ok=False))
        self.assertEqual(lost.sigma_xy, (LOST_SIGMA, LOST_SIGMA))
        self.assertEqual(lost.future_pointer_xy, (0.7, 0.3))
        self.assertIsNone(predictor.velocity_xy)

    def test_recovery_starts_fresh_with_no_old_velocity(self):
        predictor = KalmanPredictor()
        feed(predictor, ramp(rate_x=0.8))
        predictor.update(hand(4000, ok=False))
        first = predictor.update(hand(4033, 0.9, 0.5))
        self.assertEqual(first.future_pointer_xy, (0.9, 0.5))     # no leftover velocity
        self.assertGreater(max(first.sigma_xy), 0.1)              # and it knows it knows little

    def test_non_finite_pointer_counts_as_lost(self):
        predictor = KalmanPredictor()
        feed(predictor, ramp())
        for bad in (float("nan"), float("inf")):
            p = predictor.update(hand(9000, bad, 0.5))
            self.assertEqual(p.sigma_xy, (LOST_SIGMA, LOST_SIGMA))
            self.assertTrue(all(math.isfinite(v) for v in p.future_pointer_xy))

    def test_repeated_and_out_of_order_timestamps_are_ignored(self):
        predictor = KalmanPredictor()
        samples = ramp(frames=20)
        before = feed(predictor, samples)
        same = predictor.update(hand(samples[-1].t_ms, 0.99, 0.01))
        older = predictor.update(hand(samples[-1].t_ms - 100, 0.0, 0.0))
        self.assertEqual((same, older), (before, before))
        self.assertEqual(predictor.ignored_samples, 2)
        after = predictor.update(hand(samples[-1].t_ms + 33, samples[-1].pointer_xy[0] + 0.0165))
        self.assertTrue(all(math.isfinite(v) for v in (*after.future_pointer_xy, *after.sigma_xy)))
        self.assertGreater(after.t_query_ms, before.t_query_ms)

    def test_a_long_gap_restarts_the_filter(self):
        predictor = KalmanPredictor(max_gap_ms=500)
        feed(predictor, ramp(rate_x=0.8, frames=20))
        p = predictor.update(hand(10_000, 0.1, 0.9))
        self.assertEqual(p.future_pointer_xy, (0.1, 0.9))
        self.assertGreater(max(p.sigma_xy), 0.1)

    def test_bad_parameters_are_rejected(self):
        for kwargs in ({"horizon_ms": 0}, {"horizon_ms": -5}, {"process_noise": -1.0},
                       {"process_noise": float("nan")}, {"measurement_noise": 0.0},
                       {"measurement_noise": float("inf")}, {"initial_velocity_sigma": 0.0},
                       {"max_gap_ms": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                KalmanPredictor(**kwargs)

    def test_horizon_comes_from_the_config(self):
        from ghostgrip.config import load_config
        config = load_config()
        self.assertEqual(KalmanPredictor.from_config(config).horizon_ms,
                         config.ai.prediction_horizon_ms)


class TestErrorTracker(unittest.TestCase):
    @staticmethod
    def perfect_run(rate=0.5, frames=40, step=30, horizon=120, tracker=None):
        tracker = tracker or ErrorTracker()
        for i in range(frames):
            t = i * step
            x = 0.2 + rate * t / 1000.0
            future = (x + rate * horizon / 1000.0, 0.5)
            prediction = Prediction(t, horizon, future, (0.01, 0.01))
            tracker.record(hand(t, x, 0.5, vel=(rate, 0.0)), prediction)
        return tracker

    def test_perfect_forecasts_score_near_zero_and_baselines_do_not(self):
        s = self.perfect_run().summary()
        self.assertGreater(s["scored"], 25)
        self.assertLess(s["kalman"], 1e-9)
        self.assertEqual(s["within_2_sigma"], 1.0)
        self.assertLess(s["constant_velocity"], 1e-9)             # exact velocity supplied
        self.assertAlmostEqual(s["no_movement"], 0.5 * 0.120, delta=1e-6)

    def test_a_prediction_is_scored_only_after_its_target_time(self):
        tracker = ErrorTracker()
        tracker.record(hand(0, 0.2), Prediction(0, 120, (0.3, 0.5), (0.01, 0.01)))
        tracker.record(hand(30, 0.21), Prediction(30, 120, (0.3, 0.5), (0.01, 0.01)))
        self.assertEqual(tracker.scored, 0)                       # 120 ms have not passed
        tracker.record(hand(150, 0.4), Prediction(150, 120, (0.5, 0.5), (0.01, 0.01)))
        self.assertGreaterEqual(tracker.scored, 1)

    def test_wrong_forecast_gets_the_distance_it_missed_by(self):
        tracker = ErrorTracker()
        for i in range(10):
            t = i * 30
            tracker.record(hand(t, 0.5), Prediction(t, 90, (0.6, 0.5), (0.01, 0.01)))
        s = tracker.summary()
        self.assertGreater(s["scored"], 3)
        self.assertAlmostEqual(s["kalman"], 0.1, places=6)
        self.assertAlmostEqual(s["no_movement"], 0.0, places=9)

    def test_forecasts_spanning_a_tracking_loss_are_not_scored(self):
        tracker = ErrorTracker()
        for t in range(0, 301, 30):
            tracker.record(hand(t, 0.5), Prediction(t, 120, (0.5, 0.5), (0.01, 0.01)))
        for t in range(630, 1000, 30):
            tracker.record(hand(t, 0.5), Prediction(t, 120, (0.5, 0.5), (0.01, 0.01)))
        self.assertGreater(tracker.unscorable, 0)

    def test_lost_frames_and_nothing_known_predictions_are_not_recorded(self):
        tracker = ErrorTracker()
        tracker.record(hand(0, ok=False), Prediction(0, 120, (0.5, 0.5), (LOST_SIGMA, LOST_SIGMA)))
        tracker.record(hand(30), Prediction(30, 120, (0.5, 0.5), (LOST_SIGMA, LOST_SIGMA)))
        self.assertEqual((len(tracker._observations), len(tracker._pending)), (1, 0))

    def test_repeated_frame_is_not_scored_twice(self):
        tracker = ErrorTracker()
        state = hand(0)
        prediction = Prediction(0, 60, (0.5, 0.5), (0.01, 0.01))
        for _ in range(3):
            tracker.record(state, prediction)
        self.assertEqual(len(tracker._pending), 1)

    def test_buffers_are_bounded(self):
        tracker = ErrorTracker(window=10, capacity=8)
        self.perfect_run(frames=300, tracker=tracker)
        self.assertLessEqual(len(tracker._observations), 8)
        self.assertLessEqual(len(tracker._pending), 8)
        self.assertLessEqual(len(tracker._recent), 10)

    def test_worse_than_baseline_needs_enough_data_and_real_evidence(self):
        self.assertFalse(ErrorTracker().is_worse_than_baseline())            # no data: not claimed
        good = self.perfect_run()
        self.assertFalse(good.is_worse_than_baseline(min_samples=20))
        bad = ErrorTracker()
        for i in range(80):
            t = i * 30
            bad.record(hand(t, 0.5), Prediction(t, 60, (0.8, 0.5), (0.01, 0.01)))   # always 0.3 off
        self.assertTrue(bad.is_worse_than_baseline(min_samples=20))
        self.assertFalse(bad.is_worse_than_baseline(min_samples=10_000))
        moving = ErrorTracker()
        for i in range(80):
            t = i * 30
            x = 0.1 + 0.4 * t / 1000.0
            moving.record(hand(t, x), Prediction(t, 60, (x + 0.3, 0.5), (0.01, 0.01)))   # baselines miss by ~0.024
        self.assertTrue(moving.is_worse_than_baseline(min_samples=20))
        self.assertFalse(moving.is_worse_than_baseline(min_samples=20, margin=1e3))

    def test_calibration_counts_errors_inside_the_claimed_box(self):
        def run(sigma):
            tracker = ErrorTracker()
            for i in range(30):
                t = i * 30
                tracker.record(hand(t, 0.5), Prediction(t, 90, (0.55, 0.5), (sigma, sigma)))   # always 0.05 off
            return tracker.calibration()
        self.assertEqual(run(0.04), 1.0)        # 2 sigma = 0.08 covers a 0.05 miss
        self.assertEqual(run(0.01), 0.0)        # 2 sigma = 0.02 does not
        self.assertTrue(math.isnan(ErrorTracker().calibration()))

    def test_the_mean_claimed_uncertainty_is_tracked(self):
        s = self.perfect_run().summary()                 # every forecast claims sigma 0.01
        self.assertAlmostEqual(s["mean_sigma"], 0.01)
        self.assertTrue(math.isnan(ErrorTracker().mean_sigma()))

    def test_the_larger_axis_is_the_claimed_uncertainty(self):
        tracker = ErrorTracker()
        for i in range(10):
            t = i * 30
            tracker.record(hand(t, 0.5), Prediction(t, 60, (0.5, 0.5), (0.01, 0.04)))
        self.assertAlmostEqual(tracker.mean_sigma(), 0.04)

    def test_bad_tracker_parameters_are_rejected(self):
        for kwargs in ({"window": 0}, {"capacity": 0}, {"max_bracket_ms": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ErrorTracker(**kwargs)


class TestOnSyntheticTraces(unittest.TestCase):
    """Measured on synthetic data only: a regression check, not evidence about real hands."""
    MOVING = ("clean", "noisy", "fast_reach", "slow_close", "glitch", "frame_drops",
              "dropout_short", "dropout_long", "low_quality")

    def test_every_scenario_yields_finite_valid_predictions(self):
        for name in SCENARIOS:
            with self.subTest(scenario=name):
                predictor = KalmanPredictor()
                for state in make_scenario(name):
                    p = predictor.update(state)
                    for v in (*p.future_pointer_xy, *p.sigma_xy):
                        self.assertTrue(math.isfinite(v))
                    self.assertTrue(all(0.0 <= v <= 1.0 for v in p.future_pointer_xy))
                    if not state.tracking_ok:
                        self.assertEqual(p.sigma_xy, (LOST_SIGMA, LOST_SIGMA))

    def test_kalman_beats_both_baselines_when_the_hand_moves(self):
        for name in self.MOVING:
            with self.subTest(scenario=name):
                s = evaluate(make_scenario(name))
                self.assertGreater(s["scored"], 100)
                best_baseline = min(s["no_movement"], s["constant_velocity"])
                self.assertLess(s["kalman"], 0.6 * best_baseline)

    def test_kalman_loses_on_a_parked_jittery_hand_and_the_gate_notices(self):
        # Honest limitation: extrapolating jitter is worse than "stay where you are".
        predictor, tracker = KalmanPredictor(), ErrorTracker()
        for state in make_scenario("q_jitter"):
            tracker.record(state, predictor.update(state))
        s = tracker.summary()
        self.assertGreater(s["kalman"], s["no_movement"])
        self.assertTrue(tracker.is_worse_than_baseline())

    def test_claimed_uncertainty_covers_the_real_errors_when_the_hand_moves(self):
        # The uncertainty must be honest: real errors fall inside it (>= 0.85; ~0.91 is ideal),
        # and it is not absurdly large compared with the typical error.
        for name in ("clean", "noisy", "fast_reach", "slow_close", "frame_drops"):
            with self.subTest(scenario=name):
                s = evaluate(make_scenario(name))
                self.assertGreaterEqual(s["within_2_sigma"], 0.85)

    def test_uncertainty_recovers_after_a_dropout(self):
        predictor = KalmanPredictor()
        sigma_after = None
        for state in make_scenario("dropout_short"):
            p = predictor.update(state)
            if state.t_ms > 2000:
                sigma_after = max(p.sigma_xy)
        self.assertLess(sigma_after, 0.05)

    def test_the_module_command_prints_a_table(self):
        import contextlib
        import io
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            P.main(["--horizon-ms", "120"])
        text = out.getvalue()
        self.assertIn("kalman", text)
        for name in SCENARIOS:
            self.assertIn(name, text)


if __name__ == "__main__":
    unittest.main()
