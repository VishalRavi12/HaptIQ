"""Tests for src/ghostgrip/metrics/trace_report.py (the real-recording check). No camera needed."""
import contextlib
import io
import math
from pathlib import Path
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))   # ghostgrip package
sys.path.insert(0, str(REPO_ROOT))           # shared contracts.py

from contracts import HandStateV2  # noqa: E402
from ghostgrip.ai.predict import evaluate  # noqa: E402
from ghostgrip.ai.synthetic import make_scenario, save_trace  # noqa: E402
from ghostgrip.metrics import trace_report as T  # noqa: E402


def hand(t, x=0.5, y=0.5, q=0.0, ok=True):
    return HandStateV2(t, (x, y), q, 0.0, (0.0, 0.0), 1.0 if ok else 0.0, ok, 0.9 if ok else 0.0)


def result(**kw):
    base = dict(scored=100, unscorable=0, kalman=0.006, no_movement=0.025, constant_velocity=0.016,
                within_2_sigma=0.95, mean_sigma=0.02)
    base.update(kw)
    return base


class TestLoading(unittest.TestCase):
    def test_both_recording_layouts_load_to_the_same_frames(self):
        states = make_scenario("noisy")
        with tempfile.TemporaryDirectory() as tmp:
            for fmt in ("vision", "simulation"):
                path = save_trace(states, Path(tmp) / f"{fmt}.csv", fmt)
                loaded = T.load_hand_trace(path)
                self.assertEqual(len(loaded), len(states), fmt)
                self.assertAlmostEqual(loaded[60].pointer_xy[0], states[60].pointer_xy[0], places=5)
                self.assertEqual([s.tracking_ok for s in loaded], [s.tracking_ok for s in states])

    def test_a_missing_or_wrong_or_empty_file_is_a_clear_error_not_a_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "no such file"):
                T.load_hand_trace(Path(tmp) / "nope.csv")
            bad = Path(tmp) / "bad.csv"
            bad.write_text("a,b,c\n1,2,3\n")
            with self.assertRaisesRegex(ValueError, "not a hand trace"):
                T.load_hand_trace(bad)
            empty = Path(tmp) / "empty.csv"
            empty.write_text("t_ms,x,y,q,tracking_ok\n")
            with self.assertRaisesRegex(ValueError, "not a hand trace"):
                T.load_hand_trace(empty)


class TestDescribe(unittest.TestCase):
    def test_frame_rate_loss_and_gaps_are_reported(self):
        f = T.describe(make_scenario("dropout_short"))
        self.assertAlmostEqual(f.median_fps, 30, delta=2)
        self.assertEqual(f.lost_segments, 1)
        self.assertTrue(120 <= f.longest_lost_ms <= 200)
        self.assertAlmostEqual(f.lost_share, 5 / 169, places=3)
        self.assertEqual(f.valid_frames + 5, f.frames)

    def test_missing_frames_show_up_as_a_gap_not_as_loss(self):
        f = T.describe(make_scenario("frame_drops"))
        self.assertGreaterEqual(f.longest_gap_ms, 250)
        self.assertEqual(f.lost_segments, 0)

    def test_a_trace_that_ends_in_loss_counts_that_stretch(self):
        states = [hand(i * 33) for i in range(5)] + [hand(165 + i * 33, ok=False) for i in range(1, 6)]
        f = T.describe(states)
        self.assertEqual((f.lost_segments, f.longest_lost_ms), (1, states[-1].t_ms - states[5].t_ms))

    def test_repeated_timestamps_are_counted_not_divided_by(self):
        f = T.describe([hand(0), hand(0), hand(33), hand(33), hand(66)])
        self.assertEqual(f.backwards_timestamps, 2)
        self.assertAlmostEqual(f.median_fps, 1000 / 33)

    def test_an_all_lost_trace_has_no_quality_or_closure(self):
        f = T.describe([hand(i * 33, ok=False) for i in range(10)])
        self.assertEqual((f.valid_frames, f.quality_median, f.closure_min, f.jitter), (0, None, None, None))
        self.assertEqual(f.lost_share, 1.0)

    def test_nothing_to_describe_is_an_error(self):
        with self.assertRaises(ValueError):
            T.describe([])


class TestScoringAndVerdict(unittest.TestCase):
    def test_score_is_the_same_as_the_predictors_own_evaluation(self):
        states = make_scenario("noisy")
        live = evaluate(states)
        mine = T.score(states, 120)
        for key in ("kalman", "no_movement", "constant_velocity", "scored"):
            self.assertEqual(mine[key], live[key])

    def test_a_clear_win_is_said_to_be_a_win(self):
        self.assertIn("beats the best baseline", T.verdict(result(kalman=0.006)))

    def test_a_marginal_result_is_not_called_a_win(self):
        text = T.verdict(result(kalman=0.0158))
        self.assertIn("only marginally better", text)
        self.assertNotIn("beats", text)

    def test_a_loss_is_said_plainly_with_what_to_do(self):
        text = T.verdict(result(kalman=0.0172, constant_velocity=0.016, no_movement=0.025))
        self.assertIn("does NOT beat constant_velocity", text)
        self.assertIn("turn predictive assistance down", text)

    def test_the_uncertainty_is_judged_too(self):
        self.assertIn("too optimistic", T.verdict(result(within_2_sigma=0.5)))
        self.assertIn("cautious", T.verdict(result(within_2_sigma=1.0)))
        self.assertNotIn("optimistic", T.verdict(result(within_2_sigma=0.93)))

    def test_nothing_scored_is_not_a_verdict(self):
        self.assertIn("nothing could be scored", T.verdict(result(scored=0)))

    def test_on_a_parked_jittery_hand_the_verdict_is_a_loss(self):
        # The honest limitation we already know about; the tool must report it, not hide it.
        self.assertIn("does NOT beat", T.verdict(T.score(make_scenario("q_jitter"), 120)))

    def test_on_a_moving_hand_the_verdict_is_a_win(self):
        self.assertIn("beats the best baseline", T.verdict(T.score(make_scenario("clean"), 120)))


class TestSweepAndSuggestion(unittest.TestCase):
    def test_the_sweep_covers_the_grid_and_reports_the_claimed_uncertainty(self):
        rows = T.sweep(make_scenario("clean"), 120, process_noises=(0.1, 1.0), measurement_noises=(4e-6, 1e-5))
        self.assertEqual(len(rows), 4)
        for row in rows:
            self.assertTrue(all(math.isfinite(row[k]) for k in ("kalman", "within_2_sigma", "mean_sigma")))
        by_q = {r["q"]: r["mean_sigma"] for r in rows if r["r"] == 4e-6}
        self.assertGreater(by_q[1.0], by_q[0.1])             # more process noise, more claimed uncertainty

    def row(self, error, cover, sigma, q=1.0):
        return {"q": q, "r": 4e-6, "kalman": error, "within_2_sigma": cover, "mean_sigma": sigma}

    def test_the_suggestion_skips_over_optimistic_and_uselessly_cautious_settings(self):
        rows = [self.row(0.0040, 0.50, 0.003, q=0.01),        # best error, but uncertainty too optimistic
                self.row(0.0050, 1.00, 0.150, q=30),          # cautious to the point of being useless
                self.row(0.0062, 0.99, 0.030, q=1),           # sensible
                self.row(0.0077, 0.99, 0.013, q=0.1)]         # sensible, but a higher error
        self.assertEqual(T.suggest(rows)["q"], 1)

    def test_no_sensible_setting_means_no_suggestion(self):
        self.assertIsNone(T.suggest([self.row(0.004, 0.5, 0.003), self.row(0.005, 1.0, 0.2)]))
        self.assertIsNone(T.suggest([]))

    def test_the_default_settings_pass_their_own_rule_on_the_synthetic_traces(self):
        row = T.sweep(make_scenario("clean"), 120, (T.DEFAULT_PROCESS_NOISE,), (T.DEFAULT_MEASUREMENT_NOISE,))[0]
        self.assertIsNotNone(T.suggest([row]))


class TestAssistProfile(unittest.TestCase):
    def test_every_frame_is_accounted_for_and_the_gain_is_bounded(self):
        states = make_scenario("dropout_long")
        frames, reasons, mean_gain = T.assist_profile(states, 120)
        self.assertEqual((frames, sum(reasons.values())), (len(states), len(states)))
        self.assertGreater(reasons["ok"], 100)
        self.assertGreaterEqual(reasons["tracking_lost"], 15)
        self.assertTrue(0.0 < mean_gain <= 1.0)

    def test_no_usable_frames_means_no_assistance(self):
        frames, reasons, mean_gain = T.assist_profile([hand(i * 33, ok=False) for i in range(20)], 120)
        self.assertEqual((reasons, mean_gain), ({"tracking_lost": 20}, 0.0))


class TestReportText(unittest.TestCase):
    def test_a_full_report_has_every_section_and_the_caveat(self):
        text = T.format_report(make_scenario("clean"), 120, do_sweep=True, intent_scene="glass", label="my.csv")
        for part in ("my.csv", "THE RECORDING", "PREDICTOR, 120 ms ahead", "THIS RECORDING", "synthetic clean",
                     "Verdict:", "ASSISTANCE", "TUNING SWEEP", "GRASP INTENT", "HOLDING",
                     "a check, not evidence"):
            self.assertIn(part, text)

    def test_the_optional_sections_are_off_by_default(self):
        text = T.format_report(make_scenario("clean"), 120)
        self.assertNotIn("TUNING SWEEP", text)
        self.assertNotIn("GRASP INTENT", text)

    def test_a_recording_that_is_too_short_says_so_and_does_not_conclude(self):
        text = T.format_report([hand(i * 33) for i in range(30)], 120)
        self.assertIn("TOO SHORT", text)
        self.assertNotIn("Verdict", text)

    def test_a_hand_that_never_closed_gets_a_pointer_to_the_likely_cause(self):
        states = [hand(i * 33, 0.3 + 0.002 * i, 0.5, q=0.02) for i in range(120)]
        self.assertIn("closure barely changed", T.format_report(states, 120))

    def test_backwards_timestamps_are_warned_about(self):
        states = make_scenario("clean")
        states = states[:50] + [states[49]] + states[50:]
        self.assertIn("WARNING", T.format_report(states, 120))

    def test_the_horizon_is_what_was_asked(self):
        self.assertIn("200 ms ahead", T.format_report(make_scenario("clean"), 200))


class TestCommand(unittest.TestCase):
    def run_cmd(self, argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = T.main(argv)
        return code, out.getvalue()

    def test_a_recording_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = save_trace(make_scenario("noisy"), Path(tmp) / "trace.csv", "vision")
            code, text = self.run_cmd([str(path), "--horizon-ms", "150", "--sweep"])
        self.assertEqual(code, 0)
        self.assertIn("150 ms ahead", text)
        self.assertIn("TUNING SWEEP", text)

    def test_the_default_horizon_comes_from_config_yaml(self):
        from ghostgrip.config import load_config
        with tempfile.TemporaryDirectory() as tmp:
            path = save_trace(make_scenario("clean"), Path(tmp) / "t.csv", "simulation")
            _, text = self.run_cmd([str(path)])
        self.assertIn(f"{load_config().ai.prediction_horizon_ms} ms ahead", text)

    def test_a_bad_path_is_an_error_message_and_exit_1(self):
        code, text = self.run_cmd(["/no/such/trace.csv"])
        self.assertEqual(code, 1)
        self.assertIn("error:", text)

    def test_an_unknown_scene_is_a_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = save_trace(make_scenario("clean"), Path(tmp) / "t.csv", "vision")
            code, text = self.run_cmd([str(path), "--intent", "kitchen"])
        self.assertEqual(code, 1)
        self.assertIn("unknown scene", text)


if __name__ == "__main__":
    unittest.main()
