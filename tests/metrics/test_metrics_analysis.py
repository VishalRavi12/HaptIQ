"""Tests for src/ghostgrip/metrics/analysis.py. No camera or hardware needed."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))   # ghostgrip package
sys.path.insert(0, str(REPO_ROOT))           # shared contracts.py

from contracts import HandStateV2, TrialOutcome  # noqa: E402
from ghostgrip.ai.adapt import AssistDecision  # noqa: E402
from ghostgrip.ai.predict import KalmanPredictor, ErrorTracker, evaluate  # noqa: E402
from ghostgrip.ai.synthetic import make_scenario  # noqa: E402
from ghostgrip.metrics import analysis as A  # noqa: E402
from ghostgrip.metrics.analysis import (TrialRecord, assist_availability, classification_report,  # noqa: E402
                                        format_summary, prediction_error, summarize_trials, trial_records)
from ghostgrip.metrics.logger import SessionLogger, read_session  # noqa: E402


def rec(trial, cond="A", success=True, reason="", start=0, end=1000, slips=0, over=0,
        origin="synthetic", status="complete", run=0):
    return TrialRecord(run, "s", origin, trial, cond, status, success, reason,
                       start, None if status == "incomplete" else end, slips, over)


class TestTrialRecords(unittest.TestCase):
    def test_start_end_and_incomplete_events_merge_into_one_record_per_trial(self):
        with tempfile.TemporaryDirectory() as tmp:
            with SessionLogger(tmp, "s1", "synthetic") as log:
                log.begin_trial(1, "A", 100)
                log.end_trial(TrialOutcome("s1", 1, "A", True, "", 100, 900, 2, 1))
                log.begin_trial(2, "B", 1000)                  # never ends
            records = trial_records(read_session(tmp)["trials"])
        done, cut = records
        self.assertEqual((done.status, done.success, done.duration_ms, done.slip_count, done.overgrip_count),
                         ("complete", True, 800, 2, 1))
        self.assertEqual((cut.status, cut.success, cut.fail_reason, cut.duration_ms), ("incomplete", False, "logger_closed", None))
        self.assertFalse(cut.valid)

    def test_a_trial_that_started_and_was_never_closed_is_still_incomplete(self):
        rows = [dict(run="0", session_id="s", origin="real", trial_id="7", condition="C", event="start",
                     t_ms="50", success="", fail_reason="", start_ms="", end_ms="", slip_count="", overgrip_count="")]
        (r,) = trial_records(rows)
        self.assertEqual((r.status, r.fail_reason, r.start_ms), ("incomplete", "interrupted", 50))

    def test_tracking_lost_is_invalid_and_incomplete_is_invalid(self):
        self.assertTrue(rec(1).valid)
        self.assertFalse(rec(2, success=False, reason="tracking_lost").valid)
        self.assertFalse(rec(3, status="incomplete", success=False, reason="interrupted").valid)
        self.assertTrue(rec(4, success=False, reason="timeout").valid)       # a real failure


class TestSummaries(unittest.TestCase):
    def test_rates_use_valid_trials_only_and_invalid_ones_are_still_counted(self):
        trials = [rec(1, start=0, end=4000), rec(2, start=0, end=6000), rec(3, start=0, end=5000),
                  rec(4, success=False, reason="timeout"),
                  rec(5, success=False, reason="damaged_overgrip", over=1),
                  rec(6, success=False, reason="tracking_lost"),
                  rec(7, status="incomplete", success=False, reason="interrupted")]
        (s,) = summarize_trials(trials)
        self.assertEqual((s.trials, s.valid, s.tracking_lost, s.incomplete), (7, 5, 1, 1))
        self.assertEqual((s.successes, s.success_rate), (3, 3 / 5))
        self.assertEqual((s.median_time_ms, s.min_time_ms, s.max_time_ms), (5000, 4000, 6000))
        self.assertEqual((s.timeouts, s.overgrip_rate), (1, 1 / 5))
        self.assertEqual(s.failures, {"timeout": 1, "damaged_overgrip": 1})

    def test_times_come_only_from_successful_trials(self):
        trials = [rec(1, start=0, end=3000), rec(2, success=False, reason="timeout", start=0, end=30000)]
        (s,) = summarize_trials(trials)
        self.assertEqual(s.median_time_ms, 3000)

    def test_slip_and_overgrip_rates_count_trials_not_events(self):
        trials = [rec(1, slips=3), rec(2), rec(3, over=2), rec(4)]
        (s,) = summarize_trials(trials)
        self.assertEqual((s.slip_rate, s.overgrip_rate), (1 / 4, 1 / 4))

    def test_nothing_valid_gives_none_not_zero(self):
        (s,) = summarize_trials([rec(1, success=False, reason="tracking_lost")])
        self.assertEqual((s.valid, s.success_rate, s.median_time_ms, s.slip_rate, s.overgrip_rate),
                         (0, None, None, None, None))

    def test_real_and_synthetic_are_never_pooled(self):
        trials = [rec(1, origin="synthetic"), rec(2, origin="real", success=False, reason="timeout")]
        out = summarize_trials(trials)
        self.assertEqual([(s.origin, s.success_rate) for s in out], [("real", 0.0), ("synthetic", 1.0)])

    def test_conditions_come_out_in_a_to_d_order(self):
        out = summarize_trials([rec(i, cond=c) for i, c in enumerate("DBAC")])
        self.assertEqual([s.condition for s in out], list("ABCD"))

    def test_the_text_table_labels_synthetic_data_and_explains_invalid_trials(self):
        text = format_summary(summarize_trials([rec(1), rec(2, success=False, reason="tracking_lost")]))
        self.assertIn("SYNTHETIC", text)
        self.assertIn("NOT experimental results", text)
        self.assertIn("not counted as failures", text)
        real = format_summary(summarize_trials([rec(1, origin="real")]))
        self.assertIn("REAL data", real)
        self.assertNotIn("SYNTHETIC", real)

    def test_the_summary_csv_has_blank_cells_for_missing_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out" / "s.csv"
            A.write_summary_csv(path, summarize_trials([rec(1, success=False, reason="tracking_lost")]))
            (row,) = list(csv.DictReader(path.open()))
        self.assertEqual((row["valid"], row["success_rate"], row["median_time_ms"]), ("0", "", ""))

    @unittest.skipUnless(__import__("importlib").util.find_spec("matplotlib"), "matplotlib not installed")
    def test_the_plot_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p" / "plot.png"
            A.plot_success_rates(summarize_trials([rec(1, cond="A"), rec(2, cond="B", origin="real")]), path)
            self.assertGreater(path.stat().st_size, 1000)
        with self.assertRaises(ValueError):
            A.plot_success_rates([], "x.png")


class TestPredictionError(unittest.TestCase):
    def logged(self, folder, scenario="clean", condition="B", seed=0):
        states = make_scenario(scenario, seed)
        predictor, tracker = KalmanPredictor(), ErrorTracker()
        with SessionLogger(folder, "s1", "synthetic") as log:
            for s in states:
                log.log_frame(s, predictor.update(s), None, AssistDecision(0.0, "x"), condition=condition)
        return states

    def test_recomputing_from_the_log_matches_scoring_live(self):
        with tempfile.TemporaryDirectory() as tmp:
            states = self.logged(tmp)
            result = prediction_error(read_session(tmp)["frames"])
        live = evaluate(states)
        e = result[("synthetic", "B")]
        self.assertEqual(e["scored"], live["scored"])
        for method in ("kalman", "no_movement", "constant_velocity"):
            self.assertAlmostEqual(e[method], live[method], delta=2e-5)     # the log rounds to 6 decimals
        self.assertLess(e["kalman"], e["constant_velocity"])

    def test_frames_without_a_prediction_or_condition_are_not_scored(self):
        with tempfile.TemporaryDirectory() as tmp:
            with SessionLogger(tmp, "s1", "synthetic") as log:
                for s in make_scenario("clean"):
                    log.log_frame(s)                                         # no prediction, no condition
            self.assertEqual(prediction_error(read_session(tmp)["frames"]), {})

    def test_separate_runs_and_conditions_do_not_mix(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.logged(tmp, condition="A")
            self.logged(tmp, condition="B", scenario="noisy")
            result = prediction_error(read_session(tmp)["frames"])
        self.assertEqual(sorted(result), [("synthetic", "A"), ("synthetic", "B")])
        self.assertNotEqual(result[("synthetic", "A")]["kalman"], result[("synthetic", "B")]["kalman"])


class TestAssistAvailability(unittest.TestCase):
    def logged(self, plan):
        """plan: {condition: [(reason, gain), ...]}; returns the frame rows."""
        with tempfile.TemporaryDirectory() as tmp:
            with SessionLogger(tmp, "s1", "synthetic") as log:
                t = 0
                for condition, steps in plan.items():
                    for reason, gain in steps:
                        t += 33
                        hand = HandStateV2(t, (0.5, 0.5), 0.0, 0.0, (0.0, 0.0), 1.0, True, 0.9)
                        log.log_frame(hand, None, None, AssistDecision(gain, reason), condition=condition)
            return read_session(tmp)["frames"]

    def test_the_share_counts_only_frames_where_the_condition_allows_assistance(self):
        rows = self.logged({"A": [("condition_off", 0.0)] * 5,
                            "B": [("ok", 0.4)] * 6 + [("predictor_worse_than_baseline", 0.0)] * 2
                                 + [("tracking_lost", 0.0)] * 2})
        a, b = assist_availability(rows)
        self.assertEqual((a.condition, a.frames, a.permitted, a.active, a.share_active), ("A", 5, 0, 0, None))
        self.assertEqual((b.condition, b.frames, b.permitted, b.active), ("B", 10, 10, 6))
        self.assertAlmostEqual(b.share_active, 0.6)
        self.assertEqual(b.reasons, {"ok": 6, "predictor_worse_than_baseline": 2, "tracking_lost": 2})

    def test_a_gain_of_zero_is_not_active_even_with_an_ok_reason(self):
        rows = self.logged({"D": [("ok", 0.0), ("ok", 0.3)]})
        self.assertEqual(assist_availability(rows)[0].active, 1)

    def test_frames_without_a_condition_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            with SessionLogger(tmp, "s1", "synthetic") as log:
                log.log_frame(make_scenario("clean")[0], None, None, AssistDecision(0.5, "ok"))
            self.assertEqual(assist_availability(read_session(tmp)["frames"]), [])

    def test_the_results_table_says_how_much_assistance_the_conditions_really_got(self):
        rows = self.logged({"B": [("ok", 0.4)] * 3 + [("tracking_lost", 0.0)]})
        text = format_summary(summarize_trials([rec(1, cond="B")]), None, assist_availability(rows))
        self.assertIn("assistance actually on", text)
        self.assertIn("B: 75% of 4 frames", text)
        self.assertIn("tracking_lost 1", text)
        plain = format_summary(summarize_trials([rec(1, cond="B")]))
        self.assertNotIn("assistance actually on", plain)

    def test_a_condition_with_assistance_always_off_is_not_listed(self):
        rows = self.logged({"A": [("condition_off", 0.0)] * 3})
        text = format_summary(summarize_trials([rec(1, cond="A")]), None, assist_availability(rows))
        self.assertNotIn("assistance actually on", text)     # nothing was ever allowed, so nothing to say


class TestClassificationReport(unittest.TestCase):
    def test_a_perfect_run(self):
        r = classification_report(["a", "b", "a"], ["a", "b", "a"])
        self.assertEqual((r["accuracy"], r["macro_f1"], r["n"]), (1.0, 1.0, 3))
        self.assertEqual(r["per_class"]["a"], {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 2})

    def test_hand_computed_confusion(self):
        truth = ["open", "open", "open", "hold", "hold", "hold", "hold", "release"]
        pred = ["open", "open", "hold", "hold", "hold", "hold", "open", "open"]
        r = classification_report(truth, pred)
        self.assertEqual(r["confusion"]["open"], {"hold": 1, "open": 2, "release": 0})
        self.assertEqual(r["confusion"]["hold"], {"hold": 3, "open": 1, "release": 0})
        self.assertAlmostEqual(r["per_class"]["hold"]["precision"], 3 / 4)
        self.assertAlmostEqual(r["per_class"]["hold"]["recall"], 3 / 4)
        self.assertAlmostEqual(r["per_class"]["open"]["precision"], 2 / 4)
        self.assertAlmostEqual(r["per_class"]["open"]["recall"], 2 / 3)
        self.assertAlmostEqual(r["accuracy"], 5 / 8)
        self.assertEqual(r["per_class"]["release"], {"precision": None, "recall": 0.0, "f1": 0.0, "support": 1})
        # a class that exists but is never predicted counts as 0 in the macro average, not as a gap
        f1_open = 2 * (2 / 4) * (2 / 3) / ((2 / 4) + (2 / 3))
        self.assertAlmostEqual(r["macro_f1"], (f1_open + 0.75 + 0.0) / 3)

    def test_an_undefined_ratio_is_none_not_zero(self):
        r = classification_report(["a", "a"], ["a", "a"], labels=["a", "b"])
        self.assertEqual(r["per_class"]["b"], {"precision": None, "recall": None, "f1": None, "support": 0})
        self.assertEqual(r["macro_f1"], 1.0)                                 # the empty class is not averaged in

    def test_bad_input_is_rejected(self):
        for args in (([], []), (["a"], ["a", "b"])):
            with self.assertRaises(ValueError):
                classification_report(*args)
        with self.assertRaises(ValueError):
            classification_report(["a"], ["z"], labels=["a", "b"])


if __name__ == "__main__":
    unittest.main()
