"""Tests for src/ghostgrip/metrics/logger.py. No camera or hardware needed."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))   # ghostgrip package
sys.path.insert(0, str(REPO_ROOT))           # shared contracts.py

from contracts import CueEvent, DeviceState, HandStateV2, Prediction, TrialOutcome  # noqa: E402
from ghostgrip.ai.adapt import AssistDecision  # noqa: E402
from ghostgrip.ai.intent import IntentState  # noqa: E402
from ghostgrip.metrics import logger as L  # noqa: E402
from ghostgrip.metrics.logger import SessionLogger, read_session, read_table  # noqa: E402


def hand(t=100, x=0.4, y=0.6, q=0.2, ok=True, quality=0.9):
    return HandStateV2(t, (x, y), q, 0.0, (0.01, -0.02), 1.0 if ok else 0.0, ok, quality if ok else 0.0)


def outcome(trial=1, cond="A", success=True, reason="", start=100, end=900, slips=0, over=0):
    return TrialOutcome("s1", trial, cond, success, reason, start, end, slips, over)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name) / "session"
        self.addCleanup(self.tmp.cleanup)

    def open(self, session="s1", origin="synthetic"):
        return SessionLogger(self.folder, session, origin)

    def rows(self, name):
        return read_session(self.folder)[name]


class TestSetup(Base):
    def test_creates_three_tables_with_headers(self):
        self.open().close()
        for filename, fields in L.FILES.values():
            with (self.folder / filename).open(newline="") as f:
                self.assertEqual(tuple(next(csv.reader(f))), fields)

    def test_origin_must_be_stated_and_valid(self):
        for bad in (None, "", "fake", "Real"):
            with self.subTest(origin=bad), self.assertRaises(ValueError):
                SessionLogger(self.folder, "s1", bad)

    def test_session_id_must_be_anonymous_looking(self):
        for bad in ("jane@example.com", "Jane Doe", "", "a" * 65, "x/y"):
            with self.subTest(session=bad), self.assertRaises(ValueError):
                SessionLogger(self.folder, bad, "real")
        SessionLogger(self.folder, "P01_session-2", "real").close()

    def test_a_table_with_the_wrong_columns_is_refused_not_mixed(self):
        self.folder.mkdir(parents=True)
        (self.folder / "frames.csv").write_text("a,b,c\n1,2,3\n")
        with self.assertRaisesRegex(ValueError, "expected columns"):
            self.open()


class TestRows(Base):
    def test_a_full_frame_row(self):
        with self.open() as log:
            ok = log.log_frame(hand(250), Prediction(250, 120, (0.41, 0.59), (0.03, 0.04)),
                               IntentState("HOLDING", "glass", 200, False, "steady"),
                               AssistDecision(0.46, "ok"), trial_id=3, condition="B")
        self.assertTrue(ok)
        (row,) = self.rows("frames")
        self.assertEqual((row["origin"], row["session_id"], row["run"], row["trial_id"], row["condition"]),
                         ("synthetic", "s1", "0", "3", "B"))
        self.assertEqual((row["t_ms"], row["tracking_ok"], row["intent_state"]), ("250", "1", "HOLDING"))
        self.assertEqual((row["pointer_x"], row["future_x"], row["sigma_y"], row["horizon_ms"]),
                         ("0.400000", "0.410000", "0.040000", "120"))
        self.assertEqual((row["assist_gain"], row["assist_reason"]), ("0.460000", "ok"))

    def test_missing_things_are_empty_never_zero(self):
        with self.open() as log:
            log.log_frame(hand(10))
        (row,) = self.rows("frames")
        for column in ("trial_id", "condition", "intent_state", "horizon_ms", "future_x", "future_y",
                       "sigma_x", "sigma_y", "assist_gain", "assist_reason"):
            self.assertEqual(row[column], "", column)

    def test_non_finite_numbers_are_blanked_and_counted(self):
        with self.open() as log:
            log.log_frame(HandStateV2(5, (float("nan"), 0.5), float("inf"), 0.0, (0.0, 0.0), 1.0, True, 0.9))
            self.assertEqual(log.blanked, 2)
        (row,) = self.rows("frames")
        self.assertEqual((row["pointer_x"], row["closure_q"], row["pointer_y"]), ("", "", "0.500000"))

    def test_intent_can_be_given_as_a_plain_string(self):
        with self.open() as log:
            log.log_frame(hand(1), intent_state="IDLE")
        self.assertEqual(self.rows("frames")[0]["intent_state"], "IDLE")

    def test_device_rows_keep_absent_sensors_empty_and_the_arduino_clock_separate(self):
        device = DeviceState(uptime_ms=172_802, armed=True, touch=False, piezo=None,
                             ax_g=None, ay_g=None, az_g=None, rotary_raw=512, last_ack_seq=104)
        with self.open() as log:
            log.log_device(2500, device, CueEvent(104, "CONTACT", 95.0, 150))
            log.log_device(2550, device)
        first, second = self.rows("device")
        self.assertEqual((first["t_ms"], first["uptime_ms"]), ("2500", "172802"))   # two clocks, two columns
        self.assertEqual((first["piezo"], first["ax_g"], first["touch"], first["armed"]), ("", "", "0", "1"))
        self.assertEqual((first["cue_kind"], first["cue_seq"], first["cue_angle_deg"]),
                         ("CONTACT", "104", "95.000000"))
        self.assertEqual((second["cue_kind"], second["cue_seq"]), ("", ""))

    def test_a_condition_must_be_a_known_code(self):
        with self.open() as log:
            for bad in ("baseline", "E", ""):
                with self.subTest(condition=bad), self.assertRaises(ValueError):
                    log.log_frame(hand(1), condition=bad)
            with self.assertRaises(ValueError):
                log.begin_trial(1, None, 0)

    def test_nothing_can_be_written_after_closing(self):
        log = self.open()
        log.close()
        log.close()                                          # closing twice is fine
        with self.assertRaises(RuntimeError):
            log.log_frame(hand(1))


class TestTimestamps(Base):
    def test_a_backwards_timestamp_is_skipped_counted_and_never_written(self):
        with self.open() as log:
            self.assertTrue(log.log_frame(hand(100)))
            self.assertTrue(log.log_frame(hand(100)))        # equal is allowed
            self.assertFalse(log.log_frame(hand(90)))
            self.assertTrue(log.log_frame(hand(110)))
            self.assertEqual(log.rejected, 1)
        self.assertEqual([r["t_ms"] for r in self.rows("frames")], ["100", "100", "110"])

    def test_each_table_has_its_own_clock_check(self):
        device = DeviceState(1, True, False, False, 0.0, 0.0, 1.0, 0, 0)
        with self.open() as log:
            log.log_frame(hand(5000))
            self.assertTrue(log.log_device(10, device))      # device time is independent of frame time
            self.assertEqual(log.rejected, 0)


class TestTrials(Base):
    def test_start_then_end(self):
        with self.open() as log:
            log.begin_trial(1, "A", 100)
            self.assertEqual(log.open_trials, [1])
            log.end_trial(outcome(1, "A", slips=1))
            self.assertEqual(log.open_trials, [])
        start, end = self.rows("trials")
        self.assertEqual((start["event"], start["t_ms"], start["success"]), ("start", "100", ""))
        self.assertEqual((end["event"], end["success"], end["start_ms"], end["end_ms"], end["slip_count"]),
                         ("end", "1", "100", "900", "1"))

    def test_a_trial_id_cannot_be_reused_in_a_run(self):
        # Regression: ids that restarted per condition silently merged different trials into one.
        with self.open() as log:
            log.begin_trial(1, "A", 0)
            log.end_trial(outcome(1, "A", start=0, end=500))
            with self.assertRaisesRegex(ValueError, "already used"):
                log.begin_trial(1, "B", 600)

    def test_closing_flags_a_trial_that_never_ended(self):
        with self.open() as log:
            log.begin_trial(1, "A", 100)
            log.begin_trial(2, "B", 200)
            log.end_trial(outcome(2, "B", start=200, end=800))
        events = [(r["trial_id"], r["event"], r["fail_reason"]) for r in self.rows("trials")]
        self.assertIn(("1", "incomplete", "logger_closed"), events)
        self.assertNotIn(("2", "incomplete", "logger_closed"), events)

    def test_abandoning_an_unknown_trial_does_nothing(self):
        with self.open() as log:
            self.assertFalse(log.abandon_trial(99))


class TestRestart(Base):
    def crash_midway(self):
        """Write some rows and 'crash': files are flushed but close() is never called."""
        log = self.open()
        log.begin_trial(1, "A", 100)
        log.log_frame(hand(150), trial_id=1, condition="A")
        log.begin_trial(2, "B", 300)
        log.end_trial(outcome(2, "B", start=300, end=900))
        log.begin_trial(3, "C", 1000)
        log.log_frame(hand(1100), trial_id=3, condition="C")
        return log

    def test_unfinished_trials_from_an_earlier_run_are_flagged_on_restart(self):
        first = self.crash_midway()
        second = self.open()
        self.assertEqual(second.recovered, [(0, 1), (0, 3)])
        second.close()
        first._files["trials"].close()
        rows = self.rows("trials")
        flagged = [(r["run"], r["trial_id"], r["event"], r["fail_reason"]) for r in rows if r["event"] == "incomplete"]
        self.assertEqual(flagged, [("0", "1", "incomplete", "interrupted"), ("0", "3", "incomplete", "interrupted")])
        self.assertNotIn("2", [r["trial_id"] for r in rows if r["event"] == "incomplete"])

    def test_the_new_run_has_its_own_number_and_its_own_clock(self):
        first = self.crash_midway()
        with self.open() as second:
            self.assertEqual(second.run, 1)
            self.assertTrue(second.log_frame(hand(5)))       # the clock restarted at 0; that is not "backwards"
            self.assertEqual(second.rejected, 0)
        first._files["trials"].close()
        self.assertEqual(sorted({r["run"] for r in self.rows("frames")}), ["0", "1"])

    def test_earlier_data_stays_readable_and_headers_are_not_repeated(self):
        first = self.crash_midway()
        before = len(self.rows("frames"))
        self.open().close()
        first._files["trials"].close()
        self.assertEqual(len(self.rows("frames")), before)
        text = (self.folder / "frames.csv").read_text()
        self.assertEqual(text.count("run,session_id"), 1)

    def test_an_interrupted_trial_is_flagged_only_once(self):
        first = self.crash_midway()
        self.open().close()
        self.open().close()
        first._files["trials"].close()
        incomplete = [r for r in self.rows("trials") if r["event"] == "incomplete" and r["trial_id"] == "1"]
        self.assertEqual(len(incomplete), 1)

    def test_trial_ids_may_repeat_across_runs(self):
        with self.open() as a:
            a.begin_trial(1, "A", 0)
            a.end_trial(outcome(1, "A", start=0, end=400))
        with self.open() as b:                               # a new run starts numbering again
            b.begin_trial(1, "A", 0)
            b.end_trial(outcome(1, "A", start=0, end=400))
        self.assertEqual(sorted({r["run"] for r in self.rows("trials")}), ["0", "1"])

    def test_read_table_of_a_missing_file_is_empty(self):
        self.assertEqual(read_table(self.folder / "nope.csv", L.FRAME_FIELDS), [])


if __name__ == "__main__":
    unittest.main()
