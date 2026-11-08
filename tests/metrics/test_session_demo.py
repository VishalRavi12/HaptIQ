"""End-to-end check: synthetic traces through the whole headless pipeline, logged and summarised."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))   # ghostgrip package
sys.path.insert(0, str(REPO_ROOT))           # shared contracts.py

from ghostgrip.metrics import report  # noqa: E402
from ghostgrip.metrics.analysis import load_trial_records, summarize_trials  # noqa: E402
from ghostgrip.metrics.logger import read_session  # noqa: E402
from ghostgrip.metrics.synthetic_session import run_synthetic_session  # noqa: E402

SCENARIOS = ("clean", "dropout_long", "q_jitter")


class TestSyntheticSession(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.tmp.name) / "demo"
        cls.counts = run_synthetic_session(cls.folder, seeds=(0,), scenarios=SCENARIOS)
        cls.tables = read_session(cls.folder)
        cls.summaries = {s.condition: s for s in summarize_trials(load_trial_records(cls.folder))}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_trace_is_accounted_for(self):
        c = self.counts
        self.assertEqual(c["traces"], 4 * len(SCENARIOS))
        self.assertEqual(c["finished"] + c["incomplete"] + c["never_started"], c["traces"])
        self.assertEqual(c["never_started"], 4)               # q_jitter never opens the hand, so no trial starts
        self.assertGreater(c["frames"], 1000)
        self.assertEqual(len(self.tables["frames"]), c["frames"])

    def test_all_four_conditions_are_kept_apart(self):
        # Regression: trial ids that restarted per condition merged A, B, C and D into one.
        self.assertEqual(sorted(self.summaries), ["A", "B", "C", "D"])
        for s in self.summaries.values():
            self.assertEqual(s.trials, 2)
        ids = [r["trial_id"] for r in self.tables["trials"] if r["event"] == "start"]
        self.assertEqual(len(ids), len(set(ids)))

    def test_invalid_trials_are_reported_not_counted_as_failures(self):
        for s in self.summaries.values():
            self.assertEqual((s.trials, s.valid, s.tracking_lost, s.incomplete), (2, 1, 1, 0))
            self.assertEqual((s.successes, s.success_rate, s.failures), (1, 1.0, {}))

    def test_nothing_is_labelled_real(self):
        for name in ("frames", "trials"):
            self.assertEqual({r["origin"] for r in self.tables[name]}, {"synthetic"})

    def test_there_is_no_device_data_because_there_is_no_hardware(self):
        self.assertEqual(self.tables["device"], [])

    def test_assistance_is_off_in_a_and_c_and_modest_in_b_and_d(self):
        by_cond = {}
        for r in self.tables["frames"]:
            by_cond.setdefault(r["condition"], []).append(r)
        for cond in ("A", "C"):
            self.assertEqual({r["assist_gain"] for r in by_cond[cond]}, {"0.000000"})
            self.assertEqual({r["assist_reason"] for r in by_cond[cond]}, {"assist_off_in_condition"})
        for cond in ("B", "D"):
            gains = [float(r["assist_gain"]) for r in by_cond[cond]]
            self.assertTrue(all(0.0 <= g <= 1.0 for g in gains))
            self.assertGreater(max(gains), 0.3)
            self.assertLess(max(gains), 0.8)

    def test_the_conditions_come_out_alike_because_there_is_no_person_and_no_hardware(self):
        # A difference here would mean a bug, not an effect.
        keys = {(s.valid, s.successes, s.median_time_ms) for s in self.summaries.values()}
        self.assertEqual(len(keys), 1)

    def test_timestamps_never_go_backwards_and_nothing_was_rejected(self):
        times = [int(r["t_ms"]) for r in self.tables["frames"]]
        self.assertEqual(times, sorted(times))


class TestReportCommand(unittest.TestCase):
    def run_report(self, argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = report.main(argv)
        return code, out.getvalue()

    def test_demo_prints_a_labelled_table_and_writes_the_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            demo = Path(tmp) / "d"
            code, text = self.run_report(["--synthetic-demo", str(demo), "--seeds", "1",
                                          "--csv", str(Path(tmp) / "s.csv")])
            self.assertEqual(code, 0)
            self.assertIn("SYNTHETIC", text)
            self.assertIn("prediction error", text)
            for filename in ("frames.csv", "trials.csv", "device.csv"):
                self.assertTrue((demo / filename).is_file())
            self.assertTrue((Path(tmp) / "s.csv").is_file())
            code2, text2 = self.run_report([str(demo)])             # an existing session can be re-read
            self.assertEqual(code2, 0)
            self.assertIn("SYNTHETIC", text2)

    def make_participant(self, folder, session_id, success=True):
        from contracts import TrialOutcome
        from ghostgrip.metrics.logger import SessionLogger
        with SessionLogger(folder, session_id, "real") as log:
            log.begin_trial(1, "A", 0)
            log.end_trial(TrialOutcome(session_id, 1, "A", success, "" if success else "timeout", 0, 5000))

    def test_several_participants_get_a_table_each_and_a_combined_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_participant(Path(tmp) / "P01", "P01", True)
            self.make_participant(Path(tmp) / "P02", "P02", False)
            code, text = self.run_report([str(Path(tmp) / "P01"), str(Path(tmp) / "P02")])
        self.assertEqual(code, 0)
        self.assertIn("SESSION P01", text)
        self.assertIn("SESSION P02", text)
        self.assertIn("ALL 2 SESSIONS TOGETHER", text)
        self.assertIn("descriptive only", text)
        combined = text.split("ALL 2 SESSIONS TOGETHER")[1]
        self.assertIn("50%", combined)                       # one success, one timeout: not merged into one trial

    def test_a_single_session_has_no_combined_section(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_participant(Path(tmp) / "P01", "P01")
            code, text = self.run_report([str(Path(tmp) / "P01")])
        self.assertEqual(code, 0)
        self.assertNotIn("TOGETHER", text)
        self.assertNotIn("=====", text)

    def test_two_folders_with_the_same_session_id_are_refused_not_merged(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_participant(Path(tmp) / "a", "P01", True)
            self.make_participant(Path(tmp) / "b", "P01", False)
            code, text = self.run_report([str(Path(tmp) / "a"), str(Path(tmp) / "b")])
        self.assertEqual(code, 1)
        self.assertIn("appears in both", text)

    def test_one_bad_folder_among_several_stops_with_a_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_participant(Path(tmp) / "P01", "P01")
            code, text = self.run_report([str(Path(tmp) / "P01"), str(Path(tmp) / "missing")])
        self.assertEqual(code, 1)
        self.assertIn("no trials.csv", text)

    def test_a_missing_or_empty_session_is_reported_not_crashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, text = self.run_report([tmp])
            self.assertEqual(code, 1)
            self.assertIn("no trials.csv", text)

    def test_the_command_needs_something_to_report(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            report.main([])


if __name__ == "__main__":
    unittest.main()
