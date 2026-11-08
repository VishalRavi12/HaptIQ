"""Tests for src/ghostgrip/metrics/protocol.py (the volunteer study plan). No camera needed."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))   # ghostgrip package
sys.path.insert(0, str(REPO_ROOT))           # shared contracts.py

from contracts import TrialOutcome  # noqa: E402
from ghostgrip.metrics import protocol as P  # noqa: E402
from ghostgrip.metrics.logger import SessionLogger  # noqa: E402


class TestBalancedOrders(unittest.TestCase):
    def test_every_condition_appears_once_in_every_position(self):
        square = P.balanced_latin_square(4)
        for position in range(4):
            self.assertEqual(sorted(row[position] for row in square), [0, 1, 2, 3])
        for row in square:
            self.assertEqual(sorted(row), [0, 1, 2, 3])

    def test_every_condition_follows_every_other_exactly_once(self):
        # This is what makes it "balanced": a condition cannot always be handed over to the same successor.
        for n in (2, 4, 6, 8):
            pairs = [(row[i], row[i + 1]) for row in P.balanced_latin_square(n) for i in range(n - 1)]
            self.assertEqual(len(pairs), len(set(pairs)), n)
            self.assertEqual(len(pairs), n * (n - 1), n)

    def test_odd_or_tiny_sizes_are_refused(self):
        for bad in (0, 1, 3, 5):
            with self.assertRaises(ValueError):
                P.balanced_latin_square(bad)

    def test_the_four_orders_as_letters(self):
        orders = P.condition_orders()
        self.assertEqual(len(orders), 4)
        self.assertEqual(len({tuple(o) for o in orders}), 4)
        self.assertTrue(all(sorted(o) == ["A", "B", "C", "D"] for o in orders))
        self.assertEqual([o[0] for o in orders].count("A"), 1)       # one participant starts with each condition


class TestPlans(unittest.TestCase):
    def test_four_participants_use_all_four_orders_and_the_fifth_repeats_the_first(self):
        plans = P.make_plans(5)
        self.assertEqual(len({p.order for p in plans[:4]}), 4)
        self.assertEqual(plans[4].order, plans[0].order)
        self.assertEqual([p.session_id for p in plans], ["P01", "P02", "P03", "P04", "P05"])

    def test_the_trial_counts(self):
        plan = P.make_plans(1, trials_per_object=2)[0]
        self.assertEqual((plan.trials_per_condition, plan.total_trials), (6, 24))
        self.assertEqual(P.make_plans(1, trials_per_object=3)[0].total_trials, 36)

    def test_starting_at_a_later_participant_continues_the_rotation(self):
        later = P.make_plans(1, start=3)[0]
        self.assertEqual((later.session_id, later.order), ("P03", P.make_plans(3)[2].order))

    def test_bad_inputs_are_refused(self):
        for kwargs in (dict(participants=0), dict(participants=51), dict(participants=2, trials_per_object=0),
                       dict(participants=2, trials_per_object=11), dict(participants=2, start=0)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                P.make_plans(**kwargs)


class TestRunSheet(unittest.TestCase):
    def test_the_sheet_has_the_commands_the_keys_and_the_safeguards(self):
        plan = P.make_plans(1)[0]
        text = P.run_sheet(plan)
        self.assertIn("practice (not analysed)", text)               # practice trials must never be analysed
        self.assertIn("--session-id P01_practice", text)
        self.assertIn(f"--condition {plan.order[0]} --evaluation --session-id P01", text)   # measured: frozen
        self.assertNotIn("--evaluation --session-id P01_practice", text)                     # practice may learn
        for code in plan.order[1:]:
            self.assertIn(f"press {P.CONDITION_KEYS[code]}", text)
        for key in ("press 1 (glass)", "press 2 (sponge)", "press 3 (block)"):
            self.assertIn(key, text)
        self.assertIn("consent given", text)
        self.assertIn("never delete a trial", text)
        self.assertIn("--audit data/local/sessions/P01", text)

    def test_the_sheet_follows_the_planned_order(self):
        plan = P.make_plans(2)[1]
        text = P.run_sheet(plan)
        positions = [text.index(f"condition {code} (") for code in plan.order]
        self.assertEqual(positions, sorted(positions))

    def test_the_servo_warning_appears_only_with_real_hardware_and_cue_conditions(self):
        plan = P.make_plans(1)[0]
        self.assertNotIn("table stand", P.run_sheet(plan, hardware="mock"))
        self.assertIn("table stand", P.run_sheet(plan, hardware="auto"))

    def test_the_number_of_trials_asked_for_matches_the_plan(self):
        text = P.run_sheet(P.make_plans(1, trials_per_object=3)[0], trials_per_object=3)
        self.assertEqual(text.count("then 3 trial(s)"), 12)           # 3 objects x 4 conditions


GOOD_META = {"evaluation": True, "assist_profile_frozen": True, "manual_assist_gain": None,
             "servo_enabled": False, "input": "camera", "origin": "real"}


def write_meta(folder, run=0, **changes):
    (Path(folder) / f"run_{run}_meta.json").write_text(json.dumps({**GOOD_META, **changes}))


class TestAudit(unittest.TestCase):
    def log_session(self, folder, plan, counts=None, order=None, extra_session=None, meta=True):
        counts = counts or {c: plan.trials_per_condition for c in plan.order}
        order = order or plan.order
        t, trial = 0, 0
        with SessionLogger(folder, plan.session_id, "real") as log:
            for code in order:
                for _ in range(counts.get(code, 0)):
                    trial += 1
                    log.begin_trial(trial, code, t)
                    log.end_trial(TrialOutcome(plan.session_id, trial, code, True, "", t, t + 5000))
                    t += 6000
        if meta:
            write_meta(folder)

    def test_a_session_that_matches_the_plan_passes(self):
        plan = P.make_plans(1)[0]
        with tempfile.TemporaryDirectory() as tmp:
            self.log_session(tmp, plan)
            result = P.audit_session(tmp, plan)
        self.assertTrue(result.ok, result.problems)
        self.assertEqual(result.order_seen, plan.order)
        self.assertEqual(set(result.attempts.values()), {6})

    def test_missing_extra_and_misordered_blocks_are_all_reported(self):
        plan = P.make_plans(1)[0]
        with tempfile.TemporaryDirectory() as tmp:
            self.log_session(tmp, plan, counts={plan.order[0]: 6, plan.order[1]: 4, plan.order[2]: 8, plan.order[3]: 6},
                             order=tuple(reversed(plan.order)))
            problems = " | ".join(P.audit_session(tmp, plan).problems)
        self.assertIn("planned " + " -> ".join(plan.order), problems)
        self.assertIn("4 attempt(s), planned 6", problems)
        self.assertIn("8 attempts, planned 6", problems)
        self.assertIn("must be reported, not hidden", problems)

    def test_invalid_and_incomplete_attempts_still_count_as_attempts(self):
        plan = P.make_plans(1)[0]
        with tempfile.TemporaryDirectory() as tmp:
            with SessionLogger(tmp, "P01", "real") as log:
                trial = 0
                for code in plan.order:
                    for k in range(6):
                        trial += 1
                        log.begin_trial(trial, code, trial * 1000)
                        if k == 0:
                            continue                                  # left open: logged as incomplete on close
                        reason = "tracking_lost" if k == 1 else ""
                        log.end_trial(TrialOutcome("P01", trial, code, not reason, reason, trial * 1000, trial * 1000 + 500))
            write_meta(tmp)
            self.assertTrue(P.audit_session(tmp, plan).ok)

    def test_another_participants_trials_in_the_folder_are_flagged(self):
        plan = P.make_plans(1)[0]
        with tempfile.TemporaryDirectory() as tmp:
            self.log_session(tmp, plan)
            with SessionLogger(tmp, "P02", "real") as log:
                log.begin_trial(1, "A", 0)
            problems = " ".join(P.audit_session(tmp, plan).problems)
        self.assertIn("other session ids", problems)

    def test_an_empty_folder_is_a_problem_not_a_crash(self):
        plan = P.make_plans(1)[0]
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIn("no trials were logged", P.audit_session(tmp, plan).problems[0])


class TestRunMetadataAudit(unittest.TestCase):
    def problems(self, **changes):
        with tempfile.TemporaryDirectory() as tmp:
            write_meta(tmp, **changes)
            return P.audit_run_metadata(tmp)

    def test_a_frozen_camera_run_with_the_servo_off_is_clean(self):
        self.assertEqual(self.problems(), [])

    def test_each_departure_from_the_protocol_is_named(self):
        self.assertIn("not started with --evaluation", " ".join(self.problems(evaluation=False)))
        self.assertIn("was not frozen", " ".join(self.problems(assist_profile_frozen=False)))
        self.assertIn("manual assist gain", " ".join(self.problems(manual_assist_gain=0.5)))
        self.assertIn("servo_enabled", " ".join(self.problems(servo_enabled=True)))
        self.assertIn("not a live camera", " ".join(self.problems(input="replay", origin="synthetic")))
        self.assertIn("not a live camera", " ".join(self.problems(input="mouse")))

    def test_missing_information_is_a_problem_not_a_pass(self):
        self.assertIn("evaluation", " ".join(self.problems(evaluation=None)))
        self.assertIn("servo_enabled", " ".join(self.problems(servo_enabled=None)))
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIn("no run_*_meta.json", P.audit_run_metadata(tmp)[0])

    def test_an_unreadable_file_is_reported_not_a_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "run_0_meta.json").write_text("{not json")
            self.assertIn("cannot be read", P.audit_run_metadata(tmp)[0])

    def test_every_run_is_checked_not_just_the_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_meta(tmp, run=0)
            write_meta(tmp, run=1, evaluation=False, assist_profile_frozen=False)
            problems = P.audit_run_metadata(tmp)
        self.assertTrue(all(p.startswith("run_1") for p in problems))
        self.assertEqual(len(problems), 2)

    def test_the_session_audit_includes_it(self):
        plan = P.make_plans(1)[0]
        with tempfile.TemporaryDirectory() as tmp:
            TestAudit().log_session(tmp, plan)
            write_meta(tmp, evaluation=False)
            result = P.audit_session(tmp, plan)
        self.assertFalse(result.ok)
        self.assertIn("not started with --evaluation", " ".join(result.problems))


class TestCommand(unittest.TestCase):
    def run_cmd(self, argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = P.main(argv)
        return code, out.getvalue()

    def test_it_prints_the_orders_and_one_sheet_per_participant(self):
        code, text = self.run_cmd(["--participants", "3"])
        self.assertEqual(code, 0)
        self.assertIn("balanced Latin square", text)
        for sid in ("P01", "P02", "P03"):
            self.assertIn(f"PARTICIPANT {sid}", text)
        self.assertNotIn("PARTICIPANT P04", text)

    def test_the_audit_command_reports_a_match_or_a_mismatch(self):
        plan = P.make_plans(1)[0]
        with tempfile.TemporaryDirectory() as tmp:
            TestAudit().log_session(tmp, plan)
            code, text = self.run_cmd(["--audit", tmp, "--participant", "1"])
            self.assertEqual(code, 0)
            self.assertIn("matches the plan", text)
            code, text = self.run_cmd(["--audit", tmp, "--participant", "2"])     # a different participant's order
        self.assertEqual(code, 1)
        self.assertIn("does NOT match", text)

    def test_audit_needs_a_participant_number_and_bad_input_is_a_message(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            self.run_cmd(["--audit", "somewhere"])
        code, text = self.run_cmd(["--participants", "0"])
        self.assertEqual(code, 1)
        self.assertIn("error:", text)


if __name__ == "__main__":
    unittest.main()
