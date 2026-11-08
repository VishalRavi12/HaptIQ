"""Member B Phase 1 checks: the simulation against the team's shared data and settings.

- Every shared synthetic trace in data/synthetic/ (made by Member D's generator from
  Member A's feature code) replays through the real World + TrialRunner with a known,
  explained outcome. Only the CSV files are used, not D's code, so B stays independent.
- The same replay gives the same outcome at different display frame rates.
- T06 on real data: low tracking quality pauses assistance and sends no events.
- Bounded assistance gain, the trial-outcome CSV log, and config.yaml consistency.

Run:  .venv\\Scripts\\python.exe -m pytest tests/simulation -v
"""
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))  # ghostgrip package
sys.path.insert(0, str(REPO_ROOT))          # shared contracts.py

from contracts import CONTACT, HOLD, RELEASE  # noqa: E402
from ghostgrip.simulation import world as W  # noqa: E402
from ghostgrip.simulation.inputs import TracePlayer, load_trace  # noqa: E402
from ghostgrip.simulation.trial import (CONDITIONS, TrialRunner, append_outcome_csv,  # noqa: E402
                                        last_trial_id, load_outcomes_csv)
from ghostgrip.simulation.world import World  # noqa: E402

SYNTHETIC = REPO_ROOT / "data" / "synthetic"
SETTLE_MS = 2000            # keep holding the last row so the object comes to rest
GRASP = [CONTACT, HOLD, RELEASE]

# What each shared trace must do in the glass trial, and why.
EXPECTED = {
    "clean":         ("success", "ideal reach, grasp, carry, release"),
    "noisy":         ("success", "measurement jitter must not add or drop events"),
    "dropout_short": ("success", "150 ms loss is shorter than the 300 ms freeze"),
    "dropout_long":  ("tracking_lost", "600 ms loss while carrying: safe release"),
    "glitch":        ("success", "one-frame landmark spike during the reach"),
    "fast_reach":    ("success", "very fast reach still grasps normally"),
    "slow_close":    ("success", "slow closure still reaches a stable hold"),
    "low_quality":   ("success", "quality dip while carrying keeps the existing hold"),
    "frame_drops":   ("success", "250 ms of missing frames (uneven time steps)"),
    "q_jitter":      ("no_trial", "hand never opens, so the trial never starts"),
}


def replay(rows, condition="A", fps=None, assist_gain=1.0):
    """Drive World + TrialRunner with a trace.

    fps=None feeds each row once with its own time step. fps=N samples the trace like a
    display running at N frames per second (zero-order hold), as the app does.
    Returns (events, reasons, outcome, world, frames) where frames lists (t_ms, snapshot).
    """
    world = World("glass")
    cond = CONDITIONS[condition]
    world.set_assistance(cond.ai_assist, assist_gain if cond.ai_assist else 0.0)
    trial = TrialRunner("glass", cond, "test")
    events, reasons, frames, outcome = [], [], [], None

    def step(hand, dt):
        nonlocal outcome
        new = world.advance(hand, dt)
        events.extend(new)
        reasons.extend(world.event_reason(e.event_id) for e in new)
        o = trial.update(world, new, hand)
        if o is not None and outcome is None:
            outcome = o
        frames.append((hand.t_ms, world.snapshot()))

    if fps is None:
        prev = rows[0].t_ms
        for hand in rows:
            step(hand, (hand.t_ms - prev) / 1000.0)
            prev = hand.t_ms
        last = rows[-1]
        for k in range(1, SETTLE_MS * 60 // 1000 + 1):
            step(replace(last, t_ms=last.t_ms + k * 1000 // 60), 1 / 60)
    else:
        player = TracePlayer(rows)
        n = int((player.duration_ms + SETTLE_MS) * fps / 1000)
        for k in range(n + 1):
            t_ms = int(k * 1000 / fps)
            step(replace(player.sample(t_ms), t_ms=t_ms), 0.0 if k == 0 else 1.0 / fps)
    return events, reasons, outcome, world, frames


def with_open_prefix(rows, open_ms=500):
    """Prepend an open hand at the first position so the trial can start."""
    first = rows[0]
    pre = [replace(first, t_ms=t, closure_q=0.0, dq_dt=0.0) for t in range(0, open_ms, 33)]
    return pre + [replace(r, t_ms=r.t_ms - first.t_ms + open_ms) for r in rows]


class TestSharedSyntheticTraces(unittest.TestCase):
    def test_every_shared_trace_is_covered(self):
        on_disk = {p.stem for p in SYNTHETIC.glob("*.csv")}
        self.assertTrue(on_disk, "no shared traces found in data/synthetic/")
        self.assertEqual(on_disk, set(EXPECTED),
                         "a trace was added or removed: give it an expected outcome here")

    def test_each_trace_gives_its_expected_outcome(self):
        for name, (expected, why) in EXPECTED.items():
            with self.subTest(trace=name, why=why):
                events, reasons, outcome, world, _ = replay(load_trace(SYNTHETIC / f"{name}.csv"))
                kinds = [e.kind for e in events]
                glass = world.object("glass")
                if expected == "success":
                    self.assertIsNotNone(outcome)
                    self.assertTrue(outcome.success, outcome)
                    self.assertEqual(kinds, GRASP)               # exactly once each
                    self.assertEqual(reasons[-1], "hand_opened")
                    self.assertFalse(glass.damaged)
                    self.assertTrue(world.in_target(glass))
                elif expected == "tracking_lost":
                    self.assertFalse(outcome.success)
                    self.assertEqual(outcome.fail_reason, "tracking_lost")
                    self.assertEqual(kinds, GRASP)
                    self.assertEqual(reasons[-1], "tracking_lost")
                    self.assertFalse(glass.damaged)              # released, not broken
                else:
                    self.assertIsNone(outcome)
                    self.assertEqual(kinds, [])

    def test_success_traces_record_a_clean_outcome(self):
        _, _, outcome, _, _ = replay(load_trace(SYNTHETIC / "clean.csv"), condition="D")
        self.assertEqual((outcome.condition, outcome.fail_reason), ("D", ""))
        self.assertEqual((outcome.slip_count, outcome.overgrip_count), (0, 0))
        self.assertGreater(outcome.end_ms, outcome.start_ms)


class TestReplayDeterminism(unittest.TestCase):
    def test_same_outcome_at_any_display_fps(self):
        for name in ("clean", "noisy", "dropout_long", "slow_close"):
            rows = load_trace(SYNTHETIC / f"{name}.csv")
            results = set()
            for fps in (30, 60, 144):
                events, reasons, outcome, _, _ = replay(rows, fps=fps)
                results.add((tuple(e.kind for e in events), tuple(reasons),
                             outcome.success, outcome.fail_reason))
            with self.subTest(trace=name):
                self.assertEqual(len(results), 1, results)

    def test_replaying_twice_is_identical(self):
        rows = load_trace(SYNTHETIC / "noisy.csv")
        a = replay(rows)
        b = replay(rows)
        self.assertEqual([(e.kind, e.t_ms, e.simulated_grip_effort) for e in a[0]],
                         [(e.kind, e.t_ms, e.simulated_grip_effort) for e in b[0]])
        self.assertEqual(a[2], b[2])


class TestT06AndT07OnSharedData(unittest.TestCase):
    def test_T06_low_quality_pauses_assist_and_sends_nothing(self):
        rows = load_trace(SYNTHETIC / "low_quality.csv")
        dip = [r.t_ms for r in rows if r.tracking_ok and r.tracking_quality < W.QUALITY_MIN]
        self.assertTrue(dip, "trace should contain a low-quality stretch")
        events, _, outcome, _, frames = replay(rows, condition="B")
        in_dip = [s for t, s in frames if dip[0] <= t <= dip[-1]]
        self.assertTrue(in_dip)
        self.assertTrue(all(not s.reliable for s in in_dip))
        self.assertTrue(all(not s.assist_active for s in in_dip))
        self.assertFalse(any(dip[0] <= e.t_ms <= dip[-1] for e in events))
        self.assertTrue(outcome.success)      # the existing hold is kept, not dropped

    def test_T07_threshold_jitter_gives_one_contact(self):
        rows = with_open_prefix(load_trace(SYNTHETIC / "q_jitter.csv"))
        events, _, _, world, _ = replay(rows)
        self.assertEqual([e.kind for e in events], [CONTACT])
        self.assertFalse(world.object("glass").damaged)


class TestAssistanceGain(unittest.TestCase):
    def test_gain_is_bounded(self):
        w = World("glass")
        w.set_assistance(True, 5.0)
        self.assertEqual(w.assist_gain, 1.0)
        w.set_assistance(True, -3.0)
        self.assertEqual(w.assist_gain, 0.0)
        self.assertFalse(w.assist_active)          # gain 0 means no help at all

    def test_assist_only_active_while_tracking_is_reliable(self):
        w = World("glass")
        w.set_assistance(True, 1.0)
        good = load_trace(SYNTHETIC / "clean.csv")[0]
        w.advance(good, 1 / 60)
        self.assertTrue(w.snapshot().assist_active)
        w.advance(replace(good, t_ms=good.t_ms + 17, tracking_quality=0.2), 1 / 60)
        snap = w.snapshot()
        self.assertFalse(snap.assist_active)
        # The configured enlargement stays visible to loggers (Member D's adapt test reads it).
        self.assertAlmostEqual(snap.grab_radius_extra_frac, W.MAX_ASSIST_FRAC)

    def test_condition_A_ignores_gain(self):
        _, _, outcome_a, world_a, _ = replay(load_trace(SYNTHETIC / "clean.csv"), "A", assist_gain=1.0)
        self.assertFalse(world_a.assist_enabled)
        self.assertTrue(outcome_a.success)


class TestOutcomeLog(unittest.TestCase):
    def test_outcomes_round_trip_through_csv(self):
        _, _, ok, _, _ = replay(load_trace(SYNTHETIC / "clean.csv"))
        _, _, lost, _, _ = replay(load_trace(SYNTHETIC / "dropout_long.csv"))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "sub" / "trials.csv"
            append_outcome_csv(path, ok)
            append_outcome_csv(path, lost)
            self.assertEqual(load_outcomes_csv(path), [ok, lost])
            header = path.read_text().splitlines()[0]
            self.assertEqual(header.count("session_id"), 1)

    def test_trial_numbering_continues_from_the_log(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "trials.csv"
            self.assertEqual(last_trial_id(path, "s1"), 0)          # no file yet
            _, _, first, _, _ = replay(load_trace(SYNTHETIC / "clean.csv"))
            append_outcome_csv(path, replace(first, session_id="s1", trial_id=4))
            append_outcome_csv(path, replace(first, session_id="other", trial_id=9))
            self.assertEqual(last_trial_id(path, "s1"), 4)
            runner = TrialRunner("glass", CONDITIONS["A"], "s1",
                                 first_trial_id=last_trial_id(path, "s1") + 1)
            world = World("glass")
            for hand in load_trace(SYNTHETIC / "clean.csv")[:3]:
                runner.update(world, world.advance(hand, 1 / 30), hand)
            self.assertEqual(runner.trial_id, 5)


class TestMemberARecordingFormat(unittest.TestCase):
    def test_vision_recorder_csv_replays_like_the_simulation_csv(self):
        """A file written in Member A's column layout (pointer_x, closure_q, ...) loads to
        the same HandStates and gives the same events as the simulation layout."""
        from ghostgrip.simulation.inputs import save_trace
        sim_rows = load_trace(SYNTHETIC / "noisy.csv")
        with tempfile.TemporaryDirectory() as d:
            sim_copy = save_trace(Path(d) / "sim.csv", sim_rows)
            vision = Path(d) / "vision.csv"
            lines = sim_copy.read_text().splitlines()
            header = lines[0].split(",")
            renamed = {"x": "pointer_x", "y": "pointer_y", "q": "closure_q",
                       "vx": "velocity_x", "vy": "velocity_y"}
            lines[0] = ",".join(renamed.get(h, h) for h in header)
            vision.write_text("\n".join(lines) + "\n")
            vision_rows = load_trace(vision)
        self.assertEqual(vision_rows, sim_rows)
        self.assertEqual([e.kind for e in replay(vision_rows)[0]], GRASP)

    def test_unknown_columns_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "bad.csv"
            bad.write_text("t_ms,a,b\n0,1,2\n")
            with self.assertRaises(ValueError):
                load_trace(bad)


class TestConfigConsistency(unittest.TestCase):
    def test_config_yaml_matches_the_world(self):
        try:
            from ghostgrip.config import load_config
        except ImportError as exc:                 # PyYAML missing on this machine
            self.skipTest(f"config loader unavailable: {exc}")
        sim = load_config(REPO_ROOT / "config.yaml").simulation
        self.assertEqual((sim.width, sim.height), (W.WORLD_W, W.WORLD_H))
        self.assertAlmostEqual(1.0 / sim.physics_hz, W.DT)


if __name__ == "__main__":
    unittest.main()
