"""Tests for src/ghostgrip/ai/synthetic.py. No camera or hardware needed."""
import csv
import math
from pathlib import Path
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))   # ghostgrip package
sys.path.insert(0, str(REPO_ROOT))           # shared contracts.py

from contracts import HandStateV2  # noqa: E402
from ghostgrip.ai import synthetic as S  # noqa: E402
from ghostgrip.ai.synthetic import SCENARIOS, make_scenario, make_trace, piecewise, summarize  # noqa: E402

SAMPLES = REPO_ROOT / "data" / "synthetic"


def flat(x=0.5, y=0.5, q=0.0):
    return (lambda t: x), (lambda t: y), (lambda t: q)


class TestEveryScenario(unittest.TestCase):
    def test_all_scenarios_follow_the_contract(self):
        for name in SCENARIOS:
            with self.subTest(scenario=name):
                states = make_scenario(name)
                self.assertGreater(len(states), 10)
                last_t = None
                for s in states:
                    self.assertIsInstance(s, HandStateV2)
                    self.assertIsInstance(s.t_ms, int)
                    if last_t is not None:
                        self.assertGreater(s.t_ms, last_t, "timestamps must strictly increase")
                    last_t = s.t_ms
                    for value in (*s.pointer_xy, s.closure_q, s.stable_score, s.tracking_quality):
                        self.assertTrue(math.isfinite(value))
                        self.assertTrue(0.0 <= value <= 1.0, f"{value} out of range")
                    for value in (s.dq_dt, *s.velocity_xy):
                        self.assertTrue(math.isfinite(value))

    def test_loss_convention_holds_everywhere(self):
        for name in SCENARIOS:
            with self.subTest(scenario=name):
                previous = None
                for s in make_scenario(name):
                    if not s.tracking_ok:
                        self.assertEqual(s.velocity_xy, (0.0, 0.0))
                        self.assertEqual((s.dq_dt, s.stable_score, s.tracking_quality), (0.0, 0.0, 0.0))
                        if previous is not None:   # pointer and closure hold their last values
                            self.assertEqual(s.pointer_xy, previous.pointer_xy)
                            self.assertEqual(s.closure_q, previous.closure_q)
                    previous = s

    def test_same_seed_same_trace_different_seed_differs(self):
        self.assertEqual(make_scenario("noisy", 7), make_scenario("noisy", 7))
        self.assertNotEqual(make_scenario("noisy", 7), make_scenario("noisy", 8))
        self.assertEqual(make_scenario("clean", 1), make_scenario("clean", 2))   # no noise: seed irrelevant

    def test_unknown_scenario_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown scenario"):
            make_scenario("nope")


class TestScenarioShapes(unittest.TestCase):
    def test_clean_is_a_full_reach_grasp_carry_release(self):
        states = make_scenario("clean")
        self.assertTrue(all(s.tracking_ok for s in states))
        self.assertLess(states[0].closure_q, 0.05)
        self.assertGreater(max(s.closure_q for s in states), 0.45)
        self.assertLess(states[-1].closure_q, 0.05)                 # hand opened again
        self.assertAlmostEqual(states[-1].pointer_xy[0], S.TARGET_X, delta=0.01)
        reach_peak = max(abs(s.velocity_xy[0]) + abs(s.velocity_xy[1]) for s in states)
        self.assertGreater(reach_peak, 0.1)

    def test_noisy_has_visible_jitter_but_stays_valid(self):
        clean_stab = sum(s.stable_score for s in make_scenario("clean")) / 169
        noisy = make_scenario("noisy")
        self.assertTrue(all(s.tracking_ok for s in noisy))
        self.assertLess(sum(s.stable_score for s in noisy) / len(noisy), clean_stab)

    def test_short_dropout_loses_a_few_frames_then_recovers(self):
        states = make_scenario("dropout_short")
        lost = [s for s in states if not s.tracking_ok]
        self.assertTrue(3 <= len(lost) <= 7)
        self.assertTrue(states[-1].tracking_ok)
        self.assertTrue(states[0].tracking_ok)

    def test_long_dropout_is_longer_than_the_simulation_freeze(self):
        lost = [s for s in make_scenario("dropout_long") if not s.tracking_ok]
        self.assertGreaterEqual(lost[-1].t_ms - lost[0].t_ms, 500)

    def test_glitch_spike_is_rejected_as_lost_then_recovers(self):
        states = make_scenario("glitch")
        self.assertGreaterEqual(sum(1 for s in states if not s.tracking_ok), 1)
        self.assertTrue(states[-1].tracking_ok)
        # The spike never appears as a real position: no pointer jumps by more than the spike size.
        steps = [math.dist(a.pointer_xy, b.pointer_xy) for a, b in zip(states, states[1:])]
        self.assertLess(max(steps), 0.15)

    def test_fast_reach_stays_a_valid_stream(self):
        # Whether the jump gate flags frames is Member A's tuning; only validity is required here.
        summary = summarize(make_scenario("fast_reach"))
        self.assertGreater(summary.frames, 100)

    def test_slow_close_ramps_gradually(self):
        states = make_scenario("slow_close")
        rises = [b.closure_q - a.closure_q for a, b in zip(states, states[1:])]
        self.assertLess(max(rises), 0.05)            # no abrupt closure
        self.assertGreater(max(s.closure_q for s in states), 0.45)

    def test_low_quality_keeps_tracking_ok_but_drops_quality(self):
        states = make_scenario("low_quality")
        dipped = [s for s in states if s.tracking_ok and s.tracking_quality < 0.5]
        self.assertGreater(len(dipped), 5)
        self.assertTrue(all(s.tracking_ok for s in states))

    def test_frame_drops_make_uneven_steps_without_lost_rows(self):
        summary = summarize(make_scenario("frame_drops"))
        self.assertGreaterEqual(summary.longest_gap_ms, 250)
        self.assertEqual(summary.lost_frames, 0)

    def test_q_jitter_straddles_the_contact_threshold(self):
        qs = [s.closure_q for s in make_scenario("q_jitter")]
        self.assertLess(min(qs), 0.30)
        self.assertGreater(max(qs), 0.30)


class TestBuildingBlocks(unittest.TestCase):
    def test_piecewise_hits_keyframes_and_holds_ends(self):
        f = piecewise([(0, 1.0), (100, 3.0), (200, 3.0)])
        self.assertEqual((f(-5), f(0), f(100), f(500)), (1.0, 1.0, 3.0, 3.0))
        self.assertAlmostEqual(f(50), 2.0)           # smoothstep midpoint

    def test_piecewise_rejects_unordered_keyframes(self):
        with self.assertRaises(ValueError):
            piecewise([(0, 0), (100, 1), (100, 2)])
        with self.assertRaises(ValueError):
            piecewise([])

    def test_make_trace_rejects_bad_parameters(self):
        x, y, q = flat()
        for kwargs in ({"duration_ms": 0}, {"duration_ms": 100, "fps": 0},
                       {"duration_ms": 100, "noise_xy": -1}, {"duration_ms": 100, "quality": 1.5},
                       {"duration_ms": 100, "dropouts": [(50, 50)]},
                       {"duration_ms": 100, "missing_frames": [(60, 10)]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_trace(x, y, q, **kwargs)

    def test_noise_is_clipped_to_the_valid_range(self):
        x, y, q = flat(x=0.0, y=1.0, q=0.0)
        for s in make_trace(x, y, q, 1000, noise_xy=0.2, noise_q=0.2, seed=3):
            self.assertTrue(0.0 <= s.pointer_xy[0] <= 1.0 and 0.0 <= s.pointer_xy[1] <= 1.0)
            self.assertTrue(0.0 <= s.closure_q <= 1.0)

    def test_summarize(self):
        summary = summarize(make_scenario("dropout_short"))
        self.assertGreater(summary.lost_frames, 0)
        self.assertAlmostEqual(summary.duration_ms, 5600, delta=40)
        self.assertEqual(summary.min_quality_ok, 0.9)
        with self.assertRaises(ValueError):
            summarize([])


class TestSavingAndSamples(unittest.TestCase):
    def test_simulation_format_round_trips_through_the_team_loader(self):
        from ghostgrip.simulation.inputs import load_trace
        with tempfile.TemporaryDirectory() as folder:
            states = make_scenario("noisy")
            loaded = load_trace(S.save_trace(states, Path(folder) / "t.csv", "simulation"))
        self.assertEqual(len(loaded), len(states))
        for a, b in zip(states, loaded):
            self.assertEqual(a.t_ms, b.t_ms)
            self.assertAlmostEqual(a.pointer_xy[0], b.pointer_xy[0], places=5)
            self.assertAlmostEqual(a.closure_q, b.closure_q, places=5)
            self.assertEqual(a.tracking_ok, b.tracking_ok)

    def test_vision_format_round_trips_through_the_perception_reader(self):
        from ghostgrip.vision.trace import read_trace
        with tempfile.TemporaryDirectory() as folder:
            states = make_scenario("dropout_short")
            path = S.save_trace(states, Path(folder) / "t.csv", "vision")
            S.save_trace(states, path, "vision")      # saving again replaces our own file
            loaded = list(read_trace(path))
        self.assertEqual(len(loaded), len(states))
        self.assertEqual([s.tracking_ok for s in loaded], [s.tracking_ok for s in states])

    def test_bad_format_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "fmt"):
            S.save_trace(make_scenario("clean"), "x.csv", "xml")

    def test_committed_samples_match_the_generator(self):
        # Drift guard: regenerate with `PYTHONPATH=src python -m ghostgrip.ai.synthetic`.
        for name in SCENARIOS:
            with self.subTest(sample=name):
                path = SAMPLES / f"{name}.csv"
                self.assertTrue(path.is_file(), f"missing {path}; regenerate the samples")
                with path.open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
                states = make_scenario(name)
                self.assertEqual(len(rows), len(states))
                for row, s in zip(rows, states):
                    self.assertEqual(int(row["t_ms"]), s.t_ms)
                    self.assertAlmostEqual(float(row["x"]), s.pointer_xy[0], delta=1e-5)
                    self.assertAlmostEqual(float(row["q"]), s.closure_q, delta=1e-5)
                    self.assertEqual(row["tracking_ok"], str(int(s.tracking_ok)))


class TestDrivesTheSimulation(unittest.TestCase):
    """The synthetic traces must work as real input to Member B's simulation."""

    @staticmethod
    def run_glass_trial(states):
        from ghostgrip.simulation.trial import TrialRunner
        from ghostgrip.simulation.world import World
        world, trial = World("glass"), TrialRunner("glass")
        previous = None
        for s in states:
            dt = 0.0 if previous is None else (s.t_ms - previous) / 1000.0
            outcome = trial.update(world, world.advance(s, dt), s)
            if outcome is not None:
                return outcome
            previous = s.t_ms
        return None

    def test_clean_trace_completes_the_glass_trial(self):
        outcome = self.run_glass_trial(make_scenario("clean"))
        self.assertIsNotNone(outcome)
        self.assertTrue(outcome.success, outcome.fail_reason)
        self.assertEqual(outcome.overgrip_count, 0)

    def test_long_dropout_ends_the_trial_as_tracking_lost(self):
        outcome = self.run_glass_trial(make_scenario("dropout_long"))
        self.assertIsNotNone(outcome)
        self.assertFalse(outcome.success)
        self.assertEqual(outcome.fail_reason, "tracking_lost")

    def test_every_scenario_runs_through_the_world_without_error(self):
        for name in SCENARIOS:
            with self.subTest(scenario=name):
                self.run_glass_trial(make_scenario(name))     # outcome may be None; must not raise


if __name__ == "__main__":
    unittest.main()
