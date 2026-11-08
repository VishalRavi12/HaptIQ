"""Member B headless physics tests: plan 7.2 cases T01-T07 + FPS independence.

Run:  .venv\\Scripts\\python.exe -m unittest discover -s tests/simulation -v
or :  python -m pytest tests/simulation      (once pytest is installed)
"""
from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))  # ghostgrip package
sys.path.insert(0, str(REPO_ROOT))          # shared contracts.py

from contracts import CONTACT, DAMAGE, HOLD, RELEASE, SLIP  # noqa: E402
from contracts import HandStateV2 as HandState  # noqa: E402
from ghostgrip.simulation import world as W  # noqa: E402
from ghostgrip.simulation.world import World  # noqa: E402

GLASS_REST_Y = W.TABLE_Y - 30     # glass radius 30, x = 200 in the 'glass' scene
BLOCK_REST_Y = W.TABLE_Y - 34


def hs(t_s, x, y, q, ok=True, quality=1.0):
    """HandState from WORLD pixel coordinates (tests think in pixels)."""
    return HandState(int(round(t_s * 1000)), (x / W.WORLD_W, y / W.WORLD_H), q,
                     0.0, (0.0, 0.0), 1.0, ok, quality)


def lerp(a, b, f):
    f = min(1.0, max(0.0, f))
    return a + (b - a) * f


def run(world, traj, seconds, fps=60, t_start=0.0):
    """Drive the world with traj(t) -> (x, y, q[, ok[, quality]]) at a display fps."""
    events = []
    n = int(round(seconds * fps))
    for k in range(n):
        t = t_start + k / fps
        args = traj(t)
        events += world.advance(hs(t, *args), 1.0 / fps)
    return events


def kinds(events):
    return [e.kind for e in events]


def close_on(x, y, q_final, ramp_s=1.0):
    return lambda t: (x, y, lerp(0.0, q_final, t / ramp_s))


class TestPlanCases(unittest.TestCase):
    def test_T01_open_hand_no_grasp(self):
        w = World("sandbox")
        ev = []
        for o in w.objects:
            ev += run(w, lambda t, o=o: (o.x, o.y, 0.0), 0.5)
        self.assertEqual(ev, [])

    def test_T02_valid_fragile_grip_and_lift(self):
        w = World("glass")
        ev = run(w, close_on(200, GLASS_REST_Y, 0.5), 2.0)
        self.assertEqual(kinds(ev), [CONTACT, HOLD])
        ev2 = run(w, lambda t: (200, lerp(GLASS_REST_Y, GLASS_REST_Y - 150, t - 2.0), 0.5), 1.0, t_start=2.0)
        self.assertEqual(ev2, [])
        g = w.object("glass")
        self.assertTrue(g.was_lifted)
        self.assertLess(g.y, GLASS_REST_Y - 100)

    def test_T03_fragile_overgrip_breaks_once(self):
        w = World("glass")
        ev = run(w, close_on(200, GLASS_REST_Y, 0.9), 1.0)
        ev += run(w, lambda t: (200, GLASS_REST_Y, 0.9), 2.0, t_start=1.0)
        self.assertEqual(kinds(ev).count(DAMAGE), 1)
        self.assertEqual(kinds(ev), [CONTACT, HOLD, DAMAGE])
        self.assertTrue(w.object("glass").damaged)
        self.assertEqual(w.event_reason(ev[-1].event_id), "overgrip")

    def test_glass_break_point_is_q_082(self):
        """Phase 2: a firm grip (q 0.78, which broke the glass before) now holds; 0.86 breaks."""
        firm = World("glass")
        ev = run(firm, close_on(200, GLASS_REST_Y, 0.78), 1.0)
        ev += run(firm, lambda t: (200, GLASS_REST_Y, 0.78), 1.0, t_start=1.0)
        self.assertEqual(kinds(ev), [CONTACT, HOLD])
        self.assertFalse(firm.object("glass").damaged)
        hard = World("glass")
        ev = run(hard, close_on(200, GLASS_REST_Y, 0.86), 1.0)
        ev += run(hard, lambda t: (200, GLASS_REST_Y, 0.86), 1.0, t_start=1.0)
        self.assertEqual(kinds(ev), [CONTACT, HOLD, DAMAGE])

    def test_a_short_squeeze_spike_does_not_break_the_glass(self):
        """A 100 ms jump in the measured closure (camera noise) is survivable; 400 ms is not."""
        for spike_s, broken in ((0.1, False), (0.4, True)):
            w = World("glass")
            run(w, close_on(200, GLASS_REST_Y, 0.5), 1.0)
            run(w, lambda t: (200, GLASS_REST_Y, 0.98), spike_s, t_start=1.0)
            ev = run(w, lambda t: (200, GLASS_REST_Y, 0.5), 1.0, t_start=1.0 + spike_s)
            self.assertEqual(w.object("glass").damaged, broken, spike_s)
            self.assertNotIn(DAMAGE, kinds(ev))                # breaks once, during the squeeze

    def test_a_camera_hand_that_opens_only_to_024_can_grab_again(self):
        """Real hands often read q 0.20-0.24 when fully open; that must re-arm grabbing."""
        w = World("glass")
        ev = run(w, close_on(200, GLASS_REST_Y, 0.5), 1.0)
        ev += run(w, lambda t: (200, GLASS_REST_Y, 0.24), 0.5, t_start=1.0)       # release
        ev += run(w, lambda t: (200, GLASS_REST_Y, 0.5), 1.0, t_start=1.5)        # pinch again
        self.assertEqual(kinds(ev).count(CONTACT), 2)

    def test_closing_on_an_object_grabs_but_a_shut_pinch_does_not(self):
        """Real hands roam half-closed (median q 0.61 in a live session). Closing further over
        the object is a grab; arriving with a pinch already shut is not (no sticky grab)."""
        w = World("glass")
        run(w, lambda t: (480, 200, 0.45), 0.5)                                    # roam half-closed
        ev = run(w, lambda t: (lerp(480, 200, (t - 0.5) / 0.5), GLASS_REST_Y, 0.45), 0.8, t_start=0.5)
        self.assertEqual(ev, [])                                                   # arriving: no grab
        ev = run(w, lambda t: (200, GLASS_REST_Y, lerp(0.45, 0.62, (t - 1.3) / 0.3)), 0.8, t_start=1.3)
        self.assertEqual(kinds(ev), [CONTACT, HOLD])                               # closing: grab
        # a small wobble (like the shared q_jitter trace) is not a closing movement
        still = World("glass")
        run(still, lambda t: (480, 200, 0.45), 0.5)
        ev = run(still, lambda t: (200, GLASS_REST_Y, 0.45 + (0.06 if int(t * 30) % 2 else 0.0)),
                 1.5, t_start=0.5)
        self.assertEqual(ev, [])

    def test_camera_like_hand_completes_two_trials(self):
        """Real camera hands read about q 0.22 when open and pinch firmly (0.65): the trial must
        start, the glass must survive the firm pinch, and a second trial must work too."""
        from ghostgrip.simulation.trial import TrialRunner

        def camera_carry(t):
            x, y, q = carry_glass_to_target(t)
            return x, y, 0.22 + q * (0.65 - 0.22) / 0.5           # 0 -> 0.22 open, 0.5 -> 0.65

        w = World("glass")
        trial = TrialRunner("glass")
        outcomes = []
        for attempt in range(2):
            for k in range(int(4.8 * 60)):
                t = k / 60
                o = trial.update(w, w.advance(hs(attempt * 10 + t, *camera_carry(t)), 1 / 60),
                                 hs(attempt * 10 + t, *camera_carry(t)))
                if o is not None:
                    outcomes.append(o)
            w.reset()
            trial.next_trial()
        self.assertEqual([(o.success, o.fail_reason) for o in outcomes], [(True, ""), (True, "")])

    def test_T04_heavy_undergrip_never_lifts(self):
        w = World("block")
        ev = run(w, close_on(200, BLOCK_REST_Y, 0.7), 1.0)   # G = 0.525 < hold 0.75
        ev += run(w, lambda t: (200, lerp(BLOCK_REST_Y, BLOCK_REST_Y - 150, t - 1.0), 0.7), 1.0, t_start=1.0)
        self.assertNotIn(HOLD, kinds(ev))
        self.assertAlmostEqual(w.object("block").y, BLOCK_REST_Y)

    def test_T04b_heavy_slips_when_grip_weakens(self):
        w = World("block")
        ev = run(w, close_on(200, BLOCK_REST_Y, 0.9), 1.0)
        ev += run(w, lambda t: (200, lerp(BLOCK_REST_Y, BLOCK_REST_Y - 120, t - 1.0), 0.9), 1.0, t_start=1.0)
        self.assertEqual(kinds(ev), [CONTACT, HOLD])
        ev2 = run(w, lambda t: (200, BLOCK_REST_Y - 120, 0.7), 2.0, t_start=2.0)
        self.assertEqual(kinds(ev2), [SLIP, RELEASE])
        self.assertEqual(w.event_reason(ev2[-1].event_id), "slipped_out")
        self.assertAlmostEqual(w.object("block").y, BLOCK_REST_Y)   # fell back to the table

    def test_T05_tracking_loss_freezes_then_safe_release(self):
        w = World("glass")
        run(w, close_on(200, GLASS_REST_Y, 0.5), 1.0)
        run(w, lambda t: (200, GLASS_REST_Y - 100, 0.5), 0.5, t_start=1.0)
        # short dropout (100 ms): frozen, no release
        ev = run(w, lambda t: (200, GLASS_REST_Y - 100, 0.5, False), 0.1, t_start=1.5)
        self.assertEqual(ev, [])
        self.assertTrue(w.snapshot().paused)
        run(w, lambda t: (200, GLASS_REST_Y - 100, 0.5), 0.3, t_start=1.6)
        # long dropout: exactly one RELEASE, reason tracking_lost
        ev = run(w, lambda t: (200, GLASS_REST_Y - 100, 0.5, False), 0.6, t_start=1.9)
        self.assertEqual(kinds(ev), [RELEASE])
        self.assertEqual(w.event_reason(ev[0].event_id), "tracking_lost")
        self.assertIsNone(w.engaged_id)

    def test_T06_low_quality_blocks_new_grasp(self):
        w = World("glass")
        ev = run(w, lambda t: (200, GLASS_REST_Y, lerp(0, 0.5, t), True, 0.3), 1.5)
        self.assertEqual(ev, [])
        self.assertFalse(w.snapshot().reliable)

    def test_T07_q_jitter_at_contact_threshold(self):
        for fps in (30, 60):
            w = World("glass")
            run(w, lambda t: (200, GLASS_REST_Y, 0.0), 0.2, fps=fps)   # arm with open hand
            ev = run(w, lambda t: (200, GLASS_REST_Y, 0.28 if int(t * fps) % 2 else 0.32),
                     2.0, fps=fps, t_start=0.2)
            self.assertLessEqual(len(ev), 1, f"fps={fps}: {kinds(ev)}")

    def test_T07_pointer_jitter_at_edge(self):
        w = World("glass")
        edge = 200 + 30 + W.GRAB_MARGIN
        run(w, lambda t: (edge + 20, GLASS_REST_Y, 0.0), 0.2)
        ev = run(w, lambda t: (edge + (-3 if int(t * 60) % 2 else 3), GLASS_REST_Y, 0.35), 2.0, t_start=0.2)
        self.assertLessEqual(len(ev), 1, kinds(ev))

    def test_T07_effort_jitter_at_hold_threshold(self):
        w = World("glass")
        ev = run(w, close_on(200, GLASS_REST_Y, 0.42), 1.5)
        self.assertEqual(kinds(ev), [CONTACT, HOLD])
        ev = run(w, lambda t: (200, GLASS_REST_Y, 0.39 if int(t * 60) % 2 else 0.41), 2.0, t_start=1.5)
        self.assertEqual(ev, [])

    def test_slow_loosening_of_lifted_object_slips(self):
        """Opening slowly while lifted is a real slip; opening fast is a release."""
        w = World("glass")
        run(w, close_on(200, GLASS_REST_Y, 0.5), 1.0)
        run(w, lambda t: (200, lerp(GLASS_REST_Y, GLASS_REST_Y - 60, t - 1.0), 0.5), 1.0, t_start=1.0)
        ev = run(w, lambda t: (200, GLASS_REST_Y - 60, lerp(0.5, 0.33, (t - 2.0) / 2.0)), 3.0, t_start=2.0)
        self.assertEqual(kinds(ev)[0], SLIP)


def carry_glass_to_target(t):
    """Approach -> close -> lift -> carry -> lower -> open -> wait (~4.8 s)."""
    if t < 0.5:
        return (lerp(100, 200, t / 0.5), GLASS_REST_Y, 0.0)
    if t < 1.0:
        return (200, GLASS_REST_Y, lerp(0.0, 0.5, (t - 0.5) / 0.5))
    if t < 1.5:
        return (200, lerp(GLASS_REST_Y, 300, (t - 1.0) / 0.5), 0.5)
    if t < 2.5:
        return (lerp(200, 830, (t - 1.5) / 1.0), 300, 0.5)
    if t < 3.0:
        return (830, lerp(300, GLASS_REST_Y - 20, (t - 2.5) / 0.5), 0.5)
    if t < 3.3:
        return (830, GLASS_REST_Y - 20, lerp(0.5, 0.0, (t - 3.0) / 0.3))
    return (830, GLASS_REST_Y - 20, 0.0)


class TestTimingAndRobustness(unittest.TestCase):
    def test_same_outcome_at_any_display_fps(self):
        results = {}
        for fps in (20, 30, 60, 144):
            w = World("glass")
            ev = run(w, carry_glass_to_target, 4.8, fps=fps)
            g = w.object("glass")
            results[fps] = (tuple(kinds(ev)), g.damaged, w.in_target(g), round(g.x))
        first = results[20]
        self.assertEqual(first[0], (CONTACT, HOLD, RELEASE))
        self.assertTrue(first[2])
        for fps, r in results.items():
            self.assertEqual(r[:3], first[:3], f"fps {fps}: {r} vs {first}")
            self.assertLessEqual(abs(r[3] - first[3]), 12)

    def test_catchup_is_capped(self):
        w = World("glass")
        w.advance(hs(0, 100, 100, 0.0), 5.0)
        self.assertLessEqual(w.steps_run, W.MAX_CATCHUP_STEPS)

    def test_bad_inputs_do_not_crash(self):
        w = World("glass")
        w.advance(HandState(0, (float("nan"), 2.0), float("inf"), 0, (0, 0), 1, True, 1), -1.0)
        w.advance(HandState(10, (0.5, 0.5), -3.0, 0, (0, 0), 1, True, 1), float("nan"))
        snap = w.snapshot()
        self.assertTrue(0.0 <= snap.q <= 1.0)

    def test_world_has_no_pygame_import(self):
        src = Path(W.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import pygame", src)


class TestStarterBugsFixed(unittest.TestCase):
    def test_no_sticky_grab_after_pinching_in_air(self):
        w = World("glass")
        run(w, lambda t: (60, GLASS_REST_Y, 0.0), 0.2)                       # open hand (armed)
        run(w, lambda t: (60, GLASS_REST_Y, 0.5), 0.3, t_start=0.2)          # pinch in air
        ev = run(w, lambda t: (lerp(60, 200, (t - 0.5) / 0.5), GLASS_REST_Y, 0.5), 1.0, t_start=0.5)
        self.assertEqual(ev, [])

    def test_no_chain_grab_after_break(self):
        w = World("sandbox")
        ev = run(w, close_on(170, GLASS_REST_Y, 0.9), 1.0)
        ev += run(w, lambda t: (170, GLASS_REST_Y, 0.9), 0.5, t_start=1.0)   # keep squeezing
        self.assertIn(DAMAGE, kinds(ev))
        sponge = w.object("sponge")
        ev = run(w, lambda t: (lerp(170, sponge.x, t - 1.5), sponge.y, 0.9), 1.5, t_start=1.5)
        self.assertEqual(ev, [])

    def test_tracking_reacquire_on_glass_does_not_break_it(self):
        w = World("glass")
        run(w, lambda t: (480, 270, 0.0, False), 0.5)
        ev = run(w, lambda t: (200, GLASS_REST_Y, 0.5), 1.0, t_start=0.5)
        self.assertNotIn(DAMAGE, kinds(ev))

    def test_damage_is_terminal_without_release(self):
        w = World("glass")
        ev = run(w, close_on(200, GLASS_REST_Y, 0.9), 1.0)
        ev += run(w, lambda t: (200, GLASS_REST_Y, 0.9), 0.5, t_start=1.0)   # keep squeezing
        ev += run(w, lambda t: (200, GLASS_REST_Y, 0.0), 1.0, t_start=1.5)
        self.assertEqual(kinds(ev)[-1], DAMAGE)

    def test_three_objects_behave_differently_at_same_grip(self):
        """FR-03: the same q gives three different outcomes.
        q = 0.84 is above the glass break point (0.82) and below the block's hold point (0.85)."""
        outcomes = {}
        for kind, rest in (("glass", GLASS_REST_Y), ("sponge", W.TABLE_Y - 38), ("block", BLOCK_REST_Y)):
            w = World(kind)
            ev = run(w, close_on(200, rest, 0.84), 1.0)
            ev += run(w, lambda t: (200, rest, 0.84), 0.5, t_start=1.0)
            outcomes[kind] = tuple(kinds(ev))
        self.assertEqual(outcomes["glass"], (CONTACT, HOLD, DAMAGE))
        self.assertEqual(outcomes["sponge"], (CONTACT, HOLD))
        self.assertEqual(outcomes["block"], (CONTACT,))


class TestDropAndAssist(unittest.TestCase):
    def _lift_and_drop(self, height):
        w = World("glass")
        run(w, close_on(200, GLASS_REST_Y, 0.5), 1.0)
        run(w, lambda t: (200, lerp(GLASS_REST_Y, GLASS_REST_Y - height, t - 1.0), 0.5), 1.0, t_start=1.0)
        ev = run(w, lambda t: (200, GLASS_REST_Y - height, 0.0), 1.5, t_start=2.0)
        return w, ev

    def test_high_drop_breaks_glass(self):
        w, ev = self._lift_and_drop(250)
        self.assertEqual(kinds(ev), [RELEASE, DAMAGE])
        self.assertEqual(w.event_reason(ev[-1].event_id), "dropped")

    def test_low_drop_is_safe(self):
        w, ev = self._lift_and_drop(60)
        self.assertEqual(kinds(ev), [RELEASE])
        self.assertFalse(w.object("glass").damaged)

    def test_assist_enlarges_grab_zone_but_never_auto_grasps(self):
        x = 200 + 30 + W.GRAB_MARGIN + 3          # just outside the unassisted zone
        plain = World("glass")
        self.assertEqual(run(plain, close_on(x, GLASS_REST_Y, 0.5), 1.5), [])
        helped = World("glass")
        helped.set_assistance(True, 1.0)
        self.assertEqual(kinds(run(helped, close_on(x, GLASS_REST_Y, 0.5), 1.5)), [CONTACT, HOLD])
        idle = World("glass")
        idle.set_assistance(True, 1.0)
        self.assertEqual(run(idle, lambda t: (200, GLASS_REST_Y, 0.0), 2.0), [])


if __name__ == "__main__":
    unittest.main()
