"""Member B tests: trial outcomes, mouse adapter, trace replay, renderer smoke test."""
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ghostgrip.contracts import HandState  # noqa: E402
from ghostgrip.simulation.inputs import (MouseKeyboardAdapter, TracePlayer,  # noqa: E402
                                         TraceRecorder, load_trace)
from ghostgrip.simulation.trial import CONDITIONS, FINISHED, TrialRunner  # noqa: E402
from ghostgrip.simulation.world import World  # noqa: E402
from test_world import GLASS_REST_Y, carry_glass_to_target, close_on, hs, lerp  # noqa: E402


def run_trial(world, trial, traj, seconds, fps=60):
    outcome = None
    for k in range(int(seconds * fps)):
        t = k / fps
        hand = hs(t, *traj(t))
        events = world.advance(hand, 1.0 / fps)
        out = trial.update(world, events, hand)
        outcome = out or outcome
    return outcome


class TestTrial(unittest.TestCase):
    def test_success_when_carried_to_target(self):
        w = World("glass")
        tr = TrialRunner("glass", CONDITIONS["A"])
        out = run_trial(w, tr, carry_glass_to_target, 4.8)
        self.assertIsNotNone(out)
        self.assertTrue(out.success, out)
        self.assertEqual(out.condition, "A")
        self.assertEqual(out.trial_id, 1)
        self.assertEqual(tr.state, FINISHED)

    def test_failure_when_overgripped(self):
        w = World("glass")
        tr = TrialRunner("glass", CONDITIONS["D"])
        out = run_trial(w, tr, close_on(200, GLASS_REST_Y, 0.9), 2.0)
        self.assertFalse(out.success)
        self.assertEqual(out.fail_reason, "damaged_overgrip")
        self.assertEqual(out.overgrip_count, 1)

    def test_tracking_loss_invalidates_trial(self):
        w = World("glass")
        tr = TrialRunner("glass")

        def traj(t):
            if t < 1.0:
                return close_on(200, GLASS_REST_Y, 0.5)(t)
            return (200, GLASS_REST_Y, 0.5, False)
        out = run_trial(w, tr, traj, 2.0)
        self.assertEqual(out.fail_reason, "tracking_lost")

    def test_timeout(self):
        w = World("glass")
        tr = TrialRunner("glass", timeout_ms=1000)
        out = run_trial(w, tr, lambda t: (500, 200, 0.0), 1.5)
        self.assertEqual(out.fail_reason, "timeout")

    def test_release_without_dwell_is_not_success(self):
        w = World("glass")
        tr = TrialRunner("glass", dwell_ms=10_000)
        out = run_trial(w, tr, carry_glass_to_target, 4.8)
        self.assertIsNone(out)


class TestMouseAdapter(unittest.TestCase):
    def test_q_ramps_and_stays(self):
        a = MouseKeyboardAdapter()
        a.sample(0)
        for t in range(100, 1100, 100):
            s = a.sample(t, closing=True)
        self.assertAlmostEqual(s.closure_q, 0.6, places=2)
        for t in range(1100, 2100, 100):
            s = a.sample(t)
        self.assertAlmostEqual(s.closure_q, 0.6, places=2)   # holds steady without input
        self.assertGreater(a.sample(2200, opening=True).closure_q, -1e-9)

    def test_wheel_and_open(self):
        a = MouseKeyboardAdapter()
        a.sample(0)
        a.wheel(5)
        s = a.sample(1000)
        self.assertAlmostEqual(s.closure_q, 0.20, places=3)
        a.open_hand()
        self.assertAlmostEqual(a.sample(2000).closure_q, 0.0, places=3)
        a.wheel(100)
        self.assertLessEqual(a.q_target, 1.0)

    def test_tracking_lost_holds_last_pointer(self):
        a = MouseKeyboardAdapter()
        a.set_pointer_px(480, 270, 960, 540)
        a.sample(0)
        a.tracking_lost = True
        a.set_pointer_px(10, 10, 960, 540)
        s = a.sample(100)
        self.assertFalse(s.tracking_ok)
        self.assertEqual(s.pointer_xy, (0.5, 0.5))

    def test_low_quality_and_velocity(self):
        a = MouseKeyboardAdapter()
        a.set_pointer_px(0, 0, 960, 540)
        a.sample(0)
        a.low_quality = True
        a.set_pointer_px(96, 0, 960, 540)
        s = a.sample(100)
        self.assertEqual(s.tracking_quality, 0.3)
        self.assertAlmostEqual(s.velocity_xy[0], 1.0)


class TestReplay(unittest.TestCase):
    def test_replay_is_deterministic(self):
        rec = TraceRecorder()
        fps = 60
        for k in range(int(4.8 * fps)):
            rec.append(hs(5.0 + k / fps, *carry_glass_to_target(k / fps)))
        with tempfile.TemporaryDirectory() as d:
            path = rec.save(Path(d) / "trace.csv")
            rows = load_trace(path)
        self.assertEqual(rows[0].t_ms, 0)
        runs = []
        for _ in range(2):
            player = TracePlayer(rows)
            w = World("glass")
            evs = []
            for k in range(int(5.0 * fps)):
                t_ms = int(k * 1000 / fps)
                evs += [(e.event_id, e.kind) for e in w.advance(player.sample(t_ms), 1.0 / fps)]
            runs.append(evs)
        self.assertEqual(runs[0], runs[1])
        self.assertEqual([k for _, k in runs[0]], ["CONTACT", "HOLD", "RELEASE"])


class TestRendererSmoke(unittest.TestCase):
    def test_draw_every_scene_headless(self):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        try:
            import pygame
        except ImportError:
            self.skipTest("pygame not installed")
        from ghostgrip.ui.hud import PANEL_W, Hud, HudInfo
        from ghostgrip.ui.renderer_2d import Renderer2D
        pygame.init()
        try:
            surf = pygame.Surface((960 + PANEL_W, 540))
            r, h = Renderer2D(), Hud(960, 540)
            for scene in ("glass", "sponge", "block", "sandbox"):
                w = World(scene)
                w.set_assistance(True, 1.0)
                w.advance(hs(0, 200, 300, 0.3), 0.1)
                snap = w.snapshot()
                r.draw(surf, snap, ghost_world=(300, 300), banner=("SUCCESS", (0, 255, 0)))
                h.draw(surf, snap, HudInfo("B", "AI only", "mouse", scene, "RUNNING", 1, 1234,
                                           {"A": [1, 2]}, 60.0, False, True, False))
        finally:
            pygame.quit()


if __name__ == "__main__":
    unittest.main()
