"""Integration checks: every member's Phase 1 code working together, headless.

No camera and no Arduino are needed: shared synthetic traces stand in for Member A's camera,
and Member C's MockArduino (which mirrors the firmware rule for rule) stands in for the board.

- Event ids stay unique across trials, so Member C's one-cue-per-event rule keeps working.
- Member C's safety layer: cues only in conditions C/D, only when armed by the operator,
  never while tracking quality is low, servo kept at neutral while config.yaml says servo off,
  STOP after tracking loss, STOP on close.
- Member D's AI: gain bounded 0..1 and zero on tracking loss; ghost only with a real forecast.
- The app itself runs (replay + fake board + Member D's session logger) without a window.

Run:  .venv\\Scripts\\python.exe -m pytest tests/simulation/test_integration.py -v
"""
from dataclasses import replace
import os
from pathlib import Path
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))  # ghostgrip package
sys.path.insert(0, str(REPO_ROOT))          # shared contracts.py

try:
    import yaml  # noqa: F401  (config.yaml loader)
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False

from ghostgrip.simulation.inputs import TracePlayer, load_trace  # noqa: E402
from ghostgrip.simulation.trial import CONDITIONS  # noqa: E402
from ghostgrip.simulation.world import World  # noqa: E402

SYNTHETIC = REPO_ROOT / "data" / "synthetic"
FPS = 60
SETTLE_MS = 2000


class FakeClock:
    """One shared millisecond clock for the policy, the bridge and the fake board."""

    def __init__(self):
        self.t = 0

    def __call__(self):
        return self.t


def trace(name):
    return load_trace(SYNTHETIC / f"{name}.csv")


@unittest.skipUnless(HAVE_YAML, "pyyaml not installed")
class _WithConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ghostgrip.config import load_config
        cls.cfg = load_config()


class TestEventIdsAcrossTrials(unittest.TestCase):
    def test_ids_never_repeat_after_reset(self):
        world, ids = World("glass"), []
        for _ in range(2):
            player = TracePlayer(trace("clean"))
            for k in range(int((player.duration_ms + SETTLE_MS) / 1000 * FPS)):
                ids += [e.event_id for e in world.advance(player.sample(int(k * 1000 / FPS)), 1 / FPS)]
            world.reset()
        self.assertGreaterEqual(len(ids), 6)                  # two full grasps
        self.assertEqual(len(ids), len(set(ids)), "an event id was reused after reset()")
        self.assertEqual(ids, sorted(ids))


class TestHapticBridge(_WithConfig):
    def make_bridge(self, clock):
        from ghostgrip.config import to_haptic_config
        from ghostgrip.hardware.haptics import HapticPolicy
        from ghostgrip.hardware.mock_arduino import MockArduino
        from ghostgrip.hardware.serial_link import SerialLink
        from ghostgrip.ui.integration import HapticBridge
        policy = HapticPolicy(to_haptic_config(self.cfg), clock_ms=clock)
        board = MockArduino(clock_ms=clock)
        bridge = HapticBridge(SerialLink(board), policy, self.cfg.hardware.keepalive_ms, "mock", clock)
        return bridge, board

    def drive(self, name, condition, bridge, clock, world=None, arm=True):
        """Replay a trace through World + bridge in simulated time. Returns (cues, world)
        where cues lists (CueEvent, tracking_quality at that moment)."""
        world = world or World("glass")
        haptic = CONDITIONS[condition].haptic
        player = TracePlayer(trace(name))
        if arm:
            bridge.arm()
        sent, start = [], clock.t
        for k in range(int((player.duration_ms + SETTLE_MS) / 1000 * FPS)):
            t = int(k * 1000 / FPS)
            clock.t = start + t
            hand = replace(player.sample(t), t_ms=clock.t)
            events = world.advance(hand, 1 / FPS)
            bridge.update(hand)
            sent += [(cue, hand.tracking_quality) for cue in bridge.handle_events(events, hand, haptic)]
        return sent, world

    def test_conditions_without_haptics_send_nothing(self):
        for condition in ("A", "B"):
            clock = FakeClock()
            bridge, board = self.make_bridge(clock)
            sent, _ = self.drive("clean", condition, bridge, clock)
            self.assertEqual(sent, [], condition)

    def test_contact_cue_in_condition_c_keeps_servo_at_neutral(self):
        clock = FakeClock()
        bridge, board = self.make_bridge(clock)
        sent, _ = self.drive("clean", "C", bridge, clock)
        self.assertTrue(sent, "no cue was sent")
        cue = sent[0][0]
        self.assertEqual(cue.kind, "CONTACT")
        self.assertFalse(self.cfg.hardware.servo_enabled)     # safety gate not passed yet
        self.assertEqual(cue.safe_angle_deg, 90)              # so CONTACT is sent at neutral
        self.assertTrue(cue.no_contact_mode)
        self.assertEqual(board.servo_angle, 90)

    def test_board_is_never_armed_automatically(self):
        clock = FakeClock()
        bridge, board = self.make_bridge(clock)
        sent, _ = self.drive("clean", "D", bridge, clock, arm=False)
        self.assertEqual(sent, [])
        self.assertEqual(board.state, "DISARMED")

    def test_second_trial_still_gets_its_cue(self):
        clock = FakeClock()
        bridge, board = self.make_bridge(clock)
        first, world = self.drive("clean", "C", bridge, clock)
        world.reset()
        open_hand = trace("clean")[0]
        for _ in range(60):                                    # 1 s between trials, app still running
            clock.t += 1000 // FPS
            bridge.update(replace(open_hand, t_ms=clock.t))
        second, _ = self.drive("clean", "C", bridge, clock, world=world, arm=False)
        self.assertTrue(bridge.armed)
        self.assertIn("CONTACT", [c.kind for c, _ in first])
        self.assertIn("CONTACT", [c.kind for c, _ in second])

    def test_no_cue_while_tracking_quality_is_low(self):
        clock = FakeClock()
        bridge, board = self.make_bridge(clock)
        sent, _ = self.drive("low_quality", "D", bridge, clock)
        self.assertTrue(sent)
        for cue, quality in sent:
            self.assertGreaterEqual(quality, self.cfg.hardware.min_tracking_quality, cue)

    def test_tracking_loss_sends_stop(self):
        clock = FakeClock()
        bridge, board = self.make_bridge(clock)
        self.drive("dropout_long", "C", bridge, clock)
        self.assertFalse(bridge.armed)
        self.assertEqual(board.state, "DISARMED")
        self.assertIn("tracking lost", bridge.message)

    def test_close_sends_stop(self):
        clock = FakeClock()
        bridge, board = self.make_bridge(clock)
        bridge.arm()
        self.assertEqual(board.state, "ARMED")
        bridge.close()
        self.assertEqual(board.state, "DISARMED")
        self.assertFalse(board.is_open)
        bridge.close()                                         # safe twice

    def test_without_a_board_nothing_breaks(self):
        from ghostgrip.config import to_haptic_config
        from ghostgrip.hardware.haptics import HapticPolicy
        from ghostgrip.ui.integration import HapticBridge
        bridge = HapticBridge(None, HapticPolicy(to_haptic_config(self.cfg)))
        hand = trace("clean")[0]
        self.assertEqual(bridge.status, "NO BOARD")
        self.assertIsNone(bridge.update(hand))
        bridge.arm()
        bridge.stop()
        bridge.close()


class TestAIPipeline(_WithConfig):
    def test_gain_bounded_and_ghost_from_the_predictor(self):
        from ghostgrip.ui.integration import AIPipeline, lost_hand
        ai = AIPipeline(self.cfg, "glass")
        rows = trace("clean")
        gains, ghosts = [], []
        for hand in rows:
            frame = ai.update(hand, record_for_trial=True)
            gains.append(frame.decision.gain)
            ghosts.append(AIPipeline.ghost_xy(hand, frame))
        self.assertTrue(all(0.0 <= g <= 1.0 for g in gains))
        self.assertGreater(max(gains), 0.0)
        self.assertTrue(any(g is not None for g in ghosts))
        frame = ai.update(lost_hand(rows[-1].t_ms + 100, rows[-1]), record_for_trial=False)
        self.assertEqual(frame.decision.gain, 0.0)
        self.assertEqual(frame.decision.reason, "tracking_lost")
        self.assertIsNone(AIPipeline.ghost_xy(lost_hand(0, None), frame))


class TestAppHeadless(_WithConfig):
    def setUp(self):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        try:
            import pygame  # noqa: F401
        except ImportError:
            self.skipTest("pygame not installed")

    def test_data_origin_is_never_guessed_wrong(self):
        from ghostgrip.ui import app
        self.assertEqual(app.data_origin(True, None), "real")
        self.assertEqual(app.data_origin(False, None), "synthetic")       # mouse values are generated
        self.assertEqual(app.data_origin(False, SYNTHETIC / "clean.csv"), "synthetic")
        with tempfile.TemporaryDirectory() as d:
            member_a = Path(d) / "trace.csv"
            member_a.write_text("t_ms,pointer_x,pointer_y,closure_q,tracking_ok\n0,0.5,0.5,0,1\n")
            self.assertEqual(app.data_origin(False, member_a), "real")

    def test_replay_with_fake_board_and_session_log(self):
        from ghostgrip.metrics.logger import read_session
        from ghostgrip.ui import app
        with tempfile.TemporaryDirectory() as d:
            saved = app.REPO_ROOT
            app.REPO_ROOT = Path(d)                    # keep the test's log out of data/local/
            try:
                code = app.main(["--replay", "clean", "--hardware", "mock", "--condition", "D",
                                 "--session-id", "smoke", "--max-frames", "40", "--fps", "0"])
            finally:
                app.REPO_ROOT = saved
            self.assertEqual(code, 0)
            tables = read_session(Path(d) / "data" / "local" / "sessions" / "smoke")
            self.assertEqual(len(tables["frames"]), 40)
            self.assertEqual({r["origin"] for r in tables["frames"]}, {"synthetic"})
            self.assertTrue(tables["device"])          # fake board's sensor packets were logged


if __name__ == "__main__":
    unittest.main()
