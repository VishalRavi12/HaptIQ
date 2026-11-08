"""Phase 2 checklist B1-B6: the integrated app's safety and lifecycle rules, headless.

B1 quit: nothing runs after Esc/window close (no camera read, physics step or cue).
B2 scene/condition/trial reset sends STOP and needs H again; interrupted trials are logged.
B3 camera frames keep their own capture time (not the time before the blocking read).
B4 --evaluation freezes Member D's assist policy, across resets; manual gain is refused.
B5 startup failure and exceptions still STOP the board, close the camera and the log.
B6 replay end disarms; hardware.enabled=false opens no port; all conditions over two trials.

Uses Member C's MockArduino, a fake camera and a fake clock, so no hardware or webcam is needed.
Run:  .venv\\Scripts\\python.exe -m pytest tests/simulation/test_app_lifecycle.py -v
"""
from dataclasses import replace
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))  # ghostgrip package
sys.path.insert(0, str(REPO_ROOT))          # shared contracts.py

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

try:
    import pygame
    import yaml  # noqa: F401
    READY_TO_RUN = True
except ImportError:
    READY_TO_RUN = False

SYNTHETIC = REPO_ROOT / "data" / "synthetic"
FRAME_MS = 16


class FakeClock:
    def __init__(self, t=50_000):
        self.t = t

    def __call__(self):
        return self.t


class FakeCamera:
    """Stands in for Member A's LiveHandSource: read() blocks for `delay_ms` (on the fake clock)
    and returns the next recorded row stamped with the capture time, like CameraTracker does."""

    def __init__(self, clock, rows, delay_ms=0):
        self.clock, self.rows, self.delay_ms = clock, rows, delay_ms
        self.reads = 0
        self.closed = False
        self.last_observation = None
        self.reason = ""

    def read(self):
        self.reads += 1
        self.clock.t += self.delay_ms
        row = self.rows[min(self.reads - 1, len(self.rows) - 1)]
        return replace(row, t_ms=self.clock.t)

    def close(self):
        self.closed = True


@unittest.skipUnless(READY_TO_RUN, "pygame or pyyaml not installed")
class AppTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ghostgrip.config import load_config
        from ghostgrip.simulation.inputs import load_trace
        from ghostgrip.ui import app
        cls.app_mod = app
        cls.cfg = load_config()
        cls.clean = load_trace(SYNTHETIC / "clean.csv")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.object(self.app_mod, "REPO_ROOT", Path(self.tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.clock = FakeClock()

    def mock_bridge(self):
        from ghostgrip.config import to_haptic_config
        from ghostgrip.hardware.haptics import HapticPolicy
        from ghostgrip.hardware.mock_arduino import MockArduino
        from ghostgrip.hardware.serial_link import SerialLink
        from ghostgrip.ui.integration import HapticBridge
        board = MockArduino(clock_ms=self.clock)
        policy = HapticPolicy(to_haptic_config(self.cfg), clock_ms=self.clock)
        bridge = HapticBridge(SerialLink(board), policy, self.cfg.hardware.keepalive_ms, "mock", self.clock)
        return bridge, board

    def build(self, argv, bridge=None, camera=None, cfg=None):
        """An opened App whose camera and bridge are test doubles."""
        argv = list(argv) + ["--fps", "0", "--session-id", "test"]
        with mock.patch.object(self.app_mod, "make_bridge", lambda target, c: bridge), \
                mock.patch.object(self.app_mod, "open_camera", lambda a, c: camera):
            a = self.app_mod.App(self.app_mod.parse_args(argv), cfg or self.cfg, clock_ms=self.clock)
            a.open()
        self.addCleanup(a.close)
        return a

    def step(self, a, frames=1):
        for _ in range(frames):
            self.clock.t += FRAME_MS
            if not a.frame():
                return False
        return True

    def run_until(self, a, condition, limit=2000):
        for _ in range(limit):
            if condition():
                return True
            self.step(a)
        return condition()

    def count_calls(self, obj, name):
        calls = []
        original = getattr(obj, name)

        def wrapper(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)
        setattr(obj, name, wrapper)
        return calls

    def session(self):
        from ghostgrip.metrics.logger import read_session
        return read_session(Path(self.tmp.name) / "data" / "local" / "sessions" / "test")


class TestB1Quit(AppTestCase):
    def test_nothing_runs_after_quit(self):
        bridge, board = self.mock_bridge()
        camera = FakeCamera(self.clock, self.clean)
        a = self.build(["--camera", "--hardware", "mock", "--condition", "C"], bridge, camera)
        a.handle_key(pygame.K_h)
        self.step(a, 5)
        reads = camera.reads
        advances = self.count_calls(a.world, "advance")
        cues = self.count_calls(bridge, "handle_events")
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        self.assertFalse(a.frame())
        self.assertEqual(camera.reads, reads, "camera read after quit")
        self.assertEqual(advances, [], "physics step after quit")
        self.assertEqual(cues, [], "cue path ran after quit")

    def test_keys_queued_after_escape_are_ignored(self):
        bridge, board = self.mock_bridge()
        a = self.build(["--replay", "clean", "--hardware", "mock", "--condition", "C"], bridge)
        a.handle_events([pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE),
                         pygame.event.Event(pygame.KEYDOWN, key=pygame.K_h)])
        self.assertFalse(a.running)
        self.assertEqual(board.state, "DISARMED")              # the H after Esc never armed it

    def test_quit_ends_with_stop_and_closed_port(self):
        bridge, board = self.mock_bridge()
        a = self.build(["--replay", "clean", "--hardware", "mock", "--condition", "C"], bridge)
        a.handle_key(pygame.K_h)
        self.assertEqual(board.state, "ARMED")
        a.handle_key(pygame.K_ESCAPE)
        a.run()
        self.assertEqual(board.state, "DISARMED")
        self.assertFalse(board.is_open)


class TestB2ResetsDisarm(AppTestCase):
    def test_condition_c_to_a_disarms_and_needs_h_again(self):
        bridge, board = self.mock_bridge()
        a = self.build(["--replay", "clean", "--loop", "--hardware", "mock", "--condition", "C"], bridge)
        a.handle_key(pygame.K_h)
        self.assertTrue(bridge.armed)
        a.handle_key(pygame.K_F1)                               # C -> A
        self.assertFalse(bridge.armed)
        self.assertEqual(board.state, "DISARMED")
        sent = []
        original = bridge.handle_events
        bridge.handle_events = lambda ev, hand, haptic: sent.extend(original(ev, hand, haptic)) or []
        self.run_until(a, lambda: a.all_outcomes, limit=1500)
        self.assertTrue(a.all_outcomes)                         # a full trial ran in A
        self.assertEqual(sent, [])                              # with no cue at all
        a.handle_key(pygame.K_F3)                               # back to C: still disarmed
        self.assertFalse(bridge.armed)
        a.handle_key(pygame.K_h)                                # only an explicit H re-arms
        self.assertTrue(bridge.armed)

    def test_scene_change_and_trial_reset_disarm(self):
        bridge, board = self.mock_bridge()
        a = self.build(["--replay", "clean", "--hardware", "mock", "--condition", "D"], bridge)
        for key in (pygame.K_2, pygame.K_n, pygame.K_RETURN, pygame.K_r):
            a.handle_key(pygame.K_h)
            self.assertTrue(bridge.armed)
            a.handle_key(key)
            self.assertFalse(bridge.armed, pygame.key.name(key))
            self.assertEqual(board.state, "DISARMED")

    def test_interrupted_trial_is_logged_and_ids_stay_unique(self):
        bridge, board = self.mock_bridge()
        a = self.build(["--replay", "clean", "--loop", "--hardware", "mock", "--condition", "C"], bridge)
        self.assertTrue(self.run_until(a, lambda: a.trial.state == "RUNNING"))
        first = a.trial.trial_id
        a.handle_key(pygame.K_F1)                               # switch mid-trial
        self.assertTrue(self.run_until(a, lambda: a.trial.state == "RUNNING"))
        self.assertGreater(a.trial.trial_id, first)
        a.close()
        rows = self.session()["trials"]
        events = [(int(r["trial_id"]), r["event"]) for r in rows]
        self.assertIn((first, "incomplete"), events)
        starts = [t for t, e in events if e == "start"]
        self.assertEqual(len(starts), len(set(starts)))


class TestB3CameraTiming(AppTestCase):
    def test_frame_keeps_its_capture_time_after_a_slow_read(self):
        camera = FakeCamera(self.clock, self.clean, delay_ms=120)   # a slow, blocking read
        a = self.build(["--camera", "--no-log"], camera=camera)
        before_read = self.clock.t
        self.step(a)
        captured = self.clock.t - a.t0_ms                       # read finished 120 ms later
        self.assertEqual(a.last_hand.t_ms, captured)
        self.assertNotEqual(a.last_hand.t_ms, before_read + FRAME_MS - a.t0_ms)
        self.assertEqual(a.world.t_ms, captured)                 # world, AI and log share it
        previous = a.last_hand.t_ms
        self.step(a)
        self.assertEqual(a.last_hand.t_ms - previous, FRAME_MS + 120)   # elapsed = real frame gap


class TestCameraWorkspace(AppTestCase):
    def test_mapping_reaches_the_table_with_the_hand_in_view(self):
        from contracts import HandStateV2
        ws = self.app_mod.CAMERA_WORKSPACE
        x0, x1, y0, y1 = ws
        hand = HandStateV2(0, (0.5, 0.64), 0.3, 0.0, (0.2, 0.3), 1.0, True, 0.9)
        mapped = self.app_mod.map_to_workspace(hand, ws)
        self.assertAlmostEqual(mapped.pointer_xy[0], (0.5 - x0) / (x1 - x0))
        self.assertAlmostEqual(mapped.pointer_xy[1], 0.9)       # table level at 64% of the image
        self.assertAlmostEqual(mapped.velocity_xy[0], 0.2 / (x1 - x0))
        self.assertAlmostEqual(mapped.velocity_xy[1], 0.3 / (y1 - y0))
        # the target zone (world x 0.77-0.97) is reached before the hand leaves the image side
        self.assertLess(x0 + 0.97 * (x1 - x0), 0.75)
        self.assertEqual(mapped.closure_q, 0.3)
        edge = self.app_mod.map_to_workspace(replace(hand, pointer_xy=(0.0, 0.99)), ws)
        self.assertEqual(edge.pointer_xy, (0.0, 1.0))           # clamped to the world

    def test_workspace_option_is_validated(self):
        self.assertEqual(self.app_mod.parse_args(["--camera-workspace", "0.2,0.8,0.1,0.6"]).camera_workspace,
                         (0.2, 0.8, 0.1, 0.6))
        for bad in ("0.8,0.2,0.1,0.6", "0,1,0.5,0.55", "a,b,c,d", "0.1,0.9"):
            with self.assertRaises(SystemExit):
                self.app_mod.parse_args(["--camera-workspace", bad])

    def test_live_camera_frames_are_mapped(self):
        camera = FakeCamera(self.clock, [replace(self.clean[0], pointer_xy=(0.5, 0.64))])
        a = self.build(["--camera", "--no-log"], camera=camera)
        self.step(a)
        self.assertAlmostEqual(a.last_hand.pointer_xy[1], 0.9)


def cam(t, x, y, q, ok=True, quality=1.0):
    from contracts import HandStateV2
    return HandStateV2(t, (x, y), q, 0.0, (0.0, 0.0), 1.0, ok, quality)


class TestCameraStabilizer(unittest.TestCase):
    def stab(self):
        from ghostgrip.ui.integration import CameraStabilizer
        return CameraStabilizer()

    def test_one_frame_teleport_is_held_but_a_real_move_is_accepted(self):
        s = self.stab()
        for t in range(0, 300, 33):
            s.filter(cam(t, 0.30, 0.50, 0.5))
        out = s.filter(cam(330, 0.70, 0.50, 0.5))              # one wild frame
        self.assertEqual(out.pointer_xy, (0.30, 0.50))
        self.assertLess(out.tracking_quality, 0.5)             # unreliable: no grasp, no cue
        self.assertEqual(s.filter(cam(363, 0.31, 0.50, 0.5)).pointer_xy, (0.31, 0.50))
        outs = [s.filter(cam(400 + 33 * k, 0.70 + 0.01 * k, 0.50, 0.5)) for k in range(3)]
        self.assertEqual(outs[0].pointer_xy, (0.31, 0.50))     # still held ...
        self.assertEqual(outs[2].pointer_xy, (0.72, 0.50))     # ... until it persists: a real move
        self.assertEqual(s.counts["teleport"], 3)

    def test_unreliable_frames_hold_and_long_garbage_becomes_lost(self):
        s = self.stab()
        s.filter(cam(0, 0.4, 0.5, 0.7))
        out = s.filter(cam(33, 0.9, 0.1, 0.0, quality=0.2))
        self.assertEqual((out.pointer_xy, out.closure_q, out.tracking_ok), ((0.4, 0.5), 0.7, True))
        out = s.filter(cam(700, 0.9, 0.1, 0.0, quality=0.2))   # held for > 600 ms
        self.assertFalse(out.tracking_ok)                       # -> the world releases safely

    def test_fingers_toward_the_camera_or_sideways_are_held(self):
        from ghostgrip.ui.hand_3d import OPEN_POSE
        s = self.stab()
        normal = pixel_landmarks(OPEN_POSE, scale=100.0)
        for t in range(0, 300, 33):
            s.filter(cam(t, 0.4, 0.5, 0.5), normal)
        pointing = pixel_landmarks(OPEN_POSE, scale=40.0)       # hand collapses on screen
        out = s.filter(cam(330, 0.45, 0.52, 0.95), pointing)
        self.assertEqual((out.pointer_xy, out.closure_q), ((0.4, 0.5), 0.5))
        self.assertEqual(s.counts["foreshortened"], 1)
        sideways = [(x * 0.2 + 320 * 0.8, y) for x, y in normal]  # palm width squashed
        s.filter(cam(363, 0.45, 0.52, 0.95), sideways)
        self.assertEqual(s.counts["edge_on"], 1)

    def test_logged_sponge_drop_incident_no_longer_drops_it(self):
        """Pattern from a live session: holding the sponge, one lost frame, a garbage frame
        (grip 0.00, quality 0.40), two more bad frames, a lost frame, a teleport with grip 1.00,
        two lost frames, then the real hand (still gripping ~0.6) again. Before: RELEASE mid-air."""
        from ghostgrip.simulation.world import World, WORLD_H, WORLD_W
        w = World("sponge")
        sx, sy = 200 / WORLD_W, 412 / WORLD_H
        frames = [cam(t, sx, sy, 0.0) for t in range(0, 400, 33)]                 # open, on sponge
        frames += [cam(400 + t, sx, sy, min(0.77, 0.1 + t / 400)) for t in range(0, 600, 33)]
        frames += [cam(1000 + t, sx + t / 6000, sy - t / 3000, 0.77) for t in range(0, 600, 33)]
        x, y = frames[-1].pointer_xy
        t0 = frames[-1].t_ms
        frames += [cam(t0 + 33, x, y, 0.77, ok=False, quality=0.0),
                   cam(t0 + 66, x - 0.05, y + 0.01, 0.00, quality=0.40),
                   cam(t0 + 99, x - 0.06, y + 0.01, 0.09),
                   cam(t0 + 132, x - 0.06, y + 0.01, 0.18),
                   cam(t0 + 165, x, y, 0.26, ok=False, quality=0.0),
                   cam(t0 + 198, x + 0.17, y + 0.26, 1.00),
                   cam(t0 + 231, x, y, 1.0, ok=False, quality=0.0),
                   cam(t0 + 264, x, y, 1.0, ok=False, quality=0.0)]
        frames += [cam(t0 + 297 + 33 * k, x, y - 0.02, 0.6) for k in range(20)]
        for guarded in (False, True):
            s = self.stab() if guarded else None
            world, prev, events = World("sponge"), 0, []
            for h in frames:
                h2 = s.filter(h) if s else h
                events += [e.kind for e in world.advance(h2, (h.t_ms - prev) / 1000.0)]
                prev = h.t_ms
            if guarded:
                self.assertEqual(events, ["CONTACT", "HOLD"])    # still holding the sponge
                self.assertEqual(world.engaged_id, "sponge")
            else:
                self.assertIn("RELEASE", events)                 # what happened before the guard


class TestB4Evaluation(AppTestCase):
    def test_assist_gain_is_refused_with_evaluation(self):
        with self.assertRaises(SystemExit):
            self.app_mod.parse_args(["--evaluation", "--assist-gain", "0.5"])

    def finish_one_trial(self, argv):
        a = self.build(argv)
        self.assertTrue(self.run_until(a, lambda: a.all_outcomes, limit=1500))
        self.assertTrue(a.all_outcomes[0].success)
        return a

    def test_profile_frozen_across_trials_resets_and_keys(self):
        a = self.finish_one_trial(["--replay", "clean", "--loop", "--condition", "B", "--evaluation"])
        policy = a.ai.policy
        profile = policy.profile
        self.assertTrue(policy.frozen)
        self.assertEqual(policy.profile, profile)
        self.assertGreaterEqual(policy.skipped, 1)              # end_trial ran, learning skipped
        for key in (pygame.K_n, pygame.K_2, pygame.K_F4, pygame.K_1):
            a.handle_key(key)
        self.assertTrue(a.ai.policy.frozen)
        a.handle_key(pygame.K_RIGHTBRACKET)
        a.handle_key(pygame.K_LEFTBRACKET)
        self.assertIsNone(a.manual_gain)                        # manual override refused
        self.assertEqual(a.ai.policy.profile, profile)

    def test_without_evaluation_the_same_trial_does_learn(self):
        """Control: proves the frozen test above would notice learning."""
        a = self.finish_one_trial(["--replay", "clean", "--loop", "--condition", "B"])
        self.assertFalse(a.ai.policy.frozen)
        self.assertEqual(a.ai.policy.profile.trials, 1)

    def test_session_metadata_records_the_mode(self):
        import json
        bridge, _ = self.mock_bridge()
        a = self.build(["--replay", "clean", "--hardware", "mock", "--evaluation"], bridge)
        meta = json.loads((a.log_folder / f"run_{a.logger.run}_meta.json").read_text())
        self.assertTrue(meta["evaluation"])
        self.assertTrue(meta["assist_profile_frozen"])
        self.assertEqual(meta["hardware"], "mock")
        self.assertEqual(meta["origin"], "synthetic")
        self.assertFalse(meta["servo_enabled"])


class TestB5Cleanup(AppTestCase):
    def test_startup_failure_releases_board_camera_and_log(self):
        bridge, board = self.mock_bridge()
        bridge.arm()                                            # worst case: already armed
        camera = FakeCamera(self.clock, self.clean)
        argv = ["--camera", "--hardware", "mock", "--fps", "0", "--session-id", "test"]
        a = self.app_mod.App(self.app_mod.parse_args(argv), self.cfg, clock_ms=self.clock)
        with mock.patch.object(self.app_mod, "make_bridge", lambda t, c: bridge), \
                mock.patch.object(self.app_mod, "open_camera", lambda x, c: camera), \
                mock.patch.object(self.app_mod, "World", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                a.open()
        self.assertEqual(board.state, "DISARMED")
        self.assertFalse(board.is_open)
        self.assertTrue(camera.closed)
        self.assertTrue(a.logger.closed)
        self.assertFalse(pygame.get_init())

    def test_crash_inside_the_loop_still_stops_the_board(self):
        bridge, board = self.mock_bridge()
        camera = FakeCamera(self.clock, self.clean)
        a = self.build(["--camera", "--hardware", "mock", "--condition", "C"], bridge, camera)
        a.handle_key(pygame.K_h)
        self.step(a, 3)
        a.world.advance = mock.Mock(side_effect=RuntimeError("crash"))
        with self.assertRaises(RuntimeError):
            a.run()
        self.assertEqual(board.state, "DISARMED")
        self.assertFalse(board.is_open)
        self.assertTrue(camera.closed)
        self.assertTrue(a.logger.closed)

    def test_close_keeps_going_when_one_step_fails(self):
        bridge, board = self.mock_bridge()
        camera = FakeCamera(self.clock, self.clean)
        a = self.build(["--camera", "--hardware", "mock"], bridge, camera)
        camera.close = mock.Mock(side_effect=OSError("camera stuck"))
        a.close()
        self.assertFalse(board.is_open)                         # before the failing step
        self.assertTrue(a.logger.closed)                        # after it
        a.close()                                               # safe twice


class TestB6RemainingChecks(AppTestCase):
    def test_replay_end_disarms_the_board(self):
        bridge, board = self.mock_bridge()
        a = self.build(["--replay", "clean", "--hardware", "mock", "--condition", "C"], bridge)
        a.handle_key(pygame.K_h)
        self.assertTrue(self.run_until(a, lambda: a.replay_done, limit=2000))
        self.assertFalse(bridge.armed)
        self.assertIn("replay ended", bridge.message)

    def test_hardware_disabled_in_config_opens_no_port(self):
        off = replace(self.cfg, hardware=replace(self.cfg.hardware, enabled=False))
        with mock.patch("ghostgrip.ui.integration.HapticBridge.connect") as connect:
            self.assertIsNone(self.app_mod.make_bridge("COM9", off))
            self.assertIsNone(self.app_mod.make_bridge("auto", off))
            connect.assert_not_called()
        self.assertIsNotNone(self.app_mod.make_bridge("mock", off))   # the fake board opens nothing

    def test_every_condition_over_two_trials(self):
        for code in ("A", "B", "C", "D"):
            with self.subTest(condition=code):
                self.clock = FakeClock()
                bridge, board = self.mock_bridge()
                a = self.build(["--replay", "clean", "--loop", "--hardware", "mock",
                                "--condition", code], bridge)
                sent = []
                original = bridge.handle_events
                bridge.handle_events = lambda ev, hand, haptic, o=original: sent.extend(o(ev, hand, haptic)) or []
                for trial in range(2):
                    a.handle_key(pygame.K_h)                    # deliberate arming each trial
                    done = len(a.all_outcomes)
                    self.assertTrue(self.run_until(a, lambda: len(a.all_outcomes) > done, limit=1500))
                    a.handle_key(pygame.K_n)                    # next trial (disarms)
                self.assertEqual([o.success for o in a.all_outcomes], [True, True])
                contacts = [c for c in sent if c.kind == "CONTACT"]
                if code in ("C", "D"):
                    self.assertEqual(len(contacts), 2)          # one CONTACT cue per trial
                else:
                    self.assertEqual(sent, [])                  # A and B never cue
                a.close()


def fake_landmarks(curl):
    """21 (x, y) points: wrist at the bottom, fingers up; curl 0 = straight, 1 = folded."""
    pts = [(0.5, 0.85)] * 21
    for f, x in enumerate((0.40, 0.46, 0.50, 0.54, 0.58)):
        base = 1 + 4 * f
        pts[base] = (x, 0.65)
        for j in range(1, 4):
            straight = 0.65 - 0.06 * j
            folded = 0.65 - (0.03 if j == 1 else -0.02 * (j - 1))
            pts[base + j] = (x, straight * (1 - curl) + folded * curl)
    return pts


def identity3d(x, y, z=0.0):
    return (x, y, z)


def pixel_landmarks(pose, scale=100.0, ox=320.0, oy=400.0):
    return [(ox + scale * x, oy + scale * y) for x, y in pose]


class TestHand3D(unittest.TestCase):
    def posed(self, points=None, q=0.0, pointer=(300.0, 200.0), held=None):
        from ghostgrip.ui.hand_3d import Hand3D
        hand = Hand3D()
        hand.update(q, 0.0, points)
        return hand, hand.pose(pointer, identity3d, held)["joints"]

    def test_each_finger_follows_only_its_own_landmarks(self):
        from ghostgrip.ui.hand_3d import FINGERS, OPEN_POSE
        _, open_joints = self.posed(pixel_landmarks(OPEN_POSE))
        bent = list(OPEN_POSE)
        for j in FINGERS["ring"][1:]:                           # fold only the ring finger
            bent[j] = (OPEN_POSE[13][0], OPEN_POSE[13][1] - 0.1)
        hand, bent_joints = self.posed(pixel_landmarks(bent))
        self.assertTrue(hand.from_camera)
        moved = {i for i in range(21) if open_joints[i] != bent_joints[i]}
        self.assertEqual(moved, set(FINGERS["ring"][1:]))

    def test_pinch_point_is_the_pointer(self):
        from ghostgrip.ui.hand_3d import OPEN_POSE
        for points in (None, pixel_landmarks(OPEN_POSE)):
            _, j = self.posed(points, q=0.3, pointer=(300.0, 200.0))
            self.assertAlmostEqual((j[4][0] + j[8][0]) / 2, 300.0)
            self.assertAlmostEqual((j[4][1] + j[8][1]) / 2, 200.0)

    def test_hand_size_does_not_depend_on_camera_distance(self):
        from ghostgrip.ui.hand_3d import OPEN_POSE
        _, near = self.posed(pixel_landmarks(OPEN_POSE, scale=180.0))
        _, far = self.posed(pixel_landmarks(OPEN_POSE, scale=60.0))
        for a, b in zip(near, far):
            self.assertAlmostEqual(a[0], b[0], places=6)
            self.assertAlmostEqual(a[1], b[1], places=6)

    def test_held_object_fingertips_rest_on_its_surface(self):
        thumb, index = (270.0, 200.0), (330.0, 200.0)
        _, j = self.posed(q=0.5, held=(thumb, index))
        self.assertAlmostEqual(j[4][0], thumb[0])
        self.assertAlmostEqual(j[4][1], thumb[1])
        self.assertAlmostEqual(j[8][0], index[0])
        self.assertAlmostEqual(j[8][1], index[1])

    def test_an_edge_on_hand_cannot_explode(self):
        """Hand turned sideways: the palm collapses to a few pixels while the fingers stay long.
        Dividing by the tiny palm made the drawn fingers shoot off-screen."""
        import math
        from ghostgrip.ui.hand_3d import normalized_shape
        pts = [(320.0, 400.0)] + [(322.0 + i * 0.3, 395.0) for i in range(4)] * 5
        for chain_start in (1, 5, 9, 13, 17):
            for k in range(1, 4):
                pts[chain_start + k] = (325.0 + 2 * k, 400.0 - 50.0 * k)  # long fingers
        shape = normalized_shape(pts[:21])
        self.assertIsNotNone(shape)
        self.assertLess(max(math.hypot(x, y) for x, y in shape), 3.0)   # a normal hand is ~2

    def test_unreliable_frames_hold_the_last_good_pose(self):
        from ghostgrip.ui.hand_3d import OPEN_POSE, Hand3D
        hand = Hand3D()
        hand.update(0.0, 0.0, pixel_landmarks(OPEN_POSE))
        good = [list(p) for p in hand.shape]
        for _ in range(60):                                     # 1 s of unreliable frames
            hand.update(0.9, 1 / 60, None)
        self.assertEqual(hand.shape, good)
        self.assertTrue(hand.from_camera)
        for _ in range(120):                                    # 2 s more: ease to the default
            hand.update(0.9, 1 / 60, None)
        self.assertFalse(hand.from_camera)
        self.assertNotEqual(hand.shape, good)

    def test_without_a_camera_the_hand_pinches_with_q(self):
        import math
        from ghostgrip.ui.hand_3d import template
        gap_open = math.dist(template(0.0)[4], template(0.0)[8])
        gap_closed = math.dist(template(1.0)[4], template(1.0)[8])
        self.assertLess(gap_closed, 0.2 * gap_open)
        self.assertIsNone(self.posed(None)[0].from_camera or None)
        self.assertIsNone(self.posed([(0.5, 0.5)] * 5)[0].from_camera or None)   # bad landmarks


@unittest.skipUnless(READY_TO_RUN, "pygame or pyyaml not installed")
class TestRenderer3D(unittest.TestCase):
    def test_draws_every_scene_and_state_headless(self):
        from contracts import HandStateV2
        from ghostgrip.simulation.world import World
        from ghostgrip.ui.hud import PANEL_W
        from ghostgrip.ui.renderer_3d import Renderer3D
        pygame.init()
        try:
            surf = pygame.display.set_mode((960 + PANEL_W, 540))
            r = Renderer3D()
            for scene in ("glass", "sponge", "block", "sandbox"):
                w = World(scene)
                w.set_assistance(True, 1.0)
                for t in range(1, 40):
                    q = 0.95 if t > 25 else 0.0                 # squeeze hard: squash, break, slip
                    w.advance(HandStateV2(t * 16, (200 / 960, 420 / 540), q, 0.0, (0.0, 0.0), 1.0,
                                          True, 1.0), 0.016)
                r.draw(surf, w.snapshot(), ghost_world=(300.0, 300.0), banner=("SUCCESS", (0, 255, 0)),
                       hand_points=fake_landmarks(0.5))
            w.advance(HandStateV2(700, (0.5, 0.5), 0.0, 0.0, (0.0, 0.0), 0.0, False, 0.0), 0.016)
            r.draw(surf, w.snapshot())                          # tracking lost
            self.assertEqual(surf.get_clip(), surf.get_rect())  # clip restored for the HUD
        finally:
            pygame.quit()


if __name__ == "__main__":
    unittest.main()
