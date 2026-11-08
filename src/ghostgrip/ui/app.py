"""GhostGrip integrated app: hand input -> Member D's AI -> World -> 3D/2D view -> Member C's hardware.

Run from the repository root (Windows):
    .venv\\Scripts\\python.exe src\\ghostgrip\\ui\\app.py                          # mouse
    .venv\\Scripts\\python.exe src\\ghostgrip\\ui\\app.py --camera                 # live hand (Member A)
    .venv\\Scripts\\python.exe src\\ghostgrip\\ui\\app.py --camera --hardware COM4 --condition C
    .venv\\Scripts\\python.exe src\\ghostgrip\\ui\\app.py --replay clean --hardware mock --condition D
    .venv\\Scripts\\python.exe src\\ghostgrip\\ui\\app.py --camera --evaluation --session-id final_eval

Every frame, in this order:
    events = keyboard/window                   (quit stops here: nothing below runs again)
    hand   = camera / mouse / replay           (HandStateV2; camera keeps its capture time)
    ai     = predictor, assist gain, intent    (Member D; frozen with --evaluation)
    events = world.advance(hand, dt)           (Member B, gain applied, bounded)
    cues   = haptic safety gate -> Arduino     (Member C, conditions C and D only)
    trial, session log, draw                   (Member B + Member D's logger)

Safety rules kept here (AGENTS.md): the board is armed only by the H key; scene, condition and
trial resets send STOP first and need H again; quitting sends no further cue; any exit, even an
exception, sends STOP, closes the serial port, releases the camera and finalises the log.
"""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

if __package__ in (None, ""):
    # Run as a script: make the repo root (shared contracts.py) and src/ importable.
    for _p in (Path(__file__).resolve().parents[3], Path(__file__).resolve().parents[2]):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

import pygame  # noqa: E402

from ghostgrip.ai.adapt import AssistDecision  # noqa: E402
from ghostgrip.config import ConfigError, load_config  # noqa: E402
from ghostgrip.metrics.logger import SessionLogger, read_session  # noqa: E402
from ghostgrip.simulation.inputs import MouseKeyboardAdapter, TracePlayer, TraceRecorder  # noqa: E402
from ghostgrip.simulation.trial import CONDITIONS, READY, RUNNING, TrialRunner  # noqa: E402
from ghostgrip.simulation.world import SCENES, WORLD_H, WORLD_W, World  # noqa: E402
from ghostgrip.ui.hud import PANEL_W, Hud, HudInfo  # noqa: E402
from ghostgrip.ui.integration import AIPipeline, CameraStabilizer, HapticBridge, lost_hand  # noqa: E402
from ghostgrip.ui.renderer_2d import Renderer2D  # noqa: E402
from ghostgrip.ui.renderer_3d import Renderer3D  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
SYNTHETIC_DIR = REPO_ROOT / "data" / "synthetic"
GAIN_STEP = 0.1
CAMERA_VIEW_W = 240
# Part of the camera image (x0, x1, y0, y1, normalised) that spans the whole world. Objects sit
# low in the world (the table is at 83% of its height); mapped 1:1, the hand had to go to the
# bottom edge of the image, where the wrist leaves the frame and tracking becomes unreliable.
# X stops at 0.70: in a logged session tracking quality collapsed from x = 0.76 of the image
# on the way to the target (the palm and fingers extend to the right of the pinch point).
CAMERA_WORKSPACE = (0.15, 0.70, 0.10, 0.70)
VIEWS = ("3d", "2d")
SCENE_KEYS = {pygame.K_1: "glass", pygame.K_2: "sponge", pygame.K_3: "block", pygame.K_0: "sandbox"}
CONDITION_KEYS = {pygame.K_F1: "A", pygame.K_F2: "B", pygame.K_F3: "C", pygame.K_F4: "D"}


def monotonic_ms() -> int:
    """App clock. The same monotonic clock Member A's CameraTracker stamps frames with."""
    return time.monotonic_ns() // 1_000_000


# ====================================================================== setup helpers
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="GhostGrip integrated simulation")
    ap.add_argument("--scene", default="glass", choices=list(SCENES))
    ap.add_argument("--condition", default="A", choices=list(CONDITIONS))
    ap.add_argument("--camera", action="store_true",
                    help="live hand from the webcam (Member A); needs data/local/calibration.json "
                         "(python vision_preview.py --calibrate)")
    ap.add_argument("--camera-index", type=int, default=None, help="override config.yaml camera.index")
    ap.add_argument("--camera-workspace", type=parse_workspace, default=CAMERA_WORKSPACE,
                    metavar="X0,X1,Y0,Y1",
                    help="part of the camera image that spans the world (default "
                         f"{','.join(str(v) for v in CAMERA_WORKSPACE)}); lower Y1 if reaching "
                         "the table is hard")
    ap.add_argument("--replay", default=None,
                    help="CSV trace to replay instead of the mouse, or a shared synthetic trace "
                         f"name: {', '.join(synthetic_names()) or '(none found)'}")
    ap.add_argument("--loop", action="store_true", help="loop the replay")
    ap.add_argument("--hardware", nargs="?", const="auto", default=None,
                    help="send cues to the Arduino: a port such as COM4, 'auto', or 'mock' "
                         "for a fake board. Cues only in conditions C and D; press H to arm")
    ap.add_argument("--assist-gain", type=float, default=None,
                    help="fixed manual assistance gain 0..1 instead of Member D's AssistPolicy "
                         "(not allowed with --evaluation)")
    ap.add_argument("--evaluation", action="store_true",
                    help="measured comparison: Member D's assist profile is frozen (no learning "
                         "between trials) and manual assist overrides are refused")
    ap.add_argument("--view", default="3d", choices=VIEWS,
                    help="3d (default) or 2d; Tab switches while running. Same simulation in both")
    ap.add_argument("--fps", type=int, default=60, help="display frame cap (physics stays 60 Hz)")
    ap.add_argument("--session-id", default="anonymous",
                    help="anonymous id (letters, digits, - and _); never a real name")
    ap.add_argument("--no-log", action="store_true",
                    help="do not write the session log in data/local/sessions/<session-id>/")
    ap.add_argument("--max-frames", type=int, default=0, help="quit after N frames (smoke test)")
    args = ap.parse_args(argv)
    if args.camera and args.replay:
        ap.error("choose either --camera or --replay, not both")
    if args.evaluation and args.assist_gain is not None:
        ap.error("--assist-gain cannot be used with --evaluation: a measured comparison must use "
                 "Member D's frozen policy")
    return args


def parse_workspace(text):
    try:
        x0, x1, y0, y1 = (float(v) for v in str(text).split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("expected four numbers: X0,X1,Y0,Y1")
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0 and x1 - x0 >= 0.2 and y1 - y0 >= 0.2):
        raise argparse.ArgumentTypeError("need 0 <= X0 < X1 <= 1 and 0 <= Y0 < Y1 <= 1, each span >= 0.2")
    return (x0, x1, y0, y1)


def map_to_workspace(hand, workspace):
    """Camera pointer (normalised image coordinates) -> world (0-1), so the chosen part of the
    image spans the whole world. Velocities scale with it; closure and quality are unchanged."""
    x0, x1, y0, y1 = workspace
    sx, sy = x1 - x0, y1 - y0
    x = min(1.0, max(0.0, (hand.pointer_xy[0] - x0) / sx))
    y = min(1.0, max(0.0, (hand.pointer_xy[1] - y0) / sy))
    return replace(hand, pointer_xy=(x, y),
                   velocity_xy=(hand.velocity_xy[0] / sx, hand.velocity_xy[1] / sy))


def synthetic_names():
    return sorted(p.stem for p in SYNTHETIC_DIR.glob("*.csv"))


def resolve_replay(arg):
    """A path, or the bare name of a shared trace in data/synthetic/ (e.g. 'clean')."""
    path = Path(arg)
    if path.exists():
        return path
    named = SYNTHETIC_DIR / f"{arg}.csv"
    if named.exists():
        return named
    raise SystemExit(f"--replay: no file {arg!r} and no synthetic trace of that name "
                     f"(choose from: {', '.join(synthetic_names())})")


def data_origin(camera: bool, replay_path) -> str:
    """Member D's logger must say whether the hand data is real or synthetic; never guess wrong.
    real: the live camera, or a recording of a real hand (Member A's recorder layout).
    synthetic: the shared synthetic traces, and mouse/keyboard (its hand values are generated)."""
    if camera:
        return "real"
    if replay_path is None:
        return "synthetic"
    if SYNTHETIC_DIR.resolve() in Path(replay_path).resolve().parents:
        return "synthetic"
    with Path(replay_path).open() as f:
        header = f.readline()
    return "real" if "pointer_x" in header else "synthetic"


def open_camera(args, cfg):
    from ghostgrip.vision.camera import LiveHandSource   # imports OpenCV + MediaPipe only when needed
    print("Opening the camera and the hand model ...")
    try:
        source = LiveHandSource(camera_index=args.camera_index, config=cfg.camera)
        source.tracker.open()
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        raise SystemExit(f"--camera: {error}")
    return source


def make_bridge(target, cfg):
    """Member C's link, or None. config.yaml hardware.enabled=false means never open a serial
    port (the fake board opens nothing, so 'mock' stays allowed for tests and demos)."""
    if not target:
        return None
    if target.lower() != "mock" and not cfg.hardware.enabled:
        print("Hardware: config.yaml has hardware.enabled: false, so no serial port is opened. "
              "Running laptop-only (use --hardware mock for the fake board).")
        return None
    return HapticBridge.connect(target, cfg)


def git_revision():
    try:
        out = subprocess.run(["git", "describe", "--always", "--dirty", "--abbrev=12"], cwd=REPO_ROOT,
                             capture_output=True, text=True, timeout=3)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def camera_view(frame):
    """Small preview of the (already mirrored) camera image. Shown only, never saved."""
    import cv2
    h, w = frame.shape[:2]
    height = max(1, int(CAMERA_VIEW_W * h / w))
    small = cv2.cvtColor(cv2.resize(frame, (CAMERA_VIEW_W, height)), cv2.COLOR_BGR2RGB)
    return pygame.image.frombuffer(small.tobytes(), (CAMERA_VIEW_W, height), "RGB")


def draw_board_overlay(screen, font, bridge, condition):
    """Make the hardware state impossible to miss: the LED's cue flicker lasts only 150 ms."""
    if bridge is None or not condition.haptic:
        return
    if not bridge.connected:
        text, color = f"Board {bridge.status}: cues cannot reach the Arduino", (255, 110, 110)
    elif not bridge.armed:
        text, color = "Board DISARMED: click this window, then press H", (255, 200, 90)
    elif bridge.cue_recent():
        text, color = f"CUE SENT TO BOARD: {bridge.last_cue}", (110, 235, 150)
    else:
        return
    img = font.render(text, True, color)
    box = img.get_rect(center=(WORLD_W // 2 - CAMERA_VIEW_W // 2, 40)).inflate(20, 10)
    pygame.draw.rect(screen, (20, 24, 36), box, border_radius=6)
    screen.blit(img, img.get_rect(center=box.center))


# ====================================================================== the app
class App:
    """One integrated session. open() acquires everything (and releases it again if any step
    fails), frame() runs one loop iteration, close() always releases everything, STOP first."""

    def __init__(self, args, cfg, clock_ms=monotonic_ms):
        self.args, self.cfg, self.clock_ms = args, cfg, clock_ms
        self.camera = None
        self.stabilizer = CameraStabilizer()
        self._camera_offset_ms = None     # camera clock minus app clock (0 for Member A's camera)
        self.bridge = None
        self.logger = None
        self.log_folder = None
        self.recorder = None
        self.pygame_started = False
        self.running = True
        self.closed = False
        self.all_outcomes = []
        self.tally = {}
        self.source_name = "mouse"

    # ------------------------------------------------------------------ lifecycle
    def open(self) -> "App":
        try:
            self._open()
        except BaseException:
            self.close()            # B5: a failed start never leaves the board armed or the camera on
            raise
        return self

    def _open(self) -> None:
        a, cfg = self.args, self.cfg
        self.replay_path = resolve_replay(a.replay) if a.replay else None
        if a.camera:
            self.camera = open_camera(a, cfg)
        self.origin = data_origin(a.camera, self.replay_path)
        self.last_id = 0
        if not a.no_log:
            self.log_folder = REPO_ROOT / "data" / "local" / "sessions" / a.session_id
            try:
                self.logger = SessionLogger(self.log_folder, a.session_id, self.origin)
            except ValueError as error:
                raise SystemExit(f"session log: {error}")
            # Trial ids keep counting across app restarts, scene and condition changes.
            self.last_id = max((int(r["trial_id"]) for r in read_session(self.log_folder)["trials"]),
                               default=0)
        self.bridge = make_bridge(a.hardware, cfg)

        pygame.init()
        self.pygame_started = True
        self.screen = pygame.display.set_mode((WORLD_W + PANEL_W, WORLD_H))
        pygame.display.set_caption("GhostGrip" + ("  [EVALUATION - assistance frozen]" if a.evaluation else ""))
        self.clock = pygame.time.Clock()
        self.overlay_font = pygame.font.Font(None, 32)
        self.renderers = {"3d": Renderer3D(), "2d": Renderer2D()}
        self.view = a.view
        self.hud = Hud(WORLD_W, WORLD_H)

        self.world = World(a.scene)
        self.ai = AIPipeline(cfg, self.world.scene)
        if a.evaluation:
            self.ai.policy.freeze()            # B4: no learning between trials, kept across resets
        self.condition = CONDITIONS[a.condition]
        self.adapter = MouseKeyboardAdapter()
        self.player = TracePlayer.from_csv(self.replay_path, loop=a.loop) if self.replay_path else None
        self.source_name = "camera" if self.camera else ("replay" if self.player else "mouse")
        self.ghost_on = True
        self.camera_view_on = True
        self.camera_ended = False
        self.replay_done = False
        self.banner = None
        self.last_hand = None
        self.frame_ai = None
        # Manual override (--assist-gain or [ ] keys); None = Member D's AssistPolicy decides.
        self.manual_gain = None if a.assist_gain is None else min(1.0, max(0.0, a.assist_gain))
        self.frames = 0
        self.trial = self._new_trial_runner()
        self._write_session_meta()
        self.t0_ms = self.clock_ms()
        self.t_ms = 0
        self._print_start()

    def run(self) -> int:
        try:
            while self.frame():
                pass
        finally:
            self.close()
        return 0

    def close(self) -> None:
        """Release everything, in safety order, even if an earlier step fails. Safe to call twice."""
        if self.closed:
            return
        self.closed = True
        self.running = False
        for name, step in (("hardware STOP and serial close", self._close_bridge),
                           ("camera", self._close_camera),
                           ("trace", self._save_trace),
                           ("session log", self._close_logger),
                           ("window", self._close_window)):
            try:
                step()
            except Exception as error:   # keep going: the remaining resources still need releasing
                print(f"Cleanup: {name} failed: {error}")
        self._print_summary()

    def _close_bridge(self):
        if self.bridge is not None:
            self.bridge.close()                # sends STOP before the port closes

    def _close_camera(self):
        if self.camera is not None:
            self.camera.close()

    def _save_trace(self):
        if self.recorder is not None and self.recorder.rows:
            path = REPO_ROOT / "data" / "local" / time.strftime("trace_%Y%m%d_%H%M%S.csv")
            print("Trace saved:", self.recorder.save(path))
        self.recorder = None

    def _close_logger(self):
        if self.logger is not None:
            self.logger.close()                # a trial still running is logged as incomplete

    def _close_window(self):
        if self.pygame_started:
            pygame.quit()
            self.pygame_started = False

    # ------------------------------------------------------------------ one frame
    def frame(self) -> bool:
        """One loop iteration. Returns False once the app is quitting."""
        self.handle_events(pygame.event.get())
        if not self.running:
            return False                       # B1: no camera read, physics step or cue after quit

        hand = self._read_hand()
        elapsed = max(0, hand.t_ms - self.t_ms) / 1000.0
        self.t_ms = hand.t_ms
        self.last_hand = hand
        if self.recorder is not None:
            self.recorder.append(hand)

        # Member D: prediction, assistance gain, intent
        frame_ai = self.ai.update(hand, record_for_trial=self.trial.state == RUNNING)
        self.frame_ai = frame_ai
        applied = self._assist(frame_ai)
        self.world.set_assistance(self.condition.ai_assist, applied.gain)

        # physics (Member B)
        events = self.world.advance(hand, elapsed)
        for e in events:
            reason = self.world.event_reason(e.event_id)
            route = "  -> haptics" if self.condition.haptic else ""
            print(f"[{e.t_ms/1000:8.3f}s] #{e.event_id:<3d} {e.kind:7s} {e.object_id:6s} "
                  f"G={e.simulated_grip_effort:0.2f} ({reason}){route}")

        self._hardware(events, hand)
        self._trial_and_log(events, hand, frame_ai, applied)
        self._draw(hand, frame_ai)

        self.clock.tick(self.args.fps)
        self.frames += 1
        if self.args.max_frames and self.frames >= self.args.max_frames:
            self.running = False
        return self.running

    def _read_hand(self):
        now = max(self.clock_ms() - self.t0_ms, self.t_ms + 1)   # frame times strictly increase
        if self.camera is not None:
            state = None if self.camera_ended else self.camera.read()
            if state is None:
                if not self.camera_ended:
                    self.camera_ended = True
                    print("Camera stopped:", getattr(self.camera, "reason", "") or "no more frames")
                    self.banner = ("CAMERA STOPPED - restart the app", (255, 110, 110))
                    self._disarm("camera stopped")
                return lost_hand(max(self.clock_ms() - self.t0_ms, self.t_ms + 1), self.last_hand)
            # B3: keep the frame's own capture time, re-based to the app origin; never label a
            # frame with the time from before its blocking read. Member A's camera stamps frames
            # on the same monotonic clock as the app. A source on another clock (tests, other
            # cameras) is aligned once, at its first frame, to the time just after that read.
            after_read = self.clock_ms()
            if self._camera_offset_ms is None:
                gap = int(state.t_ms) - after_read
                self._camera_offset_ms = 0 if abs(gap) < 5000 else gap
            t = int(state.t_ms) - self._camera_offset_ms - self.t0_ms
            state = replace(state, t_ms=max(t, self.t_ms + 1))   # strictly increasing
            state = map_to_workspace(state, self.args.camera_workspace)
            # bad frames (jumps, spikes, fingers toward the camera) never reach the physics
            return self.stabilizer.filter(state, self._camera_points_px())
        if self.player is not None:
            hand = replace(self.player.sample(now), t_ms=now)
            if self.player.finished(now) and not self.replay_done:
                self.replay_done = True
                print("Replay finished (holding the last row).")
                self._disarm("replay ended")   # B6: a finished replay cannot keep the board armed
            return hand
        mx, my = pygame.mouse.get_pos()
        self.adapter.set_pointer_px(min(mx, WORLD_W - 1), my, WORLD_W, WORLD_H)
        buttons = pygame.mouse.get_pressed()
        keys = pygame.key.get_pressed()
        return self.adapter.sample(now, closing=buttons[0] or keys[pygame.K_w],
                                   opening=buttons[2] or keys[pygame.K_s])

    def _camera_points_px(self):
        """Member A's 21 landmarks of the current frame in pixels, or None."""
        obs = getattr(self.camera, "last_observation", None) if self.camera is not None else None
        points = getattr(obs, "points", None)
        if points is None:
            return None
        frame = getattr(obs, "frame", None)
        h, w = frame.shape[:2] if frame is not None else (480, 640)
        return [(x * w, y * h) for x, y in points]

    def _assist(self, frame_ai) -> AssistDecision:
        if not self.condition.ai_assist:
            return AssistDecision(0.0, "condition_off")
        if self.manual_gain is not None:       # never set while --evaluation is on
            return AssistDecision(self.manual_gain, "manual")
        return frame_ai.decision

    def _hardware(self, events, hand) -> None:
        if self.bridge is None:
            return
        device = self.bridge.update(hand)
        if device is not None and self.logger is not None:
            self.logger.log_device(self.t_ms, device)
        for cue in self.bridge.handle_events(events, hand, self.condition.haptic):
            if self.logger is not None and self.bridge.link.device is not None:
                self.logger.log_device(self.t_ms, self.bridge.link.device, cue)

    def _trial_and_log(self, events, hand, frame_ai, applied) -> None:
        if self.world.scene != "sandbox":
            before = self.trial.state
            outcome = self.trial.update(self.world, events, hand)
            if before == READY and self.trial.state == RUNNING:
                self.last_id = max(self.last_id, self.trial.trial_id)
                self.ai.trial_states = [hand]
                if self.logger is not None:
                    self.logger.begin_trial(self.trial.trial_id, self.condition.code, hand.t_ms)
            if outcome is not None:
                self.all_outcomes.append(outcome)
                s_n = self.tally.setdefault(outcome.condition, [0, 0])
                s_n[0] += int(outcome.success)
                s_n[1] += 1
                secs = (outcome.end_ms - outcome.start_ms) / 1000.0
                self.banner = ((f"SUCCESS  {secs:0.1f}s  (N: next trial)", (110, 235, 150))
                               if outcome.success else
                               (f"FAILED: {outcome.fail_reason}  (N: next trial)", (255, 110, 110)))
                print("TRIAL", outcome)
                self.ai.end_trial(outcome)     # skipped by Member D's policy while frozen
                if self.logger is not None:
                    self.logger.end_trial(outcome)
        if self.logger is not None:
            self.logger.log_frame(hand, frame_ai.prediction, frame_ai.intent, applied,
                                  trial_id=self.trial.trial_id if self.trial.state == RUNNING else None,
                                  condition=self.condition.code)

    def _draw(self, hand, frame_ai) -> None:
        snap = self.world.snapshot()
        ghost = AIPipeline.ghost_xy(hand, frame_ai) if self.ghost_on else None
        ghost_px = (ghost[0] * WORLD_W, ghost[1] * WORLD_H) if ghost else None
        obs = getattr(self.camera, "last_observation", None) if self.camera is not None else None
        # Member A's landmarks pose the drawn hand only (never the physics or the board); the
        # renderer ignores them on unreliable frames (the guard marks bad ones unreliable).
        points = self._camera_points_px() if hand.tracking_ok else None
        self.renderers[self.view].draw(self.screen, snap, ghost_px, self.banner, hand_points=points)
        if self.camera_view_on and obs is not None and getattr(obs, "frame", None) is not None:
            self.screen.blit(camera_view(obs.frame), (WORLD_W - CAMERA_VIEW_W - 8, 8))
        draw_board_overlay(self.screen, self.overlay_font, self.bridge, self.condition)
        trial_state = "READY (open hand)" if self.trial.state == READY else self.trial.state
        intent = frame_ai.intent
        if not self.condition.ai_assist:
            source = ""
        elif self.args.evaluation:
            source = "AI, frozen"
        else:
            source = "manual" if self.manual_gain is not None else "AI"
        self.hud.draw(self.screen, snap, HudInfo(
            condition_code=self.condition.code, condition_label=self.condition.label,
            source=self.source_name, object_kind=self.world.scene,
            trial_state=trial_state, trial_id=self.trial.trial_id,
            trial_elapsed_ms=self.trial.elapsed_ms(self.t_ms), tally=self.tally,
            fps=self.clock.get_fps(), recording=self.recorder is not None,
            ghost_on=self.ghost_on, haptic_routing=self.condition.haptic,
            intent=f"{intent.state} {intent.object_id or ''}".strip(),
            assist_source=source,
            board=self.bridge.status if self.bridge else "",
            board_detail=self.bridge.hud_detail() if self.bridge else ""))
        pygame.display.flip()

    # ------------------------------------------------------------------ input
    def handle_events(self, events) -> None:
        for ev in events:
            if ev.type == pygame.QUIT:
                self.request_quit()
                return                         # B1: ignore everything queued after a quit
            if ev.type == pygame.MOUSEWHEEL:
                self.adapter.wheel(ev.y)
            elif ev.type == pygame.KEYDOWN:
                self.handle_key(ev.key)
                if not self.running:
                    return

    def request_quit(self) -> None:
        self.running = False

    def handle_key(self, k) -> None:
        if k == pygame.K_ESCAPE:
            self.request_quit()
        elif k == pygame.K_SPACE:
            self.adapter.open_hand()
        elif k in SCENE_KEYS:
            self._disarm("scene change")       # B2: STOP first; H is needed again
            self._abandon_running("scene_changed")
            self.world.reset(SCENE_KEYS[k])
            self.ai.set_scene(self.world.scene)
            self.ai.new_trial()
            self.trial, self.banner = self._new_trial_runner(), None
        elif k in CONDITION_KEYS:
            self._disarm("condition change")
            self._abandon_running("condition_changed")
            self.condition = CONDITIONS[CONDITION_KEYS[k]]
            self.world.reset()
            self.ai.new_trial()
            self.trial, self.banner = self._new_trial_runner(), None
            print(f"Condition -> {self.condition.code} ({self.condition.label})"
                  f"{'  [cues on: press H to arm]' if self.condition.haptic and self.bridge else ''}")
        elif k in (pygame.K_n, pygame.K_RETURN, pygame.K_r):
            self._disarm("trial reset")
            self._abandon_running("reset")
            self.world.reset()
            self.ai.new_trial()
            self.trial.next_trial()
            self.banner = None
            if k == pygame.K_r:
                self.adapter.open_hand()
        elif k == pygame.K_t:
            self.adapter.tracking_lost = not self.adapter.tracking_lost
        elif k == pygame.K_l:
            self.adapter.low_quality = not self.adapter.low_quality
        elif k == pygame.K_j:
            self.adapter.jitter = not self.adapter.jitter
        elif k == pygame.K_g:
            self.ghost_on = not self.ghost_on
        elif k == pygame.K_v:
            self.camera_view_on = not self.camera_view_on
        elif k == pygame.K_TAB:
            self.view = "2d" if self.view == "3d" else "3d"
        elif k == pygame.K_h:
            if self.bridge is None:
                print("No hardware: start the app with --hardware COM4 (or mock)")
            else:
                self.bridge.arm()
        elif k == pygame.K_x:
            if self.bridge is not None:
                self.bridge.stop("X key")
        elif k in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
            if self.args.evaluation:           # B4: no manual overrides during a comparison
                print("Ignored: --evaluation keeps Member D's assistance frozen (no manual gain)")
                return
            step = GAIN_STEP if k == pygame.K_RIGHTBRACKET else -GAIN_STEP
            start = self.manual_gain if self.manual_gain is not None else self.world.assist_gain
            self.manual_gain = round(min(1.0, max(0.0, start + step)), 2)
            print(f"Assist gain -> {self.manual_gain:0.2f} (manual override of Member D's policy)"
                  f"{'' if self.condition.ai_assist else ' (no effect: condition has AI off)'}")
        elif k == pygame.K_F9:
            if self.recorder is None:
                self.recorder = TraceRecorder()
                print("Recording trace ... press F9 again to save")
            else:
                self._save_trace()

    # ------------------------------------------------------------------ helpers
    def _disarm(self, why: str) -> None:
        if self.bridge is not None and self.bridge.connected:
            self.bridge.stop(why)

    def _abandon_running(self, reason: str) -> None:
        """A trial that was started but will not finish is logged as incomplete."""
        if self.trial.state == RUNNING and self.logger is not None:
            self.logger.abandon_trial(self.trial.trial_id, self.t_ms, reason)

    def _new_trial_runner(self) -> TrialRunner:
        obj = self.world.scene if self.world.scene != "sandbox" else "glass"
        return TrialRunner(obj, self.condition, self.args.session_id, first_trial_id=self.last_id + 1)

    def _write_session_meta(self) -> None:
        """One small JSON per run next to Member D's CSV tables, so results can say exactly how
        they were produced (input, hardware mode, servo, evaluation, code version)."""
        if self.logger is None:
            return
        a, hw = self.args, self.cfg.hardware
        calibration = None
        if self.camera is not None:
            from ghostgrip.vision.camera import DEFAULT_PROFILE
            p = Path(DEFAULT_PROFILE)
            calibration = {"file": p.name, "modified_utc": datetime.fromtimestamp(
                p.stat().st_mtime, timezone.utc).isoformat(timespec="seconds") if p.exists() else None}
        hardware = "off"
        if a.hardware:
            if self.bridge is None:
                hardware = f"requested ({a.hardware}) but not opened"
            elif a.hardware.lower() == "mock":
                hardware = "mock"
            else:
                hardware = f"real ({self.bridge.port_label})" if self.bridge.connected else \
                    f"real ({self.bridge.port_label or a.hardware}) not answering"
        meta = {
            "run": self.logger.run,
            "session_id": a.session_id,
            "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "code_version": git_revision(),
            "origin": self.origin,
            "input": self.source_name,
            "replay": self.replay_path.name if self.replay_path else None,
            "hardware": hardware,
            "servo_enabled": hw.servo_enabled,
            "cue_duration_ms": hw.cue_duration_ms,
            "evaluation": a.evaluation,
            "assist_profile": self.ai.policy.profile.to_dict(),
            "assist_profile_frozen": self.ai.policy.frozen,
            "manual_assist_gain": self.manual_gain,
            "start_condition": a.condition,
            "start_scene": a.scene,
            "calibration": calibration,
            "camera_workspace": list(a.camera_workspace) if self.camera is not None else None,
            "pointer_note": "camera pointer_x/y in frames.csv are mapped through camera_workspace",
            "camera_guard": ("CameraStabilizer: bad frames hold the last good pointer/closure and are "
                             "logged with quality <= 0.3") if self.camera is not None else None,
            "note": "per-trial condition and order are in trials.csv",
        }
        path = self.log_folder / f"run_{self.logger.run}_meta.json"
        path.write_text(json.dumps(meta, indent=2))

    def _print_start(self) -> None:
        a = self.args
        print(f"GhostGrip | scene={self.world.scene} condition={self.condition.code} "
              f"({self.condition.label}) input={self.source_name}"
              f"{' ' + str(self.replay_path) if self.player else ''} view={self.view} "
              f"assist={'manual %.2f' % self.manual_gain if self.manual_gain is not None else 'Member D AssistPolicy'}"
              f"{' (EVALUATION: frozen)' if a.evaluation else ''} "
              f"hardware={self.bridge.status if self.bridge else 'off'} "
              f"log={'off' if self.logger is None else str(self.log_folder) + ' (' + self.origin + ')'}")
        if self.bridge is not None:
            print("Keys (click the GAME window first, keys typed in the terminal do nothing): "
                  "H = arm the board, X = STOP. Cues are sent only in conditions C and D (F3/F4). "
                  "Scene, condition and trial resets disarm the board.")

    def _print_summary(self) -> None:
        if self.camera is not None:
            print(f"Camera guard (frames held instead of trusted): {self.stabilizer.summary()}")
        if self.all_outcomes:
            print(f"\nSession summary ({self.source_name} input, synthetic physics):")
            for code, (s, n) in sorted(self.tally.items()):
                print(f"  Condition {code}: {s}/{n} successful")
        if self.logger is not None:
            print(f"  Session log (Member D's format): {self.log_folder}")


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        cfg = load_config()
    except ConfigError as error:
        raise SystemExit(f"config.yaml: {error}")
    return App(args, cfg).open().run()


if __name__ == "__main__":
    sys.exit(main())
