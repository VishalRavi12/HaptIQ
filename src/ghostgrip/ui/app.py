"""GhostGrip integrated app: hand input -> Member D's AI -> World -> 2D view -> Member C's hardware.

Run from the repository root (Windows):
    .venv\\Scripts\\python.exe src\\ghostgrip\\ui\\app.py                          # mouse
    .venv\\Scripts\\python.exe src\\ghostgrip\\ui\\app.py --camera                 # live hand (Member A)
    .venv\\Scripts\\python.exe src\\ghostgrip\\ui\\app.py --camera --hardware COM4 --condition C
    .venv\\Scripts\\python.exe src\\ghostgrip\\ui\\app.py --replay clean --hardware mock --condition D

Every frame, in this order:
    hand   = camera / mouse / replay           (HandStateV2, Member A or a stand-in)
    ai     = predictor, assist gain, intent    (Member D)
    events = world.advance(hand, dt)           (Member B, gain applied, bounded)
    cues   = haptic safety gate -> Arduino     (Member C, conditions C and D only)
    trial, session log, draw                   (Member B + Member D's logger)
"""
import argparse
from dataclasses import replace
from pathlib import Path
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
from ghostgrip.ui.integration import AIPipeline, HapticBridge, lost_hand  # noqa: E402
from ghostgrip.ui.renderer_2d import Renderer2D  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
SYNTHETIC_DIR = REPO_ROOT / "data" / "synthetic"
GAIN_STEP = 0.1
CAMERA_VIEW_W = 240
SCENE_KEYS = {pygame.K_1: "glass", pygame.K_2: "sponge", pygame.K_3: "block", pygame.K_0: "sandbox"}
CONDITION_KEYS = {pygame.K_F1: "A", pygame.K_F2: "B", pygame.K_F3: "C", pygame.K_F4: "D"}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="GhostGrip integrated simulation")
    ap.add_argument("--scene", default="glass", choices=list(SCENES))
    ap.add_argument("--condition", default="A", choices=list(CONDITIONS))
    ap.add_argument("--camera", action="store_true",
                    help="live hand from the webcam (Member A); needs data/local/calibration.json "
                         "(python vision_preview.py --calibrate)")
    ap.add_argument("--camera-index", type=int, default=None, help="override config.yaml camera.index")
    ap.add_argument("--replay", default=None,
                    help="CSV trace to replay instead of the mouse, or a shared synthetic trace "
                         f"name: {', '.join(synthetic_names()) or '(none found)'}")
    ap.add_argument("--loop", action="store_true", help="loop the replay")
    ap.add_argument("--hardware", nargs="?", const="auto", default=None,
                    help="send cues to the Arduino: a port such as COM4, 'auto', or 'mock' "
                         "for a fake board. Cues only in conditions C and D; press H to arm")
    ap.add_argument("--assist-gain", type=float, default=None,
                    help="fixed manual assistance gain 0..1 instead of Member D's AssistPolicy")
    ap.add_argument("--fps", type=int, default=60, help="display frame cap (physics stays 60 Hz)")
    ap.add_argument("--session-id", default="anonymous",
                    help="anonymous id (letters, digits, - and _); never a real name")
    ap.add_argument("--no-log", action="store_true",
                    help="do not write the session log in data/local/sessions/<session-id>/")
    ap.add_argument("--max-frames", type=int, default=0, help="quit after N frames (smoke test)")
    args = ap.parse_args(argv)
    if args.camera and args.replay:
        ap.error("choose either --camera or --replay, not both")
    return args


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


def camera_view(frame):
    """Small preview of the (already mirrored) camera image. Shown only, never saved."""
    import cv2
    h, w = frame.shape[:2]
    height = max(1, int(CAMERA_VIEW_W * h / w))
    small = cv2.cvtColor(cv2.resize(frame, (CAMERA_VIEW_W, height)), cv2.COLOR_BGR2RGB)
    return pygame.image.frombuffer(small.tobytes(), (CAMERA_VIEW_W, height), "RGB")


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        cfg = load_config()
    except ConfigError as error:
        raise SystemExit(f"config.yaml: {error}")

    replay_path = resolve_replay(args.replay) if args.replay else None
    camera = open_camera(args, cfg) if args.camera else None
    origin = data_origin(args.camera, replay_path)

    logger, log_folder, last_id = None, None, 0
    if not args.no_log:
        log_folder = REPO_ROOT / "data" / "local" / "sessions" / args.session_id
        try:
            logger = SessionLogger(log_folder, args.session_id, origin)
        except ValueError as error:
            raise SystemExit(f"session log: {error}")
        # Trial ids keep counting across app restarts, scene and condition changes.
        last_id = max((int(r["trial_id"]) for r in read_session(log_folder)["trials"]), default=0)

    bridge = HapticBridge.connect(args.hardware, cfg) if args.hardware else None

    pygame.init()
    screen = pygame.display.set_mode((WORLD_W + PANEL_W, WORLD_H))
    pygame.display.set_caption("GhostGrip")
    clock = pygame.time.Clock()

    world = World(args.scene)
    ai = AIPipeline(cfg, world.scene)
    renderer = Renderer2D()
    hud = Hud(WORLD_W, WORLD_H)
    condition = CONDITIONS[args.condition]
    adapter = MouseKeyboardAdapter()
    player = TracePlayer.from_csv(replay_path, loop=args.loop) if replay_path else None
    source_name = "camera" if camera else ("replay" if player else "mouse")
    recorder = None
    all_outcomes = []
    tally = {}
    ghost_on = True
    camera_view_on = True
    camera_ended = False
    banner = None
    last_hand = None
    # Manual override (--assist-gain or [ ] keys); None = Member D's AssistPolicy decides.
    manual_gain = None if args.assist_gain is None else min(1.0, max(0.0, args.assist_gain))

    def new_trial_runner():
        obj = world.scene if world.scene != "sandbox" else "glass"
        return TrialRunner(obj, condition, args.session_id, first_trial_id=last_id + 1)

    trial = new_trial_runner()

    def abandon_running(t_ms, reason):
        """A trial that was started but will not finish is logged as incomplete."""
        if trial.state == RUNNING and logger is not None:
            logger.abandon_trial(trial.trial_id, t_ms, reason)

    print(f"GhostGrip | scene={world.scene} condition={condition.code} ({condition.label}) "
          f"input={source_name}{' ' + str(replay_path) if player else ''} "
          f"assist={'manual %.2f' % manual_gain if manual_gain is not None else 'Member D AssistPolicy'} "
          f"hardware={bridge.status if bridge else 'off'} "
          f"log={'off' if logger is None else str(log_folder) + ' (' + origin + ')'}")
    if bridge is not None:
        print("Keys: H = arm the board, X = STOP. Cues are sent only in conditions C and D (F3/F4).")

    t0 = last = time.monotonic()
    t_ms = 0
    frames = 0
    running = True
    replay_done_reported = False
    try:
        while running:
            now = time.monotonic()
            elapsed, last = now - last, now
            t_ms = int((now - t0) * 1000)

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.MOUSEWHEEL:
                    adapter.wheel(ev.y)
                elif ev.type == pygame.KEYDOWN:
                    k = ev.key
                    if k == pygame.K_ESCAPE:
                        running = False
                    elif k == pygame.K_SPACE:
                        adapter.open_hand()
                    elif k in SCENE_KEYS:
                        abandon_running(t_ms, "scene_changed")
                        world.reset(SCENE_KEYS[k])
                        ai.set_scene(world.scene)
                        ai.new_trial()
                        trial, banner = new_trial_runner(), None
                    elif k in CONDITION_KEYS:
                        abandon_running(t_ms, "condition_changed")
                        condition = CONDITIONS[CONDITION_KEYS[k]]
                        world.reset()
                        ai.new_trial()
                        trial, banner = new_trial_runner(), None
                        print(f"Condition -> {condition.code} ({condition.label})"
                              f"{'  [cues on]' if condition.haptic else ''}")
                    elif k in (pygame.K_n, pygame.K_RETURN, pygame.K_r):
                        abandon_running(t_ms, "reset")
                        world.reset()
                        ai.new_trial()
                        trial.next_trial()
                        banner = None
                        if k == pygame.K_r:
                            adapter.open_hand()
                    elif k == pygame.K_t:
                        adapter.tracking_lost = not adapter.tracking_lost
                    elif k == pygame.K_l:
                        adapter.low_quality = not adapter.low_quality
                    elif k == pygame.K_j:
                        adapter.jitter = not adapter.jitter
                    elif k == pygame.K_g:
                        ghost_on = not ghost_on
                    elif k == pygame.K_v:
                        camera_view_on = not camera_view_on
                    elif k == pygame.K_h:
                        if bridge is None:
                            print("No hardware: start the app with --hardware COM4 (or mock)")
                        else:
                            bridge.arm()
                    elif k == pygame.K_x:
                        if bridge is not None:
                            bridge.stop("X key")
                    elif k in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
                        step = GAIN_STEP if k == pygame.K_RIGHTBRACKET else -GAIN_STEP
                        start = manual_gain if manual_gain is not None else world.assist_gain
                        manual_gain = round(min(1.0, max(0.0, start + step)), 2)
                        print(f"Assist gain -> {manual_gain:0.2f} (manual override of Member D's policy)"
                              f"{'' if condition.ai_assist else ' (no effect: condition has AI off)'}")
                    elif k == pygame.K_F9:
                        if recorder is None:
                            recorder = TraceRecorder()
                            print("Recording trace ... press F9 again to save")
                        else:
                            path = REPO_ROOT / "data" / "local" / time.strftime("trace_%Y%m%d_%H%M%S.csv")
                            if recorder.rows:
                                print("Trace saved:", recorder.save(path))
                            recorder = None

            # ---- 1. input source -> HandState (Member A's camera, or a stand-in) ----
            if camera is not None:
                state = None if camera_ended else camera.read()
                if state is None:
                    if not camera_ended:
                        camera_ended = True
                        print("Camera stopped:", camera.reason or "no more frames")
                        banner = ("CAMERA STOPPED - restart the app", (255, 110, 110))
                    hand = lost_hand(t_ms, last_hand)
                else:
                    hand = replace(state, t_ms=t_ms)   # one clock for world, AI, trial and log
            elif player is not None:
                hand = replace(player.sample(t_ms), t_ms=t_ms)
                if player.finished(t_ms) and not replay_done_reported:
                    print("Replay finished (holding the last row).")
                    replay_done_reported = True
            else:
                mx, my = pygame.mouse.get_pos()
                adapter.set_pointer_px(min(mx, WORLD_W - 1), my, WORLD_W, WORLD_H)
                buttons = pygame.mouse.get_pressed()
                keys = pygame.key.get_pressed()
                hand = adapter.sample(t_ms, closing=buttons[0] or keys[pygame.K_w],
                                      opening=buttons[2] or keys[pygame.K_s])
            last_hand = hand
            if recorder is not None:
                recorder.append(hand)

            # ---- 2. Member D: prediction, assistance gain, intent ----
            frame_ai = ai.update(hand, record_for_trial=trial.state == RUNNING)
            if not condition.ai_assist:
                applied = AssistDecision(0.0, "condition_off")
            elif manual_gain is not None:
                applied = AssistDecision(manual_gain, "manual")
            else:
                applied = frame_ai.decision
            world.set_assistance(condition.ai_assist, applied.gain)

            # ---- 3. physics (Member B) ----
            events = world.advance(hand, elapsed)
            for e in events:
                reason = world.event_reason(e.event_id)
                route = "  -> haptics" if condition.haptic else ""
                print(f"[{e.t_ms/1000:8.3f}s] #{e.event_id:<3d} {e.kind:7s} {e.object_id:6s} "
                      f"G={e.simulated_grip_effort:0.2f} ({reason}){route}")

            # ---- 4. Member C: safety gate -> Arduino ----
            if bridge is not None:
                device = bridge.update(hand)
                if device is not None and logger is not None:
                    logger.log_device(t_ms, device)
                for cue in bridge.handle_events(events, hand, condition.haptic):
                    if logger is not None and bridge.link.device is not None:
                        logger.log_device(t_ms, bridge.link.device, cue)

            # ---- 5. trial + session log ----
            if world.scene != "sandbox":
                before = trial.state
                outcome = trial.update(world, events, hand)
                if before == READY and trial.state == RUNNING:
                    last_id = max(last_id, trial.trial_id)
                    ai.trial_states = [hand]
                    if logger is not None:
                        logger.begin_trial(trial.trial_id, condition.code, hand.t_ms)
                if outcome is not None:
                    all_outcomes.append(outcome)
                    s_n = tally.setdefault(outcome.condition, [0, 0])
                    s_n[0] += int(outcome.success)
                    s_n[1] += 1
                    secs = (outcome.end_ms - outcome.start_ms) / 1000.0
                    banner = ((f"SUCCESS  {secs:0.1f}s  (N: next trial)", (110, 235, 150))
                              if outcome.success else
                              (f"FAILED: {outcome.fail_reason}  (N: next trial)", (255, 110, 110)))
                    print("TRIAL", outcome)
                    ai.end_trial(outcome)
                    if logger is not None:
                        logger.end_trial(outcome)
            if logger is not None:
                logger.log_frame(hand, frame_ai.prediction, frame_ai.intent, applied,
                                 trial_id=trial.trial_id if trial.state == RUNNING else None,
                                 condition=condition.code)

            # ---- 6. draw ----
            snap = world.snapshot()
            ghost = AIPipeline.ghost_xy(hand, frame_ai) if ghost_on else None
            ghost_px = (ghost[0] * WORLD_W, ghost[1] * WORLD_H) if ghost else None
            renderer.draw(screen, snap, ghost_px, banner)
            if camera is not None and camera_view_on and camera.last_observation is not None \
                    and camera.last_observation.frame is not None:
                view = camera_view(camera.last_observation.frame)
                screen.blit(view, (WORLD_W - CAMERA_VIEW_W - 8, 8))
            trial_state = trial.state
            if trial_state == READY:
                trial_state = "READY (open hand)"
            intent = frame_ai.intent
            hud.draw(screen, snap, HudInfo(
                condition_code=condition.code, condition_label=condition.label,
                source=source_name, object_kind=world.scene,
                trial_state=trial_state, trial_id=trial.trial_id,
                trial_elapsed_ms=trial.elapsed_ms(t_ms), tally=tally,
                fps=clock.get_fps(), recording=recorder is not None,
                ghost_on=ghost_on, haptic_routing=condition.haptic,
                intent=f"{intent.state} {intent.object_id or ''}".strip(),
                assist_source="manual" if manual_gain is not None else "AI",
                board=bridge.status if bridge else "",
                board_detail=bridge.last_cue if bridge else ""))
            pygame.display.flip()
            clock.tick(args.fps)
            frames += 1
            if args.max_frames and frames >= args.max_frames:
                running = False
    finally:
        if bridge is not None:
            bridge.close()                     # sends STOP before the port closes
        if camera is not None:
            camera.close()
        if recorder is not None and recorder.rows:
            path = REPO_ROOT / "data" / "local" / time.strftime("trace_%Y%m%d_%H%M%S.csv")
            print("Trace saved:", recorder.save(path))
        if logger is not None:
            logger.close()                     # a trial still running is logged as incomplete
        if all_outcomes:
            print(f"\nSession summary ({source_name} input, synthetic physics):")
            for code, (s, n) in sorted(tally.items()):
                print(f"  Condition {code}: {s}/{n} successful")
        if logger is not None:
            print(f"  Session log (Member D's format): {log_folder}")
        pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
