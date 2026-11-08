"""Member B playable demo: mouse/keyboard (or a replayed trace) -> World -> 2D view.

Run from the repository root:
    .venv\\Scripts\\python.exe src\\ghostgrip\\ui\\app.py
    .venv\\Scripts\\python.exe src\\ghostgrip\\ui\\app.py --scene sandbox --condition B --assist-gain 0.5
    .venv\\Scripts\\python.exe src\\ghostgrip\\ui\\app.py --replay clean     # shared trace by name
    .venv\\Scripts\\python.exe src\\ghostgrip\\ui\\app.py --replay data\\local\\trace_XXXX.csv

This is a stand-alone demo for Phase 1. In Phase 2, Member D's main loop does the
same four calls per frame: hand = source.sample(); events = world.advance(hand, dt);
trial.update(world, events, hand); renderer.draw(world.snapshot()).
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

from contracts import HandStateV2 as HandState  # noqa: E402
from ghostgrip.simulation.inputs import MouseKeyboardAdapter, TracePlayer, TraceRecorder  # noqa: E402
from ghostgrip.simulation.trial import (CONDITIONS, READY, TrialRunner,  # noqa: E402
                                        append_outcome_csv, last_trial_id)
from ghostgrip.simulation.world import SCENES, WORLD_H, WORLD_W, World  # noqa: E402
from ghostgrip.ui.hud import PANEL_W, Hud, HudInfo  # noqa: E402
from ghostgrip.ui.renderer_2d import Renderer2D  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
SYNTHETIC_DIR = REPO_ROOT / "data" / "synthetic"
GHOST_HORIZON_S = 0.12
GAIN_STEP = 0.1
SCENE_KEYS = {pygame.K_1: "glass", pygame.K_2: "sponge", pygame.K_3: "block", pygame.K_0: "sandbox"}
CONDITION_KEYS = {pygame.K_F1: "A", pygame.K_F2: "B", pygame.K_F3: "C", pygame.K_F4: "D"}


def stub_prediction(hand: HandState):
    """STAND-IN constant-velocity ghost until Member D's predict.py is integrated.
    Not the project's AI; it only exercises the ghost-marker drawing."""
    if not hand.tracking_ok:
        return None
    x = hand.pointer_xy[0] + hand.velocity_xy[0] * GHOST_HORIZON_S
    y = hand.pointer_xy[1] + hand.velocity_xy[1] * GHOST_HORIZON_S
    return (min(1.0, max(0.0, x)) * WORLD_W, min(1.0, max(0.0, y)) * WORLD_H)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="GhostGrip Member B simulation demo")
    ap.add_argument("--scene", default="glass", choices=list(SCENES))
    ap.add_argument("--condition", default="A", choices=list(CONDITIONS))
    ap.add_argument("--replay", default=None,
                    help="CSV trace to replay instead of the mouse, or a shared synthetic trace "
                         f"name: {', '.join(synthetic_names()) or '(none found)'}")
    ap.add_argument("--loop", action="store_true", help="loop the replay")
    ap.add_argument("--assist-gain", type=float, default=1.0,
                    help="assistance gain 0..1 for conditions B and D (stand-in for Member D's "
                         "adapt.py; clamped)")
    ap.add_argument("--fps", type=int, default=60, help="display frame cap (physics stays 60 Hz)")
    ap.add_argument("--session-id", default="anonymous")
    ap.add_argument("--no-log", action="store_true",
                    help="do not append finished trials to data/local/trials_<session>.csv")
    ap.add_argument("--max-frames", type=int, default=0, help="quit after N frames (smoke test)")
    return ap.parse_args(argv)


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


def main(argv=None) -> int:
    args = parse_args(argv)
    pygame.init()
    screen = pygame.display.set_mode((WORLD_W + PANEL_W, WORLD_H))
    pygame.display.set_caption("GhostGrip - Member B Simulation")
    clock = pygame.time.Clock()

    world = World(args.scene)
    renderer = Renderer2D()
    hud = Hud(WORLD_W, WORLD_H)
    condition = CONDITIONS[args.condition]
    adapter = MouseKeyboardAdapter()
    replay_path = resolve_replay(args.replay) if args.replay else None
    player = TracePlayer.from_csv(replay_path, loop=args.loop) if replay_path else None
    recorder = None
    all_outcomes = []
    tally = {}
    ghost_on = True
    banner = None
    # Bounded assistance gain (plan B5.03). Stand-in for Member D's adapt.assist_gain.
    gain = min(1.0, max(0.0, args.assist_gain))
    log_path = None if args.no_log else (
        REPO_ROOT / "data" / "local" / f"trials_{args.session_id}.csv")
    # Trial ids keep counting across scene switches and app restarts.
    last_id = last_trial_id(log_path, args.session_id) if log_path else 0

    def apply_condition():
        world.set_assistance(condition.ai_assist, gain if condition.ai_assist else 0.0)

    def new_trial_runner():
        obj = world.scene if world.scene != "sandbox" else "glass"
        return TrialRunner(obj, condition, args.session_id, first_trial_id=last_id + 1)

    apply_condition()
    trial = new_trial_runner()
    print(f"GhostGrip sim | scene={world.scene} condition={condition.code} ({condition.label}) "
          f"assist_gain={gain:0.2f} input={'replay ' + str(replay_path) if player else 'mouse'}")

    t0 = last = time.monotonic()
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
                        world.reset(SCENE_KEYS[k])
                        trial, banner = new_trial_runner(), None
                    elif k in CONDITION_KEYS:
                        condition = CONDITIONS[CONDITION_KEYS[k]]
                        world.reset()
                        apply_condition()
                        trial, banner = new_trial_runner(), None
                        print(f"Condition -> {condition.code} ({condition.label})")
                    elif k in (pygame.K_n, pygame.K_RETURN, pygame.K_r):
                        world.reset()
                        apply_condition()
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
                    elif k in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
                        step = GAIN_STEP if k == pygame.K_RIGHTBRACKET else -GAIN_STEP
                        gain = round(min(1.0, max(0.0, gain + step)), 2)
                        apply_condition()
                        print(f"Assist gain -> {gain:0.2f}"
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

            # ---- input source -> HandState ----
            if player is not None:
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
            if recorder is not None:
                recorder.append(hand)

            # ---- physics ----
            events = world.advance(hand, elapsed)
            for e in events:
                reason = world.event_reason(e.event_id)
                route = "  -> haptics (Member C)" if condition.haptic else ""
                print(f"[{e.t_ms/1000:8.3f}s] #{e.event_id:<3d} {e.kind:7s} {e.object_id:6s} "
                      f"G={e.simulated_grip_effort:0.2f} ({reason}){route}")

            # ---- trial ----
            if world.scene != "sandbox":
                outcome = trial.update(world, events, hand)
                if outcome is not None:
                    all_outcomes.append(outcome)
                    last_id = max(last_id, outcome.trial_id)
                    s_n = tally.setdefault(outcome.condition, [0, 0])
                    s_n[0] += int(outcome.success)
                    s_n[1] += 1
                    secs = (outcome.end_ms - outcome.start_ms) / 1000.0
                    banner = ((f"SUCCESS  {secs:0.1f}s  (N: next trial)", (110, 235, 150))
                              if outcome.success else
                              (f"FAILED: {outcome.fail_reason}  (N: next trial)", (255, 110, 110)))
                    print("TRIAL", outcome)
                    if log_path is not None:
                        append_outcome_csv(log_path, outcome)

            # ---- draw ----
            snap = world.snapshot()
            ghost = stub_prediction(hand) if ghost_on else None
            renderer.draw(screen, snap, ghost, banner)
            trial_state = trial.state
            if trial_state == READY:
                trial_state = "READY (open hand)"
            hud.draw(screen, snap, HudInfo(
                condition_code=condition.code, condition_label=condition.label,
                source="replay" if player else "mouse", object_kind=world.scene,
                trial_state=trial_state, trial_id=trial.trial_id,
                trial_elapsed_ms=trial.elapsed_ms(t_ms), tally=tally,
                fps=clock.get_fps(), recording=recorder is not None,
                ghost_on=ghost_on, haptic_routing=condition.haptic))
            pygame.display.flip()
            clock.tick(args.fps)
            frames += 1
            if args.max_frames and frames >= args.max_frames:
                running = False
    finally:
        if recorder is not None and recorder.rows:
            path = REPO_ROOT / "data" / "local" / time.strftime("trace_%Y%m%d_%H%M%S.csv")
            print("Trace saved:", recorder.save(path))
        if all_outcomes:
            print("\nSession summary (mouse/replay trials, synthetic physics):")
            for code, (s, n) in sorted(tally.items()):
                print(f"  Condition {code}: {s}/{n} successful")
            if log_path is not None:
                print(f"  Trial log (stand-in for Member D's logger): {log_path}")
        pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
