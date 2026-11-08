"""Member B playable demo: mouse/keyboard (or a replayed trace) -> World -> 2D view.

Run from the repository root:
    .venv\\Scripts\\python.exe src\\ghostgrip\\ui\\app.py
    .venv\\Scripts\\python.exe src\\ghostgrip\\ui\\app.py --scene sandbox --condition B
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
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # .../src

import pygame  # noqa: E402

from ghostgrip.contracts import HandState  # noqa: E402
from ghostgrip.simulation.inputs import MouseKeyboardAdapter, TracePlayer, TraceRecorder  # noqa: E402
from ghostgrip.simulation.trial import CONDITIONS, READY, TrialRunner  # noqa: E402
from ghostgrip.simulation.world import SCENES, WORLD_H, WORLD_W, World  # noqa: E402
from ghostgrip.ui.hud import PANEL_W, Hud, HudInfo  # noqa: E402
from ghostgrip.ui.renderer_2d import Renderer2D  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
GHOST_HORIZON_S = 0.12
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
    ap.add_argument("--replay", default=None, help="CSV trace to replay instead of the mouse")
    ap.add_argument("--loop", action="store_true", help="loop the replay")
    ap.add_argument("--fps", type=int, default=60, help="display frame cap (physics stays 60 Hz)")
    ap.add_argument("--session-id", default="anonymous")
    ap.add_argument("--max-frames", type=int, default=0, help="quit after N frames (smoke test)")
    return ap.parse_args(argv)


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
    player = TracePlayer.from_csv(args.replay, loop=args.loop) if args.replay else None
    recorder = None
    all_outcomes = []
    tally = {}
    ghost_on = True
    banner = None

    def apply_condition():
        # Stand-in gain 1.0 until Member D's adapt.assist_gain is wired in.
        world.set_assistance(condition.ai_assist, 1.0 if condition.ai_assist else 0.0)

    def new_trial_runner():
        obj = world.scene if world.scene != "sandbox" else "glass"
        return TrialRunner(obj, condition, args.session_id)

    apply_condition()
    trial = new_trial_runner()
    print(f"GhostGrip sim | scene={world.scene} condition={condition.code} ({condition.label}) "
          f"input={'replay' if player else 'mouse'}")

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
                    s_n = tally.setdefault(outcome.condition, [0, 0])
                    s_n[0] += int(outcome.success)
                    s_n[1] += 1
                    secs = (outcome.end_ms - outcome.start_ms) / 1000.0
                    banner = ((f"SUCCESS  {secs:0.1f}s  (N: next trial)", (110, 235, 150))
                              if outcome.success else
                              (f"FAILED: {outcome.fail_reason}  (N: next trial)", (255, 110, 110)))
                    print("TRIAL", outcome)

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
        pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
