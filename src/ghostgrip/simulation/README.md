# Member B: Simulation and UI

Owner: Member B. Folders: `src/ghostgrip/simulation/`, `src/ghostgrip/ui/`, `tests/simulation/`.
`world.py` has no pygame import and runs headless.

## Run (from the repository root, Windows)

```
.venv\Scripts\python.exe src\ghostgrip\ui\app.py                     # mouse, glass trial
.venv\Scripts\python.exe src\ghostgrip\ui\app.py --camera            # live hand (Member A)
.venv\Scripts\python.exe src\ghostgrip\ui\app.py --camera --hardware COM4 --condition C
.venv\Scripts\python.exe src\ghostgrip\ui\app.py --replay clean --hardware mock --condition D
.venv\Scripts\python.exe src\ghostgrip\ui\app.py --camera --evaluation --session-id final_eval
.venv\Scripts\python.exe src\ghostgrip\ui\app.py --view 2d           # flat view (Tab switches)
.venv\Scripts\python.exe src\ghostgrip\ui\app.py --scene sandbox     # all three objects
.venv\Scripts\python.exe src\ghostgrip\ui\app.py --replay data\local\trace_XXXX.csv
.venv\Scripts\python.exe -m pytest tests/simulation -v               # Member B tests (CI uses pytest)
```

Integrated loop (`ui/integration.py`, Phase 2): every frame the hand (camera, mouse or replay)
goes through Member D's `KalmanPredictor` (ghost marker), `AssistPolicy` (assistance gain in
conditions B/D; `--assist-gain` or `[` `]` override it) and `IntentMachine` (HUD), then the
World. In conditions C/D each event goes through Member C's `HapticPolicy` and `SerialLink`.
The board is armed only with **H**; **X** sends STOP; tracking lost for 300 ms sends STOP;
quitting sends STOP. While `config.yaml` has `servo_enabled: false`, CONTACT is sent at the
90 degree neutral, so only the LED shows the cue. Sessions are logged with Member D's
`SessionLogger` in `data/local/sessions/<session-id>/` (origin `real` for the camera,
`synthetic` for mouse and shared traces). Calibrate once first: `python vision_preview.py --calibrate`.

Phase 2 safety and lifecycle (checklist B1-B6, tests in `tests/simulation/test_app_lifecycle.py`):
- Esc or closing the window ends the loop at once: no further camera read, physics step or cue.
- Scene (1/2/3/0), condition (F1-F4) and trial resets (N, Enter, R) send STOP; press H again.
- Camera frames keep their own capture time (re-based to the app start), not the loop time.
- `--evaluation` freezes Member D's assist profile for the whole run and refuses `--assist-gain`
  and the `[` `]` keys. Each run writes `run_<n>_meta.json` next to the CSV tables (input,
  hardware mode, servo, evaluation, frozen profile, code version).
- Any exit, including a crash or a failed start, sends STOP, closes the serial port, releases the
  camera and closes the log. A finished replay disarms. `hardware.enabled: false` opens no port.

View: `--view 3d` (default) is a pygame-only perspective renderer (`ui/renderer_3d.py`) of the
same `WorldSnapshot`; `--view 2d` or **Tab** gives the flat view. Depth and smoothing are visual
only; physics, events and logs are identical in both.

Shared trace names for `--replay`: `clean`, `noisy`, `dropout_short`, `dropout_long`,
`glitch`, `fast_reach`, `slow_close`, `low_quality`, `frame_drops`, `q_jitter`.
`--replay` also plays Member A's real hand recordings (`python vision_preview.py --record`
writes `data/local/trace_*.csv` with `pointer_x, closure_q, ...` columns); `load_trace`
reads both layouts.
Finished trials are appended to `data/local/trials_<session-id>.csv` (gitignored; turn off
with `--no-log`). Trial ids keep counting across runs of the same session.

Controls: mouse moves the hand; hold left button or W to close, right button or S to open;
wheel sets a grip that stays; Space opens; 1/2/3/0 glass/sponge/block/sandbox; F1-F4 condition
A-D; `[` / `]` assist gain -/+ 0.1; N next trial; T lose tracking; L low quality; J jitter;
G ghost marker; F9 record a trace; Tab 3D/2D view; V camera preview; H arm board; X STOP; Esc quit.

## Phase 1 status (Phase 1 plan section 4)

| Step | What it needs | Where |
|---|---|---|
| B1 | world.py without pygame; 960x540 window; mouse -> HandState | `world.py`, `ui/app.py`, `inputs.MouseKeyboardAdapter` |
| B2 | glass / sponge / block synthetic profiles | `objects.py` |
| B3 | fixed 1/60 s step with capped catch-up; G formula; 4 states; hysteresis | `world.py` |
| B4 | one SimEvent per transition; safe release on tracking loss | `world.py` |
| B5 | cursor, ghost, grip meter, squash, slip, break, target; trial + TrialOutcome; conditions A-D; bounded assist gain; trace replay | `ui/`, `trial.py`, `inputs.py` |

| Plan test | Test in `tests/simulation/` |
|---|---|
| T01 open hand | `test_world.py::test_T01_open_hand_no_grasp` |
| T02 fragile valid grip | `test_world.py::test_T02_valid_fragile_grip_and_lift` |
| T03 fragile over-grip | `test_world.py::test_T03_fragile_overgrip_breaks_once` |
| T04 heavy under-grip | `test_world.py::test_T04_*` |
| T05 tracking lost | `test_world.py::test_T05_*`, `test_phase1.py` (`dropout_long`) |
| T06 low confidence | `test_world.py::test_T06_*`, `test_phase1.py::test_T06_*` |
| T07 contact jitter | `test_world.py::test_T07_*`, `test_phase1.py::test_T07_*` |
| Same outcome at any display FPS | `test_world.py`, `test_phase1.py::test_same_outcome_at_any_display_fps` |

### Shared synthetic traces (Member D's `data/synthetic/`)

`test_phase1.py` replays every file through the real World + TrialRunner (glass trial).
Only the CSV files are read, so this does not depend on Member D's code.

| Trace | Result |
|---|---|
| clean, noisy, glitch, fast_reach, slow_close, frame_drops, dropout_short, low_quality | success, exactly CONTACT -> HOLD -> RELEASE |
| dropout_long | fails safely: RELEASE (tracking_lost), trial `tracking_lost`, glass intact |
| q_jitter | no trial (hand never opens); with an open hand first: exactly one CONTACT |

During `low_quality`'s quality dip the world is unreliable, assistance shows PAUSED and no
event is sent; the existing hold is kept. Assistance only helps while tracking is reliable
(`config.yaml` `ai.low_confidence_behavior: disable_assist`). A test checks that
`config.yaml`'s `simulation` values match `world.py`.

### Not done in Phase 1 (by design)

- The ghost marker is still a constant-velocity stand-in, not Member D's Kalman `Prediction`.
- No physical cues: events are printed; Member C's `haptics.py` is connected in Phase 2.
- The trial CSV is a stand-in. Member D's `metrics.logger.SessionLogger.end_trial()` takes
  this `TrialOutcome` unchanged (same A-D codes); switching the app to it is Phase 2.
- The assist gain comes from `--assist-gain` / `[` `]`; Member D's `ai.adapt.AssistPolicy.gain()`
  feeds `world.set_assistance()` in Phase 2.
- 2D only; the 3D view comes later and reads the same `WorldSnapshot`.

## Integration API (what other members call)

Shared types come from the repository-root `contracts.py` (the one file every
member uses). Inside this package `HandState` and `SimEvent` are short aliases
for `contracts.HandStateV2` and `contracts.SimEventV2`. Both the repository root
and `src/` must be importable (`app.py` and the tests set this up themselves).

```python
from contracts import HandStateV2, SimEventV2, TrialOutcome, CONTACT, HOLD, SLIP, RELEASE, DAMAGE
from ghostgrip.simulation.world import World
from ghostgrip.simulation.trial import TrialRunner, CONDITIONS

world = World("glass")                       # "glass" | "sponge" | "block" | "sandbox"
world.set_assistance(cond.ai_assist, gain)   # gain 0..1 from Member D's adapt.py
events = world.advance(hand_state, elapsed_s)   # list[SimEventV2], one per transition
outcome = trial.update(world, events, hand_state) # TrialOutcome when a trial ends
                                             # condition = "A".."D", fail_reason "" on success
snap = world.snapshot()                      # read-only WorldSnapshot for any renderer
world.event_reason(event.event_id)           # e.g. "overgrip", "dropped", "tracking_lost"
```

| Member | Gives B | Gets from B |
|---|---|---|
| A (perception) | `HandStateV2` per frame: mirrored 0..1 pointer, calibrated `closure_q` | nothing; B replaces `MouseKeyboardAdapter` with A's source |
| C (hardware) | nothing | `SimEventV2`s only (never per-frame). C's `haptics.py` applies its own confidence gate using `HandStateV2.tracking_quality` |
| D (AI, metrics) | `Prediction` (ghost marker), `assist_gain` | `SimEventV2`, `TrialOutcome`, `event_reason()` for logs |

## Event semantics C and D must know

- `CONTACT` -> `HOLD` -> (`SLIP` -> `HOLD`)* -> `RELEASE` is a normal grasp.
- `DAMAGE` ends an engagement by itself. **No `RELEASE` follows it.**
- `DAMAGE` reasons: `overgrip` (squeezed a fragile object) or `dropped` (hard impact).
- `RELEASE` reasons: `hand_opened`, `moved_away`, `slipped_out`, `tracking_lost`.
- Opening the hand quickly is a release, not a slip; slow loosening of a lifted object slips.

## Synthetic parameters (not measured, not newtons)

| Object | contact q | holds at q | breaks at q |
|---|---|---|---|
| glass | 0.30 | 0.40 | > 0.82 for 0.2 s (was 0.60 before Phase 2), or dropped from > ~150 px |

A grab needs a deliberate closing, either of:
- the hand was open (`q < REARM_Q = 0.25`) since the last engagement;
- it closes by `CLOSE_GESTURE_Q = 0.12` within `CLOSE_WINDOW_MS = 500` while over the object.

In a logged live session the hand roamed half-closed (median q 0.61). Under the old "open below 0.15 first" rule, about 580 frames with the pinch over an object were refused. A pinch held shut while sliding onto an object still never grabs, and a 0.07 wobble (`q_jitter`) never grabs. `OPEN_Q = 0.15` is unchanged (Member D's intent machine uses it).

Camera input:
- **Workspace.** It is mapped through `--camera-workspace` (default `0.15,0.70,0.10,0.70`), so the table and the target are reached with the whole hand still in view.
- **Guard.** `ui/integration.py: CameraStabilizer` then holds the last good pointer and closure through:
  - low-quality frames;
  - frames with the fingers toward the camera (palm collapsed on screen) or the hand sideways;
  - one-frame position jumps or grip spikes.

  Held frames are reported as unreliable, so they cannot grab, cue or release. Garbage for more than 600 ms becomes tracking lost (safe release).
- **Result.** Replaying the logged session: grabs 4 → 16, and the mid-air sponge drop caused by a bad frame is gone. The remaining drops are deliberate openings, or the heavy block slipping below its 0.85 hold.
| sponge | 0.20 | ~0.39 | never (squashes) |
| block | 0.35 | 0.85 | never (slips below hold) |

## Stand-ins replaced in Phase 2 (ui/integration.py)

- Ghost marker: Member D's `KalmanPredictor` (was a constant-velocity stub).
- Assistance gain: Member D's `AssistPolicy` (`--assist-gain` / `[` `]` remain as a manual override).
- Hand input: Member A's `LiveHandSource` with `--camera` (mouse stays the default).
- Events: Member C's `HapticPolicy` + `SerialLink` with `--hardware` (conditions C and D).
- Trial log: Member D's `SessionLogger` (`trial.append_outcome_csv` is kept for its tests).
