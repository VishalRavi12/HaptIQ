# Member B: Simulation and UI

Owner: Member B. Folders: `src/ghostgrip/simulation/`, `src/ghostgrip/ui/`, `tests/simulation/`.
`world.py` has no pygame import and runs headless.

## Run (from the repository root, Windows)

```
.venv\Scripts\python.exe src\ghostgrip\ui\app.py                     # mouse, glass trial
.venv\Scripts\python.exe src\ghostgrip\ui\app.py --scene sandbox     # all three objects
.venv\Scripts\python.exe src\ghostgrip\ui\app.py --condition B       # A/B/C/D
.venv\Scripts\python.exe src\ghostgrip\ui\app.py --replay data\local\trace_XXXX.csv
.venv\Scripts\python.exe -m unittest discover -s tests/simulation -v
```

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
| glass | 0.30 | 0.40 | > 0.60, or dropped from > ~150 px |
| sponge | 0.20 | ~0.39 | never (squashes) |
| block | 0.35 | 0.85 | never (slips below hold) |

## Stand-ins to replace in Phase 2

- `ui/app.py: stub_prediction()` -> Member D's `predict.py` (`Prediction`).
- `ui/app.py: apply_condition()` fixed gain 1.0 -> Member D's `adapt.assist_gain`.
- `MouseKeyboardAdapter` -> Member A's live `HandStateV2` source.
- Console prints of events -> Member C's `haptics.py` + Member D's logger.
