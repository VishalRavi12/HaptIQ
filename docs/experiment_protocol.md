# Volunteer study protocol

A small feasibility study (technical doc 10.4). Write down anything that differs from this page and report it
with the results. **It cannot show that GhostGrip helps anyone**: a handful of people and a simulation with
synthetic objects cannot support a claim about real tasks, real force or rehabilitation. It can show whether
the system runs end to end with a person in the loop, and give honest descriptive numbers.

## What is compared

| Code | Condition | AI assistance | Physical cue |
|---|---|---|---|
| A | Baseline | off | off |
| B | AI only | on | off |
| C | Haptic only | off | on |
| D | AI + haptic | on | on |

- **AI assistance** enlarges an object's grab zone by at most 15% of its radius, scaled by how sure the predictor
  is (about 7% in normal use, a few pixels). It never grasps for the person. It is off whenever tracking is lost,
  quality is low, the forecast is stale, or the predictor is doing worse than a simple guess.
- **Physical cue** means what the board does on a simulation event. With the committed `config.yaml`
  (`servo_enabled: false`) the servo stays at neutral and nothing touches the person, so the cue is **lights**
  (the buzzer beeps on slip and damage only in the hardware branch that is not merged yet; check which firmware
  is on the board and write it down). So C and D test a light cue, not a tactile one. Do not describe them as
  tactile feedback.
- The assist profile is **frozen** (`ai.freeze_adaptation: true`): it starts the same for everyone and never
  changes during the session, so the four conditions stay comparable. Personalisation is built and tested but
  deliberately not part of this study.

## Who and how many

3 to 5 consenting adult volunteers, each a separate session with an **anonymous id** (`P01`, `P02`, ...). Do not
record names, faces, ages or anything else that identifies anyone. No one is paid or pressured; anyone may stop at any time.

## Consent (read aloud, then ask for a spoken yes; note the date and id only)

> This is a student prototype, not a medical device. You will move your hand in front of a webcam to pick up
> virtual objects on a screen, about 20 minutes in all. The computer saves numbers (where your fingers were, how
> closed your hand was, whether the hand was tracked), never pictures or video. The numbers are labelled with an
> anonymous id, not your name. A small board may flash a light when you touch a virtual object; nothing
> will touch your skin. You can stop at any time and ask for your numbers to be deleted. Do you agree?

## Safety

- The servo stays on the table or its stand, away from the participant, for the whole study. No human contact
  with it, ever, in this protocol. The kit's 12 V adapter is never connected to it.
- If anything looks wrong (the board resets, heat, noise, a participant is uncomfortable): press **X** in the app
  (sends STOP) or the board's stop button, and stop the session.
- Close other programs that use the camera. Same room, lighting and camera position for everyone.

## Before the first participant (once)

```
python scripts/preflight.py                  # camera, hand model, config: nothing should say FAIL
python vision_preview.py --calibrate         # calibrate for the person being tested (profile stays local)
PYTHONPATH=src python -m ghostgrip.metrics.protocol --participants 4     # print the run sheets
```

Recalibrate for each participant: hand sizes differ. Check `ai.freeze_adaptation` is `true` in `config.yaml`.

## Procedure per participant

The run sheet printed by `ghostgrip.metrics.protocol` has every command and key. In short:

1. Consent, anonymous id, calibration.
2. **Practice** (3 trials, condition A, session `P0X_practice`). Not analysed.
3. **Measured session**: four blocks, one per condition, in this participant's order (balanced Latin square, so
   every condition appears once in each position and follows every other once). In each block: object 1 (glass),
   2 (sponge), 3 (block), **2 trials each**, 6 per condition, 24 per person. Keys: F1-F4 condition, 1/2/3 object,
   N next trial, Esc quit.
4. Say the same short instructions every time: "Open your hand to start, pick up the object, carry it to the
   marked zone on the right, and let go. Take your time; you have 30 seconds."
5. Do not coach during a trial, and do not tell participants which condition they are in.
6. Afterwards run the audit and write down anything unusual.

## What counts, decided in advance

- A **trial** is one attempt from the moment the hand opens and is tracked. Every attempt is logged, including
  ones that go wrong.
- **Valid** = it finished with tracking intact. **Success** = the object came to rest, undamaged, in the target
  zone after at least 0.3 s of stable grasp, within 30 s.
- **Failure** (valid, counted): timeout, or the glass broke (over-grip or a hard drop).
- **Invalid** (reported, not counted as a failure): tracking was lost long enough to release the object
  (`tracking_lost`), or the trial was interrupted (`incomplete`).
- If the camera or board fails, note it, restart that block's session, and report both. **No trial is deleted,
  hidden or re-labelled afterwards, and slow or failed ones stay in.**
- Real and synthetic data are never added together.

## Measures (all from the logs; nothing is invented)

Success rate of valid trials; median and range of time for successful trials; share of valid trials with at
least one slip, and with at least one over-grip (both describe the simulation, not a real object); prediction
error against the two baselines; how often assistance was really on; counts of invalid and incomplete trials.

```
PYTHONPATH=src python -m ghostgrip.metrics.protocol --audit data/local/sessions/P01 --participant 1
PYTHONPATH=src python -m ghostgrip.metrics.report data/local/sessions/P01 data/local/sessions/P02 data/local/sessions/P03
```

The report prints one table per participant and then one descriptive table over all of them. The audit checks
that what was logged matches the plan (order, number of attempts) and reports any difference.

## Reporting

Show per-participant results and the descriptive total, with the counts. **No significance tests and no claim of
benefit**: with this few people and these synthetic objects they would mean nothing. State the limits: synthetic
objects, a light cue instead of touch, frozen assistance, very small sample, one room, one camera, no blinding
(the experimenter knows the condition), and a practice effect that the order balancing only partly removes.
Report failures and invalid trials as prominently as successes.

## Data

Logs are in `data/local/sessions/` and are never committed. No video is stored. Delete a participant's folder on
request. Keep the consent notes (date and id only) apart from the data.
