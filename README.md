# GhostGrip (HaptIQ)

A research prototype from a 36-hour hackathon: a webcam tracks one hand, a small predictor and rule-based
grasp detector run on top of it, a 2D simulation of three virtual objects reacts, and an Arduino turns
simulation events into a safe light (and, once verified, servo) cue.

**It is not a medical device.** No real force is measured, nothing here is clinically validated, and every
result so far comes from synthetic data. See [What is and is not done](#what-is-and-is-not-done).

```
webcam -> perception -> predictor + intent -> assist policy -> simulation -> haptic policy -> Arduino
          (HandStateV2)   (Prediction)         (gain 0-1)      (SimEventV2)   (CueEvent)       (DeviceState)
```

The shared data types are in [`contracts.py`](contracts.py). Changing them needs all four team members' approval.

## Quick start

Python **3.11** (3.12 also works). Run everything from the repository root.

```
conda create -n ghostgrip python=3.11 -y        # or: python3.11 -m venv .venv
conda activate ghostgrip                        # or: source .venv/bin/activate   (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

python -m pytest -q tests                       # all tests; no camera or hardware needed
python scripts/preflight.py --no-camera         # what can this laptop do right now?
```

`scripts/preflight.py` lists missing packages, the hand model, a saved calibration, a camera and an
Arduino, and tells you which modes will work. Drop `--no-camera` to also try opening the camera
(macOS asks for permission the first time).

Windows PowerShell: where a command below starts with `PYTHONPATH=src`, run `$env:PYTHONPATH = "src"` once
first and then the rest of the command.

## Things you can run

| What | Command | Needs |
|---|---|---|
| Simulation, mouse control | `python src/ghostgrip/ui/app.py` (`--scene sandbox`, `--condition B`, `--assist-gain 0.5`, `--no-log`) | nothing else |
| Simulation, replay a recording | `python src/ghostgrip/ui/app.py --replay data/synthetic/clean.csv` | nothing else |
| Settings check | `PYTHONPATH=src python -m ghostgrip.config` | nothing else |
| Synthetic hand traces | `PYTHONPATH=src python -m ghostgrip.ai.synthetic` | nothing else |
| Predictor vs baselines | `PYTHONPATH=src python -m ghostgrip.ai.predict` | nothing else |
| Grasp intent timeline | `PYTHONPATH=src python -m ghostgrip.ai.intent all` | nothing else |
| Whole pipeline on synthetic data, with results table | `PYTHONPATH=src python -m ghostgrip.metrics.report --synthetic-demo data/local/demo` | nothing else |
| Arduino with a fake board | `PYTHONPATH=src python -m ghostgrip.hardware.cli --mock cue CONTACT` | nothing else |
| Hand model download (once) | `python download_hand_model.py` | internet |
| Calibrate, then live hand preview | `python vision_preview.py --calibrate`, then `python vision_preview.py` | camera, hand model |
| Live hand states without a window (add `--record` to save numbers, never images) | `PYTHONPATH=src python -m ghostgrip.vision.cli --frames 150` | camera, hand model |
| Real Arduino | `PYTHONPATH=src python -m ghostgrip.hardware.cli cue CONTACT` | the board, USB data cable |
| The whole loop, live camera, fake board | `python src/ghostgrip/ui/app.py --camera --hardware mock --condition B --session-id me01` | camera, hand model, calibration |
| Record your hand (numbers, never images) | `PYTHONPATH=src python -m ghostgrip.vision.cli --frames 600 --record` | camera, hand model, calibration |
| How do the predictor and assistance do on a recording? | `PYTHONPATH=src python -m ghostgrip.metrics.trace_report data/local/trace_XXXX.csv --sweep` | a recording |
| Results of a logged session | `PYTHONPATH=src python -m ghostgrip.metrics.report data/local/sessions/me01` | a session folder |

Results from `ghostgrip.metrics.report` are labelled **synthetic** and are a pipeline check, not an experiment.
Add `--csv summary.csv` or `--plot success.png` to export them.

## What is and is not done

| Area | State |
|---|---|
| Perception: landmarks to calibrated `HandStateV2`, a reusable camera source, trace recording | Built and tested. The live numbers (about 29.6 FPS, 4.5 ms inference) were measured by its author before the camera source was refactored; re-checking the refactored preview with a real camera is still listed as open in `docs/perception_notes.md`. |
| Simulation: glass, sponge and block, trials, replay, 2D UI | Built and tested (`tests/simulation`, including every shared synthetic trace replayed at three frame rates). Each finished trial is appended to `data/local/trials_<session>.csv`. The window has been looked at by its author only. |
| Hardware: firmware, serial link, safety policy, mock board | Built and tested with the mock. Its author reports (2026-11-08) that the stop button and the servo were tried on a real board, **unloaded and inside an 80-100 degree window**: 17 of 17 button presses disarmed it, and stop worked mid-cue. Three items are still open in `docs/hardware_inventory.md`: the servo's supply voltage has **not been measured** (5 V is assumed), its neutral position has not been checked on its mount, and a second person has not watched a first trial. **No servo should touch a person until all three are done.** |
| AI core: predictor, grasp intent, bounded adaptation | Built and tested on **synthetic traces only**. The predictor beats the baselines while the hand moves and loses for a parked, jittery hand. |
| Settings, CI, logging and results, preflight | Built and tested. |
| Everything running together (camera, AI, simulation, logger, hardware link) | **Wired and run headless** on the synthetic traces with the mock board (`ui/app.py --replay clean --hardware mock`). **Not done:** a live camera with real people, the real board in the loop, and the four-condition comparison. Open: the assistance profile is not yet frozen during a measured run. |
| Real-user data and the four-condition comparison | **Not done.** Nothing here supports a claim about real people. |

The four conditions are A baseline, B AI only, C haptic only, D AI and haptic.

## Try it on your own laptop (real hand, no one else needed)

Nothing here sends anything anywhere; recordings hold numbers only and stay in `data/local/`.

```
python download_hand_model.py                        # once; needs internet
python scripts/preflight.py                          # camera, model, calibration: what is still missing?
python vision_preview.py --calibrate                 # 20 s: open hand, pinch, reach, release
PYTHONPATH=src python -m ghostgrip.vision.cli --frames 600 --record      # about 20 s of your hand
PYTHONPATH=src python -m ghostgrip.metrics.trace_report data/local/trace_XXXX.csv --sweep
python src/ghostgrip/ui/app.py --camera --hardware mock --condition B --session-id me01
PYTHONPATH=src python -m ghostgrip.metrics.report data/local/sessions/me01
```

`trace_report` says in plain words whether the predictor beats the two baselines on your recording, how often
assistance would have been on, and how its settings behave. One recording of one person is a check, not
evidence; keep it apart from the synthetic results.

## Repository layout

```
contracts.py                  shared data types (change only with all four approvals)
config.yaml                   every setting, one place; checked by src/ghostgrip/config.py
src/ghostgrip/
  vision/  ai/features.py  ai/calibration.py      perception: camera source, landmarks to HandStateV2
  ai/synthetic.py  predict.py  intent.py  adapt.py   fake traces, Kalman predictor, grasp states, assistance
  simulation/  ui/                                 virtual objects, trials, replay, Pygame window
  hardware/                                        serial link, safety policy, mock Arduino, CLI
  metrics/                                         session logging, results, plots, recorded-trace report, synthetic run
  preflight.py                                     the laptop check
firmware/ghostgrip_uno/       Arduino firmware (boots disarmed; servo limited to 80-100 degrees; its checklist is still open)
sim/wokwi/                    browser simulation of the board
data/synthetic/               fake hand traces (safe to share)       data/local/   real recordings and logs, never committed
docs/                         perception notes, hardware inventory and safety checklist
tests/                        one folder per area            scripts/   preflight launcher
main.py simulation.py grasp_ai.py hand_tracking.py ...      the original beginner starter, kept for reference
```

## Working together

- One branch per piece of work, then a pull request into `main`. CI runs `pytest` on every pull request.
- Ask someone else to review your pull request; merge one at a time, after CI passes.
- Only edit the area you own. Changes to `contracts.py` or the serial protocol go through one shared pull request.
- Stage specific files (`git add path/to/file`), never `git add .`. Never commit `.venv`, the hand model,
  recordings, video, calibration profiles or secrets (`data/local/` and `assets/*.task` are ignored).
- Re-run the tests after pulling `main`.

## Data and privacy

Camera video is never stored. Logs hold numbers only (pointer, closure, tracking, predictions, events) in
`data/local/`, under an anonymous id (letters, digits, `-`, `_`). Every row the session logger writes says whether it
is `real` or `synthetic`, and the two are never added together in a results table.

## Safety

The servo is **off in `config.yaml`**, so the app currently sends only neutral-angle cues. The firmware has the
servo enabled but limited to an 80-100 degree window, and its stop button always wins. Do not connect the kit's
12 V adapter to the servo, and do not attach the servo to a person, until the whole checklist in
`docs/hardware_inventory.md` is complete (supply voltage measured, neutral position checked on its mount, a
second person watching the first trial). The board boots disarmed, and a 500 ms watchdog disarms it if the
laptop goes quiet. A light cue is a valid final output.

## Troubleshooting

- **Wrong Python / no `mediapipe`:** use Python 3.11 and `pip install -r requirements.txt` in the active environment.
- **OpenCV and pygame warnings on macOS (`implemented in both ...`):** `pip uninstall -y opencv-python opencv-python-headless`, then
  `pip install -r requirements.txt` (it pins the build that does not clash).
- **Camera black or busy:** close Zoom, Meet and Teams, and allow the terminal under System Settings, Privacy and Security, Camera.
- **`hand_landmarker.task` missing:** `python download_hand_model.py` (the model is not in the repository).
- **Serial port busy:** close the Arduino Serial Monitor; only one program can own the port.
- **Anything else:** run `python scripts/preflight.py` first; it names what is missing.
