# 90-second demo script

Follows the technical doc's demo outline (13.1). Say only what the screen can show. Everything below runs from the
repository root; `python scripts/preflight.py` first tells you what is missing.

## Before you go on (5 minutes, not in the 90 seconds)

- [ ] `python scripts/preflight.py`: nothing says FAIL; camera, hand model and calibration say OK for **live**.
- [ ] Recalibrate for whoever will be at the camera: `python vision_preview.py --calibrate`.
- [ ] Close Zoom, Meet and Teams. Same lighting as your rehearsal. Camera view (key **V**) on, so the audience sees the hand.
- [ ] Board plugged in, servo on its stand **away from anyone**, stop button within reach. Never attach it to a person.
- [ ] This is a demo, not a measured run: do not present its numbers as results. (A measured run uses `--evaluation`.)
- [ ] The fallback is ready: `python src/ghostgrip/ui/app.py --replay clean --hardware mock` works with no camera and no board.

## The run

Start: `python src/ghostgrip/ui/app.py --camera --hardware auto --condition A --session-id demo`
(use `--hardware mock` if the board is not there; say so out loud).

| Time | Say | Do | The screen shows |
|---|---|---|---|
| 0-10 s | "A webcam can see a hand, but not whether a grasp is stable. GhostGrip tries prediction, a simple simulation and a physical cue in one loop. It is a prototype, not a medical device." | Hand in view, open. | Camera view, pointer, three objects, "A Baseline". |
| 10-25 s | "First with no help: I pick up the glass and carry it to the zone." | Key **1**, then grasp, carry, release. Squeeze too hard once if it goes well. | Grip meter, grasp, success or the glass breaking. |
| 25-42 s | "Now the AI is on. The violet ring is where the predictor thinks my hand will be in 120 ms, and it shows how sure it is." | **F2**. Repeat the glass. | Ghost marker; HUD line for intent and assist gain. |
| 42-58 s | "Now a physical cue. The first contact sends one command to the board, and the board answers." | **F4**, press **H** to arm, grasp the glass. Show the board's lights. | Board status ARMED, last cue, packets arriving. |
| 58-75 s | "Same hand, different objects: the glass breaks if squeezed, the sponge squashes, the block slips unless held hard." | Keys **2**, **3**. | Different outcomes. |
| 75-90 s | "These are the numbers from the logs, with the limits." | Show the results table. | `python -m ghostgrip.metrics.report data/local/sessions/demo` |

Hide your hand for a second during a hold if you want to show the safety behaviour: the object is released after
0.3 s, assistance goes to zero, and the board is told to stop. Press **X** at any time to stop the board.

## Say it straight (the judges may ask)

- The objects and forces are **synthetic**. Nothing measures a real grip force.
- The "physical cue" is a **light** today (the servo stays at neutral and never touches anyone); the buzzer and servo
  are not part of this demo unless the hardware notes say otherwise and you can show their checklist.
- The predictor beats a simple guess **only while the hand is moving**, and loses on a parked hand; the system turns
  assistance off in that case. State the numbers from `trace_report`, not from memory.
- The assistance is small (a few pixels of grab zone) and does not grasp for the user.
- Any result shown is labelled synthetic or real. A demo table from one session is not a study.

## If something goes wrong

| Problem | Do |
|---|---|
| Camera black or "unavailable" | Close other camera apps; or switch to the replay: `--replay clean`. |
| No board | Continue with `--hardware mock` and say so. |
| Tracking keeps dropping | Better light, plain background, hand fully in frame; or the replay. |
| Anything with the servo or board misbehaves | **X**, unplug the servo, carry on with the mock. |
| Out of time | Skip the sponge and block; end on the results slide. |

## The results slide (fill from the logs, never from memory)

1. What was built: one line and the loop diagram from the README.
2. What was measured, with counts: trials per condition, valid and invalid, success rate, median time, assistance
   actually on, prediction error against the two baselines. Label it **synthetic** or **real**.
3. What was not: no force, no tactile cue, no clinical claim, a handful of volunteers at most.
4. What is next: a real force sensor or a proper vibration actuator, a larger study, a 3D scene.
