# GhostGrip Beginner Starter Kit

A real, minimal codebase for a 4-person 36-hour hackathon project.

**Please read:** The starter gives you a mouse-controlled virtual-grasp demo,
optional MediaPipe camera input, and optional Arduino built-in LED responses.
It has a **threshold-based grasp and constant-velocity prediction BASELINE**.
It is **not yet** a trained AI classifier, adaptive therapeutic system, or
physically measured servo-haptic system.

## 0. First five minutes: team agreement

- Member 1 owns `member1_camera_test.py`, `hand_tracking.py`.
- Member 2 owns `grasp_ai.py`, `member2_prediction_test.py`.
- Member 3 owns `simulation.py`, `main.py` and project integration.
- Member 4 owns `arduino/`, `hardware.py`.
- `contracts.py` is the shared agreement. Discuss changes before editing it.
- Keep a group message with status: DONE / BLOCKED / NEXT.

## 1. Install tools

All: VS Code, Python **3.11 or 3.12**, GitHub account. 
Member 4: Arduino IDE 2.
Official setup: https://code.visualstudio.com/download ; https://www.python.org/downloads/ ;
https://docs.arduino.cc/software/ide-v2/tutorials/getting-started/ide-v2-downloading-and-installing/

Open VS Code -> File -> Open Folder -> this `GhostGrip_Beginner_Starter` folder.
Open Terminal -> New Terminal. **Always run commands from this folder**.

### Windows PowerShell

```
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe main.py
```

If you installed only 3.12, change `-3.11` to `-3.12`.
You do **not** need to activate the environment when using explicit paths.

### macOS / Linux terminal

```
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py
```

If you installed 3.12, replace `python3.11` with `python3.12`.

**Success:** A window called `GhostGrip - Beginner Demo` appears with three circles.
Put your mouse on a circle, hold left-click, move it, then release.
Press `R` to reset, `Q` to exit. This is the starting point, not the final product.

## 2. Member 1: webcam and hand tracking

1. Run the camera alone:
   `python member1_camera_test.py` (or `.venv/bin/python` / `.venv\Scripts\python.exe`).
2. You should see a camera window and your own hand. Press `Q` to close.
3. To use MediaPipe, run `python download_hand_model.py` while online.
   This uses the official Google model bundle linked in its documentation.
4. Run `python member1_hand_test.py`.
5. Move and pinch your hand in view of the camera.
6. You should see numbers changing: `x`, `y`, `pinch`, `closing_speed`.
7. Report hand tracking issues and test in good front lighting.

Understanding: `x/y` tell us *where* the hand is (0...1); `pinch` is
thumb-to-index distance relative to the hand size; `closing_speed` estimates
how quickly the measured pinch ratio decreases. No force is measured.

If camera permissions fail on macOS: System Settings -> Privacy & Security ->
Camera -> allow the terminal/VS Code application; restart that application.
Windows: Settings -> Privacy & security -> Camera.

## 3. Member 2: prediction

1. Run `python member2_prediction_test.py`.
2. Notice that `current_x` grows from 0.10 to 0.25.
3. Notice that `predicted_x` tends to be further right than `current_x`.
4. Read `grasp_ai.py`. Change `horizon_s=0.15` to `0.10`.
5. Re-run and notice the forecast moves a smaller distance.
6. Run all tests: `python -m unittest discover -s tests -v`.
7. Read `NEXT_STEPS.md` before creating a real trained grasp classifier.

Understanding: Motion prediction means estimating position a fraction of a
second into the future. The bundled one uses constant velocity. It is a
simple mathematical baseline, NOT proof of AI novelty.

## 4. Member 3: simulation

1. Run `python main.py`.
2. Put mouse cursor near glass and HOLD left mouse button.
3. Move toward the right, then release the button.
4. Watch terminal print `EVENT: GRASP glass`, then `EVENT: RELEASE glass`.
5. Press `R`; repeat with sponge and block.
6. Open `simulation.py`. Change glass color (RGB triple) or object position.
7. Run again to see your change.
8. Read `NEXT_STEPS.md` for the next physics and experiment milestones.

IMPORTANT: In this first lesson `successes` is a RELEASE count, not validated
task success. The objects are circles, not realistic biomechanics.

## 5. Member 4: Arduino built-in LED FIRST (no wiring)

1. Install Arduino IDE 2.
2. Plug Arduino Uno into the laptop using a DATA-capable USB cable.
3. Arduino IDE -> select the correct board `Arduino Uno` and port.
4. File -> Examples -> 01.Basics -> Blink.
5. Click Verify (checkmark); then Upload (right arrow).
6. The yellow built-in LED labelled L should flash.
7. Close the Blink example. Open:
   `arduino/ghostgrip_led_test/ghostgrip_led_test.ino`.
8. Click Upload.
9. Open Serial Monitor and set baud to 115200. Set line ending to Newline.
10. Type `GRASP`, press Send. Built-in LED should turn ON.
11. Type `RELEASE`, press Send. Built-in LED should turn OFF.
12. **Close Serial Monitor before Python connects**. Only one app can own the port.
13. Find your port name: Windows often `COM3`; macOS often
    `/dev/cu.usbmodem...`; Linux often `/dev/ttyACM0`.
14. From project terminal run `python main.py --port COM3`, replacing with your
    actual observed port.
15. Click a virtual object to grab; onboard LED ON. Release; LED OFF.

DO NOT connect the unknown servo to someone's finger, or guess its power
wiring. Verify exact model, operating voltage, current and suitable power
supply before trying external servo movements. See Arduino's servo guide:
https://docs.arduino.cc/learn/electronics/servo-motors/

## 6. Combine camera and simulation

After the model download and camera test are successful:

```
python main.py --camera
```

Move index finger and thumb toward each other. When pinch ratio is below the
baseline threshold (0.55), the program attempts a grasp at the virtual
pointer location. When pinch opens, it releases. Your hand must be visible.
The actual threshold can vary by person and camera; tune it with logged data.

If Arduino LED test has passed, run:

```
python main.py --camera --port COM3
```

Replace `COM3` with the actual port. This is the first complete proof of
concept: live camera -> grasp rule -> virtual interaction -> physical LED.

## 7. Project timeline

- Hours 0-2: each member runs an independent tiny program; agree on contracts.
- Hours 2-6: Member 1 hand landmarks; Member 2 prediction baseline; Member 3
  mouse simulation; Member 4 Arduino LED serial control.
- Hours 6-12: Member 3 combines the four into `main.py`; get a stable demo.
- Hours 12-20: develop/compare learned grasp state model + Kalman forecasting,
  personalization, validate safe servo system.
- Hours 20-27: instrument physical response, compare four modes, log data.
- Hours 27-32: UX polish, record fallback demo and testing evidence.
- Hours 32-36: re-run everything on demo laptop, rehearse and submit.

## 8. Troubleshooting

**`python` not found:** Windows use `py`, macOS use `python3`, or use the
virtual-environment Python executable from section 1.

**`No module named pygame` / `mediapipe`:** You installed packages into a
different interpreter. Use the `.venv` executable explicitly.

**Camera shows black / busy:** Close Zoom, Meet, Teams. Re-check operating
system camera permission. Try another index (`cv2.VideoCapture(1)`).

**`hand_landmarker.task` missing:** Run `python download_hand_model.py` with
internet. The model is NOT in the zip.

**Arduino not detected:** Try a known data USB cable, change USB port,
select Arduino Uno in the IDE, install drivers if needed, and check port.

**Serial permission/busy:** Close Arduino Serial Monitor before running Python.
Verify the port name. Linux may require membership of `dialout` group.

**No servo power / unexplained resets:** Stop external wiring. Check power
budget and grounds against the servo's documentation. Do not continue with
the unknown model without confirming electrical limits.

**Pygame window off-screen or too large:** Change `WIDTH, HEIGHT` in
`simulation.py` to fit the laptop display.

## 9. Testing and honesty

To run tests: `python -m unittest discover -s tests -v`.
The tests check baseline prediction and synthetic simulation logic.
They are NOT tests of camera accuracy or actual hardware.

**Status you can claim after following these instructions:** A toy grasp
simulator with camera-based hand position estimation and Arduino LED event
feedback. To claim predictive AI and tactile haptics, finish the steps in
`NEXT_STEPS.md` and collect real measurements.
