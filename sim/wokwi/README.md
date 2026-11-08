# GhostGrip firmware in the Wokwi simulator

Runs the real Member C firmware on a virtual Arduino Uno with a servo, LEDs, buttons and a knob,
with no hardware needed. Free, in the browser, no account required to run.

## Files
- `sketch.ino`: generated from `firmware/ghostgrip_uno` by `make_sketch.py`. Simulation only:
  servo and STOP button enabled, watchdog relaxed to 60 s so you can type commands by hand.
- `diagram.json`: the virtual wiring (same pins as the real board).

After changing the firmware, regenerate with `.venv/bin/python sim/wokwi/make_sketch.py`.

## Run it
1. Open https://wokwi.com/projects/new/arduino-uno
2. Click the `sketch.ino` tab, select all, and paste the contents of `sim/wokwi/sketch.ino`.
   (`pbcopy < sim/wokwi/sketch.ino` puts it on your clipboard.)
3. Click the `diagram.json` tab, select all, and paste `sim/wokwi/diagram.json`.
4. Press the green play button.

## What to try in the serial monitor (type a line, press Enter)
| Type | Expect |
|---|---|
| `CUE,1,CONTACT,85,300` | `ERR,1,NOT_ARMED`: boots disarmed |
| `ARM,2` | `ACK,...,ARMED`, green LED on |
| `CUE,3,CONTACT,85,300` | servo swings to 85, back to 90 after 300 ms |
| `CUE,4,CONTACT,170,300` | `ERR,4,ANGLE_RANGE`, servo does not move |
| `CUE,5,DAMAGE,85,300` | ACK, LED flickers, servo stays still (only CONTACT moves it) |
| Press the red STOP button, then `ARM,6` | `ERR,6,LATCHED` |
| `STOP,7` then `ARM,8` | re-armed |
| Wait 60 s without typing | watchdog: red LED, `FAULT` |

The touch and piezo buttons show up as `touch` and `piezo` in the `S,...` lines; the knob is `pot`.
