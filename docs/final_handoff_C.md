# Final handoff: Member C (Rishi), hardware

Branch `final-c-hardware`, 2026-11-08. Board: Adafruit Metro 328 + Grove Base Shield V2, TowerPro SG-5010
servo, Raspberry Pi Pico H used only as a 5V USB adapter for the servo. Full detail and the safety gate are
in `docs/hardware_inventory.md`. Everything below was observed on the real board unless marked "mock".

## Status by task
| Task | Result |
|---|---|
| C1 board in the app | Done. `app.py --hardware /dev/cu.usbserial-02860253 --condition C`: handshake ACK, board started DISARMED, H armed it. `device.csv` (session `final_board`) has 379 real sensor packets and `armed` 0 then 1. |
| C2 contact, X, STOP, latch, disconnect, quit | Partly done. Outside the app: STOP button mid-cue disarms and latches (ARM refused `LATCHED`, CUE refused until STOP); watchdog and serial STOP tested earlier. **Not yet done inside the app:** two contact trials, X, physical STOP, unplug, quit. Waiting for B's app changes (see below). |
| C3 readiness checks | Done. Supply measured at the servo plug: 5.23 V steady while moving on a wall USB port; laptop/hub USB dips to 4.16-4.56 V, so it must not be used. Neutral hold with the horn fitted and the first trial observed by Sindhu and Vishal. |
| C4 output mode | **Servo GO for the non-contact table demo**, 80-100 degree window, Pico on a wall USB port. Config change requested from D (below). Must not touch a person: the finger-safety item was never tested because the band was cut. |
| C5 tracking loss, reset/condition STOP | Not done. Needs B's reset/condition STOP changes and A's camera-loss test. The bridge already sends STOP after 300 ms of lost tracking (code review). |
| C6 LED D4 and touch D2 in the app | Partly done. Telemetry reaches `device.csv`. Touch changes and LED cues in the app not yet exercised. D4 holds the green LED module now; the red LED module goes on D5 (fault). |
| C7 buzzer | Done. 120 ms beep on accepted SLIP and DAMAGE cues only, through the same arming, cooldown and range checks. STOP, button, fault and watchdog silence it (5 mock tests). Turn off with `BUZZER_ENABLED 0` in `pins.h`. |
| C8 buzzer on the board | Partly done. 3 SLIP + 1 DAMAGE cues sent from the laptop gave 4 beeps (heard by the operator). Not yet checked inside the app or for silence in conditions A/B. |

## Requests
- **D (config.yaml):** `hardware: servo_enabled: true`, `servo_neutral_deg: 90`, `servo_min_deg: 80`, `servo_max_deg: 100`. Until then the app sends neutral-angle (LED/buzzer only) cues.
- **D (README):** the line saying the app sends no hardware keepalive is out of date; the HapticBridge does.
- **B:** reviewed `HapticBridge` (ui/integration.py) as safety owner. It handshakes with STOP, never arms itself, keeps the 250 ms keepalive, routes every event through HapticPolicy, STOPs on 300 ms tracking loss, on X and on close. Still open on B's list: STOP on scene/condition/trial reset (B2) and no cue after quit (B1). Tell C when those land so C2/C5 can be run on the board.

## Demo rules
- Pico on a wall USB port, Metro on the laptop. The GND jumper (shield GND to the breadboard minus rail) links them.
- Non-contact only: the servo stands on the table and must not touch anyone.
- Never connect the kit's 12 V adapter to the servo.

## Tests
`.venv/bin/python -m pytest tests -q`: 402 passed.
