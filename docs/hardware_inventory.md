# Hardware inventory and bring-up log (Member C)

Last updated: 2026-11-07. Only observed facts; nothing here is a measured performance number.

## Confirmed parts
| Part | Details | Notes |
|---|---|---|
| Board | Adafruit Metro 328 (ATmega328, CP2104 USB) | 5V logic; program as `arduino:avr:uno`; macOS port `/dev/cu.usbserial-*` |
| Shield | Grove Base Shield V2 | Power switch set to **5V** |
| Servo | TowerPro SG-5010 "DiGi hi-Speed" | Standard size, 4.8-6V (datasheet to confirm). Cross horn fitted, no screw (bundled screws did not fit; ask for an M3 horn screw) |
| Kit adapter | MOSO MSP-C2000IC12.0-24W, **12V 2A**, barrel plug | **Never connect to the servo.** Only usable on the Metro DC jack |
| Kit | Grove Starter Kit Plus, IoT Edition | Several Grove cables missing; borrowed |

## Module bring-up
| Module | Socket | Result |
|---|---|---|
| Built-in LED (blink) | pin 13 | Works: upload path proven |
| LED Socket Kit (red LED for now) | D4 | Works. Swap to green later; red goes to D5 |
| Touch Sensor v1.1 | D2 | Works: touch and release detected exactly |
| Piezo Vibration Sensor + MEAS film | D6 (film in J1) | Works after turning **R6 clockwise**; before that it reported vibration continuously |
| Grove Button | D3 | Works (2026-11-08): every press disarms and latches; ARM refused while held. `BUTTON_CONNECTED` is 1 |
| Grove Buzzer | D7 | Works (2026-11-08): 3 SLIP + 1 DAMAGE cues from the laptop gave 4 short beeps (heard by the operator). Beep is 120 ms, only on accepted SLIP/DAMAGE cues; STOP, button, fault and watchdog silence it (mock tests). Disable with `BUZZER_ENABLED 0` |
| Rotary, accelerometer, LCD | A0, I2C | Not tested (cut from the final plan) |
| Servo | D9 (shield header: 9th hole from the SCL end, verified with a pin-check sketch) | Works unloaded (2026-11-08). Powered from a Raspberry Pi Pico H used only as a 5V USB adapter (pin 40 VBUS / pin 38 GND to the breadboard rails), ground shared with the Metro. `SERVO_ENABLED` is 1 |

## Safety gate status (plan section 8)
- [x] Servo model identified (SG-5010, standard positional servo)
- [x] Supply voltage measured at the servo plug (2026-11-08, Crenova MS8233D, DC V):
  - Pico on the laptop/hub USB: 4.93 V at rest, dipping to 4.16-4.56 V while the servo moved
    (below the SG-5010's 4.8 V rating). **Do not power the servo from the laptop or hub.**
  - Pico on the desk power strip's USB charging port: **5.23 V, steady while the servo moved**
    (8/8 swings, 80-100 degrees). This is the supply to use for the demo.
  - The meter shows a few readings per second, so very short dips may not be visible.
- [x] 12V kit adapter NOT wired to the servo
- [x] Grounds shared with external supply (Metro GND to the breadboard - rail, verified with a pin-check sketch)
- [x] Travel tested unloaded within the 80-100 degree window (2026-11-08); window not widened
- [x] Emergency stop button tested mid-cue (2026-11-08): pressed during continuous 80/100 degree swings, board disarmed, then ARM refused LATCHED and CUE refused until STOP
- [x] Watchdog, STOP, latch and command rejection tested on the real Metro (LED path, 2026-11-07)
- [x] Neutral state tested with the horn attached (2026-11-08): armed and held at 90 degrees for 5 s, and for 3 s after 4 swings; no creep, twitch or buzz (observed by Sindhu and Vishal). Servo is on a table stand, not on a person
- [x] 10 repeated cycles without heat or noise (2026-11-08: 20/20 moves acknowledged; case cool and quiet by touch afterwards)
- [x] Second team member observes first trial (2026-11-08: Sindhu and Vishal watched the neutral hold and 4 swings at 80-100 degrees: small, controlled, no twitch or buzz)

**Servo decision (C4, 2026-11-08): GO for the non-contact table demo** inside the 80-100 degree window, with the Pico on a wall USB port. It still must not touch a person: the band was cut from the final plan, so the finger-safety item above has not been tested.

No human contact with the servo until every box is checked.

## Protocol notes for the team
Member C added two rules on top of plan section 6.3, after code review found real safety gaps:
- `KA,seq` is the keepalive. It refreshes the watchdog but can never arm the board. Send it at
  least every 250 ms while armed; `ARM` is only for an explicit arm.
- After the STOP button, a watchdog timeout or repeated bad packets the board is **latched**:
  `ARM` replies `ERR,seq,LATCHED` until the host sends `STOP` first.
- `STOP` is always obeyed, even with a stale or unreadable sequence number.
- Only `CONTACT` cues move the servo; `HOLD` (sent to the board as `STABLE_HOLD`), `SLIP` and `DAMAGE` are lights only; `RELEASE` has no cue.
