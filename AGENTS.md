# GhostGrip — agent instructions

Shared context for any coding agent (Claude Code, Codex, Cursor, Antigravity) working in this repo.
Full spec: `../GhostGrip_Complete_Technical_Project_Documentation.pdf`. Team guide: `README.md`, `NEXT_STEPS.md`.

## What this is
36-hour hackathon prototype. Webcam + MediaPipe hand landmarks -> grasp rules / constant-velocity
prediction -> 2D Pygame simulation (glass, sponge, block) -> Arduino Uno cue (LED now, servo later)
over serial at 115200 baud. Not a medical device; no real force is measured.

## Environment
- Python 3.11 venv at `.venv/`. Always run with `.venv/bin/python` (never system python3).
- Model asset: `assets/hand_landmarker.task` (already downloaded).
- Arduino: `arduino-cli` with core `arduino:avr`, libs Servo, Grove LCD RGB Backlight, Accelerometer_MMA7660.
  Compile: `arduino-cli compile --fqbn arduino:avr:uno arduino/<sketch>`.

## Commands
- Tests: `.venv/bin/python -m pytest tests` (all) or `-m unittest discover -s tests` + `-s tests/hardware`
- Member C hardware tool: `PYTHONPATH=src .venv/bin/python -m ghostgrip.hardware.cli [--mock] cue CONTACT`
- Firmware: `arduino-cli compile --upload -p <port> --fqbn arduino:avr:uno firmware/ghostgrip_uno`
- Wokwi simulation: see `sim/wokwi/README.md`
- Mouse demo: `.venv/bin/python main.py`; camera: `--camera`; hardware: `--port /dev/cu.usbmodemXXXX`

## Ownership (from README)
- Member 1: `hand_tracking.py`, `member1_*` — Member 2: `grasp_ai.py`, `member2_*`
- Member 3: `simulation.py`, `main.py` — Member 4: `arduino/`, `hardware.py`
- Implementation plan Member C (Rishi): `firmware/`, `src/ghostgrip/hardware/`, `tests/hardware/`,
  `sim/wokwi/`, `docs/hardware_*`
- `contracts.py` is the shared interface; do not change it without being asked.

## Rules
- Never let raw camera data drive the servo; actuator commands go through confidence gate, clamp,
  rate limit, arming state and watchdog. Firmware boots DISARMED.
- Never invent performance numbers, accuracy or force values.
- Keep changes small and match the existing beginner-friendly code style.
