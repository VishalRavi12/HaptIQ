# Member A final handoff - Sindhu

Base: `ee7be60` (latest fetched main, includes PR #18). Branch: `final-a-validation`.
Scope: PDF A1-A6. Only perception tests/notes and this handoff were changed.
No runtime defect in vision-owned code was established; no smoothing or calibration
changes were made. B owns UI glue. C owns physical verification. D owns dependencies.

## Task status (do not treat pending checks as passes)

| Task | Result | Remaining dependency |
| --- | --- | --- |
| A1 integrated camera launch | Attempted from Codex; OpenCV reported macOS camera access denied; exited with camera unavailable | Run from Sindhu's camera-authorized terminal |
| A2 glass/sponge/block transfer | Pending; synthetic mapping test passes | Sindhu performs and reports all three tasks |
| A3 hide/recover hand | Synthetic source-to-mock-bridge test passes, including sustained loss and no automatic re-arm | Live camera and real board with C |
| A4 timing | Deterministic delayed-read example passes; UI timestamp issue identified below | B applies app clock fix |
| A5 EOF/error/cleanup | Synthetic app camera path logs loss after EOF and closes source; interrupt cleanup passes | B startup/exit cleanup fixes and live close/reopen check |
| A6 evidence/final recheck | Environment and limitations recorded; real results remain blank | Final merged commit and A/C manual evidence |

## Reproducible handoff to B (no UI files edited)

1. **Capture-time preservation.** At this base, `ui/app.py` computes `t_ms`
   before `camera.read()` then overwrites `state.t_ms`. Run:

   ```bash
   .venv/bin/python -m pytest tests/vision/test_final_handoff.py -k timestamp -v
   ```

   The fake clock starts at 10000 ms, capture takes 120 ms, then the next capture
   takes 33 ms. The source returns **10120, 10153 ms**, also sent to the detector.
   A loop stamp made before the first read would label that first observation
   120 ms too early. Use source time relative to one monotonic application origin;
   ensure elapsed physics time, loss/EOF timestamps and logs remain consistent.
   Source timestamp is after frame acquisition, before inference: it is not the
   sensor exposure timestamp or end-to-end latency. A 700 ms source gap and
   missing-hand recovery reset derivatives; see the second test in this file.

2. **Startup cleanup.** `open_camera()` runs before logger creation and before
   the `try/finally` protecting the app loop. Logger failure or display startup
   failure can bypass camera cleanup. Bring resource acquisition under cleanup
   protection and make cleanup independent if another resource's close fails.
   Existing source close/interrupt behavior passes synthetic tests.

3. **Quit ordering.** Esc/window close sets `running=False`, but the current
   iteration can still perform a camera read and route events before exit.
   Break immediately after the event loop when quitting. This is especially
   relevant if capture stalls. No physical claim was made about this path.

4. **Blocked acquisition.** A synchronous `capture.read()` cannot emit a loss
   state until it returns/errors. Do not claim an instantaneous host response
   to a blocked camera. C must verify the board's independent watchdog. Do not
   add artificial fresh camera timestamps while reusing stale data.

These are ownership handoffs, not fixes included in A's PR.

## Local environment and failed live attempt

- macOS 15.3 arm64; Python 3.12.16; pygame 2.6.1; MediaPipe 0.10.35.
- Camera config index 0; model and saved calibration exist. Physical camera
  identity, lighting, resolution and current user suitability are unverified.
- Saved profile is mirrored normalized; sample counts open 120, pinch 120,
  reach 40, release 105. Reach range is not remapped by the source.
- Actual command attempted (hardware off):
  `.venv/bin/python src/ghostgrip/ui/app.py --camera --condition A --session-id final_a_acceptance --max-frames 1800`
- Result: camera permission denied in this Codex process; exit 1. No real
  transfer, integrated FPS or physical safety result can be inferred.
- Local environment contains both `opencv-contrib-python 5.0.0.93` and
  `opencv-python 4.14.0.94`; requirements pins contrib `4.12.0.88`.
  Startup emitted duplicate SDL Objective-C class warnings. **D handoff:**
  reconcile the demo environment with the agreed dependency set if this affects
  launch. No dependencies or shared requirements were changed in this PR.
  Camera permission denial is a separate observed error; package drift is not
  asserted as its cause.

## Manual acceptance - use one session and record actual outcomes

From the usual authorized project terminal, close other camera apps/previews:

```bash
.venv/bin/python src/ghostgrip/ui/app.py --camera --condition A --session-id final_a_manual
```

Open the hand to start; acquire, lift, move to target, release. Use 1/2/3 for
objects, N for a new trial. Hide the hand during a hold for over 300 ms, restore
it, then press Esc. Reopen once to confirm camera release. No hardware is
opened by this command. Do not run another camera process concurrently.

| Observation | Actual result |
| --- | --- |
| Camera identity / lighting / resolution | Pending operator report |
| Glass transfer and correct mirror | Pending |
| Sponge transfer | Pending |
| Block transfer | Pending |
| Hide/recover during hold | Pending |
| Exit and reopen | Pending |
| Real board loss/disarm/no re-arm | Pending C; not tested by camera-only run |
| Final merged commit recheck | Pending |

Logs stay under `data/local/sessions/final_a_manual/`; do not commit them.
A logged success does not establish visual smoothness or physical output.
Report failures/invalid/incomplete trials too. No new integrated performance
numbers are claimed here.

## Automated evidence

`tests/vision/test_final_handoff.py` adds seven synthetic tests:
post-capture timestamps, long-gap/recovery derivative reset, left/right mapping
into World (two cases), loss through the real bridge with a fake board and no
re-arm, capture interruption cleanup, and the actual app camera branch with
injected frames through loss/EOF, log writing and normal cleanup.

The app camera-branch test uses temporary logs. Its injected frames are
**synthetic**, even though the unchanged app labels the `--camera` source real;
those temporary rows are not experiment evidence. No physical camera/model or
serial board is used by the automated tests.

Verification on this branch:
- New handoff tests: **7 passed**.
- Full suite: **404 passed, 196 subtests passed** in the local environment.
- `git diff --check`: passed.
- No physical output or final camera acceptance was verified.

## Proposed PR

Title: `Member A: final perception validation and integration handoff`

Adds synthetic source-to-consumer coverage for camera timing, mapping,
loss/recovery, EOF and cleanup, and updates perception status for the merged
application. Records B/D handoffs without editing their files. Runtime camera
code, UI glue, shared contracts and dependency files are unchanged.

Keep the PR in draft until the manual camera acceptance results are recorded
and the final combined version is rechecked. Hardware acceptance remains C's
responsibility; synthetic fake-board checks are not a replacement.
