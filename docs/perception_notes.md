# Member A: perception progress and measurements

## Historical Phase 0 preview performance

Source: local report `data/local/performance_20261107_221323_987525.json`,
recorded at 2026-11-08 03:13:23 UTC (November 7, 10:13:23 PM EST).
These are measurements from one Phase 0 run, not estimates or averages across sessions.
They have not been remeasured after the Phase 1 camera refactor.

| Item | Result |
| --- | --- |
| Python | 3.12.16 |
| OS | macOS 15.3, arm64 |
| Camera index | 0 |
| Frame dimensions | 1920 x 1080 |
| Duration | 20.026 seconds; completed |
| Processed frames | 593 |
| Preview throughput | 29.61 FPS |
| Mean inference time | 4.50 ms |
| 95th-percentile inference time | 5.26 ms |
| Accepted tracking frames | 593 / 593 (100%) |
| Saved calibration loaded | Yes |

The measured mean and p95 inference times are below the plan's 45 ms target.
Inference timing covers the synchronous MediaPipe detection call only. It
excludes capture, image conversion, rendering and smoothing delay. FPS includes
the preview loop. These numbers do not measure end-to-end interaction latency.

Tracking availability counts accepted motion states when calibrated (otherwise
valid geometric samples). It is not landmark accuracy or model confidence.
One successful run does not establish performance in other lighting, with
other users, or during occlusion. Lighting and camera model were not recorded.

## Implemented and checked

- Webcam and 21-landmark overlay, with a mirrored thumb/index midpoint cursor.
- Tracking loss and recovery, checked manually by the user.
- Pinch gap normalized by index-MCP to pinky-MCP palm width using pixel distances.
- Open, pinch, reach and release calibration, saved under `data/local/`.
- User verified closure near 0 when open and near 1 when pinched.
- Cursor and closure smoothing with a 180 ms time constant; derivative smoothing
  with a further 220 ms time constant. These settings introduce response delay.
- Velocity and closure speed in normalized units per second; jump rejection,
  timestamp checks, and resets after tracking loss or long frame gaps.
- Rolling positional MAD stability heuristic, separate from tracking quality.
- Numeric overlay refreshes at 4 Hz. Very small displayed speeds round to zero;
  feature data retains its actual values.
- Geometric tracking-quality heuristic based on palm size and image-edge margin;
  zero for missing/invalid landmarks or rejected motion states. It is not model
  confidence and does not detect all occlusions. The user confirmed the live
  indicator responds as expected.
- Automated perception checks cover feature bounds, calibration validation,
  jitter attenuation, sustained motion, variable timestamps, loss/recovery,
  jump rejection, camera mirroring, resource cleanup and recorded source output.

Screen coordinates use the top-left origin: x increases rightward and y
increases downward. Thus vx is negative for leftward motion and vy is negative
for upward motion. Closure q stays within [0, 1]; dq/dt can have either sign.

## Reproduce

From the project root with the virtual environment active:

```bash
.venv/bin/python -m pytest tests/vision -q
.venv/bin/python vision_preview.py --measure
```

Position the entire hand in view, click the preview window, then press M.
Keep the hand visible for the 20-second measurement. Results print in the
terminal and save to a timestamped JSON file under gitignored `data/local/`.
Early termination marks a shorter run as incomplete. No video is recorded.

## Phase 1 reusable camera source

The merged root `contracts.py` is the shared interface; no contract changes are
required for this source. `LiveHandSource` emits that exact `HandStateV2` type.
Run commands from the repository root:

```bash
# Preview; Q closes the camera before another process opens it.
.venv/bin/python vision_preview.py
# Calibrate first if the saved profile is unavailable.
.venv/bin/python vision_preview.py --calibrate
# Read 150 states without a window, then release the camera.
PYTHONPATH=src .venv/bin/python -m ghostgrip.vision.cli --frames 150
# Optional numeric CSV recording, no images.
PYTHONPATH=src .venv/bin/python -m ghostgrip.vision.cli --frames 150 --record
```

Both commands read `camera.index`, `camera.mirror` and `camera.target_fps`
from `config.yaml`. `--config PATH` selects another configuration and `--camera N`
overrides its camera index. FPS is a best-effort device request, not a throughput
promise. `mirror: false` is rejected because the shared V2 coordinates must be
mirrored. No second mirror should be applied by the consumer.

For B/D integration, with `PYTHONPATH=src`:

```python
from ghostgrip.vision.camera import LiveHandSource

with LiveHandSource() as source:
    for state in source:
        if state.tracking_ok:
            consume(state)  # Replace with simulator/predictor call.
        else:
            pause_interaction()  # Replace with consumer's loss handler.
```

The default profile is `data/local/calibration.json`; pass `profile_path=` or a
profile dictionary as the first argument to use another calibration. Keep the
context manager: it releases camera/model resources on early break or errors.
Startup failures raise. A capture/detection failure emits one final lost state,
closes resources, then ends iteration (`read()` returns `None`). Missing hands
produce lost states but do not end acquisition. Consumers must stop interaction
on loss, EOF, exceptions and their own exit. This source sends no actuator commands.
The CLI returns a nonzero exit code for source failures.

On loss or rejection, tracking quality, speeds and stability are zero; pointer
and closure retain their last values. Before first detection the pointer is
(0.5, 0.5) and closure is zero. Consumers must honor `tracking_ok` even when
retained closure is nonzero. Local diagnostic text is `source.reason`.

Timestamps are strictly increasing host monotonic integer milliseconds.
Coordinates are normalized and derivatives are per second. The saved reach
`pointer_range` is not remapped; consumers use the existing full-frame coordinates.
`PerceptionPipeline.update` and `stream` remain available for supplied landmark
observations, including synthetic inputs, without owning a camera.

The preview uses the same `CameraTracker` acquisition code. Its new timing scope
includes image wrapping plus detection, excluding capture, color conversion,
rendering and smoothing delay. FPS includes the preview loop.

### Phase 2 integration status

The reusable source is now connected to the actual predictor, simulator and
logger by merged PR #17; PR #18 improves board status. The simulator also reads
Member A CSV columns. Those integration implementation tasks are no longer open.

The final Member A audit is based on `ee7be60`, on branch `final-a-validation`.
See [the final handoff](final_handoff_A.md) for numbered A1-A6 status, synthetic
tests, reproducible timing/cleanup requests to B and manual acceptance fields.

An integrated camera launch was attempted in this Codex process; macOS denied
camera access, so no real three-object transfer or integrated performance is
claimed. The saved calibration is present. Final camera checks must run in the
operator's authorized terminal; physical disarm needs C's board observation.
Existing preview performance and recordings below remain historical evidence.

## Local trace recording

Run `.venv/bin/python vision_preview.py --record` after calibration. Recording starts when
the preview opens; press Q to stop. It stores a unique CSV under `data/local/`,
including loss states and full-precision shared values before display rounding.
No video or images are stored. The folder is gitignored; do not commit real hand
recordings. Existing recordings are never overwritten, and each row is flushed.

Columns: `t_ms, pointer_x, pointer_y, closure_q, dq_dt, velocity_x, velocity_y,
stable_score, tracking_ok, tracking_quality`. Boolean tracking is encoded as 0/1.
`read_trace(path)` from `src.ghostgrip.vision.trace` yields `HandStateV2` objects
for offline consumers. It preserves timestamps but does not pace playback.

### Verified live recording

Inspected `data/local/trace_20261107_230653_728994.csv` using `read_trace`:

- 1,414 states over 48.743 seconds.
- 1,279 states with `tracking_ok=True`; 135 with `tracking_ok=False`.
- All timestamps strictly increase, and all numeric feature values are finite.
- Recorded closure ranges from 0.0 to approximately 0.9959.
- The recording loads successfully as shared `HandStateV2` objects.

This completes the live recording and CSV inspection check. The user previously
confirmed the updated preview works following adapter integration. The trace
does not independently verify visual steadiness, lighting, landmark accuracy,
or integration with the simulator and predictor. Keep this real recording local.
