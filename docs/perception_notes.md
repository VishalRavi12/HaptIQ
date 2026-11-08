# Member A: perception progress and measurements

## Measured preview performance

Source: local report `data/local/performance_20261107_221323_987525.json`,
recorded at 2026-11-08 03:13:23 UTC (November 7, 10:13:23 PM EST).
These are measurements from one run, not estimates or averages across sessions.

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
- 31 perception tests and the 4 original baseline tests passed after trace
  integration. They cover feature
  bounds, calibration validation, jitter attenuation, sustained motion, variable
  timestamps, tracking loss, and jump rejection.

Screen coordinates use the top-left origin: x increases rightward and y
increases downward. Thus vx is negative for leftward motion and vy is negative
for upward motion. Closure q stays within [0, 1]; dq/dt can have either sign.

## Reproduce

From the project root with the virtual environment active:

```bash
python -m unittest discover -s tests/vision -v
python vision_preview.py --measure
```

Position the entire hand in view, click the preview window, then press M.
Keep the hand visible for the 20-second measurement. Results print in the
terminal and save to a timestamped JSON file under gitignored `data/local/`.
Early termination marks a shorter run as incomplete. No video is recorded.

## Still required before the complete Member A handoff

- Confirm the proposed contract with the team and sync any final revisions.
- Add an explicit left/right test of the camera mirror transform. The adapter
  test already checks that mirrored input is not mirrored a second time.
- Test the final pipeline and coordinate integration with Members B and D.
- Submit the Member A pull request when the module and handoff are ready.

## Provisional shared-contract integration

PR #1 was still open when checked for this integration. Local `contracts.py`
matches the exact proposal at commit
`f3583aca65c7789969f3610ca6383377a2c8ba2c` and preserves all v1 types.
This does not constitute team approval or a GitHub merge.

`src/ghostgrip/vision/pipeline.py` exposes `PerceptionPipeline.update`, which
returns the shared `HandStateV2` class, and `stream`, which yields those objects
from incoming observations. The preview now uses this adapter. The internal
motion helper still uses dictionaries; callers of the adapter receive objects.

Example from the repository root:

```python
import json
from pathlib import Path
from src.ghostgrip.vision.pipeline import PerceptionPipeline

profile = json.loads(Path('data/local/calibration.json').read_text())
pipeline = PerceptionPipeline(profile)
# observations yields (monotonic_ms, landmarks_xy_or_None, width, height).
# Each landmark list contains 21 normalized pairs detected on a mirrored frame.
for state in pipeline.stream(observations):
    if state.tracking_ok:
        consume(state)  # Replace with the simulator or predictor call.
    else:
        pause_interaction()  # Replace with the consumer's loss handler.
```

The caller owns camera acquisition; the generator is also usable with synthetic
observations. It does not mirror twice or remap the saved pointer range.
`tracking_quality` is included on every state. On loss or rejection it is zero;
speeds and stability are also zero while pointer and closure retain their last
values. Before first detection the pointer is (0.5, 0.5) and closure is zero.
Consumers must honor `tracking_ok`, even when retained closure is nonzero.
Local diagnostic text is available as `pipeline.reason`, not a shared field.

Timestamps are host monotonic integer milliseconds. Coordinates are normalized
and derivatives are per second. Out-of-order observations are marked invalid
and reset motion history; camera callers must supply increasing timestamps.

## Local trace recording

Run `python vision_preview.py --record` after calibration. Recording starts when
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
