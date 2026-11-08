"""Standalone Member A landmark preview. Press Q to quit."""
import argparse
from pathlib import Path
import time

import cv2
import math
import json
import platform
from datetime import datetime, timezone

from src.ghostgrip.ai.features import normalized_pinch_gap, calibrated_closure, tracking_quality
from src.ghostgrip.ai.calibration import PHASES, build_profile, save_profile
from src.ghostgrip.vision.pipeline import PerceptionPipeline
from src.ghostgrip.vision.camera import CameraTracker
from src.ghostgrip.config import load_config
from src.ghostgrip.vision.trace import TraceRecorder


CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, help='Override configured camera index')
    parser.add_argument('--config', help='Settings YAML; defaults to repository config.yaml')
    parser.add_argument('--calibrate', action='store_true', help='Run a 20-second calibration')
    parser.add_argument('--measure', action='store_true', help='Press M for a 20-second performance run')
    parser.add_argument('--record', action='store_true', help='Save hand states locally as CSV (no video)')
    args = parser.parse_args()
    model = Path(__file__).parent / 'assets' / 'hand_landmarker.task'
    if not model.exists():
        raise SystemExit('Run python download_hand_model.py first.')
    profile = None
    profile_path = Path(__file__).parent / 'data/local/calibration.json'
    if profile_path.exists():
        try:
            profile = json.loads(profile_path.read_text())
            calibrated_closure(0, profile['d_open'], profile['d_closed'])
            print(f'Loaded calibration from {profile_path}')
        except (ValueError, KeyError, TypeError, OSError) as error:
            profile = None
            print(f'Cannot load calibration: {error}. Run with --calibrate.')
    if args.record and (profile is None or args.calibrate):
        raise SystemExit('Record after calibration: run --calibrate first, then --record separately.')
    pipeline = PerceptionPipeline(profile) if profile else None
    camera = CameraTracker(args.camera, model, config=load_config(args.config).camera)
    recorder = None
    measurement_start = None
    measurement_frames = 0
    measurement_valid = 0
    inference_times = []
    measurement_end = None
    try:
        camera.open()
        if args.record:
            trace_path = Path(__file__).parent / 'data/local' / (
                'trace_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.csv')
            recorder = TraceRecorder(trace_path)
            print(f'Recording hand states (no video) to {trace_path}', flush=True)
        displayed_state = None
        display_updated_at = -1000
        calibration_start = None
        samples = {phase: [] for phase in PHASES}
        note = 'SPACE: start 20-second calibration' if args.calibrate else 'Q: quit'
        if args.measure:
            note = 'Show hand, then M: start 20-second measurement; Q: quit'
        prompts = ('OPEN: spread thumb and index, hold steady',
                   'PINCH: touch thumb and index, hold steady',
                   'REACH: move left/right AND up/down',
                   'RELEASE: return to your open hand pose')
        while True:
            observation = camera.read()
            if observation is None:
                break
            timestamp = observation.t_ms
            frame = observation.frame
            if observation.error:
                if pipeline is not None:
                    loss = pipeline.update(timestamp, None, 1, 1)
                    if recorder is not None:
                        recorder.write(loss)
                print(observation.error)
                break
            inference_ms = observation.inference_ms
            status = 'TRACKING LOST - show one hand'
            sample = None
            landmark_xy = None
            quality = 0.0
            state = None
            if observation.points:
                height, width = frame.shape[:2]
                points = observation.points
                landmark_xy = points
                pixels = [(int(p[0] * (width - 1)), int(p[1] * (height - 1))) for p in points]
                for start, end in CONNECTIONS:
                    cv2.line(frame, pixels[start], pixels[end], (70, 220, 70), 2)
                for point in pixels:
                    cv2.circle(frame, point, 4, (0, 220, 255), -1)
                x = max(0.0, min(1.0, (points[4][0] + points[8][0]) / 2))
                y = max(0.0, min(1.0, (points[4][1] + points[8][1]) / 2))
                cv2.circle(frame, (int(x * (width - 1)), int(y * (height - 1))),
                           5, (255, 255, 255), 1)
                status = f'TRACKING | pointer x={x:.2f} y={y:.2f}'
                coords = [(p[0] * width, p[1] * height) for p in points]
                quality = tracking_quality([(p[0], p[1]) for p in points], width, height)
                reliable = (all(math.isfinite(v) and 0 <= v <= 1
                                for p in points for v in (p[0], p[1]))
                            and math.dist(coords[5], coords[17]) >= 10)
                if reliable:
                    gap = normalized_pinch_gap(coords[4], coords[8], coords[5], coords[17])
                    sample = (gap, x, y)
                    if profile:
                        q = calibrated_closure(gap, profile['d_open'], profile['d_closed'])
                        status += f' | q={q:.2f}'
                elif args.calibrate:
                    status = 'Show entire hand clearly, closer to camera'
            if calibration_start is not None:
                elapsed = time.monotonic() - calibration_start
                phase_index = int(elapsed // 5)
                if phase_index < len(PHASES):
                    phase = PHASES[phase_index]
                    note = f'{prompts[phase_index]} ({5 - int(elapsed % 5)}s)'
                    # Allow one second to change pose before collecting.
                    if elapsed % 5 >= 1 and sample is not None:
                        samples[phase].append(sample)
                else:
                    calibration_start = None
                    try:
                        candidate = build_profile(samples)
                        path = Path(__file__).parent / 'data/local/calibration.json'
                        save_profile(candidate, path)
                        profile = candidate
                        pipeline = PerceptionPipeline(profile)
                        displayed_state = None
                        note = 'Saved! Open: q~0; pinch: q~1. SPACE: retry; Q: quit'
                        print(f'Calibration saved to {path}')
                    except (ValueError, OSError) as error:
                        print(f'Calibration failed: {error}')
                        note = 'Calibration failed - see terminal. SPACE: retry'
            if profile:
                h, w = frame.shape[:2]
                state = pipeline.update(timestamp, landmark_xy, w, h)
                if recorder is not None:
                    recorder.write(state)
                # Refresh numbers at 4 Hz; cursor and loss indicators stay live.
                if (displayed_state is None or timestamp - display_updated_at >= 250
                        or state.tracking_ok != displayed_state.tracking_ok):
                    displayed_state = state
                    display_updated_at = timestamp
                if state.tracking_ok:
                    sx, sy = state.pointer_xy
                    h, w = frame.shape[:2]
                    cv2.circle(frame, (int(sx * (w - 1)), int(sy * (h - 1))),
                               12, (255, 80, 220), 3)
                    status = f"TRACKING | q={displayed_state.closure_q:.2f} | {pipeline.reason}"
                else:
                    status = pipeline.reason.upper() + ' - waiting for reliable hand'
                vx, vy = displayed_state.velocity_xy
                # Display-only rounding around zero; preserve actual feature data.
                vx, vy = (0.0 if abs(v) < 0.02 else v for v in (vx, vy))
                dq = displayed_state.dq_dt
                dq = 0.0 if abs(dq) < 0.05 else dq
                detail = (f"vx={vx:+.2f} vy={vy:+.2f} dq/dt={dq:+.2f}/s "
                          f"steady={displayed_state.stable_score:.1f}")
                quality = displayed_state.tracking_quality
            else:
                detail = 'Run with --calibrate to enable motion features'
            if measurement_start is not None:
                measurement_frames += 1
                measurement_valid += int(state.tracking_ok if state is not None else sample is not None)
                inference_times.append(inference_ms)
                measurement_end = time.perf_counter()
                duration = measurement_end - measurement_start
                note = (f'Measuring: {duration:.0f}/20s | '
                        f'{measurement_frames / duration:.1f} FPS | '
                        f'{sum(inference_times) / measurement_frames:.1f} ms inference')
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 140), (25, 25, 25), -1)
            cv2.putText(frame, f'quality={quality:.1f} (geometry heuristic, not confidence)',
                        (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, detail, (15, 90), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)
            cv2.putText(frame, status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2)
            cv2.putText(frame, note, (15, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)
            if recorder is not None:
                cv2.putText(frame, 'REC: hand states only | Q: stop', (15, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow('GhostGrip - Member A landmarks', frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                break
            if args.measure and key in (ord('m'), ord('M')) and measurement_start is None:
                measurement_start = time.perf_counter()
                print('Measurement started: keep your hand visible for 20 seconds.', flush=True)
            if measurement_end is not None and measurement_end - measurement_start >= 20:
                break
            if key == ord(' ') and args.calibrate:
                samples = {phase: [] for phase in PHASES}
                calibration_start = time.monotonic()
            if cv2.getWindowProperty('GhostGrip - Member A landmarks', cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if recorder is not None:
                recorder.close()
                print(f'Trace saved to {recorder.path}')
        finally:
            try:
                camera.close()
            finally:
                cv2.destroyAllWindows()
        if measurement_frames and measurement_end is not None:
            duration = measurement_end - measurement_start
            ordered = sorted(inference_times)
            report = {
                'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
                'python': platform.python_version(), 'os': platform.platform(),
                'camera_index': camera.camera_index,
                'target_fps_requested': camera.target_fps,
                'fps_request_accepted': camera.fps_request_accepted,
                'frame_size': [int(frame.shape[1]), int(frame.shape[0])] if frame is not None else None,
                'duration_s': round(duration, 3), 'completed_20_seconds': duration >= 20,
                'frames': measurement_frames,
                'processed_fps': round(measurement_frames / duration, 2),
                'mean_inference_ms': round(sum(inference_times) / measurement_frames, 2),
                'p95_inference_ms': round(ordered[math.ceil(len(ordered) * 0.95) - 1], 2),
                'valid_tracking_frames': measurement_valid,
                'tracking_availability_percent': round(100 * measurement_valid / measurement_frames, 2),
                'tracking_definition': 'Accepted motion state when calibrated; valid geometric sample otherwise',
                'calibration_loaded': profile is not None,
                'timing_scope': 'Detector call includes image wrapping/inference; excludes capture, color conversion, rendering and smoothing delay; FPS includes preview loop',
            }
            path = Path(__file__).parent / 'data/local' / (
                'performance_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.json')
            save_profile(report, path)
            print('\nPerformance results:\n' + json.dumps(report, indent=2))
            print(f'Saved to {path}')


if __name__ == '__main__':
    main()
