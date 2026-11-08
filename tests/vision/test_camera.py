"""Camera/source lifecycle tests; no real camera or model required."""
import json
from unittest.mock import Mock

import numpy as np
import pytest

from contracts import HandStateV2
from src.ghostgrip.config import CameraConfig
from src.ghostgrip.vision.camera import CameraTracker, LiveHandSource

PROFILE = {'d_open': 1.0, 'd_closed': 0.1}
CONFIG = CameraConfig(index=2, mirror=True, target_fps=30)


def points(center=0.5):
    result = [(center, 0.5)] * 21
    result[5], result[17] = (center - 0.1, 0.5), (center + 0.1, 0.5)
    result[4], result[8] = (center - 0.01, 0.5), (center + 0.01, 0.5)
    return tuple(result)


def frame():
    return np.zeros((100, 200, 3), dtype=np.uint8)


def fake_tracker(frames, detections, times=None):
    capture = Mock()
    capture.isOpened.return_value = True
    capture.set.return_value = True
    frames = iter(frames)

    def read():
        value = next(frames, None)
        if isinstance(value, BaseException):
            raise value
        return value is not None, value

    capture.read.side_effect = read
    detector = Mock()
    detector.detect.side_effect = detections
    clock = iter(times if times is not None else range(1000, 10000, 33))
    tracker = CameraTracker(config=CONFIG, capture_factory=Mock(return_value=capture),
                            detector_factory=Mock(return_value=detector), clock_ms=lambda: next(clock))
    return tracker, capture, detector


def test_mirror_and_rgb_conversion_preserve_original_frame():
    original = frame()
    original[10, 0] = [10, 20, 30]
    tracker, capture, detector = fake_tracker([original], [points()])
    with tracker:
        observed = tracker.read()
    np.testing.assert_array_equal(observed.frame[10, -1], [10, 20, 30])
    np.testing.assert_array_equal(detector.detect.call_args.args[0][10, -1], [30, 20, 10])
    np.testing.assert_array_equal(original[10, 0], [10, 20, 30])
    capture.release.assert_called_once()
    detector.close.assert_called_once()


def test_config_and_camera_override_are_used():
    tracker, capture, _ = fake_tracker([frame()], [None])
    with tracker:
        tracker.capture_factory.assert_called_once_with(2)
        assert capture.set.call_args.args[1] == 30
        assert tracker.fps_request_accepted is True
    other = CameraTracker(camera_index=1, config=CONFIG)
    assert other.camera_index == 1


def test_fps_request_rejection_is_not_a_camera_failure():
    tracker, capture, _ = fake_tracker([frame()], [None])
    capture.set.return_value = False
    with tracker:
        assert tracker.fps_request_accepted is False
        assert tracker.read().error == ''


def test_unmirrored_output_cannot_silently_violate_contract():
    with pytest.raises(ValueError, match='mirror=true'):
        CameraTracker(config=CameraConfig(0, False, 30))


def test_source_type_no_double_mirror_and_recovery():
    tracker, _, _ = fake_tracker([frame()] * 3, [points(0.7), None, points(0.3)])
    with LiveHandSource(PROFILE, tracker=tracker) as source:
        first, lost, recovered = source.read(), source.read(), source.read()
        assert isinstance(first, HandStateV2)
        assert first.pointer_xy[0] == pytest.approx(0.7)
        assert first.tracking_quality > 0
        assert not lost.tracking_ok
        assert (lost.pointer_xy, lost.closure_q) == (first.pointer_xy, first.closure_q)
        assert (lost.velocity_xy, lost.dq_dt, lost.stable_score, lost.tracking_quality) == ((0, 0), 0, 0, 0)
        assert recovered.tracking_ok
        assert recovered.velocity_xy == (0, 0)
        assert recovered.pointer_xy[0] == pytest.approx(0.3)


def test_disconnect_emits_one_loss_before_eof():
    tracker, capture, detector = fake_tracker([frame(), None], [points()])
    with LiveHandSource(PROFILE, tracker=tracker) as source:
        states = list(source)
        assert [s.tracking_ok for s in states] == [True, False]
        assert states[-1].pointer_xy == states[0].pointer_xy
        assert 'frame unavailable' in source.reason
        assert source.read() is None
    capture.release.assert_called_once()
    detector.close.assert_called_once()


def test_failure_before_first_frame_has_safe_defaults():
    tracker, _, _ = fake_tracker([None], [])
    with LiveHandSource(PROFILE, tracker=tracker) as source:
        state = source.read()
        assert not state.tracking_ok
        assert state.pointer_xy == (0.5, 0.5)
        assert state.closure_q == 0
        assert source.read() is None


@pytest.mark.parametrize('where', ['capture', 'detect'])
def test_runtime_failure_reports_loss_and_releases(where):
    tracker, capture, detector = fake_tracker(
        [RuntimeError('capture error')] if where == 'capture' else [frame()],
        [RuntimeError('detect error')])
    with LiveHandSource(PROFILE, tracker=tracker) as source:
        assert not source.read().tracking_ok
        assert 'error' in source.reason
        assert source.read() is None
    capture.release.assert_called_once()
    detector.close.assert_called_once()


def test_camera_open_failure_closes_capture():
    tracker, capture, detector = fake_tracker([], [])
    capture.isOpened.return_value = False
    with pytest.raises(RuntimeError, match='Camera unavailable'):
        tracker.open()
    capture.release.assert_called_once()
    detector.close.assert_not_called()


def test_detector_init_failure_closes_capture():
    tracker, capture, _ = fake_tracker([], [])
    tracker.detector_factory.side_effect = RuntimeError('bad model')
    with pytest.raises(RuntimeError, match='bad model'):
        tracker.open()
    capture.release.assert_called_once()


@pytest.mark.parametrize('consumer_error', [False, True])
def test_early_break_and_consumer_exception_cleanup(consumer_error):
    tracker, capture, detector = fake_tracker([frame()] * 2, [points()] * 2)
    try:
        with LiveHandSource(PROFILE, tracker=tracker) as source:
            for _ in source:
                if consumer_error:
                    raise RuntimeError('consumer error')
                break
    except RuntimeError:
        assert consumer_error
    capture.release.assert_called_once()
    detector.close.assert_called_once()


def test_clock_repeats_and_backsteps_still_send_increasing_timestamps():
    tracker, _, detector = fake_tracker([frame()] * 3, [None] * 3, [1000, 1000, 999])
    with tracker:
        assert [tracker.read().t_ms for _ in range(3)] == [1000, 1001, 1002]
    assert [call.args[1] for call in detector.detect.call_args_list] == [1000, 1001, 1002]


def test_close_is_idempotent_and_instance_cannot_reopen():
    tracker, capture, detector = fake_tracker([], [])
    tracker.open()
    tracker.close()
    tracker.close()
    assert tracker.read() is None
    with pytest.raises(RuntimeError, match='closed'):
        tracker.open()
    capture.release.assert_called_once()
    detector.close.assert_called_once()


def test_pipeline_failure_also_closes_resources():
    tracker, capture, detector = fake_tracker([frame()], [points()])
    source = LiveHandSource(PROFILE, tracker=tracker)
    source.pipeline.update = Mock(side_effect=RuntimeError('processing error'))
    with pytest.raises(RuntimeError, match='processing error'):
        source.read()
    capture.release.assert_called_once()
    detector.close.assert_called_once()


def test_profile_load_and_missing_model(tmp_path):
    profile = tmp_path / 'profile.json'
    with pytest.raises(ValueError, match='Calibration unavailable'):
        LiveHandSource(profile_path=profile)
    profile.write_text(json.dumps(PROFILE))
    tracker, _, _ = fake_tracker([frame()], [points()])
    with LiveHandSource(profile_path=profile, tracker=tracker) as source:
        assert source.read().tracking_ok
    factory = Mock()
    tracker = CameraTracker(model_path=tmp_path / 'missing.task', config=CONFIG, capture_factory=factory)
    with pytest.raises(FileNotFoundError, match='model missing'):
        tracker.open()
    factory.assert_not_called()


def test_capture_released_even_if_detector_close_raises():
    tracker, capture, detector = fake_tracker([], [])
    tracker.open()
    detector.close.side_effect = RuntimeError('close failed')
    with pytest.raises(RuntimeError, match='close failed'):
        tracker.close()
    capture.release.assert_called_once()
