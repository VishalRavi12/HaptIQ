"""Member A source-to-consumer checks. Synthetic frames; no physical evidence."""
from pathlib import Path
from unittest.mock import Mock

import pytest

from contracts import SimEventV2
from src.ghostgrip.vision.camera import LiveHandSource
from tests.vision.test_camera import PROFILE, fake_tracker, frame, points


@pytest.fixture
def consumer_imports(monkeypatch):
    # Consumers use the installed-style package name; the root contract is shared.
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / 'src'))


def test_capture_timestamp_is_after_read_not_loop_start():
    tracker, capture, detector = fake_tracker([], [points(), points()])
    clock = {'ms': 10_000}
    delays = iter([120, 33])
    tracker.clock_ms = lambda: clock['ms']

    def delayed_read():
        clock['ms'] += next(delays)
        return True, frame()

    capture.read.side_effect = delayed_read
    with LiveHandSource(PROFILE, tracker=tracker) as source:
        loop_start = clock['ms']
        first = source.read()
        second = source.read()
    assert first.t_ms - loop_start == 120
    assert second.t_ms - first.t_ms == 33
    assert [c.args[1] for c in detector.detect.call_args_list] == [10_120, 10_153]


def test_long_capture_gap_and_recovery_do_not_invent_velocity():
    tracker, _, _ = fake_tracker([frame()] * 4,
                                [points(.3), points(.31), None, points(.7)],
                                [1000, 1700, 1733, 1766])
    with LiveHandSource(PROFILE, tracker=tracker) as source:
        first, delayed, lost, recovered = [source.read() for _ in range(4)]
    assert first.tracking_ok and delayed.tracking_ok and recovered.tracking_ok
    assert delayed.velocity_xy == recovered.velocity_xy == (0, 0)
    assert delayed.dq_dt == recovered.dq_dt == 0
    assert not lost.tracking_ok and lost.tracking_quality == 0
    assert recovered.pointer_xy[0] == pytest.approx(.7)


@pytest.mark.parametrize('center', [.3, .7])
def test_already_mirrored_source_maps_once_into_world(consumer_imports, center):
    from ghostgrip.simulation.world import WORLD_H, WORLD_W, World
    tracker, _, _ = fake_tracker([frame()], [points(center)])
    with LiveHandSource(PROFILE, tracker=tracker) as source:
        hand = source.read()
    world = World('glass')
    world.advance(hand, 1 / 30)
    assert world.pointer == pytest.approx((center * WORLD_W, .5 * WORLD_H))


def test_source_loss_stops_mock_and_recovery_does_not_arm(consumer_imports):
    from ghostgrip.hardware.haptics import HapticPolicy
    from ghostgrip.hardware.mock_arduino import MockArduino
    from ghostgrip.hardware.serial_link import SerialLink
    from ghostgrip.ui.integration import HapticBridge

    clock = {'ms': 1000}
    now = lambda: clock['ms']
    board = MockArduino(clock_ms=now)
    bridge = HapticBridge(SerialLink(board), HapticPolicy(clock_ms=now), clock_ms=now)
    tracker, _, _ = fake_tracker([frame()] * 4,
                                [points(), None, None, points()],
                                [1000, 1033, 1334, 1367])
    try:
        with LiveHandSource(PROFILE, tracker=tracker) as source:
            bridge.arm()
            good = source.read()
            assert good.tracking_ok and bridge.armed
            for t in (1033, 1334):
                clock['ms'] = t
                lost = source.read()
                bridge.update(lost)
                event = SimEventV2(t, t, 'glass', 'CONTACT', .5)
                assert bridge.handle_events([event], lost, True) == []
            assert not bridge.armed
            clock['ms'] = 1367
            recovered = source.read()
            bridge.update(recovered)
            assert recovered.tracking_ok and not bridge.armed
            event = SimEventV2(2000, 1367, 'glass', 'CONTACT', .5)
            assert bridge.handle_events([event], recovered, True) == []
    finally:
        bridge.close()


def test_capture_interrupt_releases_camera_and_detector():
    tracker, capture, detector = fake_tracker([], [])
    capture.read.side_effect = KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        tracker.read()
    capture.release.assert_called_once()
    detector.close.assert_called_once()
    assert tracker.closed


def test_integrated_camera_path_logs_loss_and_closes(consumer_imports, monkeypatch, tmp_path):
    """Exercise the real app camera branch with synthetic pixels, not a webcam."""
    monkeypatch.setenv('SDL_VIDEODRIVER', 'dummy')
    monkeypatch.setenv('SDL_AUDIODRIVER', 'dummy')
    from ghostgrip.ui import app
    from ghostgrip.metrics.logger import read_session

    tracker, capture, detector = fake_tracker([frame(), frame(), None], [points(), None])
    source = LiveHandSource(PROFILE, tracker=tracker)
    monkeypatch.setattr(app, 'open_camera', lambda args, cfg: source)
    monkeypatch.setattr(app, 'REPO_ROOT', tmp_path)
    assert app.main(['--camera', '--session-id', 'synthetic_camera_check',
                     '--max-frames', '6', '--fps', '60']) == 0
    rows = read_session(tmp_path / 'data/local/sessions/synthetic_camera_check')['frames']
    assert len(rows) == 6
    assert rows[0]['tracking_ok'] == '1'
    assert all(r['tracking_ok'] == '0' for r in rows[1:])
    assert all(float(r['tracking_quality']) == 0 for r in rows[1:])
    assert all(int(b['t_ms']) > int(a['t_ms']) for a, b in zip(rows, rows[1:]))
    capture.release.assert_called_once()
    detector.close.assert_called_once()
