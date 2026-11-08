"""Exercise CLI and preview consumers without a camera or GUI."""
import json
import sys

import pytest

from src.ghostgrip.vision import cli
from src.ghostgrip.vision.camera import LiveHandSource
from src.ghostgrip.vision.trace import read_trace
from tests.vision.test_camera import PROFILE, fake_tracker, frame, points


@pytest.mark.parametrize('disconnect', [False, True])
def test_cli_records_states_and_closes(tmp_path, monkeypatch, disconnect):
    tracker, capture, detector = fake_tracker(
        [frame(), None] if disconnect else [frame()] * 3, [points()] * 3)
    source = LiveHandSource(PROFILE, tracker=tracker)
    monkeypatch.setattr(cli, 'LiveHandSource', lambda **kwargs: source)
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    assert cli.main(['--frames', '2', '--record']) == int(disconnect)
    states = list(read_trace(next((tmp_path / 'data/local').glob('trace_*.csv'))))
    assert len(states) == 2
    assert states[-1].tracking_ok is not disconnect
    capture.release.assert_called_once()
    detector.close.assert_called_once()


def test_preview_records_tracking_and_terminal_loss(tmp_path, monkeypatch):
    import vision_preview as preview
    (tmp_path / 'assets').mkdir()
    (tmp_path / 'assets/hand_landmarker.task').touch()
    (tmp_path / 'data/local').mkdir(parents=True)
    (tmp_path / 'data/local/calibration.json').write_text(json.dumps(PROFILE))
    monkeypatch.setattr(preview, '__file__', str(tmp_path / 'vision_preview.py'))
    monkeypatch.setattr(sys, 'argv', ['vision_preview.py', '--record'])
    tracker, capture, detector = fake_tracker([frame(), frame(), None], [points(), None])
    monkeypatch.setattr(preview, 'CameraTracker', lambda *args, **kwargs: tracker)
    monkeypatch.setattr(preview.cv2, 'imshow', lambda *args: None)
    monkeypatch.setattr(preview.cv2, 'waitKey', lambda *args: -1)
    monkeypatch.setattr(preview.cv2, 'getWindowProperty', lambda *args: 1)
    monkeypatch.setattr(preview.cv2, 'destroyAllWindows', lambda: None)
    preview.main()
    states = list(read_trace(next((tmp_path / 'data/local').glob('trace_*.csv'))))
    assert [s.tracking_ok for s in states] == [True, False, False]
    capture.release.assert_called_once()
    detector.close.assert_called_once()
