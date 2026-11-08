"""Camera acquisition and a calibrated shared-state source, without a GUI.

Always use a context manager so an early break also releases the camera.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import time

from ..config import load_config
from .pipeline import PerceptionPipeline

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = ROOT / 'assets/hand_landmarker.task'
DEFAULT_PROFILE = ROOT / 'data/local/calibration.json'


@dataclass(frozen=True)
class Observation:
    t_ms: int
    frame: object  # mirrored BGR image, or None on terminal failure
    points: object  # 21 normalized (x, y) pairs, or None for no hand
    inference_ms: float
    error: str = ''


class _MediaPipeDetector:
    def __init__(self, model):
        import mediapipe as mp
        self.mp = mp
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=1)
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def detect(self, rgb, t_ms):
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect_for_video(image, t_ms)
        if not result.hand_landmarks:
            return None
        return tuple((p.x, p.y) for p in result.hand_landmarks[0])

    def close(self):
        self.detector.close()


class CameraTracker:
    """Acquire mirrored observations; no windows or physical hardware commands.

    Startup failures raise. Read/detection failures emit one terminal loss
    observation, close resources and then return None. A missing hand is not
    terminal. Capture/detector factories allow hardware-free lifecycle tests.
    """

    def __init__(self, camera_index=None, model_path=DEFAULT_MODEL, *, config=None,
                 capture_factory=None, detector_factory=None, clock_ms=None):
        config = config if config is not None else load_config().camera
        if not config.mirror:
            raise ValueError('The shared HandStateV2 contract requires camera.mirror=true.')
        self.camera_index = config.index if camera_index is None else camera_index
        self.target_fps = config.target_fps
        self.model_path = Path(model_path)
        self.capture_factory = capture_factory
        self.detector_factory = detector_factory
        self.clock_ms = clock_ms or (lambda: time.monotonic_ns() // 1_000_000)
        self.capture = self.detector = None
        self.closed = False
        self.last_timestamp = -1
        self.last_error = ''
        self.fps_request_accepted = None

    def open(self):
        if self.closed:
            raise RuntimeError('Source is closed; create a new instance to reopen it.')
        if self.capture is not None:
            return self
        if self.detector_factory is None and not self.model_path.is_file():
            raise FileNotFoundError('Hand model missing. Run python download_hand_model.py.')
        import cv2
        try:
            self.capture = (self.capture_factory or cv2.VideoCapture)(self.camera_index)
            if not self.capture.isOpened():
                raise RuntimeError('Camera unavailable. Check permissions, camera index and other camera apps.')
            self.fps_request_accepted = bool(self.capture.set(cv2.CAP_PROP_FPS, self.target_fps))
            self.detector = (self.detector_factory or _MediaPipeDetector)(self.model_path)
        except BaseException:
            self.close()
            raise
        return self

    def _timestamp(self):
        self.last_timestamp = max(self.last_timestamp + 1, int(self.clock_ms()))
        return self.last_timestamp

    def read(self):
        if self.closed:
            return None
        self.open()
        import cv2
        timestamp = None
        try:
            ok, frame = self.capture.read()
            timestamp = self._timestamp()
            if not ok or frame is None:
                raise RuntimeError('Camera frame unavailable; source stopped.')
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            start = time.perf_counter()
            points = self.detector.detect(rgb, timestamp)
            inference_ms = (time.perf_counter() - start) * 1000
            return Observation(timestamp, frame, points, inference_ms)
        except Exception as error:
            if timestamp is None:
                timestamp = self._timestamp()
            self.last_error = f'{type(error).__name__}: {error}'
            self.close()
            return Observation(timestamp, None, None, 0.0, self.last_error)
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.detector is not None:
                self.detector.close()
        finally:
            if self.capture is not None:
                self.capture.release()

    def __enter__(self):
        return self.open()

    def __exit__(self, *_):
        self.close()

    def __iter__(self):
        return self

    def __next__(self):
        value = self.read()
        if value is None:
            raise StopIteration
        return value


class LiveHandSource:
    """Read/iterate HandStateV2; None/StopIteration means the source ended.

    Terminal capture/detection failure emits one loss state before EOF.
    Consumers must stop interaction on loss, EOF, exceptions and their own exit.
    """

    def __init__(self, profile=None, *, profile_path=DEFAULT_PROFILE,
                 camera_index=None, model_path=DEFAULT_MODEL, config=None, tracker=None):
        if profile is None:
            try:
                profile = json.loads(Path(profile_path).read_text())
            except (OSError, ValueError) as error:
                raise ValueError('Calibration unavailable. Run python vision_preview.py --calibrate.') from error
        self.pipeline = PerceptionPipeline(profile)
        self.tracker = tracker if tracker is not None else CameraTracker(camera_index, model_path, config=config)
        self.last_observation = None
        self.width = self.height = 1

    @property
    def reason(self):
        return self.tracker.last_error or self.pipeline.reason

    def read(self):
        observation = self.tracker.read()
        if observation is None:
            return None
        self.last_observation = observation
        if observation.frame is not None:
            self.height, self.width = observation.frame.shape[:2]
        try:
            return self.pipeline.update(observation.t_ms, observation.points, self.width, self.height)
        except BaseException:
            self.close()
            raise

    def close(self):
        self.tracker.close()

    def __enter__(self):
        self.tracker.open()
        return self

    def __exit__(self, *_):
        self.close()

    def __iter__(self):
        return self

    def __next__(self):
        state = self.read()
        if state is None:
            raise StopIteration
        return state
