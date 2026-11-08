"""Member 1 lesson 2: webcam + MediaPipe 21 hand landmarks.

Running modes: VIDEO for simple synchronous inference per frame.
Use as: tracker = HandTracker(); state = tracker.read(); tracker.close().
"""
from pathlib import Path
import math
import time

import cv2
import mediapipe as mp
from contracts import HandState


class HandTracker:
    def __init__(self, camera_index=0):
        model = Path(__file__).parent / 'assets' / 'hand_landmarker.task'
        if not model.exists():
            raise FileNotFoundError('First run: python download_hand_model.py')
        self.cam = cv2.VideoCapture(camera_index)
        if not self.cam.isOpened():
            raise RuntimeError('Webcam unavailable. Check permissions and close camera apps.')
        self.start = time.monotonic()
        self.last_timestamp = -1
        self.last_pinch = None
        self.prev_timestamp = None
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1,
        )
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def read(self):
        ok, frame = self.cam.read()
        if not ok:
            return HandState(self._timestamp(), False, 0.5, 0.5, 5.0)
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp_ms = self._timestamp()
        result = self.detector.detect_for_video(image, timestamp_ms)
        if not result.hand_landmarks:
            self.last_pinch = None
            self.prev_timestamp = None
            return HandState(timestamp_ms, False, 0.5, 0.5, 5.0)
        points = result.hand_landmarks[0]
        thumb, index, wrist, middle_base = points[4], points[8], points[0], points[9]
        distance = math.hypot(thumb.x-index.x, thumb.y-index.y)
        scale = max(0.035, math.hypot(wrist.x-middle_base.x, wrist.y-middle_base.y))
        pinch = distance / scale
        closing_speed = 0.0
        if self.last_pinch is not None and self.prev_timestamp is not None:
            dt = (timestamp_ms - self.prev_timestamp) / 1000.0
            if dt > 0:
                closing_speed = (self.last_pinch - pinch) / dt
        self.last_pinch = pinch
        self.prev_timestamp = timestamp_ms
        return HandState(timestamp_ms, True,
                         max(0.0, min(1.0, (thumb.x + index.x)/2)),
                         max(0.0, min(1.0, (thumb.y + index.y)/2)),
                         pinch, closing_speed)

    def _timestamp(self):
        ms = int((time.monotonic() - self.start) * 1000)
        self.last_timestamp = max(self.last_timestamp + 1, ms)
        return self.last_timestamp

    def close(self):
        self.detector.close()
        self.cam.release()
