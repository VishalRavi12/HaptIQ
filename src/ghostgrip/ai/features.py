"""Camera-independent perception features for Member A.

Points must use the same coordinate system. For image landmarks, use pixel
coordinates (or aspect-corrected coordinates) so distances are consistent.
"""
import math
from collections import deque
from statistics import median


def normalized_pinch_gap(thumb_tip, index_tip, index_mcp, pinky_mcp, eps=1e-6):
    """Thumb/index distance divided by palm width; guard a tiny palm scale."""
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError('eps must be finite and positive')
    points = (thumb_tip, index_tip, index_mcp, pinky_mcp)
    if any(len(point) != 2 or not all(math.isfinite(v) for v in point)
           for point in points):
        raise ValueError('Expected four finite 2D points')
    gap = math.dist(thumb_tip, index_tip)
    palm_width = math.dist(index_mcp, pinky_mcp)
    value = gap / max(palm_width, eps)
    if not math.isfinite(value):
        raise ValueError('Pinch gap is outside the supported numeric range')
    return value


def calibrated_closure(pinch_gap, d_open, d_closed, min_span=1e-3):
    """Map a normalized gap to closure: open=0, closed=1, clipped to [0, 1].

    Calibration must supply an open gap larger than the closed gap.
    This function does not measure calibration samples or tracking quality.
    """
    if not all(math.isfinite(v) for v in (pinch_gap, d_open, d_closed, min_span)):
        raise ValueError('Closure inputs must be finite')
    if min_span <= 0 or min(pinch_gap, d_open, d_closed) < 0:
        raise ValueError('Gaps must be nonnegative and min_span positive')
    if d_open - d_closed <= min_span:
        raise ValueError('Open and closed calibration gaps are too close or reversed')
    return max(0.0, min(1.0, (d_open - pinch_gap) / (d_open - d_closed)))


def tracking_quality(points, width, height):
    """Geometric usability heuristic in [0, 1], NOT model confidence.

    Expects 21 normalized (x, y) landmarks. Penalizes small palm width and
    proximity to image edges. It cannot establish landmark accuracy, detect
    every occlusion, or replace model validation. Thresholds are provisional.
    """
    if not all(math.isfinite(v) and v > 0 for v in (width, height)):
        raise ValueError('Image dimensions must be finite and positive')
    if points is None or len(points) != 21:
        return 0.0
    if any(len(p) != 2 or not all(math.isfinite(v) and 0 <= v <= 1 for v in p)
           for p in points):
        return 0.0
    palm_pixels = math.hypot((points[5][0] - points[17][0]) * width,
                             (points[5][1] - points[17][1]) * height)
    if palm_pixels < 10:
        return 0.0
    palm_fraction = palm_pixels / min(width, height)
    size_score = max(0.0, min(1.0, (palm_fraction - 0.02) / 0.06))
    margin = min(min(x, 1 - x, y, 1 - y) for x, y in points)
    edge_score = max(0.0, min(1.0, margin / 0.05))
    return min(size_score, edge_score)


class MotionFeatures:
    """Temporal features, independent of the team's shared HandState class.

    Coordinates and q are normalized; derivatives are per second. Smoothing
    uses a 180 ms time constant; speeds use an additional 220 ms filter.
    Gaps over 500 ms reset derivatives. Stability
    is a heuristic based on rolling positional MAD, not model confidence.
    """

    def __init__(self):
        self.pointer = (0.5, 0.5)
        self.q = 0.0
        self.previous_t = None
        self.raw_pointer = None
        self.history = deque(maxlen=15)
        self.velocity = (0.0, 0.0)
        self.closure_speed = 0.0

    def reset(self):
        """Keep the last position but discard temporal history."""
        self.previous_t = None
        self.raw_pointer = None
        self.history.clear()
        self.velocity = (0.0, 0.0)
        self.closure_speed = 0.0

    def _state(self, t_ms, valid=False, velocity=(0.0, 0.0), dq_dt=0.0,
               stability=0.0, reason='tracking lost'):
        return dict(t_ms=t_ms, pointer_xy=self.pointer, closure_q=self.q,
                    velocity_xy=velocity, dq_dt=dq_dt, stable_score=stability,
                    tracking_ok=valid, reason=reason)

    def update(self, t_ms, pointer=None, q=None):
        if not math.isfinite(t_ms):
            raise ValueError('Timestamp must be finite')
        if (pointer is None or q is None or len(pointer) != 2
                or not all(math.isfinite(v) and 0 <= v <= 1 for v in (*pointer, q))):
            self.reset()
            return self._state(t_ms)
        dt = None if self.previous_t is None else (t_ms - self.previous_t) / 1000
        if dt is not None and dt <= 0:
            self.reset()
            return self._state(t_ms, reason='timestamp reset')
        if dt is None or dt > 0.5:
            self.reset()
            self.pointer, self.q = tuple(pointer), q
            self.raw_pointer, self.previous_t = tuple(pointer), t_ms
            self.history.append(tuple(pointer))
            return self._state(t_ms, valid=True, reason='warming up')
        # Heuristic jump gate: a small allowance plus 2 frame-widths/second.
        # Keep the last accepted sample so a single spike cannot shift the filter.
        if math.dist(pointer, self.raw_pointer) > 0.08 + 2.0 * dt:
            self.velocity = (0.0, 0.0)
            self.closure_speed = 0.0
            return self._state(t_ms, reason='jump rejected')
        alpha = -math.expm1(-dt / 0.18)
        smoothed = tuple(old + alpha * (new - old)
                         for old, new in zip(self.pointer, pointer))
        smoothed_q = self.q + alpha * (q - self.q)
        velocity = tuple((new - old) / dt for new, old in zip(smoothed, self.pointer))
        dq_dt = (smoothed_q - self.q) / dt
        speed_alpha = -math.expm1(-dt / 0.22)
        self.velocity = tuple(old + speed_alpha * (new - old)
                              for old, new in zip(self.velocity, velocity))
        self.closure_speed += speed_alpha * (dq_dt - self.closure_speed)
        self.pointer, self.q = smoothed, smoothed_q
        self.raw_pointer, self.previous_t = tuple(pointer), t_ms
        self.history.append(tuple(pointer))
        stability = 0.0
        if len(self.history) >= 3:
            center = tuple(median(p[axis] for p in self.history) for axis in (0, 1))
            mad = median(math.dist(p, center) for p in self.history)
            stability = max(0.0, 1.0 - mad / 0.03)
        return self._state(t_ms, True, self.velocity, self.closure_speed, stability, 'tracking')
