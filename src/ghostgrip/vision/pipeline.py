"""Member A adapter to the team's proposed contracts v2 (PR #1).

Input landmarks are already mirrored normalized (x, y) pairs, as produced by
running detection on the flipped camera frame. No second mirror is applied.
"""
import math

from contracts import HandStateV2
from ..ai.features import (
    MotionFeatures, calibrated_closure, normalized_pinch_gap, tracking_quality,
)


class PerceptionPipeline:
    """Convert each landmark observation into a complete HandStateV2.

    Speeds are per second, position uses the top-left origin. Loss holds the
    last pointer/closure and zeros speeds, stability and quality. Consumers
    must check tracking_ok. Before first acquisition the pointer is (0.5, 0.5)
    and closure is zero. reason is local diagnostics, not a contract field.
    """

    def __init__(self, profile):
        calibrated_closure(0, profile['d_open'], profile['d_closed'])
        self.d_open = profile['d_open']
        self.d_closed = profile['d_closed']
        self.motion = MotionFeatures()
        self.reason = 'tracking lost'

    def update(self, t_ms, points, width, height):
        """Use host monotonic integer milliseconds; None points means loss."""
        if isinstance(t_ms, bool) or not isinstance(t_ms, int) or t_ms < 0:
            raise ValueError('t_ms must be a nonnegative integer in milliseconds')
        quality = tracking_quality(points, width, height)
        valid = (points is not None and len(points) == 21
                 and all(len(p) == 2 and all(math.isfinite(v) and 0 <= v <= 1 for v in p)
                         for p in points))
        if valid:
            pixels = [(x * width, y * height) for x, y in points]
            valid = math.dist(pixels[5], pixels[17]) >= 10
        if valid:
            gap = normalized_pinch_gap(pixels[4], pixels[8], pixels[5], pixels[17])
            q = calibrated_closure(gap, self.d_open, self.d_closed)
            pointer = tuple((points[4][axis] + points[8][axis]) / 2 for axis in (0, 1))
            values = self.motion.update(t_ms, pointer, q)
        else:
            values = self.motion.update(t_ms)
        self.reason = values.pop('reason')
        return HandStateV2(**values, tracking_quality=quality if values['tracking_ok'] else 0.0)

    def stream(self, observations):
        """Yield HandStateV2 for each (t_ms, points, width, height) observation.

        The caller owns camera acquisition; observations may also be synthetic.
        """
        for t_ms, points, width, height in observations:
            yield self.update(t_ms, points, width, height)
