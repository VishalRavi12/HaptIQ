"""Member 2 lesson 1: transparent baseline motion predictor.

IMPORTANT: This is an engineered baseline, NOT a trained ML classifier or Kalman filter.
Later compare it with a classifier and Kalman filter using recorded data.
"""
from contracts import HandState, AIState


class GraspBaseline:
    def __init__(self, horizon_s=0.15, closed_threshold=0.55):
        self.horizon_s = horizon_s
        self.closed_threshold = closed_threshold
        self.last = None

    def update(self, hand: HandState):
        if not hand.tracked:
            self.last = None
            return AIState('open', hand.x, hand.y)

        vx = vy = 0.0
        if self.last is not None:
            dt = (hand.timestamp_ms - self.last.timestamp_ms) / 1000.0
            if dt > 0:
                vx = (hand.x - self.last.x) / dt
                vy = (hand.y - self.last.y) / dt
        prediction_x = max(0.0, min(1.0, hand.x + vx * self.horizon_s))
        prediction_y = max(0.0, min(1.0, hand.y + vy * self.horizon_s))

        # Simple threshold baseline; collect data before claiming learned grasp recognition.
        if hand.pinch < self.closed_threshold:
            grasp_state = 'holding'
        elif hand.closing_speed > 0.4:
            grasp_state = 'closing'
        elif hand.closing_speed < -0.4:
            grasp_state = 'releasing'
        else:
            grasp_state = 'open'
        self.last = hand
        return AIState(grasp_state, prediction_x, prediction_y)
