"""Motion prediction: where will the pointer be a fraction of a second from now?

A constant-velocity Kalman filter (technical doc 4.4). State per axis is [position, velocity].
x and y are independent, so the 4-state filter [px, py, vx, vy] splits into two small filters;
that is mathematically the same and needs no numpy.

    predictor = KalmanPredictor(horizon_ms=120)
    prediction = predictor.update(hand_state)      # a contracts.Prediction, every frame

    tracker = ErrorTracker()                       # scores predictions against what happened
    tracker.record(hand_state, prediction)
    tracker.summary()                              # mean error: kalman vs the two baselines

Rules:
  * Tracking lost, or a non-finite pointer: the filter resets and the Prediction carries the
    maximum uncertainty (LOST_SIGMA), so assistance gain falls to zero (plan 4.5). Consumers
    that draw a ghost marker should still check hand.tracking_ok.
  * A repeated or out-of-order timestamp is ignored (never a negative time step).
  * A gap longer than max_gap_ms restarts the filter from the new sample.
  * Predicted positions are clamped to the 0-1 frame.

This is a mathematical baseline, not a trained model. The ErrorTracker exists so the team can
check it honestly: if it is not better than the baselines, say so and turn assistance down.
"""
from collections import deque
from dataclasses import dataclass
import math
from typing import Deque, Dict, Optional, Sequence, Tuple

from contracts import HandStateV2, Prediction

LOST_SIGMA = 1.0          # uncertainty (normalised units, ~ the whole frame) when nothing is known
DEFAULT_HORIZON_MS = 120

# Chosen from the synthetic scenarios only (see `python -m ghostgrip.ai.predict`); NOT validated
# on real hands. Sweep of process noise q at r = 4e-6 (forecast error / claimed sigma / share of
# real errors inside 2 sigma): q=0.01: 0.0112 / 0.0065 / 52%, q=0.1: 0.0077 / 0.0127 / 99%,
# q=1: 0.0062 / 0.031 / 100%, q=30: 0.0058 / 0.148 / 100%. Larger q is slightly more accurate
# but claims absurd uncertainty; q=1 is close to the best accuracy with a cautious sigma. Cautious
# is deliberate: synthetic paths are smoother than real hands, so real accelerations will be larger.
# For a parked, jittery hand no setting beat "stay where you are" (q_jitter); ErrorTracker catches that.
# Process noise q: how abruptly the hand may accelerate, in (normalised units/s^2)^2 * s.
# Measurement noise r: variance of the observed pointer.
DEFAULT_PROCESS_NOISE = 1.0
DEFAULT_MEASUREMENT_NOISE = 4e-6
DEFAULT_INITIAL_VELOCITY_SIGMA = 1.5   # normalised units per second, before any motion is seen
DEFAULT_MAX_GAP_MS = 500


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


class _Axis:
    """One axis of the filter: position p, velocity v, and their 2x2 covariance."""

    def __init__(self, position: float, position_var: float, velocity_var: float):
        self.p = position
        self.v = 0.0
        self.ppp = position_var     # var(p)
        self.ppv = 0.0              # cov(p, v)
        self.pvv = velocity_var     # var(v)

    @staticmethod
    def _propagate(p, v, ppp, ppv, pvv, dt, q):
        """Move the state and its covariance forward by dt seconds (F P F^T + Q)."""
        np_ = p + v * dt
        nppp = ppp + 2.0 * dt * ppv + dt * dt * pvv + q * dt ** 3 / 3.0
        nppv = ppv + dt * pvv + q * dt ** 2 / 2.0
        npvv = pvv + q * dt
        return np_, v, nppp, nppv, npvv

    def step(self, dt: float, q: float) -> None:
        self.p, self.v, self.ppp, self.ppv, self.pvv = self._propagate(
            self.p, self.v, self.ppp, self.ppv, self.pvv, dt, q)

    def correct(self, measured: float, r: float) -> None:
        s = self.ppp + r                       # innovation variance (r > 0, so s > 0)
        k0, k1 = self.ppp / s, self.ppv / s    # Kalman gain for position and velocity
        innovation = measured - self.p
        self.p += k0 * innovation
        self.v += k1 * innovation
        # P = (I - K H) P, written out so it stays symmetric.
        self.pvv -= k1 * self.ppv
        self.ppv *= (1.0 - k0)
        self.ppp *= (1.0 - k0)

    def forecast(self, horizon_s: float, q: float) -> Tuple[float, float]:
        """Position and its standard deviation horizon_s seconds ahead."""
        p, _, ppp, _, _ = self._propagate(self.p, self.v, self.ppp, self.ppv, self.pvv, horizon_s, q)
        return p, math.sqrt(max(0.0, ppp))


class KalmanPredictor:
    def __init__(self, horizon_ms: int = DEFAULT_HORIZON_MS, process_noise: float = DEFAULT_PROCESS_NOISE,
                 measurement_noise: float = DEFAULT_MEASUREMENT_NOISE,
                 initial_velocity_sigma: float = DEFAULT_INITIAL_VELOCITY_SIGMA,
                 max_gap_ms: int = DEFAULT_MAX_GAP_MS):
        if not horizon_ms > 0:
            raise ValueError("horizon_ms must be positive")
        if not (math.isfinite(process_noise) and process_noise >= 0):
            raise ValueError("process_noise must be finite and not negative")
        if not (math.isfinite(measurement_noise) and measurement_noise > 0):
            raise ValueError("measurement_noise must be finite and positive")
        if not (math.isfinite(initial_velocity_sigma) and initial_velocity_sigma > 0):
            raise ValueError("initial_velocity_sigma must be finite and positive")
        if not max_gap_ms > 0:
            raise ValueError("max_gap_ms must be positive")
        self.horizon_ms = int(horizon_ms)
        self.q = float(process_noise)
        self.r = float(measurement_noise)
        self.initial_velocity_var = float(initial_velocity_sigma) ** 2
        self.max_gap_ms = int(max_gap_ms)
        self.ignored_samples = 0     # repeated or out-of-order timestamps skipped
        self.reset()

    @classmethod
    def from_config(cls, config) -> "KalmanPredictor":
        """Build from a loaded ghostgrip.config.Config (uses ai.prediction_horizon_ms)."""
        return cls(horizon_ms=config.ai.prediction_horizon_ms)

    def reset(self) -> None:
        """Forget everything: used on tracking loss and before the first sample."""
        self._axes: Optional[Tuple[_Axis, _Axis]] = None
        self._t_ms: Optional[int] = None
        self._last: Optional[Prediction] = None

    # ------------------------------------------------------------------ main entry point
    def update(self, hand: HandStateV2) -> Prediction:
        pointer = hand.pointer_xy
        usable = (hand.tracking_ok and len(pointer) == 2
                  and all(math.isfinite(v) for v in pointer))
        if not usable:
            self.reset()
            return self._lost(hand)

        t_ms = int(hand.t_ms)
        if self._t_ms is None or t_ms - self._t_ms > self.max_gap_ms:
            self._start(pointer, t_ms)
        else:
            dt = (t_ms - self._t_ms) / 1000.0
            if dt <= 0.0:
                self.ignored_samples += 1
                return self._last          # nothing new to learn from a repeated timestamp
            for axis, measured in zip(self._axes, pointer):
                axis.step(dt, self.q)
                axis.correct(measured, self.r)
            self._t_ms = t_ms
        self._last = self._forecast(t_ms)
        return self._last

    # ------------------------------------------------------------------ internals
    def _start(self, pointer: Sequence[float], t_ms: int) -> None:
        self._axes = (_Axis(pointer[0], self.r, self.initial_velocity_var),
                      _Axis(pointer[1], self.r, self.initial_velocity_var))
        self._t_ms = t_ms

    def _forecast(self, t_ms: int) -> Prediction:
        horizon_s = self.horizon_ms / 1000.0
        (px, sx), (py, sy) = (axis.forecast(horizon_s, self.q) for axis in self._axes)
        return Prediction(t_query_ms=t_ms, horizon_ms=self.horizon_ms,
                          future_pointer_xy=(_clip01(px), _clip01(py)),
                          sigma_xy=(sx, sy))

    def _lost(self, hand: HandStateV2) -> Prediction:
        held = tuple(_clip01(v) if math.isfinite(v) else 0.5 for v in hand.pointer_xy[:2])
        return Prediction(t_query_ms=int(hand.t_ms), horizon_ms=self.horizon_ms,
                          future_pointer_xy=held, sigma_xy=(LOST_SIGMA, LOST_SIGMA))

    @property
    def velocity_xy(self) -> Optional[Tuple[float, float]]:
        """The filter's current velocity estimate (normalised units per second), if tracking."""
        return None if self._axes is None else (self._axes[0].v, self._axes[1].v)


# ====================================================================== scoring
METHODS = ("kalman", "no_movement", "constant_velocity")


@dataclass
class _Pending:
    t_target_ms: int
    guesses: Dict[str, Tuple[float, float]]
    sigma: Tuple[float, float]


class ErrorTracker:
    """Scores predictions against the pointer that was actually observed at the target time.

    Each prediction is stored in a ring buffer, then compared with the observed pointer at
    t_query + horizon (linearly interpolated between the two frames around that time). It is
    never compared with the frame it was made on (plan 4.4). Two baselines are scored on the
    same targets: "no_movement" (stay where the pointer is) and "constant_velocity" (extend
    the HandStateV2 velocity_xy), so the Kalman filter has to earn its place.
    """

    def __init__(self, window: int = 120, capacity: int = 64, max_bracket_ms: int = 150):
        if window < 1 or capacity < 1 or max_bracket_ms < 1:
            raise ValueError("window, capacity and max_bracket_ms must be positive")
        self.max_bracket_ms = max_bracket_ms
        self._observations: Deque[Tuple[int, Tuple[float, float]]] = deque(maxlen=capacity)
        self._pending: Deque[_Pending] = deque(maxlen=capacity)
        self._recent: Deque[Dict[str, float]] = deque(maxlen=window)
        self._sum = {m: 0.0 for m in METHODS}
        self._inside_2_sigma = 0
        self._sigma_sum = 0.0
        self.scored = 0
        self.unscorable = 0

    def record(self, hand: HandStateV2, prediction: Prediction) -> None:
        """Call once per frame with the state and the prediction made from it."""
        if not hand.tracking_ok:
            return
        if self._observations and int(hand.t_ms) <= self._observations[-1][0]:
            return                       # repeated or out-of-order frame: never score it twice
        self._observations.append((int(hand.t_ms), tuple(hand.pointer_xy)))
        self._resolve(int(hand.t_ms))
        if max(prediction.sigma_xy) >= LOST_SIGMA:
            return                       # a "nothing known" prediction is not a forecast
        h = prediction.horizon_ms / 1000.0
        cv = tuple(_clip01(p + v * h) for p, v in zip(hand.pointer_xy, hand.velocity_xy))
        self._pending.append(_Pending(
            prediction.t_query_ms + prediction.horizon_ms,
            {"kalman": tuple(prediction.future_pointer_xy),
             "no_movement": tuple(hand.pointer_xy),
             "constant_velocity": cv},
            tuple(prediction.sigma_xy)))

    def _observed_at(self, t_ms: int) -> Optional[Tuple[float, float]]:
        obs = self._observations
        for (t0, p0), (t1, p1) in zip(obs, list(obs)[1:]):
            if t0 <= t_ms <= t1:
                if t1 - t0 > self.max_bracket_ms:
                    return None          # frames missing or tracking lost around that moment
                u = 0.0 if t1 == t0 else (t_ms - t0) / (t1 - t0)
                return (p0[0] + u * (p1[0] - p0[0]), p0[1] + u * (p1[1] - p0[1]))
        return None

    def _resolve(self, now_ms: int) -> None:
        keep = deque(maxlen=self._pending.maxlen)
        for pending in self._pending:
            if pending.t_target_ms > now_ms:
                keep.append(pending)     # the future has not happened yet
                continue
            observed = self._observed_at(pending.t_target_ms)
            if observed is None:
                self.unscorable += 1
                continue
            errors = {m: math.dist(guess, observed) for m, guess in pending.guesses.items()}
            kx, ky = pending.guesses["kalman"]
            if (abs(kx - observed[0]) <= 2.0 * pending.sigma[0]
                    and abs(ky - observed[1]) <= 2.0 * pending.sigma[1]):
                self._inside_2_sigma += 1
            self._sigma_sum += max(pending.sigma)
            for m, e in errors.items():
                self._sum[m] += e
            self._recent.append(errors)
            self.scored += 1
        self._pending = keep

    def mean_errors(self) -> Dict[str, float]:
        """Mean absolute error (normalised units) per method over everything scored."""
        if self.scored == 0:
            return {m: float("nan") for m in METHODS}
        return {m: self._sum[m] / self.scored for m in METHODS}

    def calibration(self) -> float:
        """Share of Kalman forecasts whose real error fell inside the claimed 2-sigma box.
        A well-calibrated Gaussian gives about 0.91; much lower means overconfident, and
        near 1.0 with a large sigma means the uncertainty is needlessly cautious."""
        return float("nan") if self.scored == 0 else self._inside_2_sigma / self.scored

    def mean_sigma(self) -> float:
        """Average claimed uncertainty (the larger axis) over everything scored, normalised units.
        Compare with the mean error: much larger means the uncertainty is needlessly cautious."""
        return float("nan") if self.scored == 0 else self._sigma_sum / self.scored

    def is_worse_than_baseline(self, min_samples: int = 30, margin: float = 0.0) -> bool:
        """True when, over the recent window, the Kalman error exceeds the better baseline's by
        more than `margin` (a fraction). False when there is not enough data to say. When this
        is True, predictive assistance should be reduced or switched off (plan 4.4)."""
        if len(self._recent) < min_samples:
            return False
        n = len(self._recent)
        mean = {m: sum(r[m] for r in self._recent) / n for m in METHODS}
        best_baseline = min(mean["no_movement"], mean["constant_velocity"])
        return mean["kalman"] > best_baseline * (1.0 + margin)

    def summary(self) -> Dict[str, float]:
        out = dict(self.mean_errors())
        out["scored"] = self.scored
        out["unscorable"] = self.unscorable
        out["within_2_sigma"] = self.calibration()
        out["mean_sigma"] = self.mean_sigma()
        return out


# ====================================================================== offline evaluation
def evaluate(states: Sequence[HandStateV2], predictor: Optional[KalmanPredictor] = None,
             tracker: Optional[ErrorTracker] = None) -> Dict[str, float]:
    """Run a predictor over a whole trace and return the error summary."""
    predictor = predictor or KalmanPredictor()
    tracker = tracker or ErrorTracker()
    for state in states:
        tracker.record(state, predictor.update(state))
    return tracker.summary()


def main(argv=None):
    """Print the error table for every synthetic scenario:  PYTHONPATH=src python -m ghostgrip.ai.predict"""
    import argparse
    from .synthetic import SCENARIOS, make_scenario
    parser = argparse.ArgumentParser(description="Kalman predictor vs baselines on synthetic traces.")
    parser.add_argument("--horizon-ms", type=int, default=DEFAULT_HORIZON_MS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    print(f"mean error in normalised units, horizon {args.horizon_ms} ms (lower is better)")
    print(f"{'scenario':<16}{'scored':>7}{'kalman':>10}{'no_move':>10}{'const_vel':>11}{'in_2sigma':>11}  best")
    for name in SCENARIOS:
        s = evaluate(make_scenario(name, args.seed), KalmanPredictor(args.horizon_ms))
        if not s["scored"]:
            print(f"{name:<16}{0:>7}  (nothing to score)")
            continue
        best = min(METHODS, key=lambda m: s[m])
        print(f"{name:<16}{int(s['scored']):>7}{s['kalman']:>10.4f}{s['no_movement']:>10.4f}"
              f"{s['constant_velocity']:>11.4f}{s['within_2_sigma']:>11.2f}  {best}")
    print("in_2sigma: share of forecasts whose real error stayed inside the claimed 2-sigma box")
    print("(about 0.91 if well calibrated; 1.00 means cautious). Synthetic data only: not evidence about real hands.")


if __name__ == "__main__":
    main()
