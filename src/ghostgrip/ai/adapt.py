"""Bounded personalisation and the assistance gain (technical doc 4.5, Phase 1 plan D4).

Two small, separate jobs:

1. Every frame, decide how much assistance to give, 0 to 1 (`AssistPolicy.decide`). It is
   zero whenever anything is doubtful, and otherwise falls as the predictor's uncertainty rises:

       gain = clamp(1 - sigma / uncertainty_limit, 0, 1) * grasp_tolerance

   The simulation turns the gain into a bounded effect: its grab zone grows by at most 15% of
   an object's radius (simulation/world.py MAX_ASSIST_FRAC). Nothing here grasps for the user.

2. Between trials, learn how steady this user is (`AssistPolicy.end_trial`). A slow moving
   average of per-trial jitter (alpha 0.15, config.yaml ai.adapt_alpha), clamped to fixed limits,
   sets the user's grasp_tolerance: a shakier hand gets a larger share of the available assistance.

Rules (all enforced and tested):
  * gain is exactly 0 when tracking is lost, the pointer is unreliable, the prediction is the
    "nothing known" one or is stale, or the predictor is not trusted (it lost to a baseline);
  * the profile only changes in end_trial(), never mid-trial, and never while frozen. Freeze it
    while a condition is being measured (plan: personalisation is frozen during an experiment);
  * only valid, successful trials count, each trial at most once;
  * one outlier trial cannot move the profile by more than alpha times its allowed range;
  * every number stays inside configured limits, whatever is fed in.

jitter_max below is a PROVISIONAL bound chosen from synthetic traces (0.0004-0.001); replace it
from real recordings. The clamps are what keep the system safe, not that value.
"""
from contextlib import contextmanager
from dataclasses import dataclass, replace
import math
from typing import Dict, Optional, Sequence, Set, Tuple

from contracts import HandStateV2, Prediction, TrialOutcome
from .predict import LOST_SIGMA


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


@dataclass(frozen=True)
class AdaptationConfig:
    alpha: float = 0.15               # weight of the newest trial in the moving average
    uncertainty_limit: float = 0.1    # sigma at which assistance reaches zero (predictor settles near 0.03)
    min_quality: float = 0.5          # below this tracking_quality: no assistance (matches the simulation)
    max_prediction_age_ms: int = 250  # an older prediction is stale
    jitter_max: float = 0.004         # PROVISIONAL ceiling for per-trial jitter (normalised units)
    tolerance_min: float = 0.3        # grasp_tolerance for a perfectly steady hand
    tolerance_max: float = 1.0        # grasp_tolerance at jitter_max or above

    def __post_init__(self):
        for name in ("alpha", "uncertainty_limit", "min_quality", "jitter_max",
                     "tolerance_min", "tolerance_max"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be a finite number")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be greater than 0 and at most 1")
        if self.uncertainty_limit <= 0.0 or self.uncertainty_limit > LOST_SIGMA:
            raise ValueError("uncertainty_limit must be positive and at most the lost-tracking sigma")
        if not 0.0 <= self.min_quality <= 1.0:
            raise ValueError("min_quality must be between 0 and 1")
        if self.jitter_max <= 0.0:
            raise ValueError("jitter_max must be positive")
        if not 0.0 <= self.tolerance_min <= self.tolerance_max <= 1.0:
            raise ValueError("need 0 <= tolerance_min <= tolerance_max <= 1")
        if self.max_prediction_age_ms < 0:
            raise ValueError("max_prediction_age_ms cannot be negative")

    def tolerance_for(self, jitter: float) -> float:
        """grasp_tolerance = tolerance_min + (tolerance_max - tolerance_min) * jitter / jitter_max."""
        share = _clamp(jitter / self.jitter_max, 0.0, 1.0)
        return _clamp(self.tolerance_min + (self.tolerance_max - self.tolerance_min) * share,
                      self.tolerance_min, self.tolerance_max)


@dataclass(frozen=True)
class AdaptationProfile:
    jitter: float            # moving average of per-trial jitter, within [0, jitter_max]
    grasp_tolerance: float   # share of the available assistance this user gets
    trials: int = 0          # how many trials have updated the profile

    def to_dict(self) -> Dict[str, float]:
        return {"jitter": self.jitter, "grasp_tolerance": self.grasp_tolerance, "trials": self.trials}


@dataclass(frozen=True)
class AssistDecision:
    gain: float              # 0 to 1, for World.set_assistance(enabled, gain)
    reason: str              # why: "ok", "tracking_lost", "low_quality", ... (log it)


def neutral_profile(config: AdaptationConfig) -> AdaptationProfile:
    """A new, unknown user starts in the middle of the allowed range."""
    jitter = config.jitter_max / 2.0
    return AdaptationProfile(jitter, config.tolerance_for(jitter))


def profile_from_dict(data: Dict, config: AdaptationConfig) -> AdaptationProfile:
    """Rebuild a profile and re-derive the tolerance, rejecting out-of-range data."""
    try:
        jitter, trials = float(data["jitter"]), int(data["trials"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid profile: {error}") from error
    if not (math.isfinite(jitter) and 0.0 <= jitter <= config.jitter_max):
        raise ValueError("profile jitter is outside the allowed range")
    if trials < 0:
        raise ValueError("profile trial count cannot be negative")
    return AdaptationProfile(jitter, config.tolerance_for(jitter), trials)


def trial_jitter(states: Sequence[HandStateV2]) -> Optional[float]:
    """How shaky was the pointer during a trial? None if there is nothing to measure.

    RMS distance (normalised units) of each valid point from the straight line between its two
    neighbours, weighted for uneven time steps. Smooth motion scores about 0; wobble scores high.
    Note the perception pipeline already smooths the pointer, so this under-reports raw jitter.
    """
    total, count = 0.0, 0
    for a, b, c in zip(states, states[1:], states[2:]):
        if not (a.tracking_ok and b.tracking_ok and c.tracking_ok) or c.t_ms <= a.t_ms:
            continue
        if not (a.t_ms < b.t_ms < c.t_ms):
            continue
        u = (b.t_ms - a.t_ms) / (c.t_ms - a.t_ms)
        for i in (0, 1):
            expected = a.pointer_xy[i] + u * (c.pointer_xy[i] - a.pointer_xy[i])
            total += (b.pointer_xy[i] - expected) ** 2
        count += 1
    if count == 0:
        return None
    value = math.sqrt(total / count)
    return value if math.isfinite(value) else None


class AssistPolicy:
    def __init__(self, config: Optional[AdaptationConfig] = None,
                 profile: Optional[AdaptationProfile] = None):
        self.config = config or AdaptationConfig()
        self.profile = neutral_profile(self.config) if profile is None else self._checked(profile)
        self.frozen = False
        self._counted: Set[Tuple[str, int]] = set()
        self.updates = 0
        self.skipped = 0

    def _checked(self, profile: AdaptationProfile) -> AdaptationProfile:
        """Refuse a starting profile that is outside the configured limits."""
        c = self.config
        if not (math.isfinite(profile.jitter) and 0.0 <= profile.jitter <= c.jitter_max):
            raise ValueError("profile jitter is outside the allowed range")
        if not (math.isfinite(profile.grasp_tolerance)
                and c.tolerance_min <= profile.grasp_tolerance <= c.tolerance_max):
            raise ValueError("profile grasp_tolerance is outside the allowed range")
        if profile.trials < 0:
            raise ValueError("profile trial count cannot be negative")
        return profile

    @classmethod
    def from_config(cls, config) -> "AssistPolicy":
        """Use ai.adapt_alpha from a loaded ghostgrip.config.Config."""
        return cls(AdaptationConfig(alpha=config.ai.adapt_alpha))

    # ------------------------------------------------------------------ every frame
    def decide(self, hand: HandStateV2, prediction: Prediction,
               predictor_trusted: bool = True) -> AssistDecision:
        """How much assistance right now? Pass predictor_trusted=not tracker.is_worse_than_baseline()."""
        c = self.config
        if not hand.tracking_ok:
            return AssistDecision(0.0, "tracking_lost")
        if not predictor_trusted:
            return AssistDecision(0.0, "predictor_worse_than_baseline")
        if not (math.isfinite(hand.tracking_quality) and hand.tracking_quality >= c.min_quality):
            return AssistDecision(0.0, "low_quality")
        sigma = max(prediction.sigma_xy)
        if not math.isfinite(sigma):
            return AssistDecision(0.0, "invalid_prediction")
        if sigma >= LOST_SIGMA:
            return AssistDecision(0.0, "no_prediction")
        age = hand.t_ms - prediction.t_query_ms
        if age < 0 or age > c.max_prediction_age_ms:
            return AssistDecision(0.0, "stale_prediction")
        confidence = _clamp(1.0 - sigma / c.uncertainty_limit, 0.0, 1.0)
        if confidence <= 0.0:
            return AssistDecision(0.0, "uncertain")
        return AssistDecision(_clamp(confidence * self.profile.grasp_tolerance, 0.0, 1.0), "ok")

    def gain(self, hand: HandStateV2, prediction: Prediction, predictor_trusted: bool = True) -> float:
        return self.decide(hand, prediction, predictor_trusted).gain

    # ------------------------------------------------------------------ between trials
    def freeze(self) -> None:
        self.frozen = True

    def unfreeze(self) -> None:
        self.frozen = False

    @contextmanager
    def evaluation(self):
        """Freeze the profile while a condition is being measured, then restore the old setting."""
        was_frozen = self.frozen
        self.frozen = True
        try:
            yield self
        finally:
            self.frozen = was_frozen

    def end_trial(self, outcome: TrialOutcome, jitter: Optional[float]) -> AdaptationProfile:
        """Update the profile after one finished trial. Returns the (possibly unchanged) profile.

        Skipped, with the profile untouched, when frozen; when the trial failed or its tracking
        was lost; when the jitter is missing or not finite; or when this trial was counted before.
        """
        key = (outcome.session_id, outcome.trial_id)
        valid = (outcome.success and outcome.fail_reason == "" and jitter is not None
                 and math.isfinite(jitter) and jitter >= 0.0)
        if self.frozen or not valid or key in self._counted:
            self.skipped += 1
            return self.profile
        c = self.config
        newest = _clamp(jitter, 0.0, c.jitter_max)        # an outlier counts as at most the ceiling
        blended = (1.0 - c.alpha) * self.profile.jitter + c.alpha * newest
        new_jitter = _clamp(blended, 0.0, c.jitter_max)
        self.profile = replace(self.profile, jitter=new_jitter,
                               grasp_tolerance=c.tolerance_for(new_jitter),
                               trials=self.profile.trials + 1)
        self._counted.add(key)
        self.updates += 1
        return self.profile
