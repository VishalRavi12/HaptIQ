"""Shared interface used by all four team members.

This is the ONE contracts file for the whole team. Import from here:

    from contracts import HandStateV2, SimEventV2, TrialOutcome, CONTACT, ...

Changing anything in the v2 section needs approval from all four members
(Implementation Plan 5.1).

The file has two parts:
  * Starter v1 types (HandState, AIState, SimEvent). Unchanged so the beginner
    demo (main.py, simulation.py, grasp_ai.py, hand_tracking.py) and
    tests/test_baseline.py keep working. Reference code only.
  * Contracts v2 (Implementation Plan 6.1 = technical document 16.2). All new
    code uses these. HandStateV2 and SimEventV2 carry a V2 suffix only to avoid
    a name clash with v1. When every module has moved to v2, drop the v1 types
    and the suffix in one all-four-approved change.
"""
from dataclasses import dataclass
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Starter v1 types (reference only; do not use in new code)
#
# x/y: normalized numbers between 0 and 1.
# pinch: thumb-index distance divided by an approximate palm scale.
# ---------------------------------------------------------------------------

@dataclass
class HandState:
    timestamp_ms: int
    tracked: bool
    x: float
    y: float
    pinch: float
    closing_speed: float = 0.0  # positive means pinch is closing


@dataclass
class AIState:
    grasp_state: str  # open, closing, holding, releasing
    predicted_x: float
    predicted_y: float
    assistance: float = 0.0


@dataclass
class SimEvent:
    name: str  # CONTACT, GRASP, RELEASE, BREAK
    object_id: str
    timestamp_ms: int


# ---------------------------------------------------------------------------
# Contracts v2
#
# Agreed conventions (Implementation Plan 6.2, plus Member A's notes on PR #1):
#   - The simulator always receives a HandStateV2. tracking_ok=False signals
#     loss; there is no separate TRACKING_LOST event. On loss, pointer_xy and
#     closure_q keep their last values, and velocity_xy, dq_dt, stable_score
#     and tracking_quality become 0. Consumers must check tracking_ok.
#   - pointer_xy is normalized 0-1 and already mirrored (Member A): x grows to
#     the right, y grows downward. The simulation maps it to its 960x540 world
#     (Member B).
#   - closure_q: 0 = calibrated open hand, 1 = calibrated closed.
#   - velocity_xy and dq_dt are per second (normalized units per second).
#   - Exactly one SimEvent per state transition, never one per frame.
#   - t_ms is the host monotonic clock in milliseconds. Arduino uptime_ms is a
#     different clock and must never be subtracted from host time.
#   - Missing sensor data is None (absent), never 0 (plan 1.6).
#   - Adaptation updates at trial end only and is frozen while measuring.
#   - All v2 types are frozen (immutable). To change a value, make a new
#     object with dataclasses.replace(obj, field=value).
# ---------------------------------------------------------------------------

XY = Tuple[float, float]

# SimEventV2.kind values (technical document 3.4 / plan 7.2 step 4).
# Use these names instead of typing the strings.
CONTACT = "CONTACT"
HOLD = "HOLD"
SLIP = "SLIP"
RELEASE = "RELEASE"
DAMAGE = "DAMAGE"
SIM_EVENT_KINDS = (CONTACT, HOLD, SLIP, RELEASE, DAMAGE)

# TrialOutcome.condition values (plan 1.5).
CONDITION_BASELINE = "A"   # AI assistance OFF, physical cue OFF
CONDITION_AI = "B"         # AI assistance ON,  physical cue OFF
CONDITION_HAPTIC = "C"     # AI assistance OFF, physical cue ON
CONDITION_BOTH = "D"       # AI assistance ON,  physical cue ON
CONDITION_CODES = (CONDITION_BASELINE, CONDITION_AI, CONDITION_HAPTIC, CONDITION_BOTH)


@dataclass(frozen=True)
class HandStateV2:
    """Member A -> everyone. One per processed camera frame."""
    t_ms: int                  # host monotonic milliseconds
    pointer_xy: XY             # normalized 0-1, mirrored (thumb/index midpoint)
    closure_q: float           # 0 = calibrated open, 1 = calibrated closed
    dq_dt: float               # closure velocity per second; positive = closing
    velocity_xy: XY            # pointer velocity, normalized units per second
    stable_score: float        # 0-1, 1 = steady (low jitter)
    tracking_ok: bool          # False = no valid hand this frame
    tracking_quality: float    # 0-1 geometric usability heuristic, NOT model confidence


@dataclass(frozen=True)
class Prediction:
    """Member D -> Member B (ghost marker) and the assistance policy."""
    t_query_ms: int
    horizon_ms: int
    future_pointer_xy: XY      # normalized 0-1
    sigma_xy: XY               # forecast uncertainty, normalized units


@dataclass(frozen=True)
class SimEventV2:
    """Member B -> Member C (haptic policy) and Member D (logger).
    Exactly one per state transition."""
    event_id: int
    t_ms: int
    object_id: str
    kind: str                     # one of SIM_EVENT_KINDS
    simulated_grip_effort: float  # synthetic, dimensionless; NOT newtons


@dataclass(frozen=True)
class CueEvent:
    """Member C's haptic policy -> serial link."""
    seq: int
    kind: str                     # event name sent to the Arduino
    safe_angle_deg: float
    duration_ms: int
    no_contact_mode: bool = True  # non-contact until the safety gate passes


@dataclass(frozen=True)
class DeviceState:
    """Member C -> everyone. Parsed from Arduino 'S' packets.
    Sensor fields are None when absent, never 0."""
    uptime_ms: int                # Arduino millis(); NOT comparable with host t_ms
    armed: bool
    touch: Optional[bool]
    piezo: Optional[bool]         # binary vibration event, NOT force
    ax_g: Optional[float]
    ay_g: Optional[float]
    az_g: Optional[float]
    rotary_raw: Optional[int]
    last_ack_seq: Optional[int]


@dataclass(frozen=True)
class TrialOutcome:
    """Member B -> Member D (logger, metrics). One per finished trial.

    condition:   one of CONDITION_CODES ("A", "B", "C", "D").
    fail_reason: "" on success. Otherwise one of:
                 "damaged_overgrip"  fragile object squeezed past its break threshold
                 "damaged_dropped"   fragile object fell too fast and broke
                 "tracking_lost"     hand tracking lost during the trial
                 "timeout"           not completed within the time limit
    """
    session_id: str
    trial_id: int
    condition: str
    success: bool
    fail_reason: str
    start_ms: int
    end_ms: int
    slip_count: int = 0
    overgrip_count: int = 0
