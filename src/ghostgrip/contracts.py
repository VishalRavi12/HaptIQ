"""Frozen contracts v1 shared by all four members.

Source: Implementation Plan section 6.1 (= technical document section 16.2).
Changing this file needs approval from ALL FOUR members (plan section 5.1).

Integration decisions (plan section 6.2):
- The simulator always receives a HandState. tracking_ok=False signals loss;
  pointer_xy then holds its last value.
- Member A outputs mirrored, normalized 0..1 coordinates. Member B maps them
  to the 960x540 world.
- closure_q: 0 = calibrated open, 1 = calibrated closed.
- Exactly one SimEvent per state transition, never one per frame.
- Timestamps are host monotonic milliseconds. Arduino millis() is a different
  clock and is never subtracted from host time.
- Missing sensor data is None (absent), never 0.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

XY = Tuple[float, float]

# SimEvent.kind values (technical document 3.4 / plan 7.2 step 4)
CONTACT = "CONTACT"
HOLD = "HOLD"
SLIP = "SLIP"
RELEASE = "RELEASE"
DAMAGE = "DAMAGE"
SIM_EVENT_KINDS = (CONTACT, HOLD, SLIP, RELEASE, DAMAGE)


@dataclass(frozen=True)
class HandState:
    """Member A -> everyone. One per processed camera frame."""
    t_ms: int                  # host monotonic milliseconds
    pointer_xy: XY             # mirrored, normalized 0..1 (thumb/index midpoint)
    closure_q: float           # 0 = calibrated open, 1 = calibrated closed
    dq_dt: float               # closure velocity per second, positive = closing
    velocity_xy: XY            # pointer velocity, normalized units per second
    stable_score: float        # 0..1, 1 = steady (low jitter)
    tracking_ok: bool          # False = no valid hand this frame
    tracking_quality: float    # 0..1 confidence used to gate assistance and cues


@dataclass(frozen=True)
class Prediction:
    """Member D -> B (ghost marker) and the assistance policy."""
    t_query_ms: int
    horizon_ms: int
    future_pointer_xy: XY      # normalized 0..1
    sigma_xy: XY               # uncertainty, normalized units


@dataclass(frozen=True)
class SimEvent:
    """Member B -> C (haptic policy) and D (logger). One per transition."""
    event_id: int
    t_ms: int
    object_id: str
    kind: str                  # one of SIM_EVENT_KINDS
    simulated_grip_effort: float  # synthetic, dimensionless; NOT newtons


@dataclass(frozen=True)
class CueEvent:
    """Member C's haptic policy -> serial link."""
    seq: int
    kind: str
    safe_angle_deg: float
    duration_ms: int
    no_contact_mode: bool


@dataclass(frozen=True)
class DeviceState:
    """Member C -> everyone. Parsed from Arduino 'S' packets."""
    uptime_ms: int             # Arduino millis(); separate clock
    armed: bool
    touch: Optional[bool]
    piezo: Optional[bool]
    ax_g: Optional[float]
    ay_g: Optional[float]
    az_g: Optional[float]
    rotary_raw: Optional[int]
    last_ack_seq: Optional[int]


@dataclass(frozen=True)
class TrialOutcome:
    """Member B -> D (logger, metrics)."""
    session_id: str
    trial_id: int
    condition: str             # "A", "B", "C" or "D" (plan 1.5)
    success: bool
    fail_reason: str           # "" on success
    start_ms: int
    end_ms: int
    slip_count: int
    overgrip_count: int
