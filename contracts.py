"""Shared interface used by all four team members.

x/y: normalized numbers between 0 and 1.
pinch: thumb-index distance divided by an approximate palm scale.
"""
from dataclasses import dataclass


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
# Contracts v2 (proposal, from GhostGrip technical documentation section 16.2)
#
# The v1 types above stay unchanged so existing code keeps working.
# HandStateV2 and SimEventV2 have a V2 suffix only to avoid a name clash with
# v1. When every module has moved to v2, drop the v1 types and the suffix.
#
# Agreed conventions (to be confirmed by all four members):
#   - Tracking loss is reported with tracking_ok=False; the pointer holds its
#     last value. There is no separate TRACKING_LOST event.
#   - pointer_xy is normalized 0-1 and already mirrored (Member A). The
#     simulation maps it to world coordinates (Member B).
#   - closure_q: 0 = calibrated open hand, 1 = calibrated closed.
#   - One SimEvent per state transition, never one per frame.
#   - t_ms is the host monotonic clock. Arduino uptime_ms is a different clock
#     and must never be subtracted from host time.
#   - Adaptation updates at trial end only and is frozen while measuring.
# ---------------------------------------------------------------------------
from typing import Tuple


@dataclass
class HandStateV2:
    t_ms: int
    pointer_xy: Tuple[float, float]  # normalized 0-1, mirrored
    closure_q: float  # 0 = open, 1 = closed
    dq_dt: float  # closure velocity; positive means closing
    velocity_xy: Tuple[float, float]
    stable_score: float  # 0-1, higher is steadier
    tracking_ok: bool
    tracking_quality: float  # 0-1


@dataclass
class Prediction:
    t_query_ms: int
    horizon_ms: int
    future_pointer_xy: Tuple[float, float]
    sigma_xy: Tuple[float, float]  # forecast uncertainty


@dataclass
class SimEventV2:
    event_id: int
    t_ms: int
    object_id: str
    kind: str  # CONTACT, HOLD, SLIP, RELEASE, DAMAGE
    simulated_grip_effort: float  # simulation units, NOT newtons


@dataclass
class CueEvent:
    seq: int
    kind: str  # event name sent to the Arduino
    safe_angle_deg: float
    duration_ms: int
    no_contact_mode: bool = True  # non-contact until the safety gate passes


@dataclass
class DeviceState:
    uptime_ms: int  # Arduino clock, not comparable with host t_ms
    armed: bool
    touch: bool
    piezo: bool  # binary vibration event, not force
    ax_g: float
    ay_g: float
    az_g: float
    rotary_raw: int
    last_ack_seq: int


@dataclass
class TrialOutcome:
    session_id: str
    trial_id: int
    condition: str  # baseline, ai, haptic, both
    success: bool
    fail_reason: str
    start_ms: int
    end_ms: int
    slip_count: int = 0
    overgrip_count: int = 0
