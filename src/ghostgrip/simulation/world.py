"""Headless grasp physics. NO pygame import: this runs in tests and CI.

Side view of a table. The hand is a pointer plus two virtual finger pads whose
gap follows closure q. One object can be engaged at a time.

Per engaged object the phase is:
    NO_CONTACT -> CONTACT -> STABLE_GRASP <-> SLIPPING
                                  any -> DAMAGED (fragile over-grip)
                                  any -> released (NO_CONTACT)

Events (exactly one per transition, never per frame):
    CONTACT  engagement starts (pads touch the object)
    HOLD     effort reaches the object's hold threshold
    SLIP     effort drops below hold while the object is lifted
    RELEASE  engagement ends (hand opened, moved away, slipped out, tracking lost)
    DAMAGE   fragile object over-gripped or dropped too hard. DAMAGE ends the
             engagement by itself; no RELEASE follows it.

Anti-chatter rules (test cases T03, T07):
    - a new phase must be wanted for CONFIRM_STEPS consecutive physics steps;
    - contact/release and hold/slip use hysteresis margins;
    - after any engagement ends, the hand must open (q < OPEN_Q) before it can
      grab again, and pinching in empty air disarms grabbing (no "sticky grab").

Timing: fixed 1/60 s physics steps with capped catch-up, so the outcome does
not depend on the display frame rate.
"""
from collections import deque
from dataclasses import dataclass, field
import math
from typing import Deque, Dict, List, Optional, Tuple

from contracts import CONTACT, DAMAGE, HOLD, RELEASE, SLIP
from contracts import HandStateV2 as HandState
from contracts import SimEventV2 as SimEvent
from ghostgrip.simulation.objects import PROFILES, ObjectProfile

# ---- world geometry (plan 6.2: B maps normalized 0..1 to a 960x540 world) ----
WORLD_W, WORLD_H = 960, 540
TABLE_Y = 450.0
TARGET_ZONE = (740.0, 930.0)        # x-range on the table where objects must be placed

# ---- timing ----
PHYSICS_HZ = 60
DT = 1.0 / PHYSICS_HZ
MAX_CATCHUP_STEPS = 5               # at most ~83 ms simulated per advance() call

# ---- anti-chatter ----
CONFIRM_STEPS = 3                   # 50 ms persistence before any phase change
Q_HYST = 0.05                       # release only below contact_threshold - Q_HYST
G_HYST = 0.03                       # keep holding until effort < hold - G_HYST
GRAB_MARGIN = 14.0                  # px beyond the radius where pads can engage
EXIT_EXTRA = 16.0                   # extra px before a resting contact is released
OPEN_Q = 0.15                       # hand counts as "open" below this
AIR_PINCH_Q = 0.30                  # pinching this much in empty air disarms grabbing
RELEASE_INTENT_RATE = 0.4           # q falling faster than this per second = deliberate
                                    # release, so passing through the weak band is not a SLIP
Q_RATE_TAU_S = 0.05                 # smoothing time constant for the closure rate

# ---- assistance (bounded; full automatic grasp is prohibited) ----
MAX_ASSIST_FRAC = 0.15              # grab zone grows by at most 15% of the radius

# ---- tracking ----
TRACKING_TIMEOUT_MS = 300.0         # freeze up to this long, then safe release
QUALITY_MIN = 0.5                   # below this: unreliable, no new grasps

# ---- motion ----
GRAVITY = 1400.0                    # px/s^2, synthetic
SLIP_OUT_FRAC = 1.2                 # slipped out after sliding this * radius
PAD_OPEN_GAP = 110.0                # pad gap in px at q = 0

# ---- phase names ----
NO_CONTACT = "NO_CONTACT"
PHASE_CONTACT = "CONTACT"
STABLE_GRASP = "STABLE_GRASP"
SLIPPING = "SLIPPING"
DAMAGED = "DAMAGED"
_RELEASED = "_RELEASED"             # internal pseudo-phase


@dataclass
class ObjectState:
    object_id: str
    profile: ObjectProfile
    x: float
    y: float
    vy: float = 0.0
    damaged: bool = False
    held_ms: float = 0.0            # total time spent in STABLE_GRASP
    was_lifted: bool = False

    @property
    def rest_y(self) -> float:
        return TABLE_Y - self.profile.radius

    @property
    def lifted(self) -> bool:
        return self.y < self.rest_y - 1.0

    @property
    def resting(self) -> bool:
        return not self.lifted and self.vy == 0.0


@dataclass(frozen=True)
class ObjectView:
    object_id: str
    kind: str
    x: float
    y: float
    radius: float
    color: Tuple[int, int, int]
    damaged: bool
    compression: float              # 0..1, visual only
    phase: str
    in_target: bool


@dataclass(frozen=True)
class WorldSnapshot:
    """Read-only view for renderers (2D now, 3D later) and the HUD."""
    t_ms: int
    pointer: Tuple[float, float]    # world px
    q: float
    pad_gap: float
    tracking_ok: bool
    reliable: bool
    paused: bool
    engaged_id: Optional[str]
    phase: str
    effort: float
    objects: Tuple[ObjectView, ...]
    target_zone: Tuple[float, float]
    table_y: float
    assist_enabled: bool
    assist_gain: float
    grab_radius_extra_frac: float
    recent_events: Tuple[Tuple[int, str, str, str], ...]   # (t_ms, kind, object, reason)
    slip_count: int
    overgrip_count: int


SCENES = {
    "glass": [("glass", 200.0)],
    "sponge": [("sponge", 200.0)],
    "block": [("block", 200.0)],
    "sandbox": [("glass", 170.0), ("sponge", 340.0), ("block", 510.0)],
}


class World:
    def __init__(self, scene: str = "glass"):
        self.reset(scene)

    # ------------------------------------------------------------------ setup
    def reset(self, scene: Optional[str] = None) -> None:
        if scene is not None:
            if scene not in SCENES:
                raise ValueError(f"unknown scene {scene!r}; choose from {list(SCENES)}")
            self.scene = scene
        self.objects: List[ObjectState] = []
        for kind, x in SCENES[self.scene]:
            p = PROFILES[kind]
            self.objects.append(ObjectState(kind, p, x, TABLE_Y - p.radius))
        self.pointer = (WORLD_W / 2.0, WORLD_H / 2.0)
        self.q = 0.0
        self.q_rate = 0.0                # smoothed dq/dt measured by the world itself
        self.tracking_ok = False
        self.reliable = False
        self.t_ms = 0
        self.steps_run = 0
        self.slip_count = 0
        self.overgrip_count = 0
        self.assist_enabled = False
        self.assist_gain = 0.0
        self._acc = 0.0
        self._lost_ms = 0.0
        self._armed = False
        self._engaged: Optional[ObjectState] = None
        self._phase = NO_CONTACT
        self._effort = 0.0
        self._offset = (0.0, 0.0)
        self._slip_v = 0.0
        self._slip_off = 0.0
        self._pending: Optional[tuple] = None
        self._pending_n = 0
        self._next_event_id = 1
        self._out: List[SimEvent] = []
        self._reasons: Dict[int, str] = {}
        self._recent: Deque[Tuple[int, str, str, str]] = deque(maxlen=6)

    def set_assistance(self, enabled: bool, gain: float) -> None:
        """gain 0..1 (later from Member D's adapt.assist_gain). Bounded effect only."""
        self.assist_enabled = bool(enabled)
        self.assist_gain = min(1.0, max(0.0, float(gain)))

    # --------------------------------------------------------------- queries
    def object(self, object_id: str) -> ObjectState:
        for o in self.objects:
            if o.object_id == object_id:
                return o
        raise KeyError(object_id)

    def in_target(self, o: ObjectState) -> bool:
        return (o is not self._engaged and o.resting
                and TARGET_ZONE[0] <= o.x <= TARGET_ZONE[1])

    def event_reason(self, event_id: int) -> str:
        return self._reasons.get(event_id, "")

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def engaged_id(self) -> Optional[str]:
        return self._engaged.object_id if self._engaged else None

    def enter_radius(self, o: ObjectState) -> float:
        extra = o.profile.radius * MAX_ASSIST_FRAC * self.assist_gain if self.assist_enabled else 0.0
        return o.profile.radius + GRAB_MARGIN + extra

    # ---------------------------------------------------------------- update
    def advance(self, hand: HandState, elapsed_s: float) -> List[SimEvent]:
        """Feed the latest HandState and the real elapsed time since the last call.
        Runs as many fixed 1/60 s steps as needed (capped). Returns new events."""
        self._out = []
        self.t_ms = int(hand.t_ms)
        self.tracking_ok = bool(hand.tracking_ok)
        self.reliable = self.tracking_ok and hand.tracking_quality >= QUALITY_MIN
        if self.tracking_ok:
            x = min(1.0, max(0.0, float(hand.pointer_xy[0]))) * WORLD_W
            y = min(1.0, max(0.0, float(hand.pointer_xy[1]))) * WORLD_H
            if math.isfinite(x) and math.isfinite(y):
                self.pointer = (x, y)
        if not math.isfinite(elapsed_s) or elapsed_s < 0:
            elapsed_s = 0.0
        if self.tracking_ok:
            q = float(hand.closure_q)
            if math.isfinite(q):
                q = min(1.0, max(0.0, q))
                if elapsed_s > 0:
                    a = 1.0 - math.exp(-elapsed_s / Q_RATE_TAU_S)
                    self.q_rate += a * ((q - self.q) / elapsed_s - self.q_rate)
                self.q = q
        else:
            self.q_rate = 0.0
        self._acc = min(self._acc + elapsed_s, MAX_CATCHUP_STEPS * DT)
        while self._acc >= DT - 1e-9:
            self._acc -= DT
            self._step()
        return self._out

    def _step(self) -> None:
        self.steps_run += 1
        self._lost_ms = 0.0 if self.tracking_ok else self._lost_ms + DT * 1000.0
        self._update_free_objects()
        if self._lost_ms > 0.0:
            # Freeze the engaged object; after the timeout, release it safely.
            if self._lost_ms >= TRACKING_TIMEOUT_MS and self._engaged is not None:
                self._end_engagement(RELEASE, "tracking_lost")
            self._pending = None
            return
        if self.q < OPEN_Q:
            self._armed = True
        if self._engaged is None:
            self._try_engage()
        else:
            self._update_engaged()

    # ------------------------------------------------------------ internals
    def _emit(self, kind: str, o: ObjectState, reason: str) -> None:
        ev = SimEvent(self._next_event_id, self.t_ms, o.object_id, kind, round(self._effort, 4))
        self._next_event_id += 1
        self._reasons[ev.event_id] = reason
        self._recent.append((ev.t_ms, kind, o.object_id, reason))
        self._out.append(ev)

    def _confirm(self, candidate: tuple) -> bool:
        if candidate == self._pending:
            self._pending_n += 1
        else:
            self._pending, self._pending_n = candidate, 1
        if self._pending_n >= CONFIRM_STEPS:
            self._pending = None
            return True
        return False

    def _nearest_overlapping(self) -> Optional[ObjectState]:
        best, best_d = None, float("inf")
        px, py = self.pointer
        for o in self.objects:
            if o.damaged:
                continue
            d = math.hypot(o.x - px, o.y - py)
            if d <= self.enter_radius(o) and d < best_d:
                best, best_d = o, d
        return best

    def _try_engage(self) -> None:
        o = self._nearest_overlapping()
        if o is None:
            if self.q >= AIR_PINCH_Q:
                self._armed = False      # pinched in empty air: no sticky grab
            self._pending = None
            return
        if not self._armed or not self.reliable or self.q < o.profile.contact_threshold:
            self._pending = None
            return
        if self._confirm(("ENGAGE", o.object_id)):
            self._engaged = o
            self._phase = PHASE_CONTACT
            self._effort = o.profile.effort(self.q)
            ox, oy = o.x - self.pointer[0], o.y - self.pointer[1]
            r = o.profile.radius
            self._offset = (max(-r, min(r, ox)), max(-r, min(r, oy)))
            self._slip_v = self._slip_off = 0.0
            self._emit(CONTACT, o, "pads_touched")

    def _desired_phase(self, o: ObjectState, g: float) -> str:
        p = o.profile
        if p.break_threshold is not None and g > p.break_threshold:
            return DAMAGED
        if self.q < p.contact_threshold - Q_HYST:
            return _RELEASED
        if self._phase == PHASE_CONTACT and not o.lifted:
            if math.hypot(o.x - self.pointer[0], o.y - self.pointer[1]) > self.enter_radius(o) + EXIT_EXTRA:
                return _RELEASED
        need = p.hold_threshold - (G_HYST if self._phase == STABLE_GRASP else 0.0)
        if g >= need:
            return STABLE_GRASP
        if (self._phase == STABLE_GRASP and o.lifted
                and self.q_rate < -RELEASE_INTENT_RATE):
            return STABLE_GRASP          # opening deliberately: a release, not a slip
        return SLIPPING if o.lifted else PHASE_CONTACT

    def _update_engaged(self) -> None:
        o = self._engaged
        g = o.profile.effort(self.q)
        self._effort = g
        desired = self._desired_phase(o, g)
        if desired != self._phase:
            if self._confirm(("PHASE", desired)):
                self._transition(o, desired)
                if self._engaged is None:
                    return
        else:
            self._pending = None
        self._move_engaged(o, g)

    def _transition(self, o: ObjectState, new: str) -> None:
        if new == DAMAGED:
            o.damaged = True
            self.overgrip_count += 1
            self._end_engagement(DAMAGE, "overgrip")
            return
        if new == _RELEASED:
            reason = "hand_opened" if self.q < o.profile.contact_threshold - Q_HYST else "moved_away"
            self._end_engagement(RELEASE, reason)
            return
        if new == STABLE_GRASP:
            self._slip_v = 0.0
            self._emit(HOLD, o, "effort_reached_hold")
        elif new == SLIPPING:
            self._slip_v = 0.0
            self.slip_count += 1
            self._emit(SLIP, o, "effort_below_hold")
        # STABLE_GRASP -> CONTACT while resting on the table emits no event.
        self._phase = new

    def _end_engagement(self, kind: str, reason: str) -> None:
        o = self._engaged
        self._emit(kind, o, reason)
        self._engaged = None
        self._phase = NO_CONTACT
        self._effort = 0.0
        self._armed = False              # hand must open before the next grab
        self._pending = None

    def _clamp_to_world(self, o: ObjectState) -> None:
        r = o.profile.radius
        o.x = min(WORLD_W - r, max(r, o.x))
        o.y = min(o.rest_y, max(r, o.y))

    def _move_engaged(self, o: ObjectState, g: float) -> None:
        px, py = self.pointer
        if self._phase == STABLE_GRASP:
            o.x, o.y = px + self._offset[0], py + self._offset[1] + self._slip_off
            o.vy = 0.0
            self._clamp_to_world(o)
            o.held_ms += DT * 1000.0
            if o.lifted:
                o.was_lifted = True
        elif self._phase == SLIPPING:
            factor = min(1.0, max(0.2, 1.0 - g / o.profile.hold_threshold))
            self._slip_v += GRAVITY * factor * DT
            self._slip_off += self._slip_v * DT
            o.x, o.y = px + self._offset[0], py + self._offset[1] + self._slip_off
            self._clamp_to_world(o)
            if self._slip_off > o.profile.radius * SLIP_OUT_FRAC:
                o.vy = self._slip_v
                self._end_engagement(RELEASE, "slipped_out")
        # PHASE_CONTACT: object rests on the table and does not follow the hand.

    def _update_free_objects(self) -> None:
        for o in self.objects:
            if o is self._engaged:
                continue
            if o.lifted or o.vy != 0.0:
                o.vy += GRAVITY * DT
                o.y += o.vy * DT
                if o.y >= o.rest_y:
                    impact = o.vy
                    o.y, o.vy = o.rest_y, 0.0
                    limit = o.profile.drop_break_speed
                    if not o.damaged and limit is not None and impact > limit:
                        o.damaged = True
                        self._effort = 0.0
                        self._emit(DAMAGE, o, "dropped")

    # -------------------------------------------------------------- snapshot
    def snapshot(self) -> WorldSnapshot:
        views = []
        for o in self.objects:
            p = o.profile
            comp = 0.0
            phase = DAMAGED if o.damaged else NO_CONTACT
            if o is self._engaged:
                phase = self._phase
                span = max(1e-6, 1.0 - p.contact_threshold)
                comp = p.compressibility * min(1.0, max(0.0, (self.q - p.contact_threshold) / span))
            views.append(ObjectView(o.object_id, p.kind, o.x, o.y, p.radius, p.color,
                                    o.damaged, comp, phase, self.in_target(o)))
        if self._engaged is not None:
            e = next(v for v in views if v.object_id == self._engaged.object_id)
            gap = 2.0 * e.radius * (1.0 - e.compression)
        else:
            gap = PAD_OPEN_GAP * (1.0 - self.q) + 8.0
        return WorldSnapshot(
            t_ms=self.t_ms, pointer=self.pointer, q=self.q, pad_gap=gap,
            tracking_ok=self.tracking_ok, reliable=self.reliable,
            paused=self._lost_ms > 0.0, engaged_id=self.engaged_id,
            phase=self._phase, effort=self._effort, objects=tuple(views),
            target_zone=TARGET_ZONE, table_y=TABLE_Y,
            assist_enabled=self.assist_enabled, assist_gain=self.assist_gain,
            grab_radius_extra_frac=MAX_ASSIST_FRAC * self.assist_gain if self.assist_enabled else 0.0,
            recent_events=tuple(self._recent),
            slip_count=self.slip_count, overgrip_count=self.overgrip_count)
