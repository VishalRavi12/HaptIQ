"""Grasp intent: what is the hand doing right now? (technical doc 4.3)

A rule-based state machine, the always-available baseline before any trained model:

    IDLE -> APPROACHING -> CLOSING -> CONTACT_CANDIDATE -> HOLDING -> RELEASING -> IDLE

    machine = IntentMachine(zones_for_scene("glass"))
    state = machine.update(hand_state)         # an IntentState, every frame
    state.state, state.object_id, state.changed

Guards (so noise cannot cause "transition spam"):
  * every transition must be wanted for `confirm_frames` consecutive usable frames
    (config.yaml: ai.grip_confirm_frames);
  * hysteresis: contact is entered at contact_q but only left below contact_q - hysteresis;
  * a cooldown after returning to IDLE stops an immediate re-trigger;
  * a tiny pinch alone never counts: the pointer must also be on or near the object;
  * tracking lost: the state freezes, and after loss_timeout_ms the machine goes back to IDLE
    (it mirrors the simulation's 300 ms freeze-then-release);
  * tracking_quality below min_quality: the hand is still followed, but such a frame can never
    start or strengthen a grasp, only end one. The simulation does the same (low quality
    blocks new grasps but does not drop one already held).

The rules are the simulation's own numbers (see zones_for_scene), but intent is decided here
from the hand alone: it does not read the simulation's events. A state label is not proof of
what the person intended (plan 4.3).
"""
from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple

from contracts import HandStateV2

IDLE = "IDLE"
APPROACHING = "APPROACHING"
CLOSING = "CLOSING"
CONTACT_CANDIDATE = "CONTACT_CANDIDATE"
HOLDING = "HOLDING"
RELEASING = "RELEASING"
STATES = (IDLE, APPROACHING, CLOSING, CONTACT_CANDIDATE, HOLDING, RELEASING)

DEFAULT_WORLD_SIZE = (960, 540)    # config.yaml simulation.width / height


@dataclass(frozen=True)
class GraspZone:
    """One graspable object, in the units intent needs (the target zone is a plain parameter)."""
    object_id: str
    center_xy: Tuple[float, float]   # normalised 0-1, same space as HandStateV2.pointer_xy
    radius_px: float                 # how close the pointer must be to count as touching
    contact_q: float                 # closure at which the pads touch the object
    hold_q: float                    # closure at which the grip is firm enough to hold

    def __post_init__(self):
        if not (len(self.center_xy) == 2 and all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in self.center_xy)):
            raise ValueError("center_xy must be two numbers between 0 and 1")
        if not (math.isfinite(self.radius_px) and self.radius_px > 0):
            raise ValueError("radius_px must be positive")
        if not (0.0 < self.contact_q < self.hold_q <= 1.0):
            raise ValueError("need 0 < contact_q < hold_q <= 1")


@dataclass(frozen=True)
class IntentParams:
    confirm_frames: int = 3          # consecutive usable frames before any transition
    min_quality: float = 0.5         # below this a frame is not used (matches the simulation)
    approach_factor: float = 3.0     # "near" = within this many zone radii
    leave_factor: float = 1.3        # hysteresis: leave "near" only beyond approach_factor * this
    approach_speed_px: float = 40.0  # px/s toward the object that counts as heading for it
    close_rate: float = 0.15         # dq/dt (per second) that counts as closing
    open_q: float = 0.15             # below this the hand counts as open
    hysteresis: float = 0.05         # how far below contact_q before contact is lost
    release_timeout_ms: int = 1000   # RELEASING that goes nowhere falls back to IDLE
    cooldown_ms: int = 250           # after returning to IDLE, wait before the next approach
    loss_timeout_ms: int = 300       # unusable this long: back to IDLE (simulation: 300 ms)
    max_gap_ms: int = 500            # a longer silence discards partial evidence

    def __post_init__(self):
        if self.confirm_frames < 1:
            raise ValueError("confirm_frames must be at least 1")
        if not 0.0 <= self.min_quality <= 1.0:
            raise ValueError("min_quality must be between 0 and 1")
        for name in ("approach_factor", "leave_factor", "approach_speed_px", "close_rate"):
            if not (math.isfinite(getattr(self, name)) and getattr(self, name) > 0):
                raise ValueError(f"{name} must be positive")
        if self.leave_factor < 1.0:
            raise ValueError("leave_factor must be at least 1")
        if not (0.0 <= self.open_q < 1.0 and 0.0 <= self.hysteresis < 1.0):
            raise ValueError("open_q and hysteresis must be between 0 and 1")
        for name in ("release_timeout_ms", "cooldown_ms", "loss_timeout_ms", "max_gap_ms"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")


@dataclass(frozen=True)
class IntentState:
    state: str
    object_id: Optional[str]   # the object being approached or held; None while IDLE
    since_ms: int              # when this state was entered
    changed: bool              # True only on the frame the state changed
    reason: str                # why: "approach", "contact", "tracking_lost", "frozen", ...


class IntentMachine:
    def __init__(self, zones: Sequence[GraspZone], params: Optional[IntentParams] = None,
                 world_size: Tuple[int, int] = DEFAULT_WORLD_SIZE):
        zones = list(zones)
        if not zones:
            raise ValueError("at least one GraspZone is needed")
        if len({z.object_id for z in zones}) != len(zones):
            raise ValueError("object ids must be unique")
        if not (len(world_size) == 2 and all(v > 0 for v in world_size)):
            raise ValueError("world_size must be two positive numbers")
        self.zones = {z.object_id: z for z in zones}
        self.params = params or IntentParams()
        self.world_w, self.world_h = world_size
        self.ignored_samples = 0
        self.reset()

    @classmethod
    def from_config(cls, zones: Sequence[GraspZone], config) -> "IntentMachine":
        """Use ai.grip_confirm_frames and the simulation size from a loaded Config."""
        params = IntentParams(confirm_frames=config.ai.grip_confirm_frames)
        return cls(zones, params, (config.simulation.width, config.simulation.height))

    def reset(self) -> None:
        self._state = IDLE
        self._zone: Optional[GraspZone] = None
        self._since_ms = 0
        self._last_t: Optional[int] = None
        self._last_usable_t: Optional[int] = None
        self._unusable_since: Optional[int] = None
        self._cooldown_until = 0
        self._counters: Dict[str, int] = {}

    @property
    def state(self) -> str:
        return self._state

    # ------------------------------------------------------------------ main entry point
    def update(self, hand: HandStateV2) -> IntentState:
        t = int(hand.t_ms)
        if self._last_t is not None and t <= self._last_t:
            self.ignored_samples += 1            # repeated or out-of-order frame
            return self._snapshot(False, "ignored")
        self._last_t = t

        pointer_ok = len(hand.pointer_xy) == 2 and all(math.isfinite(v) for v in hand.pointer_xy)
        numbers_ok = all(math.isfinite(v) for v in (hand.closure_q, hand.dq_dt, *hand.velocity_xy))
        if not (hand.tracking_ok and pointer_ok and numbers_ok):
            return self._unusable(t)
        reliable = math.isfinite(hand.tracking_quality) and hand.tracking_quality >= self.params.min_quality

        self._unusable_since = None
        if self._last_usable_t is not None and t - self._last_usable_t > self.params.max_gap_ms:
            self._counters.clear()               # old evidence does not carry across a long gap
        self._last_usable_t = t
        return self._step(hand, t, reliable)

    # ------------------------------------------------------------------ internals
    def _snapshot(self, changed: bool, reason: str) -> IntentState:
        return IntentState(self._state, self._zone.object_id if self._zone else None,
                           self._since_ms, changed, reason)

    def _unusable(self, t: int) -> IntentState:
        self._counters.clear()                   # half-finished evidence is not trusted
        if self._unusable_since is None:
            self._unusable_since = t
        if self._state != IDLE and t - self._unusable_since >= self.params.loss_timeout_ms:
            return self._go(IDLE, t, "tracking_lost")
        return self._snapshot(False, "frozen")

    def _go(self, new: str, t: int, reason: str, zone: Optional[GraspZone] = None) -> IntentState:
        self._state = new
        self._since_ms = t
        self._counters.clear()
        if new == IDLE:
            self._zone = None
            self._cooldown_until = t + self.params.cooldown_ms
        elif zone is not None:
            self._zone = zone
        return self._snapshot(True, reason)

    def _persist(self, name: str, wanted: bool) -> bool:
        """True once `wanted` has held for confirm_frames consecutive usable frames."""
        self._counters[name] = self._counters.get(name, 0) + 1 if wanted else 0
        return self._counters[name] >= self.params.confirm_frames

    def _geometry(self, hand: HandStateV2, zone: GraspZone):
        """Distance to the zone centre in px, and the speed toward it in px/s."""
        px, py = hand.pointer_xy[0] * self.world_w, hand.pointer_xy[1] * self.world_h
        dx, dy = zone.center_xy[0] * self.world_w - px, zone.center_xy[1] * self.world_h - py
        dist = math.hypot(dx, dy)
        if dist < 1e-9:
            return 0.0, 0.0
        vx, vy = hand.velocity_xy[0] * self.world_w, hand.velocity_xy[1] * self.world_h
        return dist, (vx * dx + vy * dy) / dist

    def _step(self, hand: HandStateV2, t: int, reliable: bool) -> IntentState:
        p, q, dq = self.params, hand.closure_q, hand.dq_dt

        if self._state == IDLE:
            return self._from_idle(hand, t, reliable)

        z = self._zone
        dist, toward = self._geometry(hand, z)
        inside = dist <= z.radius_px
        near_limit = z.radius_px * p.approach_factor

        if self._state == APPROACHING:
            if self._persist("leave", dist > near_limit * p.leave_factor):
                return self._go(IDLE, t, "left_area")
            if self._persist("contact", reliable and inside and q >= z.contact_q):
                return self._go(CONTACT_CANDIDATE, t, "contact")     # closed fast: skip CLOSING
            if self._persist("close", reliable and q >= 0.02 and dq >= p.close_rate and q < z.contact_q):
                return self._go(CLOSING, t, "closing")

        elif self._state == CLOSING:
            if self._persist("contact", reliable and inside and q >= z.contact_q):
                return self._go(CONTACT_CANDIDATE, t, "contact")
            # Only an actively opening (or fully open) hand cancels. q below open_q alone is
            # just the start of a closing movement, not a reopening.
            if self._persist("reopen", dq <= -p.close_rate or q < 0.01):
                return self._go(APPROACHING, t, "reopened")
            if self._persist("leave", dist > near_limit * p.leave_factor):
                return self._go(IDLE, t, "left_area")

        elif self._state == CONTACT_CANDIDATE:
            if self._persist("hold", reliable and q >= z.hold_q):
                return self._go(HOLDING, t, "firm_grip")
            if self._persist("lost_contact", (not inside) or q < z.contact_q - p.hysteresis):
                return self._go(CLOSING if q >= p.open_q else APPROACHING, t, "lost_contact")

        elif self._state == HOLDING:
            # No "inside" check: carrying the object moves the hand far from where it started.
            if self._persist("release", q < z.contact_q - p.hysteresis):
                return self._go(RELEASING, t, "relaxing")

        elif self._state == RELEASING:
            if self._persist("open", q <= p.open_q):
                return self._go(IDLE, t, "opened")
            if self._persist("regrip", reliable and q >= z.hold_q):
                return self._go(HOLDING, t, "regrip")
            if t - self._since_ms >= p.release_timeout_ms:
                return self._go(IDLE, t, "release_timeout")

        return self._snapshot(False, "steady")

    def _from_idle(self, hand: HandStateV2, t: int, reliable: bool) -> IntentState:
        p = self.params
        if not reliable:
            self._counters.clear()
            return self._snapshot(False, "unreliable")
        if t < self._cooldown_until:
            self._counters.clear()
            return self._snapshot(False, "cooldown")
        best, best_dist = None, float("inf")
        for zone in self.zones.values():
            dist, toward = self._geometry(hand, zone)
            near = dist <= zone.radius_px * p.approach_factor
            if near and (dist <= zone.radius_px or toward >= p.approach_speed_px) and dist < best_dist:
                best, best_dist = zone, dist
        key = f"approach:{best.object_id}" if best else "approach"
        # evidence for other zones must not add up, so drop every other approach counter
        self._counters = {k: v for k, v in self._counters.items() if k == key}
        if self._persist(key, best is not None):
            return self._go(APPROACHING, t, "approach", best)
        return self._snapshot(False, "idle")


# ====================================================================== helpers
def zones_for_scene(scene: str = "glass") -> List[GraspZone]:
    """GraspZones for one of the simulation's scenes, from the simulation's own numbers."""
    from ghostgrip.simulation.objects import PROFILES
    from ghostgrip.simulation.world import GRAB_MARGIN, SCENES, TABLE_Y, WORLD_H, WORLD_W
    if scene not in SCENES:
        raise ValueError(f"unknown scene {scene!r}; choose from {list(SCENES)}")
    zones = []
    for kind, x_px in SCENES[scene]:
        profile = PROFILES[kind]
        zones.append(GraspZone(
            object_id=kind,
            center_xy=(x_px / WORLD_W, (TABLE_Y - profile.radius) / WORLD_H),
            radius_px=profile.radius + GRAB_MARGIN,
            contact_q=profile.contact_threshold,
            hold_q=min(1.0, profile.q_for_effort(profile.hold_threshold))))
    return zones


def timeline(states: Sequence[HandStateV2], machine: IntentMachine) -> List[Tuple[int, str, str, str]]:
    """Run a trace and return (t_ms, from_state, to_state, reason) for every transition."""
    log, previous = [], machine.state
    for hand in states:
        result = machine.update(hand)
        if result.changed:
            log.append((hand.t_ms, previous, result.state, result.reason))
            previous = result.state
    return log


def main(argv=None):
    """Print the state timeline for synthetic scenarios:  PYTHONPATH=src python -m ghostgrip.ai.intent [name ...]"""
    import argparse
    from .synthetic import SCENARIOS, make_scenario
    parser = argparse.ArgumentParser(description="Intent timeline on synthetic traces (glass scene).")
    parser.add_argument("scenarios", nargs="*", default=["clean"], help=f"any of {sorted(SCENARIOS)} or 'all'")
    args = parser.parse_args(argv)
    names = list(SCENARIOS) if "all" in args.scenarios else args.scenarios
    for name in names:
        machine = IntentMachine(zones_for_scene("glass"))
        log = timeline(make_scenario(name), machine)
        print(f"{name}: {len(log)} transitions")
        for t, old, new, reason in log:
            print(f"  {t:>5} ms  {old:<18} -> {new:<18} ({reason})")


if __name__ == "__main__":
    main()
