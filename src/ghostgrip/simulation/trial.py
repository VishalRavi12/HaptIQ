"""Trial logic and experimental conditions (technical document 9.2, 10.1).

A trial: the hand starts open and tracked -> grasp the object -> hold it
(at least DWELL_MS in a stable grasp) -> transport it -> release it so it comes
to rest inside the target zone, intact, before the timeout.

Failures: damaged (over-grip or hard drop), tracking_lost (invalid trial),
timeout. Dropping it outside the target is not a failure by itself: the user
may pick it up again until the timeout.
"""
import csv
from dataclasses import astuple, dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional

from contracts import (CONDITION_AI, CONDITION_BASELINE, CONDITION_BOTH, CONDITION_HAPTIC,
                       DAMAGE, RELEASE, TrialOutcome)
from contracts import HandStateV2 as HandState
from contracts import SimEventV2 as SimEvent
from ghostgrip.simulation.world import REARM_Q, World


@dataclass(frozen=True)
class Condition:
    code: str
    label: str
    ai_assist: bool
    haptic: bool


CONDITIONS: Dict[str, Condition] = {c.code: c for c in (
    Condition(CONDITION_BASELINE, "Baseline", ai_assist=False, haptic=False),
    Condition(CONDITION_AI, "AI only", ai_assist=True, haptic=False),
    Condition(CONDITION_HAPTIC, "Haptic only", ai_assist=False, haptic=True),
    Condition(CONDITION_BOTH, "AI + Haptic", ai_assist=True, haptic=True),
)}

READY, RUNNING, FINISHED = "READY", "RUNNING", "FINISHED"
DEFAULT_TIMEOUT_MS = 30_000
DWELL_MS = 300.0


class TrialRunner:
    def __init__(self, object_id: str, condition: Condition = CONDITIONS["A"],
                 session_id: str = "anonymous", timeout_ms: int = DEFAULT_TIMEOUT_MS,
                 dwell_ms: float = DWELL_MS, first_trial_id: int = 1):
        self.object_id = object_id
        self.condition = condition
        self.session_id = session_id
        self.timeout_ms = timeout_ms
        self.dwell_ms = dwell_ms
        self.trial_id = max(1, int(first_trial_id)) - 1   # incremented when a trial starts
        self.outcomes: List[TrialOutcome] = []
        self.next_trial()

    def next_trial(self) -> None:
        """Arm a new trial. Call World.reset() alongside this."""
        self.state = READY
        self.outcome: Optional[TrialOutcome] = None
        self.start_ms = 0
        self._slip0 = self._over0 = 0
        self._held0 = 0.0

    def elapsed_ms(self, now_ms: int) -> int:
        if self.state == RUNNING:
            return max(0, now_ms - self.start_ms)
        if self.outcome is not None:
            return self.outcome.end_ms - self.outcome.start_ms
        return 0

    def update(self, world: World, events: List[SimEvent], hand: HandState) -> Optional[TrialOutcome]:
        now = int(hand.t_ms)
        if self.state == READY:
            if hand.tracking_ok and world.q < REARM_Q:
                self.state = RUNNING
                self.trial_id += 1
                self.start_ms = now
                self._slip0, self._over0 = world.slip_count, world.overgrip_count
                self._held0 = world.object(self.object_id).held_ms
            return None
        if self.state != RUNNING:
            return None
        for ev in events:
            if ev.object_id != self.object_id:
                continue
            reason = world.event_reason(ev.event_id)
            if ev.kind == DAMAGE:
                return self._finish(world, False, f"damaged_{reason}", now)
            if ev.kind == RELEASE and reason == "tracking_lost":
                return self._finish(world, False, "tracking_lost", now)
        obj = world.object(self.object_id)
        if (not obj.damaged and world.in_target(obj)
                and obj.held_ms - self._held0 >= self.dwell_ms):
            return self._finish(world, True, "", now)
        if now - self.start_ms >= self.timeout_ms:
            return self._finish(world, False, "timeout", now)
        return None

    def _finish(self, world: World, success: bool, reason: str, now: int) -> TrialOutcome:
        self.state = FINISHED
        self.outcome = TrialOutcome(
            session_id=self.session_id, trial_id=self.trial_id,
            condition=self.condition.code, success=success, fail_reason=reason,
            start_ms=self.start_ms, end_ms=now,
            slip_count=world.slip_count - self._slip0,
            overgrip_count=world.overgrip_count - self._over0)
        self.outcomes.append(self.outcome)
        return self.outcome


# ------------------------------------------------------------------ outcome log
# STAND-IN until Member D's metrics logger exists: one CSV row per finished trial,
# columns in TrialOutcome field order. Real sessions go to the gitignored data/local/.
OUTCOME_COLUMNS = [f.name for f in fields(TrialOutcome)]


def append_outcome_csv(path, outcome: TrialOutcome) -> Path:
    """Append one TrialOutcome; writes the header first if the file is new."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(OUTCOME_COLUMNS)
        w.writerow(astuple(outcome))
    return path


def last_trial_id(path, session_id: str) -> int:
    """Highest trial_id already logged for this session (0 if none), so a new run
    of the app continues the numbering instead of repeating it."""
    path = Path(path)
    if not path.exists():
        return 0
    ids = [o.trial_id for o in load_outcomes_csv(path) if o.session_id == session_id]
    return max(ids, default=0)


def load_outcomes_csv(path) -> List[TrialOutcome]:
    """Read a file written by append_outcome_csv back into TrialOutcome objects."""
    out = []
    with Path(path).open(newline="") as f:
        for r in csv.DictReader(f):
            out.append(TrialOutcome(
                session_id=r["session_id"], trial_id=int(r["trial_id"]),
                condition=r["condition"], success=r["success"] == "True",
                fail_reason=r["fail_reason"], start_ms=int(r["start_ms"]),
                end_ms=int(r["end_ms"]), slip_count=int(r["slip_count"]),
                overgrip_count=int(r["overgrip_count"])))
    return out
