"""Input sources that produce HandState. The World never knows which one is used.

- MouseKeyboardAdapter: Member B's stand-in for the camera (plan section 2).
  Pure logic, no pygame import; the UI feeds it mouse/key state.
- TraceRecorder / TracePlayer: save and replay timestamped HandState rows for
  deterministic demos and tests (NFR-02).
- Later, Member A's live camera generator plugs in at the same place.
"""
import csv
from dataclasses import replace
from pathlib import Path
import random
from typing import Iterable, List, Optional

from contracts import HandStateV2 as HandState

TRACE_COLUMNS = ["t_ms", "x", "y", "q", "dq_dt", "vx", "vy",
                 "stable_score", "tracking_ok", "tracking_quality"]


class MouseKeyboardAdapter:
    """Turns mouse/keyboard into a continuous closure q, not a click.

    - closing held (left mouse / W): q target rises at close_rate per second
    - opening held (right mouse / S): q target falls at open_rate per second
    - wheel notches: q target +/- wheel_step, and it STAYS there
    - open_hand(): q target -> 0 (quick release)
    q follows its target at most follow_rate per second, so dq_dt is realistic.
    Toggles simulate failures: tracking loss (T05), low quality (T06), jitter (T07).
    """

    def __init__(self, close_rate=0.6, open_rate=0.6, wheel_step=0.04,
                 follow_rate=3.0, rng: Optional[random.Random] = None):
        self.close_rate = close_rate
        self.open_rate = open_rate
        self.wheel_step = wheel_step
        self.follow_rate = follow_rate
        self.rng = rng or random.Random(0)
        self.q_target = 0.0
        self.q = 0.0
        self.pointer = (0.5, 0.5)
        self.tracking_lost = False
        self.low_quality = False
        self.jitter = False
        self._last: Optional[HandState] = None

    # ---- inputs from the UI ----
    def set_pointer_px(self, px: float, py: float, view_w: float, view_h: float) -> None:
        self.pointer = (min(1.0, max(0.0, px / view_w)), min(1.0, max(0.0, py / view_h)))

    def wheel(self, notches: int) -> None:
        self.q_target = min(1.0, max(0.0, self.q_target + notches * self.wheel_step))

    def open_hand(self) -> None:
        self.q_target = 0.0

    # ---- output ----
    def sample(self, t_ms: int, closing: bool = False, opening: bool = False) -> HandState:
        prev = self._last
        dt = 0.0 if prev is None else max(0.0, (t_ms - prev.t_ms) / 1000.0)
        if closing:
            self.q_target = min(1.0, self.q_target + self.close_rate * dt)
        if opening:
            self.q_target = max(0.0, self.q_target - self.open_rate * dt)
        step = self.follow_rate * dt
        self.q += max(-step, min(step, self.q_target - self.q))

        if self.tracking_lost:
            last_xy = prev.pointer_xy if prev else self.pointer
            last_q = prev.closure_q if prev else self.q
            state = HandState(t_ms, last_xy, last_q, 0.0, (0.0, 0.0), 0.0, False, 0.0)
            self._last = state
            return state

        x, y, q = self.pointer[0], self.pointer[1], self.q
        stable = 1.0
        if self.jitter:
            x += self.rng.gauss(0.0, 0.006)
            y += self.rng.gauss(0.0, 0.006)
            q += self.rng.gauss(0.0, 0.03)
            stable = 0.4
        x, y, q = (min(1.0, max(0.0, v)) for v in (x, y, q))
        vx = vy = dq = 0.0
        if prev is not None and prev.tracking_ok and dt > 0:
            vx = (x - prev.pointer_xy[0]) / dt
            vy = (y - prev.pointer_xy[1]) / dt
            dq = (q - prev.closure_q) / dt
        quality = 0.3 if self.low_quality else 1.0
        state = HandState(t_ms, (x, y), q, dq, (vx, vy), stable, True, quality)
        self._last = state
        return state


# ------------------------------------------------------------------ traces
def save_trace(path, states: Iterable[HandState]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(TRACE_COLUMNS)
        for s in states:
            w.writerow([s.t_ms, f"{s.pointer_xy[0]:.6f}", f"{s.pointer_xy[1]:.6f}",
                        f"{s.closure_q:.6f}", f"{s.dq_dt:.6f}",
                        f"{s.velocity_xy[0]:.6f}", f"{s.velocity_xy[1]:.6f}",
                        f"{s.stable_score:.4f}", int(s.tracking_ok), f"{s.tracking_quality:.4f}"])
    return path


# Member A's recorder (vision/trace.py, `vision_preview.py --record`) writes the same values
# under longer column names. load_trace reads either layout, so a real hand recording
# replays in the simulation with no conversion step.
VISION_COLUMNS = {"x": "pointer_x", "y": "pointer_y", "q": "closure_q",
                  "vx": "velocity_x", "vy": "velocity_y"}


def load_trace(path) -> List[HandState]:
    rows = []
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        names = set(reader.fieldnames or ())
        if {"x", "y", "q"} <= names:
            col = {k: k for k in VISION_COLUMNS}                 # simulation layout
        elif set(VISION_COLUMNS.values()) <= names:
            col = dict(VISION_COLUMNS)                           # Member A's layout
        else:
            raise ValueError(f"trace {path}: unknown columns {sorted(names)}")
        for r in reader:
            rows.append(HandState(
                int(float(r["t_ms"])), (float(r[col["x"]]), float(r[col["y"]])), float(r[col["q"]]),
                float(r.get("dq_dt") or 0.0),
                (float(r.get(col["vx"]) or 0.0), float(r.get(col["vy"]) or 0.0)),
                float(r.get("stable_score") or 1.0),
                str(r["tracking_ok"]).strip().lower() in ("1", "true"),
                float(r.get("tracking_quality") or 1.0)))
    if not rows:
        raise ValueError(f"trace {path} has no rows")
    return rows


class TraceRecorder:
    def __init__(self):
        self.rows: List[HandState] = []

    def append(self, s: HandState) -> None:
        self.rows.append(s)

    def save(self, path) -> Path:
        t0 = self.rows[0].t_ms if self.rows else 0
        return save_trace(path, [replace(s, t_ms=s.t_ms - t0) for s in self.rows])


class TracePlayer:
    """Replays rows by time (zero-order hold). Timestamps are re-based to 0."""

    def __init__(self, states: List[HandState], loop: bool = False):
        if not states:
            raise ValueError("empty trace")
        t0 = states[0].t_ms
        self.rows = [replace(s, t_ms=s.t_ms - t0) for s in states]
        self.loop = loop
        self.duration_ms = self.rows[-1].t_ms
        self._i = 0

    @classmethod
    def from_csv(cls, path, loop: bool = False) -> "TracePlayer":
        return cls(load_trace(path), loop)

    def finished(self, t_ms: int) -> bool:
        return not self.loop and t_ms > self.duration_ms

    def sample(self, t_ms: int) -> HandState:
        if self.loop and self.duration_ms > 0:
            t_ms = t_ms % (self.duration_ms + 1)
            if self._i and self.rows[self._i].t_ms > t_ms:
                self._i = 0
        while self._i + 1 < len(self.rows) and self.rows[self._i + 1].t_ms <= t_ms:
            self._i += 1
        return self.rows[self._i]
