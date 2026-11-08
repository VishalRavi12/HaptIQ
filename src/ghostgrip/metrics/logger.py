"""Session logging: three separate CSV tables, safe to restart (technical doc 10.3).

    with SessionLogger("data/local/sessions/s1", "s1", origin="synthetic") as log:
        log.begin_trial(1, "A", hand.t_ms)
        log.log_frame(hand, prediction, intent_state, decision, trial_id=1, condition="A")
        log.end_trial(outcome)                       # a contracts.TrialOutcome

Tables (one file each, in the session folder):
    frames.csv   one row per processed frame: pointer, closure, tracking, intent, prediction, assistance
    device.csv   one row per Arduino sensor packet, plus the cue that was sent (if any)
    trials.csv   one row per trial EVENT: start, end, or incomplete

Rules:
  * Nothing is fabricated. A missing value is written as an empty cell, never 0, and a float that
    is not finite is blanked the same way (counted in `blanked`).
  * Every row says whether its data is "real" or "synthetic"; nothing guesses it.
  * Restarting appends to the same files. Host clocks restart at 0, so each opening of a session
    gets a new `run` number; timestamps must not decrease within a run, and a row that would
    break that is skipped and counted in `rejected` (never silently written).
  * A trial that started but never ended is flagged `incomplete` on the next start, or when the
    logger is closed, so an interrupted trial is never mistaken for a finished one.
  * Files live under data/local/ (git-ignored). No images or video are ever written; use an
    anonymous session id.
"""
import csv
import math
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Tuple

from contracts import CONDITION_CODES, CueEvent, DeviceState, HandStateV2, Prediction, TrialOutcome

ORIGINS = ("real", "synthetic")
SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

FRAME_FIELDS = ("run", "session_id", "origin", "trial_id", "condition", "t_ms",
                "pointer_x", "pointer_y", "closure_q", "velocity_x", "velocity_y",
                "tracking_ok", "tracking_quality", "intent_state",
                "horizon_ms", "future_x", "future_y", "sigma_x", "sigma_y",
                "assist_gain", "assist_reason")
DEVICE_FIELDS = ("run", "session_id", "origin", "t_ms", "uptime_ms", "armed", "touch", "piezo",
                 "ax_g", "ay_g", "az_g", "rotary_raw", "last_ack_seq",
                 "cue_seq", "cue_kind", "cue_angle_deg")
TRIAL_FIELDS = ("run", "session_id", "origin", "trial_id", "condition", "event", "t_ms",
                "success", "fail_reason", "start_ms", "end_ms", "slip_count", "overgrip_count")
TRIAL_EVENTS = ("start", "end", "incomplete")

FILES = {"frames": ("frames.csv", FRAME_FIELDS),
         "device": ("device.csv", DEVICE_FIELDS),
         "trials": ("trials.csv", TRIAL_FIELDS)}


def read_table(path, fields: Tuple[str, ...]) -> List[Dict[str, str]]:
    """Read one of the CSV tables back as a list of dicts of strings. [] if it does not exist."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != fields:
            raise ValueError(f"{path.name} does not have the expected columns")
        return list(reader)


def read_session(folder) -> Dict[str, List[Dict[str, str]]]:
    folder = Path(folder)
    return {name: read_table(folder / filename, fields) for name, (filename, fields) in FILES.items()}


class SessionLogger:
    def __init__(self, folder, session_id: str, origin: str):
        if not SESSION_ID.match(str(session_id)):
            raise ValueError("session_id may only use letters, digits, - and _ (use an anonymous id)")
        if origin not in ORIGINS:
            raise ValueError(f"origin must be one of {ORIGINS}; say whether the data is real or synthetic")
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.session_id, self.origin = session_id, origin
        self.rejected = 0          # rows skipped because their timestamp went backwards
        self.blanked = 0           # non-finite numbers written as empty cells
        self.closed = False
        self._last_t: Dict[str, int] = {}
        self._open: Dict[int, str] = {}
        self._used_trials = set()      # trial ids begun in this run
        self._files, self._writers = {}, {}
        self.recovered: List[Tuple[int, int]] = []     # (run, trial_id) of trials flagged on restart

        existing = read_session(self.folder)
        self.run = 1 + max((int(r["run"]) for rows in existing.values() for r in rows), default=-1)
        for name, (filename, fields) in FILES.items():
            path = self.folder / filename
            fresh = not path.exists() or path.stat().st_size == 0
            if not fresh:
                read_table(path, fields)               # raises if the columns do not match
            handle = path.open("a", newline="")
            writer = csv.writer(handle)
            if fresh:
                writer.writerow(fields)
                handle.flush()
            self._files[name], self._writers[name] = handle, writer
        self._flag_interrupted(existing["trials"])

    # ------------------------------------------------------------------ writing
    def _format(self, value) -> str:
        if value is None:
            return ""
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, float):
            if not math.isfinite(value):
                self.blanked += 1
                return ""
            return f"{value:.6f}"
        return str(value)

    def _write(self, table: str, values: Iterable, t_ms: Optional[int]) -> bool:
        if self.closed:
            raise RuntimeError("the logger is closed")
        if t_ms is not None:
            last = self._last_t.get(table)
            if last is not None and t_ms < last:
                self.rejected += 1
                return False
            self._last_t[table] = t_ms
        self._writers[table].writerow([self._format(v) for v in values])
        self._files[table].flush()
        return True

    @staticmethod
    def _check_condition(condition):
        if condition is not None and condition not in CONDITION_CODES:
            raise ValueError(f"condition must be one of {CONDITION_CODES} or None, got {condition!r}")

    def log_frame(self, hand: HandStateV2, prediction: Optional[Prediction] = None,
                  intent_state=None, decision=None, trial_id: Optional[int] = None,
                  condition: Optional[str] = None) -> bool:
        """Log one processed frame. `intent_state` may be a string or an intent.IntentState;
        `decision` is an adapt.AssistDecision. Returns False if the row was rejected."""
        self._check_condition(condition)
        state = getattr(intent_state, "state", intent_state)
        p = prediction
        return self._write("frames", (
            self.run, self.session_id, self.origin, trial_id, condition, int(hand.t_ms),
            hand.pointer_xy[0], hand.pointer_xy[1], hand.closure_q,
            hand.velocity_xy[0], hand.velocity_xy[1], hand.tracking_ok, hand.tracking_quality, state,
            p.horizon_ms if p else None,
            p.future_pointer_xy[0] if p else None, p.future_pointer_xy[1] if p else None,
            p.sigma_xy[0] if p else None, p.sigma_xy[1] if p else None,
            decision.gain if decision else None, decision.reason if decision else None),
            int(hand.t_ms))

    def log_device(self, t_ms: int, device: DeviceState, cue: Optional[CueEvent] = None) -> bool:
        """Log one sensor packet at host time t_ms. Absent sensors (None) stay empty, never 0.
        Arduino uptime is its own clock and is stored as-is, never mixed with host time."""
        return self._write("device", (
            self.run, self.session_id, self.origin, int(t_ms), device.uptime_ms, device.armed,
            device.touch, device.piezo, device.ax_g, device.ay_g, device.az_g, device.rotary_raw,
            device.last_ack_seq,
            cue.seq if cue else None, cue.kind if cue else None,
            cue.safe_angle_deg if cue else None), int(t_ms))

    # ------------------------------------------------------------------ trials
    def begin_trial(self, trial_id: int, condition: str, t_ms: int) -> bool:
        self._check_condition(condition)
        if condition is None:
            raise ValueError("a trial needs a condition")
        if trial_id in self._used_trials:
            raise ValueError(f"trial id {trial_id} was already used in this run; ids must be unique "
                             "or two trials would be merged into one")
        written = self._write("trials", (self.run, self.session_id, self.origin, trial_id, condition,
                                         "start", int(t_ms), None, None, None, None, None, None), int(t_ms))
        if written:
            self._open[trial_id] = condition
            self._used_trials.add(trial_id)
        return written

    def end_trial(self, outcome: TrialOutcome) -> bool:
        """Record the finished trial (a contracts.TrialOutcome). Invalid ones are kept and labelled."""
        self._check_condition(outcome.condition)
        written = self._write("trials", (
            self.run, self.session_id, self.origin, outcome.trial_id, outcome.condition, "end",
            int(outcome.end_ms), outcome.success, outcome.fail_reason, int(outcome.start_ms),
            int(outcome.end_ms), outcome.slip_count, outcome.overgrip_count), int(outcome.end_ms))
        if written:
            self._open.pop(outcome.trial_id, None)
        return written

    def abandon_trial(self, trial_id: int, t_ms: Optional[int] = None, reason: str = "abandoned") -> bool:
        """Flag a trial that began but will never finish (the trace ended, the app was closed)."""
        condition = self._open.get(trial_id)
        if condition is None:
            return False
        written = self._write("trials", (self.run, self.session_id, self.origin, trial_id, condition,
                                         "incomplete", t_ms, None, reason, None, None, None, None),
                              None if t_ms is None else int(t_ms))
        if written:
            del self._open[trial_id]
        return written

    @property
    def open_trials(self) -> List[int]:
        return sorted(self._open)

    def _flag_interrupted(self, trial_rows: List[Dict[str, str]]) -> None:
        """On restart, mark trials from earlier runs that started but never ended."""
        state: Dict[Tuple[int, int], Dict[str, str]] = {}
        for row in trial_rows:
            key = (int(row["run"]), int(row["trial_id"]))
            if row["event"] == "start":
                state[key] = row
            elif row["event"] in ("end", "incomplete"):
                state.pop(key, None)
        for (run, trial_id), start in sorted(state.items()):
            self._writers["trials"].writerow([run, start["session_id"], start["origin"], trial_id,
                                              start["condition"], "incomplete", "", "", "interrupted",
                                              "", "", "", ""])
            self.recovered.append((run, trial_id))
        if state:
            self._files["trials"].flush()

    # ------------------------------------------------------------------ closing
    def close(self) -> None:
        """Flag any trial still open as incomplete, then close the files. Safe to call twice."""
        if self.closed:
            return
        for trial_id in list(self._open):
            self.abandon_trial(trial_id, reason="logger_closed")
        self.closed = True
        for handle in self._files.values():
            handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
