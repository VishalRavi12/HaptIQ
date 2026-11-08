"""Results from the session tables (technical doc 10.2, Phase 1 plan D5).

Definitions, kept honest:
  * A trial is VALID when it finished and its tracking was not lost. A "tracking_lost" trial is
    INVALID, not a failure (plan: predefine what is valid; do not hide or reclassify). Interrupted
    trials are INCOMPLETE. Both are counted and reported, never dropped.
  * success rate    = successful trials / valid trials
  * task time       = end - start of SUCCESSFUL valid trials: median and range
  * slip rate       = valid trials with at least one simulated slip / valid trials
  * over-grip rate  = valid trials with at least one simulated over-grip / valid trials
  * prediction error is recomputed from the frame log with predict.ErrorTracker (same method as
    the live check), against the observed pointer at t + horizon.
  * Real and synthetic data are NEVER pooled: every summary is per (origin, condition).
  * Slips and over-grips are properties of the simulation model, not of a real object.

Nothing here invents a number. If there is nothing to compute, the value is None.
"""
from dataclasses import dataclass
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from contracts import CONDITION_CODES, HandStateV2, Prediction
from .logger import read_table, FRAME_FIELDS, TRIAL_FIELDS
from ..ai.predict import ErrorTracker, LOST_SIGMA


# ====================================================================== trials
@dataclass(frozen=True)
class TrialRecord:
    run: int
    session_id: str
    origin: str
    trial_id: int
    condition: str
    status: str                 # "complete" or "incomplete"
    success: bool
    fail_reason: str
    start_ms: Optional[int]
    end_ms: Optional[int]
    slip_count: int
    overgrip_count: int

    @property
    def valid(self) -> bool:
        return self.status == "complete" and self.fail_reason != "tracking_lost"

    @property
    def duration_ms(self) -> Optional[int]:
        if self.start_ms is None or self.end_ms is None:
            return None
        return self.end_ms - self.start_ms


def _int(text: str, default: Optional[int] = None) -> Optional[int]:
    return default if text == "" else int(float(text))


def trial_records(rows: Iterable[Dict[str, str]]) -> List[TrialRecord]:
    """Merge trials.csv event rows (start / end / incomplete) into one record per trial."""
    started: Dict[Tuple[int, int], Dict[str, str]] = {}
    done: Dict[Tuple[int, int], TrialRecord] = {}
    for row in rows:
        key = (int(row["run"]), int(row["trial_id"]))
        event = row["event"]
        if event == "start":
            started[key] = row
        elif event == "end":
            done[key] = TrialRecord(
                key[0], row["session_id"], row["origin"], key[1], row["condition"], "complete",
                row["success"] == "1", row["fail_reason"], _int(row["start_ms"]), _int(row["end_ms"]),
                _int(row["slip_count"], 0), _int(row["overgrip_count"], 0))
            started.pop(key, None)
        elif event == "incomplete":
            base = started.pop(key, None) or row
            done[key] = TrialRecord(
                key[0], row["session_id"], row["origin"], key[1], row["condition"], "incomplete",
                False, row["fail_reason"] or "interrupted", _int(base["t_ms"]), None, 0, 0)
    for key, row in started.items():             # never closed, never recovered: still incomplete
        done[key] = TrialRecord(key[0], row["session_id"], row["origin"], key[1], row["condition"],
                                "incomplete", False, "interrupted", _int(row["t_ms"]), None, 0, 0)
    return [done[k] for k in sorted(done)]


@dataclass(frozen=True)
class ConditionSummary:
    origin: str
    condition: str
    trials: int                       # everything logged for this condition
    valid: int
    tracking_lost: int                # invalid: not counted as failures
    incomplete: int                   # invalid: interrupted
    successes: int
    success_rate: Optional[float]     # successes / valid, None when valid == 0
    median_time_ms: Optional[float]   # successful valid trials only
    min_time_ms: Optional[int]
    max_time_ms: Optional[int]
    timeouts: int
    slip_rate: Optional[float]
    overgrip_rate: Optional[float]
    failures: Dict[str, int]          # valid failed trials by reason


def summarize_trials(records: Sequence[TrialRecord]) -> List[ConditionSummary]:
    """One summary per (origin, condition), real and synthetic kept apart."""
    groups: Dict[Tuple[str, str], List[TrialRecord]] = {}
    for record in records:
        groups.setdefault((record.origin, record.condition), []).append(record)
    order = {c: i for i, c in enumerate(CONDITION_CODES)}
    summaries = []
    for (origin, condition) in sorted(groups, key=lambda k: (k[0], order.get(k[1], 99))):
        group = groups[(origin, condition)]
        valid = [r for r in group if r.valid]
        wins = [r for r in valid if r.success]
        times = [r.duration_ms for r in wins if r.duration_ms is not None]
        failures: Dict[str, int] = {}
        for r in valid:
            if not r.success:
                failures[r.fail_reason or "unknown"] = failures.get(r.fail_reason or "unknown", 0) + 1
        n = len(valid)
        summaries.append(ConditionSummary(
            origin=origin, condition=condition, trials=len(group), valid=n,
            tracking_lost=sum(1 for r in group if r.status == "complete" and r.fail_reason == "tracking_lost"),
            incomplete=sum(1 for r in group if r.status == "incomplete"),
            successes=len(wins), success_rate=len(wins) / n if n else None,
            median_time_ms=median(times) if times else None,
            min_time_ms=min(times) if times else None, max_time_ms=max(times) if times else None,
            timeouts=failures.get("timeout", 0),
            slip_rate=sum(1 for r in valid if r.slip_count >= 1) / n if n else None,
            overgrip_rate=sum(1 for r in valid if r.overgrip_count >= 1) / n if n else None,
            failures=failures))
    return summaries


# ====================================================================== prediction error
def prediction_error(frame_rows: Iterable[Dict[str, str]]) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Mean forecast error per (origin, condition), recomputed from frames.csv.

    Rows are replayed in order through predict.ErrorTracker, one tracker per (run, session), so
    the forecast is compared with the pointer actually observed at t + horizon. Frames logged
    without a prediction or condition are skipped for scoring."""
    trackers: Dict[Tuple[int, str, str, str], ErrorTracker] = {}
    for row in frame_rows:
        if row["future_x"] == "" or row["condition"] == "" or row["horizon_ms"] == "":
            continue
        key = (int(row["run"]), row["session_id"], row["origin"], row["condition"])
        tracker = trackers.setdefault(key, ErrorTracker())
        t = int(row["t_ms"])
        ok = row["tracking_ok"] == "1"
        px, py = float(row["pointer_x"]), float(row["pointer_y"])
        hand = HandStateV2(t, (px, py), float(row["closure_q"] or 0.0), 0.0,
                           (float(row["velocity_x"] or 0.0), float(row["velocity_y"] or 0.0)),
                           1.0 if ok else 0.0, ok, float(row["tracking_quality"] or 0.0))
        sigma = (float(row["sigma_x"] or LOST_SIGMA), float(row["sigma_y"] or LOST_SIGMA))
        prediction = Prediction(t, int(row["horizon_ms"]), (float(row["future_x"]), float(row["future_y"])), sigma)
        tracker.record(hand, prediction)
    merged: Dict[Tuple[str, str], Dict[str, float]] = {}
    for (_, _, origin, condition), tracker in trackers.items():
        if tracker.scored == 0:
            continue
        slot = merged.setdefault((origin, condition), {"scored": 0, "kalman": 0.0, "no_movement": 0.0,
                                                         "constant_velocity": 0.0, "inside": 0.0})
        for method in ("kalman", "no_movement", "constant_velocity"):
            slot[method] += tracker.mean_errors()[method] * tracker.scored
        slot["inside"] += tracker.calibration() * tracker.scored
        slot["scored"] += tracker.scored
    result = {}
    for key, slot in merged.items():
        n = slot["scored"]
        result[key] = {"scored": int(n), "kalman": slot["kalman"] / n,
                       "no_movement": slot["no_movement"] / n,
                       "constant_velocity": slot["constant_velocity"] / n,
                       "within_2_sigma": slot["inside"] / n}
    return result


# ====================================================================== assistance availability
@dataclass(frozen=True)
class AssistAvailability:
    origin: str
    condition: str
    frames: int                 # all logged frames of this condition
    permitted: int              # frames where the condition allows assistance (not "condition_off")
    active: int                 # permitted frames that actually had a gain above 0
    reasons: Dict[str, int]     # permitted frames by reason ("ok", "tracking_lost", ...)

    @property
    def share_active(self) -> Optional[float]:
        return self.active / self.permitted if self.permitted else None


def assist_availability(frame_rows: Iterable[Dict[str, str]]) -> List[AssistAvailability]:
    """How often assistance was actually on, per (origin, condition), from frames.csv.

    A condition with AI assistance can still have it off for stretches (tracking lost, low quality,
    a stale or unknown forecast, or the predictor losing to a baseline). Without this, a comparison
    of A with B or D would not say how much assistance B and D really got. Conditions that never
    allow assistance (A, C) report share None."""
    counts: Dict[Tuple[str, str], Dict] = {}
    for row in frame_rows:
        if row["condition"] == "":
            continue
        slot = counts.setdefault((row["origin"], row["condition"]),
                                 {"frames": 0, "permitted": 0, "active": 0, "reasons": {}})
        slot["frames"] += 1
        reason = row["assist_reason"]
        if reason in ("", "condition_off", "assist_off_in_condition"):
            continue
        slot["permitted"] += 1
        slot["reasons"][reason] = slot["reasons"].get(reason, 0) + 1
        if row["assist_gain"] != "" and float(row["assist_gain"]) > 0.0:
            slot["active"] += 1
    order = {c: i for i, c in enumerate(CONDITION_CODES)}
    return [AssistAvailability(origin, condition, v["frames"], v["permitted"], v["active"], v["reasons"])
            for (origin, condition), v in sorted(counts.items(), key=lambda kv: (kv[0][0], order.get(kv[0][1], 99)))]


# ====================================================================== intent F1
def classification_report(truth: Sequence[str], predicted: Sequence[str],
                          labels: Optional[Sequence[str]] = None) -> Dict:
    """Confusion matrix, per-class precision / recall / F1 and macro F1 for labelled intent data.

    `truth` must come from a human annotating recorded action windows; this code never makes
    labels. An undefined ratio is None, not 0; a class with examples that is never predicted
    correctly scores F1 0 and counts in the macro average."""
    if len(truth) != len(predicted):
        raise ValueError("truth and predicted must have the same length")
    if not truth:
        raise ValueError("no labelled examples")
    classes = list(labels) if labels is not None else sorted(set(truth) | set(predicted))
    confusion = {t: {p: 0 for p in classes} for t in classes}
    for t, p in zip(truth, predicted):
        if t not in confusion or p not in confusion[t]:
            raise ValueError(f"label {t!r} / {p!r} is not in the label set {classes}")
        confusion[t][p] += 1
    per_class, f1s = {}, []
    for c in classes:
        tp = confusion[c][c]
        predicted_c = sum(confusion[t][c] for t in classes)
        support = sum(confusion[c].values())
        precision = tp / predicted_c if predicted_c else None
        recall = tp / support if support else None
        if support == 0:
            f1 = None                       # no examples of this class: nothing to score or average
        elif precision is None or precision + recall == 0:
            f1 = 0.0                        # it exists but was never predicted correctly: that is a 0, not a gap
        else:
            f1 = 2 * precision * recall / (precision + recall)
        per_class[c] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        if f1 is not None and support:
            f1s.append(f1)
    correct = sum(confusion[c][c] for c in classes)
    return {"labels": classes, "confusion": confusion, "per_class": per_class,
            "macro_f1": sum(f1s) / len(f1s) if f1s else None,
            "accuracy": correct / len(truth), "n": len(truth)}


# ====================================================================== loading and formatting
def load_trial_records(folder) -> List[TrialRecord]:
    from pathlib import Path
    return trial_records(read_table(Path(folder) / "trials.csv", TRIAL_FIELDS))


def load_frame_rows(folder) -> List[Dict[str, str]]:
    from pathlib import Path
    return read_table(Path(folder) / "frames.csv", FRAME_FIELDS)


def _pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{100 * value:.0f}%"


def _ms(value) -> str:
    return "n/a" if value is None else f"{value:.0f}"


def format_summary(summaries: Sequence[ConditionSummary],
                   errors: Optional[Dict[Tuple[str, str], Dict[str, float]]] = None,
                   availability: Optional[Sequence[AssistAvailability]] = None) -> str:
    """A plain-text results table. Synthetic rows are labelled; real and synthetic are never pooled."""
    lines = []
    for origin in sorted({s.origin for s in summaries}):
        title = "SYNTHETIC data: pipeline check, NOT experimental results" if origin == "synthetic" \
            else "REAL data"
        lines += [title,
                  f"{'cond':<5}{'trials':>7}{'valid':>7}{'lost':>6}{'incompl':>8}{'success':>9}"
                  f"{'median ms':>11}{'range ms':>15}{'slip':>7}{'overgrip':>10}"]
        for s in (x for x in summaries if x.origin == origin):
            rng = "n/a" if s.min_time_ms is None else f"{s.min_time_ms}-{s.max_time_ms}"
            lines.append(f"{s.condition:<5}{s.trials:>7}{s.valid:>7}{s.tracking_lost:>6}{s.incomplete:>8}"
                         f"{_pct(s.success_rate):>9}{_ms(s.median_time_ms):>11}{rng:>15}"
                         f"{_pct(s.slip_rate):>7}{_pct(s.overgrip_rate):>10}")
            if s.failures:
                lines.append("      failed (valid trials): " + ", ".join(f"{k} x{v}" for k, v in sorted(s.failures.items())))
        if errors:
            lines.append("  prediction error (mean, normalised units; lower is better):")
            for (o, condition), e in sorted(errors.items()):
                if o == origin:
                    lines.append(f"    {condition}: kalman {e['kalman']:.4f}  no-movement {e['no_movement']:.4f}  "
                                 f"constant-velocity {e['constant_velocity']:.4f}  "
                                 f"(in 2-sigma {100 * e['within_2_sigma']:.0f}%, n={e['scored']})")
        shown = [a for a in (availability or []) if a.origin == origin and a.permitted]
        if shown:
            lines.append("  assistance actually on (share of the frames where the condition allows it):")
            for a in shown:
                why = ", ".join(f"{k} {v}" for k, v in sorted(a.reasons.items(), key=lambda kv: -kv[1]))
                lines.append(f"    {a.condition}: {_pct(a.share_active)} of {a.permitted} frames  ({why})")
        lines.append("")
    lines.append("valid = finished with tracking intact; 'lost' (tracking_lost) and 'incompl' trials are "
                 "reported, not counted as failures. Slip and over-grip describe the simulation, not a real object.")
    return "\n".join(lines)


def write_summary_csv(path, summaries: Sequence[ConditionSummary]) -> None:
    import csv
    from pathlib import Path
    fields = ("origin", "condition", "trials", "valid", "tracking_lost", "incomplete", "successes",
              "success_rate", "median_time_ms", "min_time_ms", "max_time_ms", "timeouts",
              "slip_rate", "overgrip_rate")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for s in summaries:
            writer.writerow(["" if getattr(s, f) is None else getattr(s, f) for f in fields])


def plot_success_rates(summaries: Sequence[ConditionSummary], path) -> None:
    """Bar chart of success rate per condition, one panel per origin, with the valid-trial counts.
    Synthetic panels carry a watermark. Needs matplotlib (not required for anything else)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("plotting needs matplotlib (pip install matplotlib)") from error
    from pathlib import Path
    origins = sorted({s.origin for s in summaries})
    if not origins:
        raise ValueError("nothing to plot")
    fig, axes = plt.subplots(1, len(origins), figsize=(5.2 * len(origins), 3.8), squeeze=False)
    for ax, origin in zip(axes[0], origins):
        rows = [s for s in summaries if s.origin == origin]
        rates = [0 if s.success_rate is None else 100 * s.success_rate for s in rows]
        bars = ax.bar([s.condition for s in rows], rates, color="#3b6fb6")
        for bar, s in zip(bars, rows):
            label = "no valid trials" if s.valid == 0 else f"n={s.valid}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, label, ha="center", fontsize=8)
        ax.set_ylim(0, 115)
        ax.set_ylabel("success rate of valid trials (%)")
        ax.set_xlabel("condition (A baseline, B AI, C haptic, D both)")
        ax.set_title(f"{origin} data")
        if origin == "synthetic":
            ax.text(0.5, 0.5, "SYNTHETIC\nnot experimental results", transform=ax.transAxes, ha="center",
                    va="center", fontsize=15, color="gray", alpha=0.45, rotation=20)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
