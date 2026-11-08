"""How do the predictor and the assistance behave on a RECORDED hand trace? (real-data check)

    PYTHONPATH=src python -m ghostgrip.metrics.trace_report data/local/trace_XXXX.csv
    PYTHONPATH=src python -m ghostgrip.metrics.trace_report data/local/trace_XXXX.csv --sweep

Takes a recording from `python -m ghostgrip.vision.cli --record` (or any trace the simulation can
replay: both column layouts are read) and prints, in plain words:

  * the recording itself: length, real frame rate, tracking lost, quality, closure range, jitter;
  * the Kalman predictor against the two baselines on THIS recording, with how often the real
    error stayed inside its claimed uncertainty, and a verdict that says so when it does not win;
  * how often the assistance policy would have been on, and why not when it was off;
  * with --sweep, how the result changes with the predictor's noise settings, so they can be
    re-tuned from real data instead of the synthetic traces they were first chosen from;
  * with --intent SCENE, the grasp-intent timeline against that simulation scene's objects.

It only reads the file. A recording is numbers (pointer, closure, tracking), never images.
It is a check of one recording, not an experiment: nothing here supports a claim on its own.
"""
import argparse
from dataclasses import dataclass
from statistics import median
from typing import Dict, List, Optional, Sequence, Tuple

from contracts import HandStateV2
from ..ai.adapt import AssistPolicy, trial_jitter
from ..ai.predict import DEFAULT_MEASUREMENT_NOISE, DEFAULT_PROCESS_NOISE, ErrorTracker, KalmanPredictor

MIN_VALID_FRAMES = 60            # about two seconds at 30 FPS; less says nothing
WIN_MARGIN = 0.95                # "better" means at least 5% lower error than the best baseline
MIN_COVERAGE = 0.85              # fewer errors than this inside 2 sigma: uncertainty is over-optimistic
MAX_CAUTION = 6.0                # claimed uncertainty more than this many times the real error: useless caution


def load_hand_trace(path) -> List[HandStateV2]:
    """Read a recorded trace in either team layout; raises ValueError with a clear message."""
    from ..simulation.inputs import load_trace
    try:
        return load_trace(path)
    except FileNotFoundError:
        raise ValueError(f"no such file: {path}") from None
    except (KeyError, ValueError) as error:
        raise ValueError(f"{path} is not a hand trace the team tools can read ({error})") from error


@dataclass(frozen=True)
class TraceFacts:
    frames: int
    valid_frames: int
    duration_s: float
    median_fps: Optional[float]
    longest_gap_ms: int
    lost_share: float
    lost_segments: int
    longest_lost_ms: int
    quality_median: Optional[float]
    closure_min: Optional[float]
    closure_max: Optional[float]
    jitter: Optional[float]
    backwards_timestamps: int


def describe(states: Sequence[HandStateV2]) -> TraceFacts:
    if not states:
        raise ValueError("empty trace")
    dts = [b.t_ms - a.t_ms for a, b in zip(states, states[1:]) if b.t_ms > a.t_ms]
    backwards = sum(1 for a, b in zip(states, states[1:]) if b.t_ms <= a.t_ms)
    valid = [s for s in states if s.tracking_ok]
    segments, longest, start = 0, 0, None
    for s in states:                                            # runs of tracking_ok == False
        if not s.tracking_ok and start is None:
            start, segments = s.t_ms, segments + 1
        if s.tracking_ok and start is not None:
            longest, start = max(longest, s.t_ms - start), None
    if start is not None:
        longest = max(longest, states[-1].t_ms - start)
    qs = [s.closure_q for s in valid]
    return TraceFacts(
        frames=len(states), valid_frames=len(valid),
        duration_s=(states[-1].t_ms - states[0].t_ms) / 1000.0,
        median_fps=(1000.0 / median(dts)) if dts and median(dts) > 0 else None,
        longest_gap_ms=max(dts) if dts else 0,
        lost_share=1.0 - len(valid) / len(states), lost_segments=segments, longest_lost_ms=longest,
        quality_median=median(s.tracking_quality for s in valid) if valid else None,
        closure_min=min(qs) if qs else None, closure_max=max(qs) if qs else None,
        jitter=trial_jitter(states), backwards_timestamps=backwards)


def score(states: Sequence[HandStateV2], horizon_ms: int, process_noise: float = DEFAULT_PROCESS_NOISE,
          measurement_noise: float = DEFAULT_MEASUREMENT_NOISE) -> Dict[str, float]:
    """Mean error of the Kalman predictor and both baselines on one trace."""
    predictor = KalmanPredictor(horizon_ms, process_noise, measurement_noise)
    tracker = ErrorTracker()
    for state in states:
        tracker.record(state, predictor.update(state))
    return tracker.summary()


def verdict(result: Dict[str, float]) -> str:
    if not result["scored"]:
        return "nothing could be scored (no usable stretch long enough to compare forecasts)"
    best_name, best = min(((m, result[m]) for m in ("no_movement", "constant_velocity")), key=lambda kv: kv[1])
    ratio = result["kalman"] / best if best > 0 else float("inf")
    if result["kalman"] <= best * WIN_MARGIN:
        text = f"the Kalman predictor beats the best baseline ({best_name}) by {100 * (1 - ratio):.0f}% on this recording"
    elif result["kalman"] <= best:
        text = f"the Kalman predictor is only marginally better than {best_name} on this recording"
    else:
        text = (f"the Kalman predictor does NOT beat {best_name} on this recording "
                f"({100 * (ratio - 1):.0f}% worse); the plan says to turn predictive assistance down then")
    cover = result["within_2_sigma"]
    if cover < 0.85:
        text += f"; its uncertainty is too optimistic (only {100 * cover:.0f}% of errors inside 2 sigma)"
    elif cover > 0.99 and result["kalman"] > 0:
        text += f"; its uncertainty is cautious ({100 * cover:.0f}% of errors inside 2 sigma)"
    return text


def assist_profile(states: Sequence[HandStateV2], horizon_ms: int) -> Tuple[int, Dict[str, int], float]:
    """What the assistance policy would do along the trace: (frames, reasons, mean gain when on)."""
    predictor, tracker, policy = KalmanPredictor(horizon_ms), ErrorTracker(), AssistPolicy()
    reasons: Dict[str, int] = {}
    gains: List[float] = []
    for state in states:
        prediction = predictor.update(state)
        tracker.record(state, prediction)
        decision = policy.decide(state, prediction, predictor_trusted=not tracker.is_worse_than_baseline())
        reasons[decision.reason] = reasons.get(decision.reason, 0) + 1
        if decision.gain > 0:
            gains.append(decision.gain)
    return len(states), reasons, (sum(gains) / len(gains) if gains else 0.0)


def sweep(states: Sequence[HandStateV2], horizon_ms: int,
          process_noises: Sequence[float] = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0),
          measurement_noises: Sequence[float] = (1e-6, 4e-6, 1e-5, 1e-4)) -> List[Dict[str, float]]:
    rows = []
    for q in process_noises:
        for r in measurement_noises:
            result = score(states, horizon_ms, q, r)
            if result["scored"]:
                rows.append({"q": q, "r": r, "kalman": result["kalman"], "within_2_sigma": result["within_2_sigma"],
                         "mean_sigma": result["mean_sigma"]})
    return rows


def suggest(rows: Sequence[Dict[str, float]]) -> Optional[Dict[str, float]]:
    """Lowest-error setting whose uncertainty is neither over-optimistic (under 85% of errors inside
    2 sigma) nor uselessly cautious (claimed sigma more than 6x the real error)."""
    ok = [r for r in rows if r["within_2_sigma"] >= MIN_COVERAGE and r["mean_sigma"] <= MAX_CAUTION * r["kalman"]]
    return min(ok, key=lambda r: r["kalman"]) if ok else None


def synthetic_reference(horizon_ms: int) -> Dict[str, Dict[str, float]]:
    from ..ai.synthetic import make_scenario
    return {name: score(make_scenario(name), horizon_ms) for name in ("clean", "noisy")}


def format_report(states: Sequence[HandStateV2], horizon_ms: int, do_sweep: bool = False,
                  intent_scene: Optional[str] = None, label: str = "recording") -> str:
    facts = describe(states)
    out = [f"GhostGrip trace report: {label}", "",
           "THE RECORDING",
           f"  {facts.frames} frames over {facts.duration_s:.1f} s"
           + (f", about {facts.median_fps:.0f} frames per second" if facts.median_fps else "")
           + f"; longest gap between frames {facts.longest_gap_ms} ms",
           f"  hand tracked in {facts.valid_frames} frames; lost {100 * facts.lost_share:.0f}% "
           f"({facts.lost_segments} stretch(es), longest {facts.longest_lost_ms} ms)"]
    if facts.quality_median is not None:
        out.append(f"  median tracking quality {facts.quality_median:.2f}; closure from {facts.closure_min:.2f} "
                   f"to {facts.closure_max:.2f}"
                   + (f"; jitter {facts.jitter:.5f}" if facts.jitter is not None else ""))
    if facts.backwards_timestamps:
        out.append(f"  WARNING: {facts.backwards_timestamps} repeated or backwards timestamp(s); the tools skip them")
    if facts.valid_frames < MIN_VALID_FRAMES:
        out += ["", f"TOO SHORT: only {facts.valid_frames} tracked frames. Record at least {MIN_VALID_FRAMES} "
                    "(a few seconds with the hand in view) before drawing any conclusion."]
        return "\n".join(out)
    if facts.closure_max - facts.closure_min < 0.2:
        out.append("  NOTE: closure barely changed. Was the calibration saved, and did you open and pinch the hand?")

    result = score(states, horizon_ms)
    reference = synthetic_reference(horizon_ms)
    out += ["", f"PREDICTOR, {horizon_ms} ms ahead (mean error, normalised units; lower is better)",
            f"  {'':<22}{'kalman':>9}{'no-movement':>13}{'const-velocity':>16}{'in 2-sigma':>12}{'claimed +-':>12}",
            f"  {'THIS RECORDING':<22}{result['kalman']:>9.4f}{result['no_movement']:>13.4f}"
            f"{result['constant_velocity']:>16.4f}{100 * result['within_2_sigma']:>11.0f}%{result['mean_sigma']:>12.4f}"]
    for name, ref in reference.items():
        out.append(f"  {'synthetic ' + name:<22}{ref['kalman']:>9.4f}{ref['no_movement']:>13.4f}"
                   f"{ref['constant_velocity']:>16.4f}{100 * ref['within_2_sigma']:>11.0f}%{ref['mean_sigma']:>12.4f}")
    out += [f"  forecasts scored: {int(result['scored'])}, skipped as unscorable: {int(result['unscorable'])}",
            f"  Verdict: {verdict(result)}."]

    frames, reasons, mean_gain = assist_profile(states, horizon_ms)
    on = reasons.get("ok", 0)
    why = ", ".join(f"{k} {v}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]))
    out += ["", "ASSISTANCE (a new, average user, if this were a condition with AI assistance)",
            f"  on in {on} of {frames} frames ({100 * on / frames:.0f}%), mean gain {mean_gain:.2f} when on",
            f"  by reason: {why}"]

    if do_sweep:
        rows = sweep(states, horizon_ms)
        best = suggest(rows)
        out += ["", "TUNING SWEEP (predictor noise settings on THIS recording)",
                f"  {'q (process)':>12}{'r (measure)':>13}{'error':>9}{'in 2-sigma':>12}{'claimed +-':>12}"]
        for row in sorted(rows, key=lambda r: r["kalman"])[:8]:
            mark = "  <- current" if (row["q"], row["r"]) == (DEFAULT_PROCESS_NOISE, DEFAULT_MEASUREMENT_NOISE) else ""
            out.append(f"  {row['q']:>12g}{row['r']:>13.0e}{row['kalman']:>9.4f}{100 * row['within_2_sigma']:>11.0f}%"
                       f"{row['mean_sigma']:>12.4f}{mark}")
        if best:
            out.append(f"  Lowest error with sensible uncertainty (at least 85% of errors inside 2 sigma, claimed sigma at most "
                       f"6x the error): q={best['q']:g}, r={best['r']:.0e}, error {best['kalman']:.4f}. Compare with the "
                       f"current q={DEFAULT_PROCESS_NOISE:g}, r={DEFAULT_MEASUREMENT_NOISE:.0e}; change the defaults only "
                       f"if several recordings agree.")
        else:
            out.append("  No setting gave sensible uncertainty on this recording; do not tune from it.")

    if intent_scene:
        from ..ai.intent import IntentMachine, timeline, zones_for_scene
        log = timeline(states, IntentMachine(zones_for_scene(intent_scene)))
        out += ["", f"GRASP INTENT against the {intent_scene} scene's objects "
                    "(only meaningful if your hand actually went to where the objects are)",
                f"  {len(log)} transition(s)"]
        out += [f"  {t:>7} ms  {old:<18} -> {new:<18} ({reason})" for t, old, new, reason in log]
    out += ["", "One recording of one person is a check, not evidence. Repeat on several, and never mix it with synthetic results."]
    return "\n".join(out)


def main(argv=None) -> int:
    from ..config import load_config
    parser = argparse.ArgumentParser(description="Predictor and assistance behaviour on a recorded hand trace.")
    parser.add_argument("trace", help="CSV from `python -m ghostgrip.vision.cli --record` (data/local/trace_*.csv)")
    parser.add_argument("--horizon-ms", type=int, default=None, help="forecast horizon (default: config.yaml)")
    parser.add_argument("--sweep", action="store_true", help="also try other predictor noise settings")
    parser.add_argument("--intent", metavar="SCENE", help="also print the intent timeline for glass/sponge/block/sandbox")
    args = parser.parse_args(argv)
    try:
        states = load_hand_trace(args.trace)
        horizon = args.horizon_ms or load_config().ai.prediction_horizon_ms
        print(format_report(states, horizon, args.sweep, args.intent, label=args.trace))
    except ValueError as error:
        print(f"error: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
