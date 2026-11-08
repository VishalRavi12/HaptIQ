"""Synthetic hand traces: fake but realistic HandStateV2 streams, no camera needed.

Used to build and test the predictor, intent logic, adaptation and metrics before the
live camera is wired in, and to replay deterministic demos in the simulation (plan 4.3, NFR-02).

How the traces are made: a "true" hand path (pointer x/y and closure q over time) is turned
into noisy observations, which are fed through Member A's own MotionFeatures. So smoothing,
velocities, stability, jump rejection and the tracking-loss convention are exactly what the
real perception pipeline produces. Only tracking_quality is synthetic (a settable number).

Loss convention (contracts.py): pointer and closure hold their last values; velocities,
stability and quality become 0; tracking_ok is False.

Every trace is deterministic: the same seed gives the same frames, so tests never flake.

    from ghostgrip.ai.synthetic import SCENARIOS, make_scenario
    states = make_scenario("noisy", seed=3)          # list of HandStateV2

Write the sample files:   PYTHONPATH=src python -m ghostgrip.ai.synthetic
"""
import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import random
from typing import Callable, Dict, List, Sequence, Tuple

from contracts import HandStateV2
from .features import MotionFeatures

DEFAULT_FPS = 30

# Geometry of the glass scene in simulation/world.py, in normalised 0-1 coordinates
# (world is 960 x 540; glass rests at (200, 420) px; the target zone is x 740-930 px).
# tests/ai/test_synthetic.py replays the clean trace through the real World to keep this honest.
GLASS_XY = (200 / 960, 420 / 540)
TARGET_X = 835 / 960
GRASP_Q = 0.50       # glass holds at q >= 0.40 and breaks above 0.60 (simulation/objects.py)

Keyframes = Sequence[Tuple[float, float]]


# ------------------------------------------------------------------ building blocks
def _ease(a: float, b: float, u: float) -> float:
    """Smooth start and stop between a and b, u in [0, 1]."""
    u = min(1.0, max(0.0, u))
    return a + (b - a) * (u * u * (3.0 - 2.0 * u))


def piecewise(keyframes: Keyframes) -> Callable[[float], float]:
    """A function of time (ms) that moves smoothly through (time_ms, value) keyframes."""
    frames = list(keyframes)
    if not frames or any(b[0] <= a[0] for a, b in zip(frames, frames[1:])):
        raise ValueError("keyframe times must be strictly increasing")

    def value_at(t_ms: float) -> float:
        if t_ms <= frames[0][0]:
            return frames[0][1]
        for (t0, v0), (t1, v1) in zip(frames, frames[1:]):
            if t_ms <= t1:
                return _ease(v0, v1, (t_ms - t0) / (t1 - t0))
        return frames[-1][1]

    return value_at


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _in_any(t_ms: float, ranges) -> bool:
    return any(start <= t_ms < end for start, end, *_ in ranges)


def make_trace(x_fn, y_fn, q_fn, duration_ms: int, *, fps: int = DEFAULT_FPS,
               noise_xy: float = 0.0, noise_q: float = 0.0,
               dropouts: Sequence[Tuple[int, int]] = (),
               glitches: Sequence[Tuple[int, float, float]] = (),
               missing_frames: Sequence[Tuple[int, int]] = (),
               quality: float = 0.9,
               quality_dips: Sequence[Tuple[int, int, float]] = (),
               seed: int = 0) -> List[HandStateV2]:
    """Turn a true hand path into the HandStateV2 stream perception would produce.

    x_fn, y_fn, q_fn   functions of time in ms returning the true pointer and closure
    noise_xy, noise_q  standard deviation of Gaussian measurement noise (normalised units)
    dropouts           (start_ms, end_ms): the hand is not detected; tracking is lost
    glitches           (at_ms, dx, dy): a one-frame landmark spike added to the pointer
    missing_frames     (start_ms, end_ms): frames never arrive at all (uneven time steps)
    quality            tracking_quality while tracking is valid
    quality_dips       (start_ms, end_ms, value): tracking_quality drops to value
    """
    if duration_ms <= 0:
        raise ValueError("duration_ms must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if noise_xy < 0 or noise_q < 0:
        raise ValueError("noise levels cannot be negative")
    if not 0.0 <= quality <= 1.0:
        raise ValueError("quality must be between 0 and 1")
    for start, end, *_ in (*dropouts, *missing_frames, *quality_dips):
        if end <= start:
            raise ValueError("a time range must end after it starts")
    frame_ms = 1000.0 / fps
    rng = random.Random(seed)
    motion = MotionFeatures()
    states: List[HandStateV2] = []
    for i in range(int(duration_ms * fps / 1000) + 1):
        t_ms = round(i * frame_ms)
        if _in_any(t_ms, missing_frames):
            continue
        if _in_any(t_ms, dropouts):
            pointer = q = None
        else:
            x = x_fn(t_ms) + rng.gauss(0.0, noise_xy)
            y = y_fn(t_ms) + rng.gauss(0.0, noise_xy)
            for at_ms, dx, dy in glitches:
                if at_ms <= t_ms < at_ms + frame_ms:
                    x, y = x + dx, y + dy
            pointer = (_clip01(x), _clip01(y))
            q = _clip01(q_fn(t_ms) + rng.gauss(0.0, noise_q))
        values = motion.update(t_ms, pointer, q)
        values.pop("reason")
        level = quality
        for start, end, dip in quality_dips:
            if start <= t_ms < end:
                level = dip
        states.append(HandStateV2(**values, tracking_quality=level if values["tracking_ok"] else 0.0))
    return states


# ------------------------------------------------------------------ the scenarios
def _grasp_task(reach_ms=(400, 1500), close_ms=(1500, 1900), start_xy=(0.05, 0.40)):
    """Reach the glass, close the hand, lift, carry to the target zone, lower, let go."""
    gx, gy = GLASS_XY
    r0, r1 = reach_ms
    c0, c1 = close_ms
    end = c1 + 3700
    x = piecewise([(0, start_xy[0]), (r0, start_xy[0]), (r1, gx), (c1 + 400, gx),
                   (c1 + 900, gx), (c1 + 2100, TARGET_X), (c1 + 2600, TARGET_X), (end, TARGET_X)])
    y = piecewise([(0, start_xy[1]), (r0, start_xy[1]), (r1, gy), (c1 + 400, gy),
                   (c1 + 900, 0.62), (c1 + 2100, 0.62), (c1 + 2600, 0.76), (end, 0.76)])
    q = piecewise([(0, 0.0), (c0, 0.0), (c1, GRASP_Q), (c1 + 2900, GRASP_Q),
                   (c1 + 3200, 0.0), (end, 0.0)])
    return x, y, q, end


def clean(seed: int = 0) -> List[HandStateV2]:
    """Ideal task, no noise: reach, grasp, carry, release. Succeeds in the glass trial."""
    x, y, q, end = _grasp_task()
    return make_trace(x, y, q, end, seed=seed)


def noisy(seed: int = 0) -> List[HandStateV2]:
    """The same task with typical measurement jitter."""
    x, y, q, end = _grasp_task()
    return make_trace(x, y, q, end, noise_xy=0.004, noise_q=0.012, seed=seed)


def dropout_short(seed: int = 0) -> List[HandStateV2]:
    """Hand lost for 150 ms mid-reach: shorter than the simulation's 300 ms freeze."""
    x, y, q, end = _grasp_task()
    return make_trace(x, y, q, end, dropouts=[(1000, 1150)], seed=seed)


def dropout_long(seed: int = 0) -> List[HandStateV2]:
    """Hand lost for 600 ms while carrying: the simulation releases safely (tracking_lost)."""
    x, y, q, end = _grasp_task()
    return make_trace(x, y, q, end, dropouts=[(3000, 3600)], seed=seed)


def glitch(seed: int = 0) -> List[HandStateV2]:
    """One-frame landmark spike during the reach (a sudden coordinate jump)."""
    x, y, q, end = _grasp_task()
    return make_trace(x, y, q, end, glitches=[(1200, 0.35, -0.25)], seed=seed)


def fast_reach(seed: int = 0) -> List[HandStateV2]:
    """A very fast reach (about 0.6 of the frame in 150 ms). Stress test for the jump gate."""
    x, y, q, end = _grasp_task(reach_ms=(400, 550), start_xy=(0.05, 0.10))
    return make_trace(x, y, q, end, seed=seed)


def slow_close(seed: int = 0) -> List[HandStateV2]:
    """The hand closes over 2.5 s instead of 0.4 s (slow closure)."""
    x, y, q, end = _grasp_task(close_ms=(1500, 4000))
    return make_trace(x, y, q, end, noise_xy=0.002, noise_q=0.006, seed=seed)


def low_quality(seed: int = 0) -> List[HandStateV2]:
    """Tracking stays valid but tracking_quality drops to 0.3 while carrying."""
    x, y, q, end = _grasp_task()
    return make_trace(x, y, q, end, quality_dips=[(3000, 3600, 0.3)], seed=seed)


def frame_drops(seed: int = 0) -> List[HandStateV2]:
    """Frames stop arriving for 250 ms: uneven time steps, no tracking_ok=False row."""
    x, y, q, end = _grasp_task()
    return make_trace(x, y, q, end, missing_frames=[(1000, 1250)], seed=seed)


def q_jitter(seed: int = 0) -> List[HandStateV2]:
    """Hand parked on the glass with closure hovering around the contact threshold (0.30)."""
    gx, gy = GLASS_XY
    return make_trace(lambda t: gx, lambda t: gy, lambda t: 0.30, 3000,
                      noise_xy=0.002, noise_q=0.04, seed=seed)


SCENARIOS: Dict[str, Callable[[int], List[HandStateV2]]] = {
    "clean": clean, "noisy": noisy, "dropout_short": dropout_short, "dropout_long": dropout_long,
    "glitch": glitch, "fast_reach": fast_reach, "slow_close": slow_close,
    "low_quality": low_quality, "frame_drops": frame_drops, "q_jitter": q_jitter,
}


def make_scenario(name: str, seed: int = 0) -> List[HandStateV2]:
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario {name!r}; choose from {sorted(SCENARIOS)}")
    return SCENARIOS[name](seed)


# ------------------------------------------------------------------ describing and saving
@dataclass(frozen=True)
class TraceSummary:
    frames: int
    duration_ms: int
    lost_frames: int
    longest_gap_ms: int      # largest time step between two frames
    min_quality_ok: float    # lowest tracking_quality among valid frames (1.0 if none)


def summarize(states: Sequence[HandStateV2]) -> TraceSummary:
    if not states:
        raise ValueError("empty trace")
    gaps = [b.t_ms - a.t_ms for a, b in zip(states, states[1:])]
    ok_quality = [s.tracking_quality for s in states if s.tracking_ok]
    return TraceSummary(len(states), states[-1].t_ms - states[0].t_ms,
                        sum(1 for s in states if not s.tracking_ok),
                        max(gaps) if gaps else 0,
                        min(ok_quality) if ok_quality else 1.0)


def save_trace(states: Sequence[HandStateV2], path, fmt: str = "simulation") -> Path:
    """Write a trace as CSV. The team has two formats; pick the reader's:

    "simulation"  columns x, y, q, vx, vy ... (read by simulation TracePlayer / load_trace)
    "vision"      columns pointer_x, closure_q, velocity_x ... (read by vision.trace.read_trace)
    """
    path = Path(path)
    if fmt == "simulation":
        from ghostgrip.simulation.inputs import save_trace as write
        return write(path, states)
    if fmt == "vision":
        from ghostgrip.vision.trace import TraceRecorder
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()        # the vision recorder never overwrites; replace our own sample
        with TraceRecorder(path) as recorder:
            for state in states:
                recorder.write(state)
        return path
    raise ValueError("fmt must be 'simulation' or 'vision'")


def save_samples(folder, seed: int = 0, fmt: str = "simulation") -> List[Path]:
    """Write every scenario to <folder>/<name>.csv and return the paths."""
    folder = Path(folder)
    return [save_trace(make_scenario(name, seed), folder / f"{name}.csv", fmt) for name in SCENARIOS]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Write the synthetic sample traces.")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[3] / "data" / "synthetic"))
    parser.add_argument("--format", choices=("simulation", "vision"), default="simulation")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    for path in save_samples(args.out, args.seed, args.format):
        info = summarize(make_scenario(path.stem, args.seed))
        print(f"{path.name:<20} {info.frames:>4} frames  {info.duration_ms:>5} ms  "
              f"lost {info.lost_frames:>3}  longest gap {info.longest_gap_ms} ms")


if __name__ == "__main__":
    main()
