"""Build and save a calibration profile from validated (gap, x, y) samples."""
import json
import math
from pathlib import Path
from statistics import median

from .features import calibrated_closure


PHASES = ('open', 'pinch', 'reach', 'release')


def build_profile(samples):
    for phase in PHASES:
        rows = samples.get(phase, [])
        if len(rows) < 3:
            raise ValueError(f'{phase}: need at least 3 reliable samples; retry calibration')
        for row in rows:
            if (len(row) != 3 or not all(math.isfinite(v) for v in row)
                    or row[0] < 0 or not all(0 <= v <= 1 for v in row[1:])):
                raise ValueError(f'{phase}: invalid sample')
    opened = median(row[0] for row in samples['open'])
    closed = median(row[0] for row in samples['pinch'])
    calibrated_closure(closed, opened, closed, min_span=0.1)
    released = median(row[0] for row in samples['release'])
    if abs(released - opened) > 0.5 * (opened - closed):
        raise ValueError('Release did not return near the open position; retry calibration')
    xs = sorted(row[1] for row in samples['reach'])
    ys = sorted(row[2] for row in samples['reach'])
    # Trim extreme endpoints when enough observations are available.
    trim = int(len(xs) * 0.05)
    bounds = {'x_min': xs[trim], 'x_max': xs[-trim - 1],
              'y_min': ys[trim], 'y_max': ys[-trim - 1]}
    if bounds['x_max'] - bounds['x_min'] < 0.05 or bounds['y_max'] - bounds['y_min'] < 0.05:
        raise ValueError('Reach farther both horizontally and vertically; retry calibration')
    return {'version': 1, 'd_open': opened, 'd_closed': closed,
            'pointer_range': bounds, 'coordinates': 'mirrored_normalized',
            'sample_counts': {phase: len(samples[phase]) for phase in PHASES}}


def save_profile(profile, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(profile, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)
