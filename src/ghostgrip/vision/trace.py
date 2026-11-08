"""Local CSV recording of shared HandStateV2 values, without images."""
import csv
from pathlib import Path

from contracts import HandStateV2


FIELDS = ('t_ms', 'pointer_x', 'pointer_y', 'closure_q', 'dq_dt',
          'velocity_x', 'velocity_y', 'stable_score', 'tracking_ok', 'tracking_quality')


class TraceRecorder:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Never overwrite an earlier recording.
        self.file = self.path.open('x', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(FIELDS)
        self.file.flush()

    def write(self, state):
        if not isinstance(state, HandStateV2):
            raise TypeError('Expected HandStateV2')
        self.writer.writerow((state.t_ms, *state.pointer_xy, state.closure_q,
                              state.dq_dt, *state.velocity_xy, state.stable_score,
                              int(state.tracking_ok), state.tracking_quality))
        self.file.flush()

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def read_trace(path):
    """Yield stored shared states for offline consumers (no playback pacing)."""
    with Path(path).open(newline='') as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError('Unsupported trace columns')
        for row in reader:
            if row['tracking_ok'] not in ('0', '1'):
                raise ValueError('tracking_ok must be 0 or 1')
            yield HandStateV2(
                t_ms=int(row['t_ms']),
                pointer_xy=(float(row['pointer_x']), float(row['pointer_y'])),
                closure_q=float(row['closure_q']), dq_dt=float(row['dq_dt']),
                velocity_xy=(float(row['velocity_x']), float(row['velocity_y'])),
                stable_score=float(row['stable_score']),
                tracking_ok=row['tracking_ok'] == '1',
                tracking_quality=float(row['tracking_quality']),
            )
