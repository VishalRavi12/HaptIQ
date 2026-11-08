from pathlib import Path
import tempfile
import unittest

from contracts import HandStateV2
from src.ghostgrip.vision.trace import TraceRecorder, read_trace


class TestTrace(unittest.TestCase):
    def test_round_trip_preserves_motion_and_loss(self):
        states = [HandStateV2(1000, (0.3, 0.4), 0.7, -0.3, (0.2, -0.1), 0.8, True, 0.9),
                  HandStateV2(1033, (0.3, 0.4), 0.7, 0, (0, 0), 0, False, 0)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.csv'
            with TraceRecorder(path) as recorder:
                for state in states:
                    recorder.write(state)
                # Flushed data is readable even before a normal shutdown.
                self.assertEqual(list(read_trace(path)), states)
            self.assertEqual(list(read_trace(path)), states)
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                TraceRecorder(path)
            self.assertEqual(path.read_bytes(), original)

    def test_wrong_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.csv'
            path.write_text('x,y\n1,2\n')
            with self.assertRaises(ValueError):
                list(read_trace(path))
