import json
from pathlib import Path
import tempfile
import unittest

from src.ghostgrip.ai.calibration import build_profile, save_profile


def valid_samples():
    return {'open': [(1.0, 0.5, 0.5)] * 3,
            'pinch': [(0.1, 0.5, 0.5)] * 3,
            'reach': [(0.5, 0.2, 0.2), (0.5, 0.5, 0.5), (0.5, 0.8, 0.8)],
            'release': [(0.95, 0.5, 0.5)] * 3}


class TestCalibration(unittest.TestCase):
    def test_profile_round_trip(self):
        profile = build_profile(valid_samples())
        self.assertEqual(profile['d_open'], 1.0)
        self.assertEqual(profile['d_closed'], 0.1)
        self.assertEqual(profile['pointer_range']['x_min'], 0.2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'local/profile.json'
            save_profile(profile, path)
            self.assertEqual(json.loads(path.read_text()), profile)

    def test_insufficient_samples(self):
        samples = valid_samples()
        samples['pinch'] = samples['pinch'][:2]
        with self.assertRaises(ValueError):
            build_profile(samples)

    def test_degenerate_calibration(self):
        samples = valid_samples()
        samples['pinch'] = samples['open']
        with self.assertRaises(ValueError):
            build_profile(samples)

    def test_failed_release_and_stationary_reach(self):
        for phase in ('release', 'reach'):
            samples = valid_samples()
            samples[phase] = [(0.1, 0.5, 0.5)] * 3
            with self.assertRaises(ValueError):
                build_profile(samples)

    def test_nonfinite_sample(self):
        samples = valid_samples()
        samples['open'][0] = (float('nan'), 0.5, 0.5)
        with self.assertRaises(ValueError):
            build_profile(samples)
