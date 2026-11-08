"""Run: python -m unittest discover -s tests -v"""
import unittest
from contracts import HandState
from grasp_ai import GraspBaseline
from simulation import Simulation


class TestBaseline(unittest.TestCase):
    def test_predicts_motion_to_the_right(self):
        model = GraspBaseline(horizon_s=0.1)
        model.update(HandState(0, True, 0.20, 0.5, 1.0))
        result = model.update(HandState(100, True, 0.30, 0.5, 1.0))
        self.assertAlmostEqual(result.predicted_x, 0.40)

    def test_grasp_threshold(self):
        model = GraspBaseline()
        output = model.update(HandState(0, True, 0.5, 0.5, 0.20))
        self.assertEqual(output.grasp_state, 'holding')

    def test_tracking_loss(self):
        model = GraspBaseline()
        result = model.update(HandState(0, False, 0.5, 0.5, 0.0))
        self.assertEqual(result.grasp_state, 'open')

    def test_mouse_grab_release(self):
        sim = Simulation()
        events = sim.update(230, 300, True)
        self.assertEqual(events[0].name, 'GRASP')
        events = sim.update(230, 300, False)
        self.assertEqual(events[0].name, 'RELEASE')
        self.assertEqual(sim.successes, 1)


if __name__ == '__main__':
    unittest.main()
