"""Run: .venv/bin/python -m pytest tests/hardware  (or: python -m unittest discover -s tests/hardware)"""
import time
import unittest

import tests_path  # noqa: F401  (adds src/ to the path)
from ghostgrip.hardware.haptics import (ConfigError, HapticPolicy, LED_ONLY_ANGLE,
                                        validate_config)
from ghostgrip.hardware.mock_arduino import MockArduino
from ghostgrip.hardware.protocol import Ack, Err
from ghostgrip.hardware.serial_link import SerialLink


class FakeClock:
    def __init__(self):
        self.ms = 0

    def __call__(self):
        return self.ms


VERIFIED = {'servo_enabled': True, 'servo_neutral_deg': 90, 'servo_min_deg': 80, 'servo_max_deg': 100}


class TestSerialLink(unittest.TestCase):
    def test_arm_cue_stop_round_trip(self):
        board = MockArduino()
        link = SerialLink(board)
        self.assertIsInstance(link.arm(), Ack)
        self.assertTrue(link.armed)
        policy = HapticPolicy()
        decision = policy.filter(1, 'CONTACT', 0.9, link.armed)
        self.assertIsInstance(link.cue(decision.cue), Ack)
        self.assertIsInstance(link.stop(), Ack)
        self.assertFalse(link.armed)
        self.assertEqual(board.state, 'DISARMED')

    def test_board_error_is_reported_not_raised(self):
        link = SerialLink(MockArduino())
        reply = link.cue(HapticPolicy().filter(1, 'CONTACT', 0.9, True).cue)
        self.assertEqual(reply.reason, 'NOT_ARMED')
        self.assertIsInstance(link.last_error, Err)

    def test_dead_board_marks_disconnected_without_crashing(self):
        board = MockArduino()
        link = SerialLink(board, ack_timeout_s=0.02, retries=1)
        link.arm()
        board.dead = True
        self.assertIsNone(link.arm())
        self.assertFalse(link.connected)
        self.assertFalse(link.armed)

    def test_sequence_numbers_increase_across_retries(self):
        board = MockArduino()
        link = SerialLink(board, ack_timeout_s=0.02, retries=2)
        board.dead = True
        link.arm()
        board.dead = False
        self.assertIsInstance(link.arm(), Ack)  # fresh seq is accepted after the failed ones

    def test_poll_reads_sensor_packets(self):
        board = MockArduino()
        link = SerialLink(board)
        board.touch = True
        board.sample()
        link.poll()
        self.assertTrue(link.device.touch)
        self.assertIsNone(link.device.ax_g)


class TestHapticPolicy(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()

    def test_placeholder_config_refuses_servo(self):
        with self.assertRaises(ConfigError):
            validate_config({'servo_enabled': True})
        with self.assertRaises(ConfigError):
            validate_config({**VERIFIED, 'servo_neutral_deg': 120})

    def test_led_only_mode_never_moves_servo(self):
        policy = HapticPolicy(clock_ms=self.clock)
        decision = policy.filter(1, 'CONTACT', 0.9, True)
        self.assertEqual(decision.cue.safe_angle_deg, LED_ONLY_ANGLE)
        self.assertTrue(decision.cue.no_contact_mode)

    def test_contact_angle_is_clamped(self):
        policy = HapticPolicy({**VERIFIED, 'contact_offset_deg': 50}, clock_ms=self.clock)
        self.assertEqual(policy.filter(1, 'CONTACT', 0.9, True).cue.safe_angle_deg, 100)

    def test_only_contact_moves_servo(self):
        policy = HapticPolicy(VERIFIED, clock_ms=self.clock)
        self.assertEqual(policy.filter(1, 'DAMAGE', 0.9, True).cue.safe_angle_deg, 90)

    def test_gates(self):
        policy = HapticPolicy(clock_ms=self.clock)
        self.assertEqual(policy.filter(1, 'CONTACT', 0.9, False).reason, 'NOT_ARMED')
        self.assertEqual(policy.filter(1, 'CONTACT', 0.2, True).reason, 'LOW_CONFIDENCE')
        self.assertEqual(policy.filter(1, 'CONTACT', None, True).reason, 'LOW_CONFIDENCE')
        self.assertEqual(policy.filter(1, 'RELEASE', 0.9, True).reason, 'UNMAPPED')

    def test_one_cue_per_event_and_cooldown(self):
        policy = HapticPolicy(clock_ms=self.clock)
        self.assertEqual(policy.filter(1, 'CONTACT', 0.9, True).action, 'cue')
        self.clock.ms += 1000
        self.assertEqual(policy.filter(1, 'CONTACT', 0.9, True).reason, 'DUPLICATE')
        self.assertEqual(policy.filter(2, 'SLIP', 0.9, True).action, 'cue')
        self.clock.ms += 100
        self.assertEqual(policy.filter(3, 'SLIP', 0.9, True).reason, 'COOLDOWN')

    def test_tracking_loss_requests_stop(self):
        policy = HapticPolicy(clock_ms=self.clock)
        self.assertEqual(policy.filter(9, 'TRACKING_LOST', 0.0, True).action, 'stop')


class Unplugged:
    """A transport that behaves like pyserial after the USB cable is pulled."""
    in_waiting = property(lambda self: (_ for _ in ()).throw(OSError('device gone')))

    def write(self, data):
        raise OSError('device gone')

    def readline(self):
        raise OSError('device gone')

    def close(self):
        raise OSError('device gone')


class TestLinkReviewFindings(unittest.TestCase):
    def test_keepalive_does_not_rearm_after_button(self):
        board = MockArduino()
        link = SerialLink(board)
        link.arm()
        board.press_button()
        self.assertIsInstance(link.keepalive(), Err)
        self.assertFalse(link.armed)
        self.assertEqual(board.state, 'DISARMED')

    def test_cue_is_never_retried(self):
        board = MockArduino()
        link = SerialLink(board, ack_timeout_s=0.02, retries=3)
        link.arm()
        board.dead = True
        before = link.seq
        link.cue(HapticPolicy().filter(1, 'CONTACT', 0.9, True).cue)
        self.assertEqual(link.seq, before + 1)

    def test_unplugged_cable_never_raises(self):
        link = SerialLink(Unplugged())
        self.assertIsNone(link.arm())
        link.poll()
        self.assertIsNone(link.stop())
        link.close()
        self.assertFalse(link.connected)

    def test_poll_returns_quickly_with_busy_board(self):
        board = MockArduino()
        link = SerialLink(board)
        for _ in range(200):
            board.sample()
        start = time.monotonic()
        link.poll()
        self.assertLess(time.monotonic() - start, 0.1)
        self.assertIsNotNone(link.device)

    def test_last_ack_seq_tracks_acknowledged_commands(self):
        board = MockArduino()
        link = SerialLink(board)
        link.arm()
        board.sample()
        link.poll()
        self.assertEqual(link.device.last_ack_seq, 1)

    def test_nan_confidence_is_rejected(self):
        policy = HapticPolicy()
        self.assertEqual(policy.filter(1, 'CONTACT', float('nan'), True).reason, 'LOW_CONFIDENCE')


if __name__ == '__main__':
    unittest.main()
