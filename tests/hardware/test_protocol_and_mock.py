"""Run: .venv/bin/python -m pytest tests/hardware  (or: python -m unittest discover -s tests/hardware)"""
import unittest

import tests_path  # noqa: F401  (adds src/ to the path)
from ghostgrip.hardware.mock_arduino import MockArduino, SERVO_NEUTRAL_DEG
from ghostgrip.hardware.protocol import (Ack, CueEvent, DeviceState, Err, Ready,
                                         encode_cue, parse_line)


class FakeClock:
    def __init__(self):
        self.ms = 0

    def __call__(self):
        return self.ms


def drain(board):
    lines = []
    while True:
        raw = board.readline()
        if not raw:
            return lines
        lines.append(parse_line(raw))


class TestProtocol(unittest.TestCase):
    def test_encode_cue(self):
        cue = CueEvent(104, 'CONTACT', 82, 90)
        self.assertEqual(encode_cue(cue), b'CUE,104,CONTACT,82,90\n')

    def test_parse_ack_err_ready(self):
        self.assertEqual(parse_line(b'ACK,104,172802,OK\n'), Ack(104, 172802, 'OK'))
        self.assertEqual(parse_line('ERR,7,ANGLE_RANGE'), Err(7, 'ANGLE_RANGE'))
        self.assertIsInstance(parse_line('READY,ghostgrip_uno,v1'), Ready)

    def test_missing_accelerometer_is_none_not_zero(self):
        state = parse_line('S,1200,NA,NA,NA,1,0,512,1')
        self.assertIsInstance(state, DeviceState)
        self.assertIsNone(state.ax_g)
        self.assertTrue(state.piezo)
        self.assertFalse(state.touch)
        self.assertTrue(state.armed)

    def test_noise_is_ignored(self):
        self.assertIsNone(parse_line(b'\xff\xfe garbage'))
        self.assertIsNone(parse_line('ACK,notanumber,1,OK'))


class TestMockBoardSafety(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.board = MockArduino(clock_ms=self.clock)
        drain(self.board)  # discard READY

    def send(self, text):
        self.board.write((text + '\n').encode())
        return drain(self.board)

    def test_boots_disarmed_and_rejects_cue(self):
        self.assertEqual(self.board.state, 'DISARMED')
        self.assertEqual(self.send('CUE,1,CONTACT,85,100'), [Err(1, 'NOT_ARMED')])
        self.assertEqual(self.board.servo_angle, SERVO_NEUTRAL_DEG)

    def test_arm_then_cue_moves_then_returns_to_neutral(self):
        self.send('ARM,1')
        reply = self.send('CUE,2,CONTACT,85,100')
        self.assertIsInstance(reply[0], Ack)
        self.assertEqual(self.board.servo_angle, 85)
        self.clock.ms += 100
        self.board.tick()
        self.assertEqual(self.board.state, 'ARMED')
        self.assertEqual(self.board.servo_angle, SERVO_NEUTRAL_DEG)

    def test_out_of_range_angle_never_moves(self):
        self.send('ARM,1')
        self.assertEqual(self.send('CUE,2,CONTACT,170,100'), [Err(2, 'ANGLE_RANGE')])
        self.assertEqual(self.board.servo_angle, SERVO_NEUTRAL_DEG)

    def test_stale_sequence_rejected(self):
        self.send('ARM,5')
        self.assertEqual(self.send('CUE,5,CONTACT,85,100'), [Err(5, 'STALE_SEQ')])
        self.assertEqual(self.send('CUE,3,CONTACT,85,100'), [Err(3, 'STALE_SEQ')])
        self.assertEqual(self.board.servo_angle, SERVO_NEUTRAL_DEG)

    def test_malformed_packets_rejected_and_repeated_ones_fault(self):
        self.send('ARM,1')
        for text in ['HELLO', 'CUE,x,CONTACT,85,100', 'CUE,2,CONTACT,eighty,100', 'X' * 60, 'JUMP,3']:
            self.assertIsInstance(self.send(text)[0], Err)
        self.assertEqual(self.board.state, 'FAULT')
        self.assertEqual(self.board.servo_angle, SERVO_NEUTRAL_DEG)

    def test_stop_wins_mid_cue_and_no_auto_rearm(self):
        self.send('ARM,1')
        self.send('CUE,2,CONTACT,85,400')
        self.assertEqual(self.board.state, 'CUE')
        self.assertIsInstance(self.send('STOP,3')[0], Ack)
        self.assertEqual(self.board.state, 'DISARMED')
        self.assertEqual(self.board.servo_angle, SERVO_NEUTRAL_DEG)
        self.clock.ms += 5000
        self.board.tick()
        self.assertEqual(self.board.state, 'DISARMED')
        self.assertEqual(self.send('CUE,4,CONTACT,85,100'), [Err(4, 'NOT_ARMED')])

    def test_button_disarms(self):
        self.send('ARM,1')
        self.board.press_button()
        self.assertEqual(self.board.state, 'DISARMED')

    def test_watchdog_faults_when_host_goes_quiet(self):
        self.send('ARM,1')
        self.send('CUE,2,CONTACT,85,400')
        self.clock.ms += 501
        self.board.tick()
        self.assertEqual(self.board.state, 'FAULT')
        self.assertEqual(self.board.servo_angle, SERVO_NEUTRAL_DEG)

    def test_keepalive_prevents_watchdog(self):
        self.send('ARM,1')
        for seq in range(2, 12):
            self.clock.ms += 200
            self.assertIsInstance(self.send(f'KA,{seq}')[0], Ack)
        self.assertEqual(self.board.state, 'ARMED')

    def test_cooldown_between_cues(self):
        self.send('ARM,1')
        self.send('CUE,2,CONTACT,85,100')
        self.clock.ms += 150
        self.board.tick()
        self.assertEqual(self.send('CUE,3,CONTACT,85,100'), [Err(3, 'COOLDOWN')])
        self.clock.ms += 300
        self.send('KA,4')  # keepalive, as a real host sends, so the watchdog stays quiet
        self.clock.ms += 100
        self.assertIsInstance(self.send('CUE,5,CONTACT,85,100')[0], Ack)

    def test_bad_duration_rejected(self):
        self.send('ARM,1')
        self.assertEqual(self.send('CUE,2,CONTACT,85,0'), [Err(2, 'DURATION_RANGE')])
        self.assertEqual(self.send('CUE,3,CONTACT,85,5000'), [Err(3, 'DURATION_RANGE')])


class TestReviewFindings(unittest.TestCase):
    """Regression tests for defects found by the Codex and Cursor reviews."""

    def setUp(self):
        self.clock = FakeClock()
        self.board = MockArduino(clock_ms=self.clock)
        drain(self.board)

    def send(self, data):
        self.board.write(data if isinstance(data, bytes) else (data + '\n').encode())
        return drain(self.board)

    def test_stop_with_stale_or_bad_seq_still_disarms(self):
        self.send('ARM,10')
        self.assertIsInstance(self.send('STOP,9')[0], Ack)
        self.assertEqual(self.board.state, 'DISARMED')
        self.send('STOP,11')
        self.send('ARM,12')
        self.assertIsInstance(self.send('STOP,x')[0], Ack)
        self.assertEqual(self.board.state, 'DISARMED')

    def test_keepalive_never_arms(self):
        self.assertEqual(self.send('KA,1'), [Err(1, 'NOT_ARMED')])
        self.assertEqual(self.board.state, 'DISARMED')

    def test_watchdog_latches_until_stop(self):
        self.send('ARM,1')
        self.clock.ms += 600
        self.board.tick()
        self.assertEqual(self.send('ARM,2'), [Err(2, 'LATCHED')])
        self.send('STOP,3')
        self.assertIsInstance(self.send('ARM,4')[0], Ack)

    def test_button_latches_and_held_button_blocks_arm(self):
        self.send('ARM,1')
        self.board.press_button(held=True)
        self.assertEqual(self.send('STOP,2')[0].status, 'DISARMED')
        self.assertEqual(self.send('ARM,3'), [Err(3, 'BUTTON_HELD')])
        self.board.release_button()
        self.assertIsInstance(self.send('ARM,4')[0], Ack)

    def test_empty_fields_rejected(self):
        self.send('ARM,1')
        self.assertEqual(self.send('CUE,2,,CONTACT,85')[0].reason, 'MALFORMED')
        self.assertEqual(self.send('ARM,,5')[0].reason, 'MALFORMED')
        self.assertEqual(self.board.servo_angle, SERVO_NEUTRAL_DEG)

    def test_unknown_event_rejected(self):
        self.send('ARM,1')
        self.assertEqual(self.send('CUE,2,BOGUS,85,100'), [Err(2, 'BAD_EVENT')])
        self.assertEqual(self.board.servo_angle, SERVO_NEUTRAL_DEG)

    def test_only_contact_moves_servo(self):
        self.send('ARM,1')
        self.assertIsInstance(self.send('CUE,2,DAMAGE,85,100')[0], Ack)
        self.assertEqual(self.board.servo_angle, SERVO_NEUTRAL_DEG)

    def test_control_bytes_and_huge_numbers_rejected(self):
        self.send('ARM,1')
        self.assertEqual(self.send(b'CUE,2,CONTACT,85,100\x00junk\n')[0].reason, 'MALFORMED')
        self.assertEqual(self.send('KA,100000001')[0].reason, 'BAD_SEQ')

    def test_partial_writes_wait_for_newline(self):
        self.assertEqual(self.send(b'AR'), [])
        self.assertIsInstance(self.send(b'M,1\n')[0], Ack)

    def test_blank_lines_ignored(self):
        self.assertEqual(self.send(b'\n\r\n'), [])


class TestBuzzer(unittest.TestCase):
    """C7: a short beep on accepted SLIP/DAMAGE cues, silenced by expiry, STOP, button and watchdog."""

    def setUp(self):
        self.clock = FakeClock()
        self.board = MockArduino(clock_ms=self.clock)
        drain(self.board)

    def send(self, text):
        self.board.write((text + '\n').encode())
        return drain(self.board)

    def armed_beep(self, event='SLIP'):
        self.send('ARM,1')
        self.assertIsInstance(self.send(f'CUE,2,{event},90,300')[0], Ack)

    def test_slip_and_damage_beep_then_expire(self):
        for event in ('SLIP', 'DAMAGE'):
            self.setUp()
            self.armed_beep(event)
            self.assertTrue(self.board.buzzer_on, event)
            self.clock.ms += 120
            self.board.tick()
            self.assertFalse(self.board.buzzer_on, event)

    def test_beep_never_outlasts_a_short_cue(self):
        self.send('ARM,1')
        self.send('CUE,2,DAMAGE,90,50')
        self.clock.ms += 50
        self.board.tick()
        self.assertFalse(self.board.buzzer_on)

    def test_contact_and_hold_do_not_beep(self):
        self.send('ARM,1')
        self.send('CUE,2,CONTACT,85,100')
        self.assertFalse(self.board.buzzer_on)
        self.clock.ms += 600
        self.send('KA,3')
        self.send('CUE,4,STABLE_HOLD,90,100')
        self.assertFalse(self.board.buzzer_on)

    def test_disarmed_board_never_beeps(self):
        self.assertEqual(self.send('CUE,1,SLIP,90,300'), [Err(1, 'NOT_ARMED')])
        self.assertFalse(self.board.buzzer_on)

    def test_stop_button_and_watchdog_silence_it(self):
        self.armed_beep()
        self.send('STOP,3')
        self.assertFalse(self.board.buzzer_on)

        self.setUp()
        self.armed_beep()
        self.board.press_button()
        self.assertFalse(self.board.buzzer_on)

        self.setUp()
        self.board.write(b'ARM,1\n')
        self.board.write(b'CUE,2,SLIP,90,500\n')
        drain(self.board)
        self.clock.ms += 100   # still beeping: the beep is capped at 120 ms
        self.board.tick()
        self.assertTrue(self.board.buzzer_on)
        self.board.beep_end_ms = 10_000    # pretend a long beep, so only the watchdog can end it
        self.board.cue_end_ms = 10_000
        self.clock.ms += 501
        self.board.tick()
        self.assertEqual(self.board.state, 'FAULT')
        self.assertFalse(self.board.buzzer_on)


if __name__ == '__main__':
    unittest.main()
