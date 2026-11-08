"""In-process fake of firmware/ghostgrip_uno, so tests and demos run without a board.

It mirrors the firmware rule for rule: boots DISARMED, STOP always wins, ARM needs STOP first
after a latch, stale or malformed commands are rejected, only CONTACT moves the servo, and the
watchdog faults if the host goes quiet. SLIP and DAMAGE beep briefly; STOP and faults silence it. Keep the two in step when either changes.
Behaves like a pyserial port: write(bytes), readline() -> bytes (b'' means timeout), in_waiting.
"""
from collections import deque
import time

WATCHDOG_MS = 500
CUE_COOLDOWN_MS = 500
MAX_CUE_MS = 500
BEEP_MS = 120
MAX_LINE = 48
MAX_FIELDS = 5
MAX_NUMBER = 100_000_000
MAX_BAD_PACKETS = 5
SERVO_MIN_DEG, SERVO_MAX_DEG, SERVO_NEUTRAL_DEG = 80, 100, 90
CUE_EVENTS = {'CONTACT', 'STABLE_HOLD', 'SLIP', 'DAMAGE'}
BEEP_EVENTS = {'SLIP', 'DAMAGE'}


def _number(text):
    """Firmware parseNumber: digits only, at most MAX_NUMBER. Returns None otherwise."""
    if not text.isascii() or not text.isdigit() or int(text) > MAX_NUMBER:
        return None
    return int(text)


class MockArduino:
    def __init__(self, clock_ms=None):
        self.clock_ms = clock_ms or (lambda: int(time.monotonic() * 1000))
        self.boot_ms = self.clock_ms()
        self.state = 'DISARMED'
        self.latched = False
        self.last_seq = -1
        self.bad_packets = 0
        self.last_host_ms = 0
        self.last_cue_start_ms = None
        self.cue_end_ms = 0
        self.servo_angle = SERVO_NEUTRAL_DEG
        self.buzzer_on = False
        self.beep_end_ms = 0
        self.button_held = False
        self.touch = False
        self.piezo = False
        self.dead = False      # True simulates a pulled USB cable: writes vanish, nothing comes back
        self.rx = bytearray()  # bytes received but not yet newline-terminated
        self.outbox = deque(['READY,mock_arduino,v1'])
        self.is_open = True

    def uptime(self):
        return self.clock_ms() - self.boot_ms

    def armed(self):
        return self.state in ('ARMED', 'CUE')

    # pyserial-like interface
    @property
    def in_waiting(self):
        self.tick()
        return 0 if self.dead else sum(len(line) + 1 for line in self.outbox)

    def write(self, data):
        if self.dead:
            return len(data)
        self.rx.extend(data)
        while b'\n' in self.rx:
            raw, _, rest = self.rx.partition(b'\n')
            self.rx = bytearray(rest)
            self._receive_line(bytes(raw).replace(b'\r', b''))
        return len(data)

    def readline(self):
        self.tick()
        if self.dead or not self.outbox:
            return b''
        return (self.outbox.popleft() + '\n').encode('ascii')

    def close(self):
        self.is_open = False

    # firmware behaviour
    def tick(self):
        if self.button_held and self.state != 'DISARMED':
            self._disarm('DISARMED')
            self.latched = True
        now = self.uptime()
        if self.armed() and now - self.last_host_ms > WATCHDOG_MS:
            self._disarm('FAULT')
            self.latched = True
        if self.buzzer_on and now >= self.beep_end_ms:
            self.buzzer_on = False
        if self.state == 'CUE' and now >= self.cue_end_ms:
            self.servo_angle = SERVO_NEUTRAL_DEG
            self.buzzer_on = False
            self.state = 'ARMED'

    def press_button(self, held=False):
        """Momentary press by default; held=True keeps it down until release_button()."""
        self.button_held = True
        self.tick()
        self.button_held = held

    def release_button(self):
        self.button_held = False

    def sample(self):
        self.outbox.append(f'S,{self.uptime()},NA,NA,NA,{int(self.piezo)},{int(self.touch)},512,'
                           f'{int(self.armed())}')

    def _disarm(self, next_state):
        self.servo_angle = SERVO_NEUTRAL_DEG
        self.buzzer_on = False
        self.state = next_state

    def _ack(self, seq):
        self.outbox.append(f'ACK,{seq},{self.uptime()},{self.state}')

    def _err(self, seq, reason):
        self.outbox.append(f'ERR,{seq},{reason}')

    def _bad(self, seq, reason):
        self._err(seq, reason)
        self.bad_packets += 1
        if self.bad_packets >= MAX_BAD_PACKETS and self.state != 'DISARMED':
            self._disarm('FAULT')
            self.latched = True

    def _receive_line(self, raw):
        if not raw:
            return  # firmware ignores blank lines
        if len(raw) > MAX_LINE or any(b < 32 or b > 126 for b in raw):
            self._bad(-1, 'MALFORMED')
            return
        self.tick()
        self._handle(raw.decode('ascii'))

    def _handle(self, line):
        fields = line.split(',')
        if len(fields) < 2 or len(fields) > MAX_FIELDS or '' in fields:
            self._bad(-1, 'MALFORMED')
            return
        cmd, seq = fields[0], _number(fields[1])

        if cmd == 'STOP' and len(fields) == 2:
            self._disarm('DISARMED')
            self.latched = False
            self.bad_packets = 0
            if seq is not None and seq > self.last_seq:
                self.last_seq = seq
            self._ack(-1 if seq is None else seq)
            return

        if seq is None:
            self._bad(-1, 'BAD_SEQ')
            return
        if seq <= self.last_seq:
            self._bad(seq, 'STALE_SEQ')
            return

        if cmd == 'ARM' and len(fields) == 2:
            self.last_seq = seq
            if self.button_held:
                self._err(seq, 'BUTTON_HELD')
                return
            if self.latched:
                self._err(seq, 'LATCHED')
                return
            self.last_host_ms = self.uptime()
            self.bad_packets = 0
            if not self.armed():
                self.state = 'ARMED'
            self._ack(seq)
            return

        if cmd == 'KA' and len(fields) == 2:
            self.last_seq = seq
            if not self.armed():
                self._err(seq, 'NOT_ARMED')
                return
            self.last_host_ms = self.uptime()
            self._ack(seq)
            return

        if cmd == 'CUE' and len(fields) == 5:
            event = fields[2]
            angle, duration = _number(fields[3]), _number(fields[4])
            if event not in CUE_EVENTS:
                self._bad(seq, 'BAD_EVENT')
                return
            if angle is None or duration is None:
                self._bad(seq, 'BAD_FIELD')
                return
            self.last_seq = seq
            if self.state != 'ARMED':
                self._err(seq, 'BUSY' if self.state == 'CUE' else 'NOT_ARMED')
                return
            if not SERVO_MIN_DEG <= angle <= SERVO_MAX_DEG:
                self._bad(seq, 'ANGLE_RANGE')
                return
            if duration == 0 or duration > MAX_CUE_MS:
                self._bad(seq, 'DURATION_RANGE')
                return
            now = self.uptime()
            if self.last_cue_start_ms is not None and now - self.last_cue_start_ms < CUE_COOLDOWN_MS:
                self._err(seq, 'COOLDOWN')
                return
            self.last_host_ms = now
            self.last_cue_start_ms = now
            self.cue_end_ms = now + duration
            self.state = 'CUE'
            if event == 'CONTACT':
                self.servo_angle = angle
            if event in BEEP_EVENTS:
                self.buzzer_on = True
                self.beep_end_ms = now + min(duration, BEEP_MS)
            self.bad_packets = 0
            self._ack(seq)
            return

        self._bad(seq, 'UNKNOWN_CMD')
