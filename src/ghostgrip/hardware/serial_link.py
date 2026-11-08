"""Laptop side of the Arduino link: port auto-detect, sequence numbers, ACK timeout and retry.

A missing board, a dropped ACK or a pulled cable never raises into the simulation. The link
reports connected=False and the app keeps running laptop-only (plan: NFR-01 graceful failure).
"""
import time

from .protocol import (BAUD, Ack, CueEvent, DeviceState, Err, Ready, encode_arm,
                       encode_cue, encode_keepalive, encode_stop, parse_line)

PORT_HINTS = ('usbserial', 'usbmodem', 'ttyACM', 'ttyUSB', 'COM')
POLL_BUDGET_S = 0.02


def find_port():
    """Return the first port that looks like an Arduino, or None."""
    from serial.tools import list_ports
    for port in list_ports.comports():
        if any(hint in port.device for hint in PORT_HINTS):
            return port.device
    return None


def open_serial(port=None, boot_wait_s=3.0):
    """Open the real board. Opening the port resets an Uno, so wait for its READY line."""
    import serial
    port = port or find_port()
    if port is None:
        raise RuntimeError('No Arduino serial port found. Is the USB cable plugged in?')
    transport = serial.Serial(port, BAUD, timeout=0.05, write_timeout=0.2)
    deadline = time.monotonic() + boot_wait_s
    while time.monotonic() < deadline:
        if isinstance(parse_line(transport.readline()), Ready):
            break
    return transport


class SerialLink:
    def __init__(self, transport, ack_timeout_s=0.3, retries=1):
        self.transport = transport
        self.ack_timeout_s = ack_timeout_s
        self.retries = retries
        self.seq = 0
        self.connected = True
        self.armed = False
        self.device = None          # latest DeviceState from S packets
        self.last_ack_seq = -1      # latest sequence number the board acknowledged
        self.last_error = None      # latest Err from the board
        self.io_failed = False      # set when the transport raised during the current call

    def _next_seq(self):
        self.seq += 1
        return self.seq

    def _lost(self):
        self.connected = False
        self.armed = False
        self.io_failed = True

    def _write(self, data):
        try:
            self.transport.write(data)
            return True
        except Exception:  # pyserial raises SerialException/OSError when the cable is pulled
            self._lost()
            return False

    def _readline(self):
        try:
            return self.transport.readline()
        except Exception:
            self._lost()
            return b''

    def _handle(self, message):
        if isinstance(message, DeviceState):
            message.last_ack_seq = self.last_ack_seq
            self.device = message
            self.armed = message.armed
        elif isinstance(message, Err):
            self.last_error = message

    def poll(self):
        """Read only what is already waiting (sensor packets, errors); never blocks for long."""
        deadline = time.monotonic() + POLL_BUDGET_S
        while time.monotonic() < deadline:
            try:
                if not self.transport.in_waiting:
                    return
            except Exception:
                self._lost()
                return
            message = parse_line(self._readline())
            if message is not None:
                self._handle(message)

    def _send(self, make_payload, retries):
        """Send with a fresh sequence number; wait for ACK or ERR; retry on silence."""
        self.io_failed = False
        for _ in range(retries + 1):
            seq = self._next_seq()
            if not self._write(make_payload(seq)):
                return None
            deadline = time.monotonic() + self.ack_timeout_s
            while time.monotonic() < deadline:
                message = parse_line(self._readline())
                if self.io_failed:
                    return None
                if isinstance(message, (Ack, Err)) and message.seq == seq:
                    self.connected = True
                    if isinstance(message, Ack):
                        self.last_ack_seq = seq
                    else:
                        self.last_error = message
                    return message
                if message is not None:
                    self._handle(message)
        self._lost()
        return None

    def arm(self):
        """Explicit arm. After a button press, watchdog or fault the board needs stop() first."""
        reply = self._send(encode_arm, self.retries)
        self.armed = isinstance(reply, Ack) and reply.status in ('ARMED', 'CUE')
        return reply

    def keepalive(self):
        """Feed the board watchdog. Call at least every 250 ms while armed. Never re-arms."""
        if not self.armed:
            return None
        reply = self._send(encode_keepalive, self.retries)
        if not isinstance(reply, Ack):
            self.armed = False
        return reply

    def stop(self):
        reply = self._send(encode_stop, self.retries)
        self.armed = False
        return reply

    def cue(self, cue: CueEvent):
        """Send one cue. Never retried: a lost ACK must not make the same cue play twice."""
        def payload(seq):
            cue.seq = seq
            return encode_cue(cue)
        return self._send(payload, retries=0)

    def close(self):
        try:
            if self.connected:
                self.stop()
        finally:
            try:
                self.transport.close()
            except Exception:
                pass
