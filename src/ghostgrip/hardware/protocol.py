"""Serial protocol v1 (plan section 6.3): ASCII, newline-delimited, 115200 baud.

Member C addition to the plan's protocol: KA,seq is a keepalive that refreshes the board
watchdog but cannot arm it, so a stopped board is never re-armed by accident.

CueEvent and DeviceState follow the frozen contracts in plan section 6.1. They live here
until the shared contracts.py v1 lands; then import them from there instead.
"""
from dataclasses import dataclass
from typing import Optional

BAUD = 115200


@dataclass
class CueEvent:
    seq: int
    kind: str              # CONTACT, STABLE_HOLD, SLIP, DAMAGE
    safe_angle_deg: int
    duration_ms: int
    no_contact_mode: bool = True


@dataclass
class DeviceState:
    uptime_ms: int
    armed: bool
    touch: bool
    piezo: bool
    ax_g: Optional[float]   # None means the accelerometer is absent, never zero
    ay_g: Optional[float]
    az_g: Optional[float]
    rotary_raw: int
    last_ack_seq: int = -1


@dataclass
class Ack:
    seq: int
    uptime_ms: int
    status: str


@dataclass
class Err:
    seq: int
    reason: str


@dataclass
class Ready:
    firmware: str


def encode_arm(seq):
    return f'ARM,{seq}\n'.encode('ascii')


def encode_keepalive(seq):
    """Refreshes the board watchdog while armed. Unlike ARM it can never re-arm the board."""
    return f'KA,{seq}\n'.encode('ascii')


def encode_stop(seq):
    return f'STOP,{seq}\n'.encode('ascii')


def encode_cue(cue: CueEvent):
    return f'CUE,{cue.seq},{cue.kind},{cue.safe_angle_deg},{cue.duration_ms}\n'.encode('ascii')


def _accel(text):
    return None if text == 'NA' else float(text)


def parse_line(raw):
    """Turn one line from the board into Ack, Err, DeviceState or Ready. Returns None for noise."""
    if isinstance(raw, bytes):
        raw = raw.decode('ascii', errors='ignore')
    fields = raw.strip().split(',')
    try:
        if fields[0] == 'ACK' and len(fields) == 4:
            return Ack(int(fields[1]), int(fields[2]), fields[3])
        if fields[0] == 'ERR' and len(fields) == 3:
            return Err(int(fields[1]), fields[2])
        if fields[0] == 'S' and len(fields) == 9:
            return DeviceState(
                uptime_ms=int(fields[1]),
                ax_g=_accel(fields[2]), ay_g=_accel(fields[3]), az_g=_accel(fields[4]),
                piezo=fields[5] == '1', touch=fields[6] == '1',
                rotary_raw=int(fields[7]), armed=fields[8] == '1')
        if fields[0] == 'READY':
            return Ready(','.join(fields[1:]))
    except ValueError:
        return None
    return None
