"""Load and check config.yaml, the team's single source of truth for settings.

    from ghostgrip.config import load_config
    cfg = load_config()                  # reads config.yaml at the repo root
    cfg.ai.prediction_horizon_ms         # 120

Check the file from a terminal:   PYTHONPATH=src python -m ghostgrip.config

Rules this loader enforces:
  * every key must be present and no unknown key is allowed (typos fail loudly);
  * values must have the right type and sit inside a sensible range;
  * the servo cannot be enabled while any servo angle is still the placeholder
    "VERIFY", and the angles must stay inside the firmware window (plan sections 6.5, 8.4).
"""
from dataclasses import dataclass
import math
from pathlib import Path
import re

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[2] / 'config.yaml'

PLACEHOLDER = 'VERIFY'

# Must match firmware/ghostgrip_uno/pins.h (SERVO_MIN_DEG, SERVO_MAX_DEG) and MAX_CUE_MS in the
# firmware. The board rejects anything outside these, so the host must never ask for it.
FIRMWARE_ANGLE_WINDOW = (80, 100)
FIRMWARE_MAX_CUE_MS = 500

LOW_CONFIDENCE_BEHAVIORS = ('disable_assist',)
PARTICIPANT_ID = re.compile(r'^[A-Za-z0-9_-]{1,32}$')


class ConfigError(ValueError):
    """The configuration file is missing, malformed or unsafe."""


# ---------------------------------------------------------------- small checkers
def _int(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f'{name} must be a whole number, got {value!r}')
    if not low <= value <= high:
        raise ConfigError(f'{name} must be between {low} and {high}, got {value}')
    return value


def _float(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ConfigError(f'{name} must be a finite number, got {value!r}')
    if not low <= value <= high:
        raise ConfigError(f'{name} must be between {low} and {high}, got {value}')
    return float(value)


def _bool(value, name):
    if not isinstance(value, bool):
        raise ConfigError(f'{name} must be true or false, got {value!r}')
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f'{name} must be non-empty text, got {value!r}')
    return value.strip()


def _angle(value, name):
    """A servo angle is a whole number of degrees, or the placeholder VERIFY."""
    if value == PLACEHOLDER:
        return PLACEHOLDER
    return _int(value, name, 0, 180)


def _section(data, name, keys):
    """Return data[name] after checking it has exactly the expected keys."""
    section = data.get(name)
    if not isinstance(section, dict):
        raise ConfigError(f'section "{name}" is missing or is not a group of settings')
    missing = [k for k in keys if k not in section]
    unknown = [k for k in section if k not in keys]
    if missing:
        raise ConfigError(f'{name}: missing setting(s) {missing}')
    if unknown:
        raise ConfigError(f'{name}: unknown setting(s) {unknown}; check the spelling')
    return section


# ---------------------------------------------------------------- the settings
@dataclass(frozen=True)
class CameraConfig:
    index: int
    mirror: bool
    target_fps: int


@dataclass(frozen=True)
class SimulationConfig:
    width: int
    height: int
    physics_hz: int


@dataclass(frozen=True)
class AIConfig:
    prediction_horizon_ms: int
    grip_confirm_frames: int
    adapt_alpha: float
    low_confidence_behavior: str
    freeze_adaptation: bool


@dataclass(frozen=True)
class HardwareConfig:
    enabled: bool
    port: str
    baud: int
    watchdog_ms: int
    keepalive_ms: int
    cue_cooldown_ms: int
    cue_duration_ms: int
    min_tracking_quality: float
    contact_offset_deg: int
    servo_enabled: bool
    servo_neutral_deg: object   # int, or "VERIFY" while the servo is still off
    servo_min_deg: object
    servo_max_deg: object


@dataclass(frozen=True)
class ExperimentConfig:
    save_raw_video: bool
    participant_id: str


@dataclass(frozen=True)
class Config:
    camera: CameraConfig
    simulation: SimulationConfig
    ai: AIConfig
    hardware: HardwareConfig
    experiment: ExperimentConfig


def _build_hardware(data):
    h = _section(data, 'hardware', (
        'enabled', 'port', 'baud', 'watchdog_ms', 'keepalive_ms', 'cue_cooldown_ms',
        'cue_duration_ms', 'min_tracking_quality', 'contact_offset_deg',
        'servo_enabled', 'servo_neutral_deg', 'servo_min_deg', 'servo_max_deg'))
    hardware = HardwareConfig(
        enabled=_bool(h['enabled'], 'hardware.enabled'),
        port=_text(h['port'], 'hardware.port'),
        baud=_int(h['baud'], 'hardware.baud', 1200, 2_000_000),
        watchdog_ms=_int(h['watchdog_ms'], 'hardware.watchdog_ms', 100, 60_000),
        keepalive_ms=_int(h['keepalive_ms'], 'hardware.keepalive_ms', 10, 60_000),
        cue_cooldown_ms=_int(h['cue_cooldown_ms'], 'hardware.cue_cooldown_ms', 0, 60_000),
        cue_duration_ms=_int(h['cue_duration_ms'], 'hardware.cue_duration_ms', 1, FIRMWARE_MAX_CUE_MS),
        min_tracking_quality=_float(h['min_tracking_quality'], 'hardware.min_tracking_quality', 0.0, 1.0),
        contact_offset_deg=_int(h['contact_offset_deg'], 'hardware.contact_offset_deg', 0, 20),
        servo_enabled=_bool(h['servo_enabled'], 'hardware.servo_enabled'),
        servo_neutral_deg=_angle(h['servo_neutral_deg'], 'hardware.servo_neutral_deg'),
        servo_min_deg=_angle(h['servo_min_deg'], 'hardware.servo_min_deg'),
        servo_max_deg=_angle(h['servo_max_deg'], 'hardware.servo_max_deg'))

    if hardware.keepalive_ms * 2 > hardware.watchdog_ms:
        raise ConfigError('hardware.keepalive_ms must be at most half of hardware.watchdog_ms, '
                          'or one late keepalive disarms the board')
    if hardware.servo_enabled:
        if not hardware.enabled:
            raise ConfigError('hardware.servo_enabled needs hardware.enabled to be true')
        angles = (hardware.servo_min_deg, hardware.servo_neutral_deg, hardware.servo_max_deg)
        if PLACEHOLDER in angles:
            raise ConfigError('servo angles are still VERIFY; refusing to enable the servo '
                              'until the hardware safety gate (plan section 8) has passed')
        low, high = FIRMWARE_ANGLE_WINDOW
        if not low <= hardware.servo_min_deg <= hardware.servo_neutral_deg <= hardware.servo_max_deg <= high:
            raise ConfigError(f'servo angles must satisfy {low} <= min <= neutral <= max <= {high} '
                              '(the firmware rejects anything outside this window)')
    return hardware


def parse_config(data):
    """Check a settings dictionary and return a Config. Raises ConfigError on any problem."""
    if not isinstance(data, dict):
        raise ConfigError('the configuration must be a set of named sections')
    sections = ('camera', 'simulation', 'ai', 'hardware', 'experiment')
    unknown = [k for k in data if k not in sections]
    if unknown:
        raise ConfigError(f'unknown section(s) {unknown}; allowed: {list(sections)}')

    c = _section(data, 'camera', ('index', 'mirror', 'target_fps'))
    camera = CameraConfig(_int(c['index'], 'camera.index', 0, 20),
                          _bool(c['mirror'], 'camera.mirror'),
                          _int(c['target_fps'], 'camera.target_fps', 1, 120))

    s = _section(data, 'simulation', ('width', 'height', 'physics_hz'))
    simulation = SimulationConfig(_int(s['width'], 'simulation.width', 160, 4096),
                                  _int(s['height'], 'simulation.height', 90, 4096),
                                  _int(s['physics_hz'], 'simulation.physics_hz', 10, 240))

    a = _section(data, 'ai', ('prediction_horizon_ms', 'grip_confirm_frames', 'adapt_alpha',
                              'low_confidence_behavior', 'freeze_adaptation'))
    behavior = a['low_confidence_behavior']
    if behavior not in LOW_CONFIDENCE_BEHAVIORS:
        raise ConfigError(f'ai.low_confidence_behavior must be one of {LOW_CONFIDENCE_BEHAVIORS}, '
                          f'got {behavior!r}')
    alpha = _float(a['adapt_alpha'], 'ai.adapt_alpha', 0.0, 1.0)
    if alpha == 0.0:
        raise ConfigError('ai.adapt_alpha must be greater than 0')
    ai = AIConfig(_int(a['prediction_horizon_ms'], 'ai.prediction_horizon_ms', 10, 500),
                  _int(a['grip_confirm_frames'], 'ai.grip_confirm_frames', 1, 30),
                  alpha, behavior, _bool(a['freeze_adaptation'], 'ai.freeze_adaptation'))

    e = _section(data, 'experiment', ('save_raw_video', 'participant_id'))
    participant = _text(e['participant_id'], 'experiment.participant_id')
    if not PARTICIPANT_ID.match(participant):
        raise ConfigError('experiment.participant_id may only use letters, digits, - and _ '
                          '(1-32 characters); use an anonymous ID, never a name or email')
    experiment = ExperimentConfig(_bool(e['save_raw_video'], 'experiment.save_raw_video'), participant)

    return Config(camera, simulation, ai, _build_hardware(data), experiment)


def load_config(path=None):
    """Read, parse and check a YAML settings file (default: config.yaml at the repo root)."""
    path = Path(path) if path is not None else DEFAULT_PATH
    try:
        text = path.read_text(encoding='utf-8')
    except OSError as error:
        raise ConfigError(f'cannot read {path}: {error}') from error
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ConfigError(f'{path} is not valid YAML: {error}') from error
    return parse_config(data)


def to_haptic_config(config):
    """The settings ghostgrip.hardware.haptics.HapticPolicy expects, from a loaded Config."""
    h = config.hardware
    return {
        'servo_enabled': h.servo_enabled,
        'servo_neutral_deg': h.servo_neutral_deg,
        'servo_min_deg': h.servo_min_deg,
        'servo_max_deg': h.servo_max_deg,
        'contact_offset_deg': h.contact_offset_deg,
        'cue_duration_ms': h.cue_duration_ms,
        'cue_cooldown_ms': h.cue_cooldown_ms,
        'min_tracking_quality': h.min_tracking_quality,
    }


def main():
    config = load_config()
    print(f'{DEFAULT_PATH.name}: OK')
    for name, section in vars(config).items():
        print(f'  {name}: {section}')
    if not config.hardware.servo_enabled:
        print('  servo is OFF (as it must be until the hardware safety gate has passed)')


if __name__ == '__main__':
    main()
