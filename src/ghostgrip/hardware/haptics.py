"""Haptic safety policy: decides whether a simulation event may become a physical cue.

Every gate from plan section 7.3 lives here: verified config, arming, confidence,
one cue per event, cooldown, angle clamp and the event-to-output table.
Raw camera data never reaches this module; only typed simulation events do.
"""
from dataclasses import dataclass
import math
import time

from .protocol import CueEvent

PLACEHOLDER = 'VERIFY'

DEFAULT_CONFIG = {
    'servo_enabled': False,          # True only after the servo passes the section 8 safety gate
    'servo_neutral_deg': PLACEHOLDER,
    'servo_min_deg': PLACEHOLDER,
    'servo_max_deg': PLACEHOLDER,
    'contact_offset_deg': 5,
    'cue_duration_ms': 150,
    'cue_cooldown_ms': 500,
    'min_tracking_quality': 0.6,
}

# Firmware neutral angle, sent with LED-only cues. The firmware also ignores the angle for
# every event except CONTACT, so these cues cannot move the servo.
LED_ONLY_ANGLE = 90

# Event -> does it move the servo? Everything else is LED/buzzer only on the board.
SERVO_EVENTS = {'CONTACT'}
# Simulation event kind (contracts.SIM_EVENT_KINDS) -> event name the firmware understands.
# RELEASE has no cue.
CUE_EVENTS = {'CONTACT': 'CONTACT', 'HOLD': 'STABLE_HOLD', 'SLIP': 'SLIP', 'DAMAGE': 'DAMAGE'}
STOP_EVENTS = {'TRACKING_LOST', 'STOP'}


class ConfigError(ValueError):
    pass


def validate_config(config):
    """Refuse servo use while any servo value is still the VERIFY placeholder."""
    merged = {**DEFAULT_CONFIG, **config}
    if merged['servo_enabled']:
        for key in ('servo_neutral_deg', 'servo_min_deg', 'servo_max_deg'):
            if merged[key] == PLACEHOLDER or not isinstance(merged[key], int):
                raise ConfigError(f'{key} is not verified; refusing to enable the servo')
        if not merged['servo_min_deg'] <= merged['servo_neutral_deg'] <= merged['servo_max_deg']:
            raise ConfigError('servo_neutral_deg must sit between servo_min_deg and servo_max_deg')
    return merged


@dataclass
class Decision:
    action: str            # 'cue', 'stop' or 'suppressed'
    reason: str
    cue: CueEvent = None


class HapticPolicy:
    def __init__(self, config=None, clock_ms=None):
        self.config = validate_config(config or {})
        self.clock_ms = clock_ms or (lambda: int(time.monotonic() * 1000))
        self.last_cue_ms = None
        self.cued_event_ids = set()

    def _angle_for(self, kind):
        c = self.config
        if not c['servo_enabled'] or kind not in SERVO_EVENTS:
            return LED_ONLY_ANGLE
        target = c['servo_neutral_deg'] + c['contact_offset_deg']
        return max(c['servo_min_deg'], min(c['servo_max_deg'], target))

    def filter(self, event_id, kind, tracking_quality, link_armed):
        """Return a Decision for one simulation event (one per state transition)."""
        if kind in STOP_EVENTS:
            return Decision('stop', kind)
        if kind not in CUE_EVENTS:
            return Decision('suppressed', 'UNMAPPED')
        if not link_armed:
            return Decision('suppressed', 'NOT_ARMED')
        if (not isinstance(tracking_quality, (int, float)) or not math.isfinite(tracking_quality)
                or tracking_quality < self.config['min_tracking_quality']):
            return Decision('suppressed', 'LOW_CONFIDENCE')
        if event_id in self.cued_event_ids:
            return Decision('suppressed', 'DUPLICATE')
        now = self.clock_ms()
        if self.last_cue_ms is not None and now - self.last_cue_ms < self.config['cue_cooldown_ms']:
            return Decision('suppressed', 'COOLDOWN')

        self.last_cue_ms = now
        self.cued_event_ids.add(event_id)
        cue = CueEvent(seq=0, kind=CUE_EVENTS[kind], safe_angle_deg=self._angle_for(kind),
                       duration_ms=self.config['cue_duration_ms'],
                       no_contact_mode=not self.config['servo_enabled'])
        return Decision('cue', 'OK', cue)
