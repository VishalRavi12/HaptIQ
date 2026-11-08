"""Phase 2 glue: connects every member's Phase 1 code to the live simulation.

Nothing here changes another member's module; it only calls their public API.

    AIPipeline   Member D: KalmanPredictor (ghost), ErrorTracker (is it better than a guess?),
                 AssistPolicy (assistance gain, learns between trials), IntentMachine (HUD/log)
    HapticBridge Member C: SimEventV2 -> HapticPolicy safety gate -> SerialLink -> Arduino
    lost_hand    a safe "no hand" HandStateV2 when Member A's camera stops for good

Safety (AGENTS.md): every physical cue goes through Member C's HapticPolicy (arming, tracking
quality, one cue per event, cooldown, angle clamp) and then the firmware's own checks
(arming, angle window, cooldown, 500 ms watchdog). The board is never armed automatically:
only the operator's H key arms it. Tracking lost for TRACKING_LOSS_STOP_MS sends STOP.
"""
import dataclasses
from dataclasses import dataclass
import time
from typing import List, Optional

from contracts import HandStateV2 as HandState
from contracts import Prediction
from ghostgrip.ai.adapt import AssistDecision, AssistPolicy, trial_jitter
from ghostgrip.ai.intent import IntentMachine, IntentState, zones_for_scene
from ghostgrip.ai.predict import LOST_SIGMA, ErrorTracker, KalmanPredictor
from ghostgrip.config import to_haptic_config
from ghostgrip.hardware.haptics import HapticPolicy
from ghostgrip.hardware.mock_arduino import MockArduino
from ghostgrip.hardware.protocol import Ack, Err
from ghostgrip.hardware.serial_link import SerialLink, find_port, open_serial

# Same as the simulation's tracking timeout (world.TRACKING_TIMEOUT_MS): a loss this long
# releases the virtual object and disarms the board. Shorter blips cannot cue anyway,
# because HapticPolicy refuses any cue below hardware.min_tracking_quality.
TRACKING_LOSS_STOP_MS = 300
MOCK_SAMPLE_MS = 50          # the real firmware sends a sensor packet every 50 ms


def lost_hand(t_ms: int, last: Optional[HandState]) -> HandState:
    """No usable hand: keep the last pointer, quality 0, tracking_ok False (shared contract)."""
    xy = last.pointer_xy if last is not None else (0.5, 0.5)
    q = last.closure_q if last is not None else 0.0
    return HandState(t_ms, xy, q, 0.0, (0.0, 0.0), 0.0, False, 0.0)


# ====================================================================== Member D
@dataclass
class AIFrame:
    prediction: Prediction
    decision: AssistDecision
    intent: IntentState
    trusted: bool               # False while the Kalman filter is worse than a simple guess


class AIPipeline:
    """Member D's AI, run once per frame on the same HandStateV2 the world gets."""

    def __init__(self, config, scene: str):
        self.config = config
        self.predictor = KalmanPredictor.from_config(config)
        self.tracker = ErrorTracker()
        self.policy = AssistPolicy.from_config(config)
        self.intent = IntentMachine.from_config(zones_for_scene(scene), config)
        self.trial_states: List[HandState] = []
        self.last: Optional[AIFrame] = None

    def set_scene(self, scene: str) -> None:
        self.intent = IntentMachine.from_config(zones_for_scene(scene), self.config)

    def new_trial(self) -> None:
        """World was reset: forget intent evidence and the per-trial hand history."""
        self.intent.reset()
        self.trial_states = []

    def update(self, hand: HandState, record_for_trial: bool) -> AIFrame:
        prediction = self.predictor.update(hand)
        self.tracker.record(hand, prediction)
        trusted = not self.tracker.is_worse_than_baseline()
        decision = self.policy.decide(hand, prediction, predictor_trusted=trusted)
        intent = self.intent.update(hand)
        if record_for_trial:
            self.trial_states.append(hand)
        self.last = AIFrame(prediction, decision, intent, trusted)
        return self.last

    def end_trial(self, outcome) -> None:
        """Between trials only: let the assist policy learn how steady this user is."""
        self.policy.end_trial(outcome, trial_jitter(self.trial_states))
        self.trial_states = []

    @staticmethod
    def ghost_xy(hand: HandState, frame: Optional[AIFrame]):
        """Predicted pointer (0-1), or None when there is no real forecast to draw."""
        if frame is None or not hand.tracking_ok:
            return None
        if max(frame.prediction.sigma_xy) >= LOST_SIGMA:
            return None
        return frame.prediction.future_pointer_xy


# ====================================================================== Member C
class HapticBridge:
    """Simulation events -> Member C's safety gate -> Arduino. Never raises into the app."""

    def __init__(self, link: Optional[SerialLink], policy: HapticPolicy, keepalive_ms: int = 250,
                 port_label: str = "", clock_ms=None):
        self.link = link
        self.policy = policy
        self.keepalive_ms = keepalive_ms
        self.port_label = port_label
        self.clock_ms = clock_ms or (lambda: int(time.monotonic() * 1000))
        self.message = "" if link else "no board: running laptop-only"
        self.last_cue = ""
        self._last_ka_ms: Optional[int] = None
        self._last_sample_ms: Optional[int] = None
        self._lost_since: Optional[int] = None
        self._last_device = None

    # ------------------------------------------------------------------ setup
    @classmethod
    def connect(cls, target: str, config, clock_ms=None) -> "HapticBridge":
        """target: 'mock' (fake board), 'auto' (find the Arduino), or a port such as 'COM4'."""
        policy = HapticPolicy(to_haptic_config(config))
        keepalive = config.hardware.keepalive_ms
        if target.lower() == "mock":
            return cls(SerialLink(MockArduino()), policy, keepalive, "mock board", clock_ms)
        port = None if target.lower() == "auto" else target
        if port is None and str(config.hardware.port).upper() != "AUTO":
            port = config.hardware.port
        port = port or find_port()
        if port is None:
            print("Hardware: no Arduino serial port found (is the USB driver installed?). "
                  "Running laptop-only.")
            return cls(None, policy, keepalive, "", clock_ms)
        print(f"Hardware: opening {port} (the board restarts, about 2 s) ...")
        try:
            transport = open_serial(port)
        except Exception as error:   # busy port, unplugged cable, missing driver
            print(f"Hardware: could not open {port}: {error}. Running laptop-only.")
            return cls(None, policy, keepalive, "", clock_ms)
        print(f"Hardware: connected on {port}. Board is DISARMED; press H to arm.")
        return cls(SerialLink(transport), policy, keepalive, port, clock_ms)

    # ------------------------------------------------------------------ state
    @property
    def connected(self) -> bool:
        return self.link is not None and self.link.connected

    @property
    def armed(self) -> bool:
        return self.connected and self.link.armed

    @property
    def status(self) -> str:
        if self.link is None:
            return "NO BOARD"
        if not self.link.connected:
            return "DISCONNECTED"
        return "ARMED" if self.link.armed else "DISARMED"

    # ------------------------------------------------------------------ operator keys
    def arm(self) -> None:
        if not self.connected:
            self.message = "cannot arm: no board connected"
            return
        reply = self.link.arm()
        if isinstance(reply, Ack) and self.link.armed:
            self._last_ka_ms = self.clock_ms()
            self._lost_since = None
            self.message = "ARMED (X = STOP)"
        elif isinstance(reply, Err) and reply.reason == "LATCHED":
            self.message = "ARM refused: board latched. Press X (STOP), then H"
        elif isinstance(reply, Err) and reply.reason == "BUTTON_HELD":
            self.message = "ARM refused: release the STOP button"
        elif isinstance(reply, Err):
            self.message = f"ARM refused: {reply.reason}"
        else:
            self.message = "ARM failed: board did not answer"
        print("Hardware:", self.message)

    def stop(self, why: str = "operator") -> None:
        if not self.connected:
            return
        self.link.stop()
        self.message = f"STOP ({why}); press H to arm again"
        print("Hardware:", self.message)

    # ------------------------------------------------------------------ every frame
    def update(self, hand: HandState):
        """Watchdog keepalive, tracking-loss STOP and sensor packets. Returns a new
        DeviceState when the board sent one since the last call, else None."""
        if not self.connected:
            return None
        now = self.clock_ms()

        if hand.tracking_ok:
            self._lost_since = None
        elif self._lost_since is None:
            self._lost_since = now
        if (self.link.armed and self._lost_since is not None
                and now - self._lost_since >= TRACKING_LOSS_STOP_MS):
            self.stop("tracking lost")

        if self.link.armed and (self._last_ka_ms is None or now - self._last_ka_ms >= self.keepalive_ms):
            self._last_ka_ms = now
            reply = self.link.keepalive()
            if not isinstance(reply, Ack):
                why = reply.reason if isinstance(reply, Err) else "no answer"
                self.message = f"board disarmed itself ({why}); press X, then H"
                print("Hardware:", self.message)

        if isinstance(self.link.transport, MockArduino):
            if self._last_sample_ms is None or now - self._last_sample_ms >= MOCK_SAMPLE_MS:
                self._last_sample_ms = now
                self.link.transport.sample()
        self.link.poll()
        device = self.link.device
        if device is None or device is self._last_device:
            return None
        self._last_device = device
        return device

    def handle_events(self, events, hand: HandState, haptic_on: bool) -> list:
        """Send each simulation event through the safety gate. Returns the cues that the
        board accepted, as CueEvents carrying the real sequence number (for the log)."""
        sent = []
        if not haptic_on or self.link is None:
            return sent
        for event in events:
            decision = self.policy.filter(event.event_id, event.kind, hand.tracking_quality,
                                          self.armed)
            if decision.action == "stop":
                self.stop(event.kind.lower())
                continue
            if decision.action != "cue":
                if decision.reason != "UNMAPPED":       # RELEASE has no cue on purpose
                    print(f"  haptic: {event.kind} not sent ({decision.reason})")
                continue
            reply = self.link.cue(decision.cue)
            if isinstance(reply, Ack):
                cue = dataclasses.replace(decision.cue, seq=self.link.seq)
                sent.append(cue)
                self.last_cue = f"{cue.kind} sent"
                print(f"  haptic: {cue.kind} cue sent (angle {cue.safe_angle_deg}, "
                      f"{cue.duration_ms} ms)")
            else:
                why = reply.reason if isinstance(reply, Err) else "no answer"
                self.last_cue = f"{decision.cue.kind} refused ({why})"
                print(f"  haptic: board refused {decision.cue.kind} ({why})")
        return sent

    def close(self) -> None:
        """Sends STOP and closes the port. Safe to call more than once."""
        if self.link is not None:
            self.link.close()
            self.link.connected = False
