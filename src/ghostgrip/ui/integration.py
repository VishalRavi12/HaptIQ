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
from collections import Counter
import dataclasses
from dataclasses import dataclass
import math
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


# ====================================================================== Member A input guard
class CameraStabilizer:
    """Stops bad camera frames from reaching the physics (camera input only).

    Found in a logged live session: after a one-frame tracking dropout the next frame often put
    the hand 30% of the screen away (MediaPipe can also jump to another person's hand) or
    reported a nonsense closure (0.00 or 1.00), which dropped a held sponge mid-air; and a hand
    turned with its fingers toward the camera collapses on screen, making closure and pointer
    meaningless. This guard:
      * holds the last good pointer and closure through unreliable frames (quality below the
        simulation's 0.5, palm collapsed to < 60% of its usual size, or hand edge-on) and
        reports them as unreliable, so no new grasp, cue or release can come from them;
      * accepts a sudden big jump of the pointer or closure only once it persists for
        CONFIRM_FRAMES consistent frames (a real move), never from a single frame;
      * limits how fast the closure may change (a real full pinch still takes 0.15 s);
      * reports tracking lost after HOLD_MAX_MS of held frames, so the world releases safely.
    Real tracking losses pass straight through. Nothing here invents a confident measurement.
    """

    MAX_JUMP = 0.15            # world fraction in one frame; more is a mis-detection unless it persists
    MAX_Q_JUMP = 0.35          # closure change in one frame; more is a spike unless it persists
    CONFIRM_FRAMES = 3
    CONSISTENT_XY = 0.06
    CONSISTENT_Q = 0.12
    MAX_Q_RATE = 4.0           # closure units per second
    HOLD_MAX_MS = 600
    HELD_QUALITY = 0.3         # reported for held frames: below the simulation's QUALITY_MIN
    FORESHORTENED = 0.6        # palm smaller than this share of its usual on-screen size
    EDGE_ON = 0.35             # palm width / palm length below this: hand turned sideways
    QUALITY_MIN = 0.5

    def __init__(self):
        self.last: Optional[HandState] = None
        self.palm_typ: Optional[float] = None
        self.counts: Counter = Counter()
        self._candidates: List[HandState] = []
        self._held_since: Optional[int] = None

    # ------------------------------------------------------------------ main entry
    def filter(self, hand: HandState, points_px=None) -> HandState:
        if not hand.tracking_ok:
            self._candidates = []
            self._held_since = None
            return hand                                   # real loss: the world freezes, then releases
        problem = self._quality_problem(hand, points_px)
        if problem is None and self.last is not None:
            problem = self._jump_problem(hand)
        if problem is not None:
            self.counts[problem] += 1
            return self._hold(hand)
        return self._accept(hand, points_px, slew=True)

    def summary(self) -> str:
        if not self.counts:
            return "no frames needed holding"
        names = {"low_quality": "low quality", "foreshortened": "fingers toward camera",
                 "edge_on": "hand sideways", "teleport": "position jumps", "q_spike": "grip spikes"}
        return ", ".join(f"{n} {names.get(k, k)}" for k, n in self.counts.most_common())

    # ------------------------------------------------------------------ checks
    def _quality_problem(self, hand: HandState, points_px) -> Optional[str]:
        if not (math.isfinite(hand.tracking_quality) and hand.tracking_quality >= self.QUALITY_MIN):
            return "low_quality"
        if points_px is not None and len(points_px) == 21:
            from ghostgrip.ui.hand_3d import _palm_size
            palm = _palm_size(points_px)
            length = math.dist(points_px[0], points_px[9])
            width = math.dist(points_px[5], points_px[17])
            if self.palm_typ is not None and palm < self.FORESHORTENED * self.palm_typ:
                return "foreshortened"
            if length > 1e-6 and width / length < self.EDGE_ON:
                return "edge_on"
        return None

    def _jump_problem(self, hand: HandState) -> Optional[str]:
        jump = math.dist(hand.pointer_xy, self.last.pointer_xy)
        q_jump = abs(hand.closure_q - self.last.closure_q)
        if jump <= self.MAX_JUMP and q_jump <= self.MAX_Q_JUMP:
            self._candidates = []
            return None
        # a big change counts only once it repeats consistently (a real move, not one bad frame)
        prev = self._candidates[-1] if self._candidates else None
        if prev is not None and (math.dist(hand.pointer_xy, prev.pointer_xy) <= self.CONSISTENT_XY
                                 and abs(hand.closure_q - prev.closure_q) <= self.CONSISTENT_Q):
            self._candidates.append(hand)
        else:
            self._candidates = [hand]
        if len(self._candidates) >= self.CONFIRM_FRAMES:
            self._candidates = []
            return None
        return "teleport" if jump > self.MAX_JUMP else "q_spike"

    # ------------------------------------------------------------------ outputs
    def _accept(self, hand: HandState, points_px, slew: bool) -> HandState:
        out = hand
        if slew and self.last is not None:
            dt = max(0.0, (hand.t_ms - self.last.t_ms) / 1000.0)
            step = self.MAX_Q_RATE * dt
            q = self.last.closure_q + max(-step, min(step, hand.closure_q - self.last.closure_q))
            out = dataclasses.replace(hand, closure_q=q)
        if points_px is not None and len(points_px) == 21:
            from ghostgrip.ui.hand_3d import _palm_size
            palm = _palm_size(points_px)
            self.palm_typ = palm if self.palm_typ is None else self.palm_typ + 0.05 * (palm - self.palm_typ)
        self.last = out
        self._held_since = None
        return out

    def _hold(self, hand: HandState) -> HandState:
        if self.last is None:                             # nothing good seen yet: just mark it
            return dataclasses.replace(hand, tracking_quality=min(hand.tracking_quality, self.HELD_QUALITY))
        if self._held_since is None:
            self._held_since = hand.t_ms
        if hand.t_ms - self._held_since > self.HOLD_MAX_MS:
            return lost_hand(hand.t_ms, self.last)         # garbage for too long: treat as lost
        return dataclasses.replace(self.last, t_ms=hand.t_ms, dq_dt=0.0, velocity_xy=(0.0, 0.0),
                                   tracking_quality=min(hand.tracking_quality, self.HELD_QUALITY))


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
        self.last_cue_ms: Optional[int] = None     # when the board last accepted a cue
        self.packets = 0                            # sensor packets received (proof the link works)
        self._last_packet_ms: Optional[int] = None
        self._hinted_not_armed = False
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
            bridge = cls(SerialLink(MockArduino()), policy, keepalive, "mock board", clock_ms)
            bridge.handshake()
            return bridge
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
        bridge = cls(SerialLink(transport), policy, keepalive, port, clock_ms)
        bridge.handshake()
        return bridge

    def handshake(self) -> bool:
        """Prove the GhostGrip firmware is really there: STOP always gets an ACK and only ever
        leaves the board DISARMED, so it is the one command that is safe to send blindly."""
        reply = self.link.stop()
        if isinstance(reply, Ack):
            print(f"Hardware: {self.port_label} answered (ACK, status {reply.status}). The board is "
                  "DISARMED: click the game window, then press H to arm.")
            return True
        self.message = "port opened but the GhostGrip firmware did not answer"
        print(f"Hardware: {self.port_label} opened, but the GhostGrip firmware did not answer. "
              "Check that this is the Arduino's port (not the Pico) and that the firmware is uploaded.")
        return False

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

    @property
    def link_alive(self) -> bool:
        """True while the board's sensor packets (20 per second) keep arriving."""
        return (self.connected and self._last_packet_ms is not None
                and self.clock_ms() - self._last_packet_ms <= 1000)

    def cue_recent(self, window_ms: int = 700) -> bool:
        return self.last_cue_ms is not None and self.clock_ms() - self.last_cue_ms <= window_ms

    def hud_detail(self) -> str:
        """Short text for the HUD next to the status."""
        if not self.connected:
            return ""
        link = "link OK" if self.link_alive else "no data!"
        if not self.link.armed:
            return f"{link}  press H"
        return f"{link}  {self.last_cue}".rstrip()

    # ------------------------------------------------------------------ operator keys
    def arm(self) -> None:
        if not self.connected:
            self.message = "cannot arm: no board connected"
            return
        reply = self.link.arm()
        if isinstance(reply, Ack) and self.link.armed:
            self._last_ka_ms = self.clock_ms()
            self._lost_since = None
            self._hinted_not_armed = False
            self.message = "ARMED: LED should now be steady (X = STOP)"
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
        self.packets += 1
        self._last_packet_ms = now
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
                if (decision.reason == "NOT_ARMED" and self.connected
                        and not self._hinted_not_armed):
                    self._hinted_not_armed = True
                    print("  -> the board is DISARMED: click the GAME window (not the terminal), "
                          "then press H")
                continue
            reply = self.link.cue(decision.cue)
            if isinstance(reply, Ack):
                cue = dataclasses.replace(decision.cue, seq=self.link.seq)
                sent.append(cue)
                self.last_cue = f"{cue.kind} sent"
                self.last_cue_ms = self.clock_ms()
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
