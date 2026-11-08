"""Side panel: condition, trial, hand state, grip meter with object bands, events."""
from dataclasses import dataclass
from typing import Dict, List, Optional

import pygame

from ghostgrip.simulation.objects import PROFILES
from ghostgrip.simulation.world import WorldSnapshot
from ghostgrip.ui.renderer_2d import TEXT, phase_color

PANEL_W = 320
PANEL_BG = (28, 34, 52)
DIM = (150, 160, 185)


@dataclass
class HudInfo:
    condition_code: str
    condition_label: str
    source: str                 # "mouse", "replay", later "camera"
    object_kind: str
    trial_state: str
    trial_id: int
    trial_elapsed_ms: int
    tally: Dict[str, List[int]]  # condition -> [successes, trials]
    fps: float
    recording: bool
    ghost_on: bool
    haptic_routing: bool
    intent: str = ""            # Member D's intent state, e.g. "HOLDING glass"
    assist_source: str = ""     # "AI" (Member D's AssistPolicy) or "manual"
    board: str = ""             # Member C's board: NO BOARD / DISARMED / ARMED; "" = no hardware
    board_detail: str = ""      # last cue result


HELP = [
    "Mouse: move   Left/W close   Right/S open",
    "Wheel grip  Space open  G ghost  F9 rec",
    "1/2/3/0: glass/sponge/block/all  N: next",
    "F1-F4: condition A-D   [ ]: manual assist",
    "T/L/J: lose tracking/low quality/jitter",
    "H arm  X STOP board  V camera  Esc quit",
]
BOARD_COLORS = {"ARMED": (110, 230, 150), "DISARMED": (255, 200, 90)}


class Hud:
    def __init__(self, x0: int, height: int):
        self.x0 = x0
        self.h = height
        self.font = pygame.font.Font(None, 21)
        self.head = pygame.font.Font(None, 26)

    def draw(self, surf: pygame.Surface, snap: WorldSnapshot, info: HudInfo) -> None:
        pygame.draw.rect(surf, PANEL_BG, (self.x0, 0, PANEL_W, self.h))
        x, y = self.x0 + 12, 10
        y = self._line(surf, "GhostGrip  |  Member B sim", x, y, self.head, TEXT, 26)
        y = self._line(surf, f"Condition {info.condition_code}: {info.condition_label}"
                             f"{'  [haptics routed]' if info.haptic_routing else ''}", x, y)
        y = self._line(surf, f"Input: {info.source}   Object: {info.object_kind}"
                             f"   FPS {info.fps:4.0f}", x, y, color=DIM)
        state = info.trial_state if info.object_kind != "sandbox" else "SANDBOX (no trial)"
        y = self._line(surf, f"Trial #{info.trial_id} {state}  {info.trial_elapsed_ms/1000:5.1f}s", x, y)
        tally = "  ".join(f"{c}:{s}/{n}" for c, (s, n) in sorted(info.tally.items()))
        y = self._line(surf, f"Success  {tally or '-'}", x, y, color=DIM)
        y += 6

        if not snap.tracking_ok:
            trk, col = "LOST", (255, 110, 110)
        elif not snap.reliable:
            trk, col = "UNRELIABLE", (255, 200, 90)
        else:
            trk, col = "OK", (110, 230, 150)
        y = self._line(surf, f"Tracking: {trk}", x, y, color=col)
        y = self._line(surf, f"Phase: {snap.phase}", x, y, color=phase_color(snap.phase))
        y = self._line(surf, f"q = {snap.q:0.2f}   effort G = {snap.effort:0.2f} (synthetic)", x, y)
        if not snap.assist_enabled:
            assist = "OFF"
        elif snap.assist_active:
            assist = f"ON gain {snap.assist_gain:0.2f}"
        else:
            assist = f"PAUSED gain {snap.assist_gain:0.2f}"   # unreliable tracking or gain 0
        source = f" ({info.assist_source})" if info.assist_source and snap.assist_enabled else ""
        y = self._line(surf, f"Assist: {assist}{source}", x, y, color=DIM)
        y = self._line(surf, f"Intent: {info.intent or '-'}   Ghost: {'ON' if info.ghost_on else 'OFF'}"
                             f"{'   REC' if info.recording else ''}", x, y, color=DIM)
        if info.board:
            y = self._line(surf, f"Board: {info.board}  {info.board_detail}", x, y,
                           color=BOARD_COLORS.get(info.board, (255, 110, 110)))
        y += 4
        kind = snap.engaged_id or (info.object_kind if info.object_kind in PROFILES else None)
        y = self._grip_meter(surf, x, y, snap.q, kind)
        y += 8
        y = self._line(surf, "Recent events (sent once each):", x, y, color=DIM)
        for t, kind_, obj, reason in list(snap.recent_events)[-3:]:
            y = self._line(surf, f"{kind_:7s} {obj:6s} {reason}", x + 6, y)
        y = max(y + 6, self.h - len(HELP) * 19 - 8)
        for h in HELP:
            y = self._line(surf, h, x, y, color=DIM, step=19)

    def _grip_meter(self, surf, x, y, q, kind: Optional[str]) -> int:
        w, h = PANEL_W - 30, 18
        pygame.draw.rect(surf, (45, 52, 75), (x, y, w, h))
        if kind in PROFILES:
            p = PROFILES[kind]
            def qx(v):
                return x + int(w * min(1.0, max(0.0, v)))
            hold_q = p.q_for_effort(p.hold_threshold)
            brk_q = p.q_for_effort(p.break_threshold) if p.break_threshold else 1.0
            pygame.draw.rect(surf, (60, 100, 140), (qx(p.contact_threshold), y, qx(hold_q) - qx(p.contact_threshold), h))
            pygame.draw.rect(surf, (60, 150, 95), (qx(hold_q), y, qx(brk_q) - qx(hold_q), h))
            if p.break_threshold:
                pygame.draw.rect(surf, (150, 60, 60), (qx(brk_q), y, x + w - qx(brk_q), h))
            label = (f"{kind}: contact {p.contact_threshold:.2f}  hold {hold_q:.2f}"
                     + (f"  break {brk_q:.2f}" if p.break_threshold else ""))
        else:
            label = "grip meter (choose an object)"
        mx = x + int(w * q)
        pygame.draw.line(surf, (255, 255, 255), (mx, y - 3), (mx, y + h + 3), 3)
        surf.blit(self.font.render(label, True, DIM), (x, y + h + 4))
        return y + h + 24

    def _line(self, surf, text, x, y, font=None, color=TEXT, step=21):
        surf.blit((font or self.font).render(text, True, color), (x, y))
        return y + step
