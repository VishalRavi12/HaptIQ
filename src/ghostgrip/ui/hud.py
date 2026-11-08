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


HELP = [
    "Mouse move: hand position",
    "Left hold / W: close   Right hold / S: open",
    "Wheel: set grip (stays)   Space: open hand",
    "1 glass  2 sponge  3 block  0 sandbox",
    "F1-F4: condition A-D   N: next trial",
    "T: lose tracking  L: low quality  J: jitter",
    "G: ghost  F9: record trace  R: reset  Esc: quit",
]


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
        assist = f"ON gain {snap.assist_gain:0.2f}" if snap.assist_enabled else "OFF"
        y = self._line(surf, f"Assist: {assist}   Ghost: {'ON' if info.ghost_on else 'OFF'}"
                             f"{'   REC' if info.recording else ''}", x, y, color=DIM)
        y += 4
        kind = snap.engaged_id or (info.object_kind if info.object_kind in PROFILES else None)
        y = self._grip_meter(surf, x, y, snap.q, kind)
        y += 8
        y = self._line(surf, "Recent events (sent once each):", x, y, color=DIM)
        for t, kind_, obj, reason in list(snap.recent_events)[-5:]:
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
