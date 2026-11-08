"""Pygame 2D renderer. Reads a WorldSnapshot and draws it; never changes physics.

A future renderer_3d.py (Ursina) implements the same draw(snapshot, ...) idea.
"""
from typing import Optional, Tuple

import pygame

from ghostgrip.simulation.world import (DAMAGED, PHASE_CONTACT, SLIPPING, STABLE_GRASP,
                                        WORLD_H, WORLD_W, WorldSnapshot)

BG = (18, 24, 38)
TABLE = (58, 46, 40)
TABLE_EDGE = (92, 74, 62)
TARGET = (60, 170, 110)
TEXT = (225, 232, 245)
GHOST = (200, 130, 255)
PHASE_COLORS = {
    "NO_CONTACT": (230, 240, 255),
    PHASE_CONTACT: (90, 210, 240),
    STABLE_GRASP: (100, 235, 150),
    SLIPPING: (255, 170, 60),
    DAMAGED: (240, 80, 80),
}


class Renderer2D:
    def __init__(self):
        self.font = pygame.font.Font(None, 22)
        self.big = pygame.font.Font(None, 44)
        self._overlay = pygame.Surface((WORLD_W, WORLD_H), pygame.SRCALPHA)

    def draw(self, surf: pygame.Surface, snap: WorldSnapshot,
             ghost_world: Optional[Tuple[float, float]] = None,
             banner: Optional[Tuple[str, Tuple[int, int, int]]] = None) -> None:
        surf.fill(BG, pygame.Rect(0, 0, WORLD_W, WORLD_H))
        self._overlay.fill((0, 0, 0, 0))
        self._draw_table(surf, snap)
        for obj in snap.objects:
            self._draw_object(surf, obj, snap)
        surf.blit(self._overlay, (0, 0))
        if ghost_world is not None and snap.tracking_ok:
            gx, gy = int(ghost_world[0]), int(ghost_world[1])
            pygame.draw.circle(surf, GHOST, (gx, gy), 9, 2)
            pygame.draw.line(surf, GHOST, (int(snap.pointer[0]), int(snap.pointer[1])), (gx, gy), 1)
        self._draw_hand(surf, snap)
        if snap.paused:
            self._center_text(surf, "TRACKING LOST - interaction frozen", (255, 120, 120), 70)
        elif not snap.reliable:
            self._center_text(surf, "UNRELIABLE TRACKING - no new grasps", (255, 200, 90), 70)
        if banner:
            self._center_text(surf, banner[0], banner[1], 120, big=True)

    # ---------------------------------------------------------------- parts
    def _draw_table(self, surf, snap):
        ty = int(snap.table_y)
        pygame.draw.rect(surf, TABLE, (0, ty, WORLD_W, WORLD_H - ty))
        pygame.draw.line(surf, TABLE_EDGE, (0, ty), (WORLD_W, ty), 3)
        x0, x1 = snap.target_zone
        pygame.draw.rect(self._overlay, (*TARGET, 55), (int(x0), ty - 150, int(x1 - x0), 150))
        pygame.draw.rect(surf, TARGET, (int(x0), ty - 150, int(x1 - x0), 150), 2)
        surf.blit(self.font.render("TARGET ZONE", True, TARGET), (int(x0) + 8, ty - 146))

    def _draw_object(self, surf, o, snap):
        x, y, r = o.x, o.y, o.radius
        if o.damaged:
            self._draw_shards(surf, o, snap.table_y)
            return
        if o.kind == "glass":
            w, h = r * 1.5, r * 2.0
            rect = pygame.Rect(0, 0, int(w), int(h))
            rect.center = (int(x), int(y))
            pygame.draw.rect(self._overlay, (*o.color, 110), rect, border_radius=6)
            pygame.draw.rect(surf, o.color, rect, 2, border_radius=6)
            pygame.draw.ellipse(surf, o.color, (rect.x, rect.y - 5, rect.w, 10), 2)
            pygame.draw.line(surf, (235, 250, 255), (rect.x + 6, rect.y + 8), (rect.x + 6, rect.bottom - 8), 2)
        elif o.kind == "sponge":
            c = o.compression
            w, h = 2 * r * (1.0 - c), 2 * r * (1.0 + 0.3 * c)
            rect = pygame.Rect(0, 0, int(w), int(h))
            rect.midbottom = (int(x), int(y + r))
            pygame.draw.rect(surf, o.color, rect, border_radius=8)
            for i in range(6):
                hx = rect.x + (i % 3 + 0.5) * rect.w / 3
                hy = rect.y + (i // 3 + 0.5) * rect.h / 2
                pygame.draw.circle(surf, (205, 165, 50), (int(hx), int(hy)), max(2, int(r * 0.12)))
        else:  # block
            rect = pygame.Rect(0, 0, int(2 * r), int(2 * r))
            rect.center = (int(x), int(y))
            pygame.draw.rect(surf, o.color, rect, border_radius=3)
            pygame.draw.rect(surf, (100, 108, 125), rect, 3, border_radius=3)
            lab = self.font.render("HEAVY", True, (40, 45, 60))
            surf.blit(lab, lab.get_rect(center=rect.center))
        name = self.font.render(o.kind, True, TEXT)
        surf.blit(name, name.get_rect(center=(int(x), int(snap.table_y) + 16)))
        if o.in_target:
            pygame.draw.circle(surf, TARGET, (int(x), int(y - r - 14)), 6)

    def _draw_shards(self, surf, o, table_y):
        ty = int(table_y)
        for i in range(7):
            sx = o.x + (i - 3) * o.radius * 0.45
            h = 6 + (i * 37 % 11)
            pts = [(sx - 6, ty), (sx + 5, ty), (sx + (i % 3 - 1) * 3, ty - h)]
            pygame.draw.polygon(surf, o.color, pts)
        lab = self.font.render(f"{o.kind}: BROKEN", True, (240, 90, 90))
        surf.blit(lab, lab.get_rect(center=(int(o.x), ty + 16)))

    def _draw_hand(self, surf, snap):
        px, py = snap.pointer
        color = PHASE_COLORS.get(snap.phase, TEXT)
        if not snap.tracking_ok:
            color = (130, 130, 140)
        if snap.assist_active and snap.engaged_id is None:
            pygame.draw.circle(surf, (120, 120, 200), (int(px), int(py)),
                               int(14 + 40 * snap.grab_radius_extra_frac), 1)
        half = snap.pad_gap / 2.0
        for sx in (-1, 1):
            pad = pygame.Rect(0, 0, 12, 26)
            pad.center = (int(px + sx * (half + 6)), int(py))
            pygame.draw.rect(surf, color, pad, border_radius=5)
        pygame.draw.line(surf, color, (int(px - half - 6), int(py - 16)), (int(px + half + 6), int(py - 16)), 2)
        pygame.draw.circle(surf, color, (int(px), int(py)), 3)

    def _center_text(self, surf, text, color, y, big=False):
        f = self.big if big else self.font
        img = f.render(text, True, color)
        rect = img.get_rect(center=(WORLD_W // 2, y))
        bg = rect.inflate(20, 10)
        pygame.draw.rect(surf, (10, 14, 24), bg, border_radius=6)
        surf.blit(img, rect)


def phase_color(phase: str) -> Tuple[int, int, int]:
    return PHASE_COLORS.get(phase, TEXT)
