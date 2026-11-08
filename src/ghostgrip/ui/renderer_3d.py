"""3D view of the simulation (pygame only, no new dependency).

Draws the same read-only WorldSnapshot as Renderer2D, through a perspective camera: a wooden
table with the target zone, a transparent glass, a sponge that squashes, a heavy steel block
and a two-pad gripper, with flat shading, soft contact shadows and anti-aliased edges.

Physics stays exactly as it is (x = left/right, y = up/down in a 960 x 540 world); depth is
visual only. Positions are smoothed for drawing only (35 ms time constant), so motion looks
fluid at camera frame rates; the simulation, events, trials and logs never see these values.
"""
import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import pygame
from pygame import gfxdraw

from ghostgrip.simulation.world import (DAMAGED, PHASE_CONTACT, SLIPPING, STABLE_GRASP,
                                        TABLE_Y, WORLD_H, WORLD_W, WorldSnapshot)
from ghostgrip.ui.hand_3d import Hand3D
from ghostgrip.ui.renderer_2d import GHOST, PHASE_COLORS, TARGET, TEXT

CONTACT_PHASES = (PHASE_CONTACT, STABLE_GRASP, SLIPPING, DAMAGED)
FINGER_RADIUS = 7.0                    # fingertips rest on an object's surface, not inside it

Vec = Tuple[float, float, float]
Color = Tuple[int, int, int]


def _sub(a: Vec, b: Vec) -> Vec:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Vec, b: Vec) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vec, b: Vec) -> Vec:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _norm(a: Vec) -> Vec:
    n = math.sqrt(_dot(a, a)) or 1.0
    return (a[0] / n, a[1] / n, a[2] / n)


LIGHT = _norm((-0.45, 0.85, -0.40))   # key light from upper left, slightly in front
AMBIENT = 0.40
SMOOTH_TAU_S = 0.035                  # visual smoothing only
SNAP_DISTANCE = 150.0                 # bigger jumps (reset, teleport) are not smoothed
TABLE_FRONT_Z, TABLE_BACK_Z, TABLE_HALF_W = -170.0, 260.0, 640.0
GLASS_ALPHA = 70
LOST_COLOR = (130, 130, 140)
STEEL = (150, 160, 180)


def to3d(x: float, y: float, z: float = 0.0) -> Vec:
    """World px (x right, y down) -> 3D (x right, y up from the table top, z away from viewer)."""
    return (x - WORLD_W / 2.0, TABLE_Y - y, z)


def _shade(color: Sequence[int], normal: Vec, alpha: int = 255, boost: float = 1.0):
    k = (AMBIENT + (1.0 - AMBIENT) * max(0.0, _dot(normal, LIGHT))) * boost
    return (min(255, int(color[0] * k)), min(255, int(color[1] * k)),
            min(255, int(color[2] * k)), alpha)


def _mix(a: Sequence[int], b: Sequence[int], t: float) -> Color:
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


class Camera:
    """Fixed perspective camera a little above the table, looking slightly down."""

    def __init__(self, eye: Vec = (0.0, 330.0, -1000.0), target: Vec = (0.0, 170.0, 0.0),
                 focal: float = 950.0, center: Tuple[float, float] = (WORLD_W / 2.0, 290.0)):
        self.eye = eye
        self.fwd = _norm(_sub(target, eye))
        self.right = _norm(_cross((0.0, 1.0, 0.0), self.fwd))
        self.up = _cross(self.fwd, self.right)
        self.f = focal
        self.cx, self.cy = center

    def depth(self, p: Vec) -> float:
        return max(1.0, _dot(_sub(p, self.eye), self.fwd))

    def project(self, p: Vec) -> Tuple[float, float, float]:
        d = _sub(p, self.eye)
        z = max(1.0, _dot(d, self.fwd))
        return (self.cx + self.f * _dot(d, self.right) / z, self.cy - self.f * _dot(d, self.up) / z, z)

    def scale(self, p: Vec) -> float:
        """Screen pixels per world unit at point p."""
        return self.f / self.depth(p)

    def facing(self, point: Vec, normal: Vec) -> bool:
        return _dot(normal, _sub(self.eye, point)) > 0.0


def _ip(p) -> Tuple[int, int]:
    return (int(round(max(-9999.0, min(9999.0, p[0])))), int(round(max(-9999.0, min(9999.0, p[1])))))


class Renderer3D:
    """Same draw() signature as Renderer2D, so the app can switch views at any time."""

    def __init__(self):
        self.cam = Camera()
        self.font = pygame.font.Font(None, 22)
        self.small = pygame.font.Font(None, 19)
        self.big = pygame.font.Font(None, 44)
        self._static: Optional[pygame.Surface] = None
        self._smooth: Dict[str, List[float]] = {}
        self._last_t: Optional[float] = None
        self._faces: List[tuple] = []
        self.hand = Hand3D()

    # ================================================================== public
    def draw(self, surf: pygame.Surface, snap: WorldSnapshot,
             ghost_world: Optional[Tuple[float, float]] = None,
             banner: Optional[Tuple[str, Tuple[int, int, int]]] = None,
             hand_points=None) -> None:
        """hand_points: Member A's 21 (x, y) landmarks of the current camera frame, or None.
        Used only to pose the drawn fingers; nothing here feeds back into the simulation."""
        now = time.perf_counter()
        dt = 0.0 if self._last_t is None else min(0.1, now - self._last_t)
        self._last_t = now
        if self._static is None:
            self._static = self._build_static(snap)

        old_clip = surf.get_clip()
        surf.set_clip(pygame.Rect(0, 0, WORLD_W, WORLD_H))
        surf.blit(self._static, (0, 0))

        hx, hy = self._smoothed("hand", snap.pointer, dt)
        gap = self._smoothed("gap", (snap.pad_gap, 0.0), dt)[0]
        objects = []
        for o in snap.objects:
            x, y = (o.x, o.y) if o.damaged else self._smoothed(o.object_id, (o.x, o.y), dt)
            if o.phase == SLIPPING:
                x += 2.0 * math.sin(now * 55.0)            # visible shudder while slipping
            objects.append((o, x, y))

        # contact shadows first: they lie on the table under everything else
        for o, x, y in objects:
            if not o.damaged:
                self._shadow(surf, x, y + o.radius, o.radius * 0.95)
        if snap.tracking_ok:
            self._shadow(surf, hx, hy + 13.0, 16.0 + gap * 0.25)

        self._faces = []
        for o, x, y in objects:
            self._object(o, x, y, snap)
        # only trust landmarks from reliable frames: a half-visible or edge-on hand reports
        # distorted points (the same frames the simulation already refuses for new grasps)
        self._hand(snap, hx, hy, gap, objects, dt, hand_points if snap.reliable else None)
        self._faces.sort(key=lambda f: -f[0])               # painter's algorithm: far to near
        for _, pts, fill, edge in self._faces:
            self._poly(surf, pts, fill, edge)

        for o, x, y in objects:
            self._labels(surf, o, x, y)
        self._ghost_and_assist(surf, snap, (hx, hy), ghost_world)
        if snap.paused:
            self._center_text(surf, "TRACKING LOST - interaction frozen", (255, 120, 120), 70)
        elif not snap.reliable:
            self._center_text(surf, "UNRELIABLE TRACKING - no new grasps", (255, 200, 90), 70)
        if banner:
            self._center_text(surf, banner[0], banner[1], 120, big=True)
        surf.set_clip(old_clip)

    # ================================================================== smoothing
    def _smoothed(self, key: str, target: Tuple[float, float], dt: float) -> Tuple[float, float]:
        cur = self._smooth.get(key)
        if cur is None or math.hypot(target[0] - cur[0], target[1] - cur[1]) > SNAP_DISTANCE or dt <= 0:
            self._smooth[key] = [float(target[0]), float(target[1])]
            return (float(target[0]), float(target[1]))
        a = 1.0 - math.exp(-dt / SMOOTH_TAU_S)
        cur[0] += (target[0] - cur[0]) * a
        cur[1] += (target[1] - cur[1]) * a
        return (cur[0], cur[1])

    # ================================================================== static scene
    def _build_static(self, snap: WorldSnapshot) -> pygame.Surface:
        bg = pygame.Surface((WORLD_W, WORLD_H)).convert() if pygame.display.get_surface() \
            else pygame.Surface((WORLD_W, WORLD_H))
        # back wall: vertical gradient with a soft light pool
        for y in range(WORLD_H):
            t = min(1.0, y / 400.0)
            pygame.draw.line(bg, _mix((14, 18, 32), (46, 54, 80), t), (0, y), (WORLD_W, y))
        for r in range(360, 20, -10):              # many faint discs = smooth radial light
            gfxdraw.filled_circle(bg, WORLD_W // 2, 160, r, (120, 140, 190, 2))

        cam = self.cam
        top = [to3d(-TABLE_HALF_W + WORLD_W / 2, TABLE_Y, TABLE_FRONT_Z),
               to3d(TABLE_HALF_W + WORLD_W / 2, TABLE_Y, TABLE_FRONT_Z),
               to3d(TABLE_HALF_W + WORLD_W / 2, TABLE_Y, TABLE_BACK_Z),
               to3d(-TABLE_HALF_W + WORLD_W / 2, TABLE_Y, TABLE_BACK_Z)]
        wood = (126, 90, 62)
        self._poly(bg, [cam.project(p) for p in top], _shade(wood, (0, 1, 0)), None)
        # wood grain: faint lines running left to right, slightly uneven
        for k in range(18):
            z = TABLE_FRONT_Z + 8 + k * (TABLE_BACK_Z - TABLE_FRONT_Z - 16) / 17.0
            wob = 3.0 * math.sin(k * 1.7)
            a = cam.project((-TABLE_HALF_W, 0.0, z + wob))
            b = cam.project((TABLE_HALF_W, 0.0, z - wob))
            shade = (92, 64, 44, 70) if k % 3 else (150, 112, 80, 60)
            pygame.draw.aaline(bg, shade[:3], _ip(a), _ip(b))
        front = [(-TABLE_HALF_W, 0.0, TABLE_FRONT_Z), (TABLE_HALF_W, 0.0, TABLE_FRONT_Z),
                 (TABLE_HALF_W, -90.0, TABLE_FRONT_Z), (-TABLE_HALF_W, -90.0, TABLE_FRONT_Z)]
        self._poly(bg, [cam.project(p) for p in front], _shade((96, 68, 48), (0, 0, -1)), None)
        edge_a = cam.project((-TABLE_HALF_W, 0.0, TABLE_FRONT_Z))
        edge_b = cam.project((TABLE_HALF_W, 0.0, TABLE_FRONT_Z))
        pygame.draw.line(bg, (176, 134, 96), _ip(edge_a), _ip(edge_b), 2)

        # target zone: a glowing mat on the table plus a faint glass panel behind it
        x0, x1 = (v - WORLD_W / 2.0 for v in snap.target_zone)
        mat = [(x0, 0.6, -120.0), (x1, 0.6, -120.0), (x1, 0.6, 120.0), (x0, 0.6, 120.0)]
        self._poly(bg, [cam.project(p) for p in mat], (*TARGET, 70), (*TARGET, 230))
        panel = [(x0, 0.0, 120.0), (x1, 0.0, 120.0), (x1, 150.0, 120.0), (x0, 150.0, 120.0)]
        self._poly(bg, [cam.project(p) for p in panel], (*TARGET, 26), (*TARGET, 110))
        lab = self.font.render("TARGET ZONE", True, TARGET)
        lp = cam.project(((x0 + x1) / 2.0, 150.0, 120.0))
        bg.blit(lab, lab.get_rect(midbottom=(int(lp[0]), int(lp[1]) - 4)))
        return bg

    # ================================================================== primitives
    def _add(self, depth: float, pts, fill, edge=None) -> None:
        self._faces.append((depth, pts, fill, edge))

    def _box(self, center: Vec, size: Vec, color: Sequence[int], alpha: int = 255,
             edge_dark: float = 0.7, boost: float = 1.0) -> None:
        cx, cy, cz = center
        hx, hy, hz = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0
        faces = (
            ((1, 0, 0), [(cx + hx, cy - hy, cz - hz), (cx + hx, cy - hy, cz + hz),
                         (cx + hx, cy + hy, cz + hz), (cx + hx, cy + hy, cz - hz)]),
            ((-1, 0, 0), [(cx - hx, cy - hy, cz + hz), (cx - hx, cy - hy, cz - hz),
                          (cx - hx, cy + hy, cz - hz), (cx - hx, cy + hy, cz + hz)]),
            ((0, 1, 0), [(cx - hx, cy + hy, cz - hz), (cx + hx, cy + hy, cz - hz),
                         (cx + hx, cy + hy, cz + hz), (cx - hx, cy + hy, cz + hz)]),
            ((0, -1, 0), [(cx - hx, cy - hy, cz + hz), (cx + hx, cy - hy, cz + hz),
                          (cx + hx, cy - hy, cz - hz), (cx - hx, cy - hy, cz - hz)]),
            ((0, 0, -1), [(cx - hx, cy - hy, cz - hz), (cx + hx, cy - hy, cz - hz),
                          (cx + hx, cy + hy, cz - hz), (cx - hx, cy + hy, cz - hz)]),
            ((0, 0, 1), [(cx + hx, cy - hy, cz + hz), (cx - hx, cy - hy, cz + hz),
                         (cx - hx, cy + hy, cz + hz), (cx + hx, cy + hy, cz + hz)]),
        )
        for normal, quad in faces:
            mid = tuple(sum(p[i] for p in quad) / 4.0 for i in range(3))
            if not self.cam.facing(mid, normal):
                continue
            fill = _shade(color, normal, alpha, boost)
            edge = (int(fill[0] * edge_dark), int(fill[1] * edge_dark), int(fill[2] * edge_dark), alpha)
            self._add(self.cam.depth(mid), [self.cam.project(p) for p in quad], fill, edge)

    def _cylinder(self, base: Vec, radius: float, height: float, color: Sequence[int],
                  alpha: int, segments: int = 26) -> None:
        bx, by, bz = base
        ring = [(math.cos(2 * math.pi * i / segments), math.sin(2 * math.pi * i / segments))
                for i in range(segments)]
        bottom = [(bx + radius * c, by, bz + radius * s) for c, s in ring]
        top = [(bx + radius * c, by + height, bz + radius * s) for c, s in ring]
        for i in range(segments):
            j = (i + 1) % segments
            mid_a = 2 * math.pi * (i + 0.5) / segments
            normal = (math.cos(mid_a), 0.0, math.sin(mid_a))
            quad = [bottom[i], bottom[j], top[j], top[i]]
            mid = tuple(sum(p[k] for p in quad) / 4.0 for k in range(3))
            front = self.cam.facing(mid, normal)
            fill = _shade(color, normal if front else (-normal[0], 0.0, -normal[2]),
                          alpha if front else alpha // 2, 1.15 if front else 0.8)
            self._add(self.cam.depth(mid), [self.cam.project(p) for p in quad], fill, False)
        # clean silhouette edges left and right instead of a wireframe of every segment
        for side in (-1.0, 1.0):
            x = bx + side * radius
            edge = [(x - 0.8, by, bz), (x + 0.8, by, bz), (x + 0.8, by + height, bz), (x - 0.8, by + height, bz)]
            self._add(self.cam.depth((x, by, bz)) - 0.2, [self.cam.project(p) for p in edge],
                      (205, 232, 250, 170), False)
        top_mid = (bx, by + height, bz)
        self._add(self.cam.depth(top_mid) + 0.5, [self.cam.project(p) for p in top],
                  _shade(color, (0, 1, 0), alpha // 2, 1.2), (225, 245, 255, 200))
        self._add(self.cam.depth(base) + 0.4, [self.cam.project(p) for p in bottom],
                  None, (200, 230, 250, 120))

    def _poly(self, surf: pygame.Surface, pts, fill, edge) -> None:
        if len(pts) < 3:
            return
        ip = [_ip(p) for p in pts]
        if fill is not None:
            gfxdraw.filled_polygon(surf, ip, fill)
        if edge is False:                           # explicitly no outline (glass panels)
            return
        if edge is not None:
            gfxdraw.aapolygon(surf, ip, edge)
        elif fill is not None:
            gfxdraw.aapolygon(surf, ip, fill)

    def _shadow(self, surf: pygame.Surface, x: float, y_bottom: float, radius: float) -> None:
        height = TABLE_Y - y_bottom                      # distance above the table top
        if height < -6.0:
            return
        height = max(0.0, height)
        for grow, alpha in ((1.35, 0.45), (1.0, 1.0)):   # soft outer ring + darker core
            r = radius * grow * (1.0 + height / 260.0)
            a = int(95 * alpha * max(0.12, 1.0 - height / 320.0))
            pts = [self.cam.project((x - WORLD_W / 2.0 + r * math.cos(t), 0.4, r * 0.55 * math.sin(t)))
                   for t in (2 * math.pi * k / 22 for k in range(22))]
            gfxdraw.filled_polygon(surf, [_ip(p) for p in pts], (0, 0, 0, a))

    # ================================================================== objects
    def _object(self, o, x: float, y: float, snap: WorldSnapshot) -> None:
        r = o.radius
        if o.damaged:
            self._shards(o, x)
            return
        if o.kind == "glass":
            base = to3d(x, y + r)
            self._cylinder(base, 0.75 * r, 2.0 * r, o.color, GLASS_ALPHA)
            # specular streak on the front-left of the glass
            rad, h = 0.75 * r, 2.0 * r
            ang = math.radians(235)
            p0 = (base[0] + rad * math.cos(ang), base[1] + 0.12 * h, base[2] + rad * math.sin(ang))
            ang2 = math.radians(247)
            p1 = (base[0] + rad * math.cos(ang2), base[1] + 0.12 * h, base[2] + rad * math.sin(ang2))
            streak = [p0, p1, (p1[0], p1[1] + 0.76 * h, p1[2]), (p0[0], p0[1] + 0.76 * h, p0[2])]
            self._add(self.cam.depth(base) - rad - 1.0, [self.cam.project(p) for p in streak],
                      (245, 252, 255, 120), False)
        elif o.kind == "sponge":
            c = o.compression
            w, h, d = 2 * r * (1.0 - c), 2 * r * (1.0 + 0.3 * c), 2 * r * 0.85
            bx, by, bz = to3d(x, y + r)
            center = (bx, by + h / 2.0, bz)
            self._box(center, (w, h, d), o.color, edge_dark=0.78)
            pore = (196, 150, 40, 255)
            for i in range(6):                     # pores on the front face
                px = bx - w / 2 + (i % 3 + 0.5) * w / 3.0
                py = by + (i // 3 + 0.5) * h / 2.0
                pr = max(2.0, r * 0.11) * (1.0 - 0.4 * c)
                ring = [(px + pr * math.cos(t), py + pr * math.sin(t) * 1.1, bz - d / 2.0 - 0.3)
                        for t in (2 * math.pi * k / 8 for k in range(8))]
                self._add(self.cam.depth((px, py, bz - d / 2.0)) - 0.5,
                          [self.cam.project(p) for p in ring], pore, None)
            for i in range(4):                     # and a few on the top
                px = bx - w / 2 + (i + 0.5) * w / 4.0
                pz = bz + ((i * 37) % 9 - 4) * d / 12.0
                pr = max(2.0, r * 0.09)
                ring = [(px + pr * math.cos(t), by + h + 0.3, pz + pr * math.sin(t))
                        for t in (2 * math.pi * k / 8 for k in range(8))]
                self._add(self.cam.depth((px, by + h, pz)) - 0.5,
                          [self.cam.project(p) for p in ring], _shade(pore, (0, 1, 0)), None)
        else:  # heavy block
            size = 2 * r
            center = to3d(x, y)
            self._box(center, (size, size, size * 0.9), STEEL, edge_dark=0.55)
            cx, cy, cz = center
            for k in (-0.28, 0.28):                # two machined grooves on the front face
                gy = cy + k * size
                groove = [(cx - size / 2 + 4, gy - 2, cz - size * 0.45 - 0.3),
                          (cx + size / 2 - 4, gy - 2, cz - size * 0.45 - 0.3),
                          (cx + size / 2 - 4, gy + 2, cz - size * 0.45 - 0.3),
                          (cx - size / 2 + 4, gy + 2, cz - size * 0.45 - 0.3)]
                self._add(self.cam.depth((cx, gy, cz - size * 0.45)) - 0.5,
                          [self.cam.project(p) for p in groove], (96, 104, 122, 255), None)

    def _shards(self, o, x: float) -> None:
        for i in range(9):
            sx = x - WORLD_W / 2.0 + (i - 4) * o.radius * 0.38
            z0 = ((i * 53) % 40) - 20.0
            h = 6.0 + (i * 37 % 13)
            tri = [(sx - 6.0, 0.0, z0), (sx + 6.0, 0.0, z0 + 4.0), (sx + (i % 3 - 1) * 3.0, h, z0 + 2.0)]
            self._add(self.cam.depth((sx, 0.0, z0)), [self.cam.project(p) for p in tri],
                      (*o.color, 170), (230, 245, 255, 220))

    # ================================================================== hand
    def _hand(self, snap: WorldSnapshot, hx: float, hy: float, gap: float, objects, dt: float,
              hand_points) -> None:
        """A realistic hand (ui/hand_3d.py). Thumb and index tips sit on the contact pads;
        while an object is held they rest on its surface instead of inside it."""
        held_tips, z_shift = None, 0.0
        held = next(((o, x) for o, x, _ in objects if o.object_id == snap.engaged_id), None)
        if held is not None and not held[0].damaged:
            o = held[0]
            half_width = {"glass": 0.75 * o.radius,
                          "sponge": o.radius * (1.0 - o.compression)}.get(o.kind, o.radius)
            tip_off = min(gap / 2.0 + 2.0, half_width + FINGER_RADIUS)
            held_tips = ((hx - tip_off, hy), (hx + tip_off, hy))
            front = {"sponge": -0.85 * o.radius, "block": -0.9 * o.radius}.get(o.kind)
            if front is not None:                       # boxes: hold them by the front edges
                z_shift = front + FINGER_RADIUS + 3.0
        z_shift = self._smoothed("hand_z", (z_shift, 0.0), dt)[0]
        self.hand.update(snap.q, dt, hand_points)
        pose = self.hand.pose((hx, hy), to3d, held_tips, z_shift)
        phase_color = PHASE_COLORS.get(snap.phase) if snap.phase in CONTACT_PHASES else None
        self._faces.extend(self.hand.parts(self.cam, pose, phase_color, snap.tracking_ok))

    # ================================================================== overlays
    def _labels(self, surf: pygame.Surface, o, x: float, y: float) -> None:
        cam = self.cam
        base = cam.project((x - WORLD_W / 2.0, 0.0, TABLE_FRONT_Z + 25.0))
        if o.damaged:
            lab = self.font.render(f"{o.kind}: BROKEN", True, (240, 90, 90))
        else:
            lab = self.font.render(o.kind, True, TEXT)
        surf.blit(lab, lab.get_rect(center=(int(base[0]), int(base[1]))))
        if o.damaged:
            return
        if o.kind == "block":
            c = to3d(x, y)
            p = cam.project((c[0], c[1], c[2] - o.radius * 0.9 - 1.0))
            heavy = self.small.render("HEAVY", True, (44, 50, 64))
            surf.blit(heavy, heavy.get_rect(center=(int(p[0]), int(p[1]))))
        if o.in_target:
            top = cam.project(to3d(x, y - o.radius - 16.0))
            gfxdraw.filled_circle(surf, int(top[0]), int(top[1]), 6, (*TARGET, 255))
            gfxdraw.aacircle(surf, int(top[0]), int(top[1]), 6, (*TARGET, 255))

    def _ghost_and_assist(self, surf: pygame.Surface, snap: WorldSnapshot,
                          hand_xy: Tuple[float, float], ghost_world) -> None:
        cam = self.cam
        hp = to3d(*hand_xy)
        hs = cam.project(hp)
        if snap.assist_active and snap.engaged_id is None:
            rr = int((14 + 40 * snap.grab_radius_extra_frac) * cam.scale(hp))
            gfxdraw.aacircle(surf, int(hs[0]), int(hs[1]), max(4, rr), (120, 120, 210))
        if ghost_world is not None and snap.tracking_ok:
            gp = to3d(ghost_world[0], ghost_world[1])
            gs = cam.project(gp)
            rad = max(4, int(9 * cam.scale(gp)))
            pygame.draw.aaline(surf, GHOST, _ip(hs), _ip(gs))
            gfxdraw.filled_circle(surf, int(gs[0]), int(gs[1]), rad, (*GHOST, 70))
            gfxdraw.aacircle(surf, int(gs[0]), int(gs[1]), rad, (*GHOST, 230))

    def _center_text(self, surf, text, color, y, big=False):
        f = self.big if big else self.font
        img = f.render(text, True, color)
        rect = img.get_rect(center=(WORLD_W // 2, y))
        bg = rect.inflate(20, 10)
        pygame.draw.rect(surf, (10, 14, 24), bg, border_radius=6)
        surf.blit(img, rect)
