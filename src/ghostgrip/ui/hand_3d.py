"""A realistic hand for the 3D view, built from the 21 hand landmarks.

With the live camera the drawn hand IS the user's hand: every joint comes from Member A's 21
landmarks of the current frame, so each finger moves on its own and the hand has the same
orientation as the real one (upright, tilted, pinching, fist ...). Without a camera (mouse,
replays) a natural upright hand is posed from the closure q (thumb and index pinch together,
the other fingers relax a little).

Placement follows the shared contract: the pointer is the midpoint of the thumb tip (4) and
index tip (8), so the drawn pinch is exactly where the simulation's pinch is. The hand is drawn
at a fixed real-hand size, whatever the distance to the camera. While an object is held, the
thumb and index tips are bent the last few millimetres onto its surface.
Visual only: nothing here feeds back into the physics, the cues or the logs.
"""
import math
from typing import Dict, List, Optional, Sequence, Tuple

Vec = Tuple[float, float, float]
P2 = Tuple[float, float]

HAND_SIZE = 78.0            # world px from wrist (0) to middle knuckle (9): a real-sized hand
SMOOTH_TAU_S = 0.05         # landmark jitter smoothing, visual only

SKIN = (228, 182, 150)
SKIN_EDGE = (156, 108, 82)
PAD = (240, 196, 176)
SLEEVE = (54, 66, 96)
LOST = (150, 150, 160)

FINGERS = {                 # landmark chains, base to tip (MediaPipe numbering)
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
RADII = {                   # bone radii in world px, base bone first
    "thumb": (13.0, 10.8, 9.6),
    "index": (9.8, 9.0, 8.2),
    "middle": (10.1, 9.3, 8.5),
    "ring": (9.5, 8.7, 7.9),
    "pinky": (8.4, 7.7, 6.9),
}
LAYER_Z = {"palm": 5.0, "pinky": 3.0, "ring": 2.0, "middle": 1.0, "index": 0.0, "thumb": -2.0}

# Upright hand, palm toward the viewer, thumb on the left (as in the mirrored camera view).
# Units: wrist (0) to middle knuckle (9) = 1; y grows downward like image coordinates.
OPEN_POSE: List[P2] = [
    (0.00, 0.00),
    (-0.32, -0.14), (-0.58, -0.36), (-0.76, -0.58), (-0.90, -0.78),
    (-0.27, -0.95), (-0.32, -1.36), (-0.35, -1.61), (-0.37, -1.83),
    (0.00, -1.00), (0.00, -1.46), (0.00, -1.73), (0.00, -1.97),
    (0.23, -0.93), (0.28, -1.33), (0.31, -1.58), (0.33, -1.79),
    (0.43, -0.80), (0.53, -1.08), (0.59, -1.27), (0.63, -1.44),
]
PINCH_POSE: List[P2] = [    # thumb and index tips meet; the other fingers relax inward
    (0.00, 0.00),
    (-0.32, -0.14), (-0.55, -0.40), (-0.66, -0.66), (-0.62, -0.92),
    (-0.27, -0.95), (-0.40, -1.30), (-0.54, -1.24), (-0.62, -1.02),
    (0.00, -1.00), (-0.02, -1.42), (-0.07, -1.60), (-0.12, -1.56),
    (0.23, -0.93), (0.24, -1.30), (0.20, -1.44), (0.15, -1.40),
    (0.43, -0.80), (0.47, -1.06), (0.45, -1.18), (0.41, -1.15),
]


def _lerp(a: P2, b: P2, t: float) -> P2:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _mix(a: Sequence[int], b: Sequence[int], t: float) -> Tuple[int, int, int]:
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def template(q: float) -> List[P2]:
    """Natural upright hand for a closure q (0 open .. 1 closed), in wrist-to-knuckle units."""
    t = max(0.0, min(1.0, q))
    t = t * t * (3.0 - 2.0 * t)                    # ease in and out
    return [_lerp(a, b, t) for a, b in zip(OPEN_POSE, PINCH_POSE)]


# Every bone as (parent, child), parents first, so a chain can be rebuilt joint by joint.
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10),
         (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18),
         (18, 19), (19, 20)]
MAX_BONE = {bone: 1.35 * math.dist(OPEN_POSE[bone[0]], OPEN_POSE[bone[1]]) for bone in BONES}
HOLD_LAST_S = 1.5           # keep the last good camera pose this long through bad frames


def _palm_size(pts: Sequence[P2]) -> float:
    """Hand size from several palm distances (each scaled to wrist-to-middle-knuckle units).
    One distance alone collapses when the hand turns sideways and made the fingers explode;
    the largest of four stays steady."""
    return max(math.dist(pts[0], pts[9]), math.dist(pts[0], pts[5]),
               1.10 * math.dist(pts[0], pts[17]), 1.40 * math.dist(pts[5], pts[17]))


def normalized_shape(points) -> Optional[List[P2]]:
    """Landmarks (any pixel scale, y down) -> wrist at the origin, hand size 1, every bone no
    longer than 1.35x a normal hand's. None when the points cannot describe a hand."""
    if points is None or len(points) != 21:
        return None
    try:
        pts = [(float(p[0]), float(p[1])) for p in points]
    except (TypeError, ValueError, IndexError):
        return None
    if not all(math.isfinite(v) for p in pts for v in p):
        return None
    ref = _palm_size(pts)
    if ref < 1e-6:
        return None
    x0, y0 = pts[0]
    rel = [((x - x0) / ref, (y - y0) / ref) for x, y in pts]
    out: List[P2] = [rel[0]] * 21
    for parent, child in BONES:                     # rebuild with clamped bone lengths
        dx, dy = rel[child][0] - rel[parent][0], rel[child][1] - rel[parent][1]
        length = math.hypot(dx, dy)
        limit = MAX_BONE[(parent, child)]
        if length > limit:
            dx, dy = dx * limit / length, dy * limit / length
        out[child] = (out[parent][0] + dx, out[parent][1] + dy)
    return out


class Hand3D:
    """Smoothed hand shape -> 3D joints -> shaded faces for the renderer's depth sort."""

    def __init__(self):
        self.shape: Optional[List[List[float]]] = None
        self.from_camera = False
        self._since_camera = 0.0

    # ------------------------------------------------------------------ shape
    def update(self, q: float, dt: float, points=None) -> None:
        """Follow the real hand's landmarks if given (pass None for unreliable frames),
        otherwise the q-posed template. Brief gaps keep the last good camera pose."""
        target = normalized_shape(points)
        if target is not None:
            self.from_camera, self._since_camera = True, 0.0
        else:
            self._since_camera += max(0.0, dt)
            if self.from_camera and self.shape is not None and self._since_camera < HOLD_LAST_S:
                return                              # hold still rather than glitch
            self.from_camera = False
            target = template(q)
        if self.shape is None or dt <= 0:
            self.shape = [list(p) for p in target]
            return
        a = 1.0 - math.exp(-dt / SMOOTH_TAU_S)
        for cur, (tx, ty) in zip(self.shape, target):
            cur[0] += (tx - cur[0]) * a
            cur[1] += (ty - cur[1]) * a

    # ------------------------------------------------------------------ geometry
    def pose(self, pointer: P2, to3d, held_tips: Optional[Tuple[P2, P2]] = None,
             z_shift: float = 0.0) -> Dict[str, object]:
        """World joints. `pointer` (world px) is where the thumb/index midpoint goes.
        held_tips = (thumb tip, index tip) world positions while an object is held.
        z_shift moves the whole hand toward (-) or away from (+) the viewer."""
        base_to3d = to3d

        def to3d(x, y, z=0.0):
            return base_to3d(x, y, z + z_shift)
        if self.shape is None:
            self.update(0.0, 0.0)
        shape = self.shape
        mid = ((shape[4][0] + shape[8][0]) / 2.0, (shape[4][1] + shape[8][1]) / 2.0)
        world = [[pointer[0] + (x - mid[0]) * HAND_SIZE, pointer[1] + (y - mid[1]) * HAND_SIZE]
                 for x, y in shape]
        if held_tips is not None:                   # bend thumb and index onto the surface
            for name, want in (("thumb", held_tips[0]), ("index", held_tips[1])):
                chain = FINGERS[name]
                dx = want[0] - world[chain[-1]][0]
                dy = want[1] - world[chain[-1]][1]
                for joint, weight in zip(chain, (0.0, 0.25, 0.6, 1.0)):
                    world[joint][0] += dx * weight
                    world[joint][1] += dy * weight

        layer = {i: LAYER_Z["palm"] for i in range(21)}
        for name, chain in FINGERS.items():
            for i in chain[1:]:
                layer[i] = LAYER_Z[name]
        joints = [to3d(x, y, layer[i]) for i, (x, y) in enumerate(world)]

        # wrist and forearm continue away from the fingers, receding slightly
        wx, wy = world[0]
        ux, uy = wx - world[9][0], wy - world[9][1]
        n = math.hypot(ux, uy) or 1.0
        ux, uy = ux / n, uy / n
        side_x, side_y = -uy, ux                    # across the wrist
        thumb_side = 1.0 if (world[1][0] - wx) * side_x + (world[1][1] - wy) * side_y > 0 else -1.0
        half_wrist = 0.36 * HAND_SIZE
        wrist_t = to3d(wx + side_x * half_wrist * thumb_side, wy + side_y * half_wrist * thumb_side, 6.0)
        wrist_p = to3d(wx - side_x * half_wrist * thumb_side, wy - side_y * half_wrist * thumb_side, 6.0)
        web = _lerp(tuple(world[5]), tuple(world[2]), 0.45)
        # heel of the hand: a rounded edge into the wrist instead of a straight cut
        heel = [to3d(wx + ux * 0.16 * HAND_SIZE - side_x * half_wrist * thumb_side * s,
                     wy + uy * 0.16 * HAND_SIZE - side_y * half_wrist * thumb_side * s, 6.0)
                for s in (0.6, 0.0, -0.6)]
        outline = [wrist_t, joints[1], joints[2], to3d(web[0], web[1], LAYER_Z["palm"]),
                   joints[5], joints[9], joints[13], joints[17], wrist_p] + heel
        cuff = to3d(wx + ux * 0.95 * HAND_SIZE, wy + uy * 0.95 * HAND_SIZE, 14.0)
        elbow = to3d(wx + ux * 2.4 * HAND_SIZE, wy + uy * 2.4 * HAND_SIZE, 40.0)
        wrist3 = to3d(wx + ux * 0.1 * HAND_SIZE, wy + uy * 0.1 * HAND_SIZE, 7.0)
        return {"joints": joints, "outline": outline, "wrist": wrist3, "cuff": cuff,
                "elbow": elbow, "pinch": to3d(pointer[0], pointer[1], 0.0)}

    # ------------------------------------------------------------------ faces
    def parts(self, cam, pose, phase_color=None, tracking_ok: bool = True) -> List[tuple]:
        alpha = 255 if tracking_ok else 130
        ref = cam.depth(pose["pinch"])
        edge = (*SKIN_EDGE, alpha) if tracking_ok else (110, 110, 120, alpha)
        faces: List[tuple] = []

        def tone(color, depth, boost=1.0):
            if not tracking_ok:
                color = LOST
            k = max(0.8, min(1.12, 1.05 - (depth - ref) / 420.0)) * boost
            return (min(255, int(color[0] * k)), min(255, int(color[1] * k)),
                    min(255, int(color[2] * k)), alpha)

        # forearm (skin) then sleeve, both behind the hand
        d, pts = capsule(cam, pose["wrist"], pose["cuff"], 0.30 * HAND_SIZE, 0.33 * HAND_SIZE)
        faces.append((d + 3.0, pts, tone(SKIN, d, 0.96), edge))
        d, pts = capsule(cam, pose["cuff"], pose["elbow"], 0.40 * HAND_SIZE, 0.44 * HAND_SIZE)
        faces.append((d + 3.0, pts, tone(SLEEVE, d), (30, 36, 54, alpha)))

        # palm
        outline = pose["outline"]
        pd = sum(cam.depth(p) for p in outline) / len(outline)
        faces.append((pd, [cam.project(p) for p in outline], tone(SKIN, pd, 1.03), edge))

        # fingers: one rounded bone per segment, each finger on its own
        joints = pose["joints"]
        for name, chain in FINGERS.items():
            radii = RADII[name]
            bones = list(zip(chain, chain[1:]))
            for i, (a, b) in enumerate(bones):
                ra = radii[i]
                rb = radii[i + 1] if i + 1 < len(radii) else radii[-1] * 0.92
                d, pts = capsule(cam, joints[a], joints[b], ra, rb)
                color = SKIN
                if i == len(bones) - 1 and name in ("thumb", "index") and phase_color is not None:
                    color = _mix(SKIN, phase_color, 0.45)     # contact state on the fingertips
                faces.append((d, pts, tone(color, d), edge))
            # soft fingertip pad
            a, b = joints[chain[-2]], joints[chain[-1]]
            tip = (a[0] + (b[0] - a[0]) * 0.8, a[1] + (b[1] - a[1]) * 0.8, b[2] - 1.0)
            d, pts = disc(cam, tip, radii[-1] * 0.55)
            faces.append((d - 0.6, pts, tone(PAD, d), None))
        return faces


def capsule(cam, a: Vec, b: Vec, ra: float, rb: float, steps: int = 7):
    """Screen outline of a rounded bone from a to b (radii in world px)."""
    pa, pb = cam.project(a), cam.project(b)
    sa, sb = ra * cam.f / pa[2], rb * cam.f / pb[2]
    dx, dy = pb[0] - pa[0], pb[1] - pa[1]
    if math.hypot(dx, dy) < 0.5:
        return pb[2], [(pb[0] + sb * math.cos(t), pb[1] + sb * math.sin(t))
                       for t in (2 * math.pi * k / 14 for k in range(14))]
    phi = math.atan2(dy, dx)
    pts = [(pb[0] + sb * math.cos(phi - math.pi / 2 + math.pi * k / steps),
            pb[1] + sb * math.sin(phi - math.pi / 2 + math.pi * k / steps)) for k in range(steps + 1)]
    pts += [(pa[0] + sa * math.cos(phi + math.pi / 2 + math.pi * k / steps),
             pa[1] + sa * math.sin(phi + math.pi / 2 + math.pi * k / steps)) for k in range(steps + 1)]
    return (pa[2] + pb[2]) / 2.0, pts


def disc(cam, center: Vec, radius: float, steps: int = 12):
    p = cam.project(center)
    s = radius * cam.f / p[2]
    return p[2], [(p[0] + s * math.cos(t), p[1] + s * 0.85 * math.sin(t))
                  for t in (2 * math.pi * k / steps for k in range(steps))]
