"""Object profiles for glass, sponge and block.

ALL VALUES ARE SYNTHETIC teaching parameters chosen so the three objects behave
differently. They are NOT measured material properties, and effort G is a
dimensionless number, NOT newtons (technical document 5.3, 18.1).

Grip effort (technical document 5.3):
    G = stiffness * max(0, q - contact_threshold)

Required holding effort follows the two-contact friction idea from technical
document 5.4 (normal force ~ m*g / (2*mu)), used only to ORDER the objects:
    hold_threshold = HOLD_K * virtual_mass / (2 * friction)

Resulting grip bands (q = closure, 0 open .. 1 closed):
    glass : contact 0.30, holds at q >= 0.40, breaks at q > 0.82 held for 0.2 s (raised from
            0.60 in Phase 2: real camera grips broke it too easily). Kept below the block's
            0.85 so one grip still gives three different outcomes (glass breaks, sponge
            squashes, block slips).
    sponge: contact 0.20, holds at q >= ~0.39, never breaks, squashes visibly
    block : contact 0.35, holds only at q >= 0.85, slips below that
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

HOLD_K = 0.2  # scales m/(2*mu) into effort units


@dataclass(frozen=True)
class ObjectProfile:
    kind: str
    radius: float                     # world px; collision circle and half-size for drawing
    contact_threshold: float          # q at which the virtual pads touch the surface
    stiffness: float                  # effort gained per unit of q beyond contact
    virtual_mass: float               # synthetic
    friction: float                   # synthetic
    break_threshold: Optional[float]  # effort above which it is damaged; None = unbreakable
    compressibility: float            # 0 rigid .. 1 very soft (visual squash only)
    drop_break_speed: Optional[float]  # impact speed (px/s) that damages it; None = never
    color: Tuple[int, int, int]

    @property
    def hold_threshold(self) -> float:
        return HOLD_K * self.virtual_mass / (2.0 * self.friction)

    def effort(self, q: float) -> float:
        return self.stiffness * max(0.0, q - self.contact_threshold)

    def q_for_effort(self, effort: float) -> float:
        """Closure q that produces a given effort (used to draw grip-meter bands)."""
        return self.contact_threshold + effort / self.stiffness


PROFILES: Dict[str, ObjectProfile] = {
    "glass": ObjectProfile(
        kind="glass", radius=30, contact_threshold=0.30, stiffness=2.0,
        virtual_mass=1.0, friction=0.5, break_threshold=1.04,     # effort 1.04 = q 0.82
        compressibility=0.0, drop_break_speed=650.0, color=(120, 200, 245)),
    "sponge": ObjectProfile(
        kind="sponge", radius=38, contact_threshold=0.20, stiffness=0.8,
        virtual_mass=0.9, friction=0.6, break_threshold=None,
        compressibility=0.55, drop_break_speed=None, color=(245, 205, 75)),
    "block": ObjectProfile(
        kind="block", radius=34, contact_threshold=0.35, stiffness=1.5,
        virtual_mass=3.0, friction=0.4, break_threshold=None,
        compressibility=0.0, drop_break_speed=None, color=(150, 160, 180)),
}
