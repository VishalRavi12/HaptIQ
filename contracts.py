"""Shared interface used by all four team members.

x/y: normalized numbers between 0 and 1.
pinch: thumb-index distance divided by an approximate palm scale.
"""
from dataclasses import dataclass


@dataclass
class HandState:
    timestamp_ms: int
    tracked: bool
    x: float
    y: float
    pinch: float
    closing_speed: float = 0.0  # positive means pinch is closing


@dataclass
class AIState:
    grasp_state: str  # open, closing, holding, releasing
    predicted_x: float
    predicted_y: float
    assistance: float = 0.0


@dataclass
class SimEvent:
    name: str  # CONTACT, GRASP, RELEASE, BREAK
    object_id: str
    timestamp_ms: int
