"""Member 3: very simple Pygame grab-and-drop simulation.

Mouse is the controller. Hold left mouse button near an object to pick it up.
The objects demonstrate simplified synthetic properties, not real material physics.
"""
from dataclasses import dataclass
import math
import time
from contracts import SimEvent

WIDTH, HEIGHT = 900, 600


@dataclass
class SimObject:
    name: str
    x: float
    y: float
    color: tuple
    radius: int
    fragile: bool = False
    broken: bool = False


class Simulation:
    def __init__(self):
        self.objects = [
            SimObject('glass', 230, 300, (85, 195, 245), 33, fragile=True),
            SimObject('sponge', 450, 300, (245, 205, 75), 40),
            SimObject('block', 680, 300, (165, 175, 195), 37),
        ]
        self.held = None
        self.events = []
        self.attempts = 0
        self.successes = 0

    def reset(self):
        self.__init__()

    def update(self, x, y, gripping, movement_speed=0.0):
        """x/y are screen coordinates; movement_speed in pixels/sec."""
        self.events = []
        if self.held is None and gripping:
            for obj in self.objects:
                if obj.broken:
                    continue
                if math.hypot(obj.x-x, obj.y-y) < obj.radius+20:
                    self.attempts += 1
                    self.held = obj
                    self.events.append(SimEvent('GRASP', obj.name, now_ms()))
                    break
        if self.held is not None:
            if not gripping:
                self.events.append(SimEvent('RELEASE', self.held.name, now_ms()))
                self.successes += 1
                self.held = None
            else:
                self.held.x, self.held.y = x, y
                # In this beginner demo a fragile object breaks on extreme travel speed.
                if self.held.fragile and movement_speed > 1200:
                    self.held.broken = True
                    self.events.append(SimEvent('BREAK', self.held.name, now_ms()))
                    self.held = None
        return self.events

    def draw(self, screen, font, x, y, gripping, predicted=None):
        import pygame
        screen.fill((20, 27, 43))
        for obj in self.objects:
            if obj.broken:
                continue
            center = (int(obj.x), int(obj.y))
            pygame.draw.circle(screen, obj.color, center, obj.radius)
            label = font.render(obj.name, True, (15, 20, 30))
            screen.blit(label, label.get_rect(center=center))
        pygame.draw.circle(screen, (110, 240, 170) if gripping else (230, 245, 255),
                           (int(x), int(y)), 15, 3)
        if predicted is not None:
            pygame.draw.circle(screen, (220, 135, 255),
                               (int(predicted[0]), int(predicted[1])), 10, 2)
        lines = [
            'GhostGrip BEGINNER SIMULATION (mouse control)',
            'Move cursor near a circle. Hold LEFT MOUSE to grab; release to drop.',
            'R: reset     Q/ESC: quit    Shift: move faster to test fragile behavior',
            f'Successful releases: {self.successes} / attempted grabs: {self.attempts}',
            'Color circle = virtual object; ring = controlled hand point',
            'No trained AI or physical force modeling in this initial milestone.',
        ]
        for i, line in enumerate(lines):
            screen.blit(font.render(line, True, (220, 229, 245)), (20, 15+i*26))


def now_ms():
    return int(time.monotonic()*1000)
