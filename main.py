"""Main beginner demo. Uses mouse by default; optionally uses webcam and Arduino LED.

Mouse: python main.py
Camera: python main.py --camera
Arduino: python main.py --port COM3      (Windows, adjust port)
Both: python main.py --camera --port COM3
"""
import argparse
import math
import time

import pygame
from simulation import Simulation, WIDTH, HEIGHT
from hardware import HardwareLink
from grasp_ai import GraspBaseline


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', action='store_true', help='Use real MediaPipe camera input')
    parser.add_argument('--port', default=None, help='Arduino serial port, e.g., COM3')
    args = parser.parse_args()
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('GhostGrip - Beginner Demo')
    font = pygame.font.SysFont(None, 23)
    clock = pygame.time.Clock()
    simulation = Simulation()
    hardware = HardwareLink(args.port)
    hand_tracker = None
    model = GraspBaseline()
    previous_xy = None
    previous_time = None
    running = True
    try:
        if args.camera:
            from hand_tracking import HandTracker
            hand_tracker = HandTracker()
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_r:
                        simulation.reset()

            if args.camera:
                hand = hand_tracker.read()
                prediction = model.update(hand)
                # If tracking is lost, disengage grasp immediately.
                if hand.tracked:
                    x, y = hand.x * WIDTH, hand.y * HEIGHT
                    gripping = prediction.grasp_state == 'holding'
                    predicted = (prediction.predicted_x*WIDTH,
                                 prediction.predicted_y*HEIGHT)
                else:
                    x, y = WIDTH/2, HEIGHT/2
                    gripping = False
                    predicted = None
            else:
                x, y = pygame.mouse.get_pos()
                gripping = pygame.mouse.get_pressed()[0]
                predicted = None

            now = time.monotonic()
            speed = 0
            if previous_xy is not None and previous_time is not None:
                dt = now - previous_time
                if dt > 0:
                    speed = math.hypot(x-previous_xy[0], y-previous_xy[1])/dt
            previous_xy, previous_time = (x, y), now
            events = simulation.update(x, y, gripping, speed)
            for ev in events:
                print('EVENT:', ev.name, ev.object_id)
                hardware.send(ev.name)
            simulation.draw(screen, font, x, y, gripping, predicted)
            pygame.display.flip()
            clock.tick(30)
    finally:
        if hand_tracker:
            hand_tracker.close()
        hardware.close()
        pygame.quit()


if __name__ == '__main__':
    run()
