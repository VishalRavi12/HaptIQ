"""Member 1 lesson 2: log hand movement. Requires the downloaded .task model."""
import time
from hand_tracking import HandTracker

tracker = HandTracker()
print('Move your hand in front of the camera; press Ctrl+C to stop.')
try:
    while True:
        state = tracker.read()
        if state.tracked:
            print(f'x={state.x:.2f} y={state.y:.2f} pinch={state.pinch:.2f} '
                  f'closing_speed={state.closing_speed:.2f}')
        else:
            print('No hand found')
        time.sleep(0.15)
except KeyboardInterrupt:
    print('Stopped')
finally:
    tracker.close()
