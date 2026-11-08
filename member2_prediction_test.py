"""Member 2 lesson 1: no webcam required. Watch predictions for sample points."""
from contracts import HandState
from grasp_ai import GraspBaseline

model = GraspBaseline(horizon_s=0.15)
for ms, x in [(0, 0.10), (100, 0.15), (200, 0.20), (300, 0.25)]:
    current = HandState(ms, True, x, 0.50, 1.0)
    prediction = model.update(current)
    print(f't={ms:3d}ms current_x={x:.2f} predicted_x={prediction.predicted_x:.2f}')
print('This is a mathematical baseline, not a trained AI model.')
