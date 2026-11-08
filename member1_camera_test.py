"""Member 1 lesson 1: display a webcam feed. Press Q to quit."""
import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError('No webcam. Close Zoom/Teams and check camera permissions.')
print('Camera started. Press Q in the camera window to quit.')
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print('Camera frame failed.'); break
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, 'GhostGrip camera test - press Q', (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow('Member 1: Camera Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
