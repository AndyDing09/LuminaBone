"""Probe camera indices 0-9 and report which ones open successfully."""
import cv2

for i in range(10):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            print(f"Camera {i}: OK  ({w}x{h})")
        else:
            print(f"Camera {i}: opens but no frame")
        cap.release()
    else:
        print(f"Camera {i}: not available")
