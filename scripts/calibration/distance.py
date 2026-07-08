"""
Distance Estimator using Checkerboard + Camera Calibration
===========================================================
Uses a calibrated camera and a known checkerboard to estimate
the real-world distance from the camera to the board.

Usage:
    python scripts/calibration/distance.py --calib calibration/calibration.txt --rows 7 --cols 9 --size 25
"""

import cv2
import numpy as np
import argparse
import json
import os
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_CALIBRATION = os.path.join(PROJECT_ROOT, "calibration", "calibration.txt")


def load_calibration(path):
    if path.lower().endswith(".npz"):
        data = np.load(path)
        camera_matrix = data["camera_matrix"]
        dist_coeffs = data["dist_coeffs"]
        error = float(data["reprojection_error"])
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
        dist_coeffs = np.asarray(data["dist_coeffs"], dtype=np.float64)
        error = float(data["reprojection_error"])
    print(f"Loaded calibration: reprojection error = {error:.4f} px")
    return camera_matrix, dist_coeffs


def make_objpoints(rows, cols, square_size_mm):
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size_mm
    return objp


def estimate_distance(frame, camera_matrix, dist_coeffs, objp, board_size, criteria):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray, board_size,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not found:
        return None, None, frame

    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    ok, rvec, tvec = cv2.solvePnP(objp, corners2, camera_matrix, dist_coeffs)
    if not ok:
        return None, None, frame

    # Distance = magnitude of translation vector (in mm, same units as square_size_mm)
    distance_mm = float(np.linalg.norm(tvec))
    distance_cm = distance_mm / 10.0

    vis = frame.copy()
    cv2.drawChessboardCorners(vis, board_size, corners2, found)
    cv2.drawFrameAxes(vis, camera_matrix, dist_coeffs, rvec, tvec, 30)

    label = f"Distance: {distance_cm:.1f} cm  ({distance_mm/10:.0f} mm)"
    cv2.rectangle(vis, (0, 0), (420, 45), (0, 0, 0), -1)
    cv2.putText(vis, label, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)

    return distance_mm, distance_cm, vis


def main():
    parser = argparse.ArgumentParser(description="Real-time distance estimation via checkerboard")
    parser.add_argument('--calib', default=DEFAULT_CALIBRATION,
                        help='Calibration file, .txt JSON or legacy .npz (default: calibration/calibration.txt)')
    parser.add_argument('--rows', type=int, default=7,
                        help='Inner corners per row')
    parser.add_argument('--cols', type=int, default=9,
                        help='Inner corners per column')
    parser.add_argument('--size', type=float, default=25.0,
                        help='Square size in mm (default: 25)')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index (default: 0)')
    parser.add_argument('--offset', type=float, default=-12.7,
                        help='Constant correction in cm to add to every reading (e.g. --offset -12.7 to subtract 5 inches)')
    args = parser.parse_args()

    camera_matrix, dist_coeffs = load_calibration(args.calib)
    objp = make_objpoints(args.rows, args.cols, args.size)
    board_size = (args.cols, args.rows)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {args.camera}")
        sys.exit(1)

    print("Live distance estimation running. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        dist_mm, dist_cm, vis = estimate_distance(
            frame, camera_matrix, dist_coeffs, objp, board_size, criteria
        )

        if dist_mm is None:
            cv2.putText(vis, "Checkerboard not detected", (8, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)
        else:
            dist_cm_corrected = dist_cm + args.offset
            dist_in_corrected = dist_cm_corrected / 2.54
            label = f"Distance: {dist_cm_corrected:.1f} cm  /  {dist_in_corrected:.1f} in"
            cv2.rectangle(vis, (0, 0), (480, 45), (0, 0, 0), -1)
            cv2.putText(vis, label, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
            print(f"\rDistance: {dist_cm_corrected:6.1f} cm  /  {dist_in_corrected:5.1f} in", end='', flush=True)

        cv2.imshow("Distance Estimator — Q to quit", vis)
        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
            break

    print()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
