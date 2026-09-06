"""
ChArUco camera calibration for the endoscope (camera 1, 640x480).

Why ChArUco over the plain checkerboard: the ArUco markers give each corner a
unique ID, so calibration works from PARTIAL board views -- essential for the
endoscope's tiny FOV where the whole board rarely fits. It also yields the board
POSE per image (rvecs/tvecs), which the photometric light fit (calib_lights.py)
needs. Per the project protocol: ChArUco on half a rigid coplanar sheet, blank
matte white on the other half -> intrinsics AND photometry from the same poses.

Usage:
    # 0. don't know the ArUco dictionary? point it at one captured image:
    python Charuco_Calibration.py --detect-dict "calibration/charuco/*.jpg"

    # 1. capture (live ChArUco preview; SPACE=grab, Q=done):
    python Charuco_Calibration.py --capture --camera 1 --width 640 --height 480 \
        --squares-x 5 --squares-y 7 --square-mm 6 --marker-mm 4.5 --dict 5x5_100

    # 2. or calibrate existing images:
    python Charuco_Calibration.py --images "calibration/charuco/*.jpg" \
        --squares-x 5 --squares-y 7 --square-mm 6 --marker-mm 4.5 --dict 5x5_100

Requires: pip install opencv-python  (cv2 >= 4.7 for the CharucoDetector API)
"""

import os
import sys
import glob
import json
import argparse
from datetime import datetime

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from Endoscope_Calibration import open_camera_at   # reuse the UVC negotiator

PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CALIB_DIR = os.path.join(PROJECT_ROOT, "calibration")

# name -> cv2.aruco predefined dictionary id. A board with S squares needs a
# dictionary of at least S/2 markers, so a 34x25 board (~425 markers) needs a
# _1000 dictionary (or ARUCO_ORIGINAL = 1024).
DICTS = {
    "4x4_50": cv2.aruco.DICT_4X4_50, "4x4_100": cv2.aruco.DICT_4X4_100,
    "4x4_250": cv2.aruco.DICT_4X4_250, "4x4_1000": cv2.aruco.DICT_4X4_1000,
    "5x5_50": cv2.aruco.DICT_5X5_50, "5x5_100": cv2.aruco.DICT_5X5_100,
    "5x5_250": cv2.aruco.DICT_5X5_250, "5x5_1000": cv2.aruco.DICT_5X5_1000,
    "6x6_50": cv2.aruco.DICT_6X6_50, "6x6_100": cv2.aruco.DICT_6X6_100,
    "6x6_250": cv2.aruco.DICT_6X6_250, "6x6_1000": cv2.aruco.DICT_6X6_1000,
    "7x7_100": cv2.aruco.DICT_7X7_100, "7x7_1000": cv2.aruco.DICT_7X7_1000,
    "original": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


def make_board(sx, sy, square_mm, marker_mm, dict_name):
    adict = cv2.aruco.getPredefinedDictionary(DICTS[dict_name])
    board = cv2.aruco.CharucoBoard((sx, sy), float(square_mm), float(marker_mm),
                                   adict)
    return board, adict


def detect_dict(pattern, sx, sy):
    """Try every dictionary on a few images; report which finds the most markers.
    Board layout (sx, sy) is only used to sanity-check the max possible markers."""
    files = sorted(glob.glob(pattern))[:5]
    if not files:
        print(f"no images match {pattern}")
        return
    print(f"testing {len(files)} image(s); a {sx}x{sy} board has "
          f"{(sx * sy) // 2} markers max\n")
    for name, did in DICTS.items():
        adict = cv2.aruco.getPredefinedDictionary(did)
        det = cv2.aruco.ArucoDetector(adict, cv2.aruco.DetectorParameters())
        total = 0
        for f in files:
            g = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)
            _, ids, _ = det.detectMarkers(g)
            total += 0 if ids is None else len(ids)
        flag = "  <-- likely your board" if total >= len(files) else ""
        print(f"  {name:10s}: {total:3d} markers found{flag}")
    print("\nPick the dictionary with a strong, consistent count and pass it "
          "as --dict.")


def calibrate(pattern, board, adict, save_corners=None):
    detector = cv2.aruco.CharucoDetector(board)
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"ERROR: no images match {pattern}"); sys.exit(1)
    all_obj, all_img, size = [], [], None
    used = 0
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        cc, ci, mc, mi = detector.detectBoard(gray)
        if cc is None or len(cc) < 6:              # need a few corners per view
            print(f"  [skip] {os.path.basename(f)}: "
                  f"{0 if cc is None else len(cc)} corners")
            continue
        obj, imgp = board.matchImagePoints(cc, ci)
        if obj is None or len(obj) < 6:
            continue
        all_obj.append(obj); all_img.append(imgp); used += 1
        print(f"  [ok]   {os.path.basename(f)}: {len(cc)} corners")
        if save_corners:
            os.makedirs(save_corners, exist_ok=True)
            vis = img.copy()
            cv2.aruco.drawDetectedCornersCharuco(vis, cc, ci)
            cv2.imwrite(os.path.join(save_corners, f"cc_{os.path.basename(f)}"), vis)

    if used < 8:
        print(f"\nWARNING: only {used} usable views (aim for 20-40, varied "
              "distance 15-45mm and tilt to 40deg).")
    if used < 4:
        print("ERROR: not enough views to calibrate."); sys.exit(1)

    print(f"\nCalibrating from {used} views ...")
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(all_obj, all_img, size,
                                                     None, None)
    return dict(camera_matrix=K, dist_coeffs=dist, image_size=size,
                reprojection_error=float(rms), num_images_used=used,
                rvecs=[r.ravel().tolist() for r in rvecs],
                tvecs=[t.ravel().tolist() for t in tvecs])


def capture(save_dir, board, camera, width, height):
    """Live ChArUco capture; SPACE grabs a frame, Q finishes."""
    os.makedirs(save_dir, exist_ok=True)
    detector = cv2.aruco.CharucoDetector(board)
    cap, aw, ah = open_camera_at(camera, width, height)
    session = datetime.now().strftime("%Y%m%d_%H%M%S")
    n = 0
    print("\nSPACE = grab (needs board detected)   Q = done and calibrate")
    print("Vary distance 15-45mm and tilt to ~40deg; keep exposure FIXED.\n")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cc, ci, mc, mi = detector.detectBoard(gray)
        vis = frame.copy()
        ncc = 0 if cc is None else len(cc)
        if mc is not None:
            cv2.aruco.drawDetectedMarkers(vis, mc, mi)
        if ncc:
            cv2.aruco.drawDetectedCornersCharuco(vis, cc, ci)
        col = (0, 255, 0) if ncc >= 6 else (0, 0, 255)
        cv2.putText(vis, f"corners:{ncc}  saved:{n}  SPACE=grab Q=done",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        cv2.imshow("ChArUco capture", vis)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), ord("Q"), 27):
            break
        if k == ord(" ") and ncc >= 6:
            p = os.path.join(save_dir, f"charuco_{session}_{n:03d}.jpg")
            cv2.imwrite(p, frame); n += 1
            print(f"  saved {p}  ({ncc} corners)")
        elif k == ord(" "):
            print("  board not detected well enough (need >=6 corners)")
    cap.release(); cv2.destroyAllWindows()
    if n == 0:
        print("no frames captured."); sys.exit(1)
    return os.path.join(save_dir, f"charuco_{session}_*.jpg")


def save_txt(path, r):
    payload = {
        "camera_matrix": r["camera_matrix"].tolist(),
        "dist_coeffs": r["dist_coeffs"].tolist(),
        "image_size": list(r["image_size"]),
        "reprojection_error": r["reprojection_error"],
        "num_images_used": r["num_images_used"],
        "board_poses": {"rvecs": r["rvecs"], "tvecs": r["tvecs"]},
        "method": "charuco",
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2); f.write("\n")


def main():
    ap = argparse.ArgumentParser(description="ChArUco calibration for the endoscope")
    ap.add_argument("--images", default=None, help='glob of images to calibrate')
    ap.add_argument("--capture", action="store_true", help="capture from camera")
    ap.add_argument("--detect-dict", default=None,
                    help="glob of images: try every dictionary and report matches")
    ap.add_argument("--camera", type=int, default=1)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--capture-dir", default=os.path.join(CALIB_DIR, "charuco"))
    ap.add_argument("--squares-x", type=int, required=False, default=34)
    ap.add_argument("--squares-y", type=int, required=False, default=25)
    ap.add_argument("--square-mm", type=float, default=6.0)
    ap.add_argument("--marker-mm", type=float, default=4.5)
    ap.add_argument("--dict", default="5x5_1000", choices=sorted(DICTS))
    ap.add_argument("--save-corners", default=None)
    ap.add_argument("--output", default=os.path.join(CALIB_DIR,
                    "charuco_calibration.txt"))
    a = ap.parse_args()

    if a.detect_dict:
        detect_dict(a.detect_dict, a.squares_x, a.squares_y)
        return

    board, adict = make_board(a.squares_x, a.squares_y, a.square_mm,
                              a.marker_mm, a.dict)
    print(f"board: {a.squares_x}x{a.squares_y} squares, {a.square_mm}mm square, "
          f"{a.marker_mm}mm marker, dict {a.dict}")

    pattern = a.images
    if a.capture:
        pattern = capture(a.capture_dir, board, a.camera, a.width, a.height)
    if not pattern:
        ap.error("provide --images, or --capture, or --detect-dict")

    r = calibrate(pattern, board, adict, a.save_corners)
    K = r["camera_matrix"]
    print("\n" + "=" * 56)
    print(f"image size          {r['image_size']}")
    print(f"views used          {r['num_images_used']}")
    print(f"reprojection error  {r['reprojection_error']:.4f} px  (good < 0.5)")
    print(f"fx {K[0,0]:.2f}   fy {K[1,1]:.2f}   cx {K[0,2]:.2f}   cy {K[1,2]:.2f}")
    print("=" * 56)
    save_txt(a.output, r)
    print(f"saved -> {a.output}   (includes per-view board poses)")


if __name__ == "__main__":
    main()
