"""
Camera Calibration using Checkerboard Pattern
==============================================
Calibrates a camera using multiple images of a checkerboard pattern.
Outputs the camera matrix, distortion coefficients, and reprojection error.

Usage:
    python scripts/calibration/Endoscope_Calibration.py --images "calibration/calib_images/*.jpg" --rows 6 --cols 9 --size 25
    python scripts/calibration/Endoscope_Calibration.py --capture --width 1280 --height 720 --rows 6 --cols 9 --size 25
    python scripts/calibration/Endoscope_Calibration.py --capture --camera 1 --width 1280 --height 720 --upscale

Requirements:
    pip install opencv-python numpy
"""

import cv2
import numpy as np
import glob
import argparse
import json
import os
import sys
from datetime import datetime


DEFAULT_CAPTURE_WIDTH = 1280
DEFAULT_CAPTURE_HEIGHT = 720
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CALIBRATION_DIR = os.path.join(PROJECT_ROOT, "calibration")

# Windows capture backends and pixel formats to try when negotiating a
# resolution. Many UVC cameras only offer their higher modes as MJPG, and
# some accept a mode under MSMF that they refuse under DirectShow.
CAPTURE_BACKENDS = [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)]
CAPTURE_FOURCCS = ["MJPG", "YUY2", None]


def _try_open(camera_index, backend, fourcc, width, height):
    """Open the camera with one backend/fourcc combo and measure the real
    frame size from a grabbed frame (CAP_PROP values can lie)."""
    cap = cv2.VideoCapture(camera_index, backend)
    if not cap.isOpened():
        cap.release()
        return None, 0, 0
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        return None, 0, 0
    h, w = frame.shape[:2]
    return cap, w, h


def open_camera_at(camera_index, width, height):
    """
    Open a camera and negotiate the requested resolution.

    Tries every backend/fourcc combination and returns as soon as one
    delivers exactly width x height. If none does (the camera hardware
    doesn't offer that mode), reopens the combination that produced the
    largest frame and returns that.

    Returns (cap, actual_width, actual_height).
    """
    best_combo = None
    best_size = (0, 0)
    for backend_name, backend in CAPTURE_BACKENDS:
        for fourcc in CAPTURE_FOURCCS:
            cap, w, h = _try_open(camera_index, backend, fourcc, width, height)
            if cap is None:
                continue
            desc = f"{backend_name}/{fourcc or 'default'}"
            if (w, h) == (width, height):
                print(f"Camera {camera_index}: {w}x{h} via {desc}")
                return cap, w, h
            cap.release()
            if w * h > best_size[0] * best_size[1]:
                best_combo = (backend_name, backend, fourcc)
                best_size = (w, h)

    if best_combo is None:
        print(f"ERROR: Cannot open camera {camera_index} with any backend")
        sys.exit(1)

    backend_name, backend, fourcc = best_combo
    cap, w, h = _try_open(camera_index, backend, fourcc, width, height)
    if cap is None:
        print(f"ERROR: Camera {camera_index} stopped responding while reopening")
        sys.exit(1)
    print(f"WARNING: camera {camera_index} does not offer {width}x{height} in hardware.")
    print(f"         Best available mode: {w}x{h} via {backend_name}/{fourcc or 'default'}")
    print("         List its real modes with:")
    print('           ffmpeg -hide_banner -list_options true -f dshow -i video="<camera name>"')
    return cap, w, h


def calibrate_camera(image_pattern, board_rows, board_cols, square_size_mm,
                     show_corners=False, save_dir=None):
    """
    Calibrate camera from checkerboard images.

    Args:
        image_pattern: glob pattern for input images (e.g. "images/*.jpg")
        board_rows: number of INNER corners per row (squares - 1)
        board_cols: number of INNER corners per column (squares - 1)
        square_size_mm: physical size of a square in mm (for real-world units)
        show_corners: display each image with detected corners
        save_dir: optional directory to save corner-detection visualizations

    Returns:
        dict with camera_matrix, dist_coeffs, rvecs, tvecs, reprojection_error
    """
    # Termination criteria for corner sub-pixel refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points: (0,0,0), (1,0,0), (2,0,0) ... scaled by square size
    # These represent the checkerboard in its own coordinate system (Z=0 plane)
    objp = np.zeros((board_rows * board_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)
    objp *= square_size_mm

    # Arrays to store object points and image points from all images
    objpoints = []  # 3D points in real-world space
    imgpoints = []  # 2D points in image plane

    images = sorted(glob.glob(image_pattern))
    if not images:
        print(f"ERROR: No images found matching pattern: {image_pattern}")
        sys.exit(1)

    print(f"Found {len(images)} images. Processing...")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    img_shape = None
    successful = 0

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"  [SKIP] Could not read {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]  # (width, height)

        # Find the checkerboard corners
        ret, corners = cv2.findChessboardCorners(
            gray, (board_cols, board_rows),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret:
            # Refine corner locations to sub-pixel accuracy
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            successful += 1
            print(f"  [OK]   {os.path.basename(fname)}")

            if show_corners or save_dir:
                vis = img.copy()
                cv2.drawChessboardCorners(vis, (board_cols, board_rows), corners_refined, ret)
                if save_dir:
                    out_path = os.path.join(save_dir, f"corners_{os.path.basename(fname)}")
                    cv2.imwrite(out_path, vis)
                if show_corners:
                    cv2.imshow('Corners', vis)
                    cv2.waitKey(300)
        else:
            print(f"  [FAIL] {os.path.basename(fname)} - corners not found")

    if show_corners:
        cv2.destroyAllWindows()

    if successful < 5:
        print(f"\nWARNING: Only {successful} usable images. 10-20+ is recommended for good results.")
        if successful < 3:
            print("ERROR: Not enough images to calibrate.")
            sys.exit(1)

    print(f"\nCalibrating with {successful} images...")

    # Calibrate
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )

    # Compute mean reprojection error
    total_error = 0
    for i in range(len(objpoints)):
        projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i],
                                          camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], projected, cv2.NORM_L2) / len(projected)
        total_error += error
    mean_error = total_error / len(objpoints)

    return {
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coeffs,
        'rvecs': rvecs,
        'tvecs': tvecs,
        'reprojection_error': mean_error,
        'image_size': img_shape,
        'num_images_used': successful,
    }


def print_results(results):
    K = results['camera_matrix']
    d = results['dist_coeffs'].ravel()
    print("\n" + "=" * 60)
    print("CALIBRATION RESULTS")
    print("=" * 60)
    print(f"Image size:           {results['image_size']}")
    print(f"Images used:          {results['num_images_used']}")
    print(f"Reprojection error:   {results['reprojection_error']:.4f} pixels")
    print(f"  (good < 0.5, acceptable < 1.0, poor > 1.0)")
    print(f"\nCamera matrix (K):")
    print(f"  fx = {K[0,0]:.2f}    cx = {K[0,2]:.2f}")
    print(f"  fy = {K[1,1]:.2f}    cy = {K[1,2]:.2f}")
    print(f"\nDistortion coefficients (k1, k2, p1, p2, k3):")
    print(f"  {d}")
    print("=" * 60)


def _as_list(value):
    """Convert numpy values into plain Python values for text/JSON output."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def calibration_text_path(path):
    """Use .txt calibration files even if an old .npz path is supplied."""
    root, ext = os.path.splitext(path)
    if ext.lower() == ".npz":
        return root + ".txt"
    if not ext:
        return path + ".txt"
    return path


def save_calibration_txt(path, results):
    """Write calibration numbers as readable JSON text."""
    payload = {
        "camera_matrix": _as_list(results["camera_matrix"]),
        "dist_coeffs": _as_list(results["dist_coeffs"]),
        "image_size": _as_list(np.asarray(results["image_size"])),
        "reprojection_error": float(results["reprojection_error"]),
        "num_images_used": int(results["num_images_used"]),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def previous_reprojection_error(path, expected_size=None):
    """Read an older calibration error from either .txt JSON or legacy .npz.

    If expected_size is provided and the older calibration used a different
    image size, treat it as not comparable so a same-resolution calibration can
    replace it.
    """
    if not os.path.exists(path):
        return float("inf")
    try:
        if path.lower().endswith(".npz"):
            data = np.load(path)
            error = float(data["reprojection_error"])
            image_size = data["image_size"] if "image_size" in data.files else None
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            error = float(data["reprojection_error"])
            image_size = data.get("image_size")

        if expected_size is not None and image_size is not None:
            old_size = tuple(int(v) for v in image_size)
            want_size = tuple(int(v) for v in expected_size)
            if old_size != want_size:
                print(f"Previous calibration was for {old_size[0]}x{old_size[1]}; "
                      f"new calibration is {want_size[0]}x{want_size[1]}, so it will be saved.")
                return float("inf")
        return error
    except Exception:
        return float("inf")


def undistort_demo(results, sample_image_path, output_path):
    """Save a before/after comparison using the calibration."""
    img = cv2.imread(sample_image_path)
    if img is None:
        return
    h, w = img.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(
        results['camera_matrix'], results['dist_coeffs'], (w, h), 1, (w, h)
    )
    undistorted = cv2.undistort(img, results['camera_matrix'],
                                 results['dist_coeffs'], None, new_K)
    comparison = np.hstack([img, undistorted])
    cv2.imwrite(output_path, comparison)
    print(f"Saved before/after comparison to: {output_path}")


def capture_images(save_dir, board_rows, board_cols, camera_index=0, target=20,
                   frame_width=DEFAULT_CAPTURE_WIDTH,
                   frame_height=DEFAULT_CAPTURE_HEIGHT,
                   upscale=False):
    """
    Live webcam capture for calibration images.
    SPACE — capture frame   Q — quit and proceed

    If the camera hardware cannot deliver the requested resolution and
    upscale=True, every frame is resized to the requested size before corner
    detection and saving, so the calibration is valid for a live pipeline
    that applies the same resize.
    """
    os.makedirs(save_dir, exist_ok=True)
    cap, actual_w, actual_h = open_camera_at(camera_index, frame_width, frame_height)

    upscaling = upscale and (actual_w, actual_h) != (frame_width, frame_height)
    out_w, out_h = (frame_width, frame_height) if upscaling else (actual_w, actual_h)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    count = 0
    session = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"calib_{out_w}x{out_h}_{session}_"

    print(f"\nCamera open. Point at checkerboard ({board_cols}x{board_rows} inner corners).")
    print(f"Requested capture size: {frame_width}x{frame_height}")
    print(f"Sensor capture size:    {actual_w}x{actual_h}")
    if upscaling:
        print(f"Upscaling frames {actual_w}x{actual_h} -> {out_w}x{out_h} (interpolated).")
        print("NOTE: this adds no real detail. The calibration will only be valid for")
        print("      live frames resized the exact same way (cv2.resize to "
              f"{out_w}x{out_h}).")
    elif (actual_w, actual_h) != (frame_width, frame_height):
        print("WARNING: Camera hardware does not offer the requested resolution.")
        print("         Capturing at the size above. Re-run with --upscale to save")
        print(f"         interpolated {frame_width}x{frame_height} images instead.")
    print(f"SPACE = capture  |  Q = done and calibrate  |  target: {target} images\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if upscaling:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, (board_cols, board_rows),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        display = frame.copy()
        if found:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(display, (board_cols, board_rows), corners2, found)
            cv2.putText(display, f"DETECTED  [{count}/{target}]  SPACE=capture",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(display, f"Not detected  [{count}/{target}]",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Capture — SPACE to save, Q to finish", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q'):
            break
        if key == ord(' ') and found:
            path = os.path.join(save_dir, f"{prefix}{count:03d}.jpg")
            cv2.imwrite(path, frame)
            count += 1
            print(f"  Saved {path}")
        elif key == ord(' ') and not found:
            print("  Checkerboard not detected — move it into view and try again")

    cap.release()
    cv2.destroyAllWindows()

    if count == 0:
        print("No images captured. Exiting.")
        sys.exit(1)
    print(f"\nCaptured {count} images to '{save_dir}'")
    return os.path.join(save_dir, f"{prefix}*.jpg")


def main():
    parser = argparse.ArgumentParser(description="Camera calibration via checkerboard")
    parser.add_argument('--images', default=None,
                        help='Glob pattern for input images, e.g. "calib/*.jpg"')
    parser.add_argument('--capture', action='store_true',
                        help='Capture calibration images from webcam interactively')
    parser.add_argument('--capture-dir',
                        default=os.path.join(CALIBRATION_DIR, 'calib_images_1280x720'),
                        help='Directory to save captured images (default: calibration/calib_images_1280x720)')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index (default: 0)')
    parser.add_argument('--width', type=int, default=DEFAULT_CAPTURE_WIDTH,
                        help=f'Capture width in pixels (default: {DEFAULT_CAPTURE_WIDTH})')
    parser.add_argument('--height', type=int, default=DEFAULT_CAPTURE_HEIGHT,
                        help=f'Capture height in pixels (default: {DEFAULT_CAPTURE_HEIGHT})')
    parser.add_argument('--upscale', action='store_true',
                        help='If the camera cannot deliver --width x --height in '
                             'hardware, resize frames to that size before saving '
                             '(interpolated; live pipeline must resize the same way)')
    parser.add_argument('--rows', type=int, default=6,
                        help='Inner corners per row (squares per row minus 1)')
    parser.add_argument('--cols', type=int, default=9,
                        help='Inner corners per column (squares per column minus 1)')
    parser.add_argument('--size', type=float, default=25.0,
                        help='Square size in mm (default: 25)')
    parser.add_argument('--show', action='store_true',
                        help='Display each image with detected corners')
    parser.add_argument('--save-corners', default=None,
                        help='Directory to save corner visualizations')
    parser.add_argument('--output', default=os.path.join(CALIBRATION_DIR, 'calibration.txt'),
                        help='Text output file for calibration data (default: calibration/calibration.txt)')
    parser.add_argument('--undistort-sample', default=None,
                        help='Sample image to undistort as before/after demo')
    args = parser.parse_args()

    if not args.capture and args.images is None:
        parser.error("provide --images or use --capture to take photos from webcam")

    image_pattern = args.images
    if args.capture:
        image_pattern = capture_images(
            save_dir=args.capture_dir,
            board_rows=args.rows,
            board_cols=args.cols,
            camera_index=args.camera,
            frame_width=args.width,
            frame_height=args.height,
            upscale=args.upscale,
        )

    results = calibrate_camera(
        image_pattern=image_pattern,
        board_rows=args.rows,
        board_cols=args.cols,
        square_size_mm=args.size,
        show_corners=args.show,
        save_dir=args.save_corners,
    )

    print_results(results)

    output_path = calibration_text_path(args.output)
    args.output = output_path

    # Only save if this run is better than what's already on disk
    prev_error = previous_reprojection_error(output_path, expected_size=results['image_size'])

    new_error = results['reprojection_error']
    if new_error < prev_error:
        save_calibration_txt(output_path, results)
        if prev_error == float('inf'):
            print(f"\nCalibration saved to: {output_path}  (error: {new_error:.4f} px)")
        else:
            print(f"\nCalibration saved to: {args.output}  (improved: {prev_error:.4f} → {new_error:.4f} px)")
        print("Load it later with: json.load(open('calibration/calibration.txt'))")
    else:
        print(f"\nNot saved — previous calibration was better ({prev_error:.4f} px) than this run ({new_error:.4f} px)")

    if args.undistort_sample:
        undistort_demo(results, args.undistort_sample, 'undistort_comparison.jpg')


if __name__ == '__main__':
    main()
