#!/usr/bin/env python3


import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    import serial
except ImportError:
    serial = None

N_LEDS = 4
FLUSH = 6          # frames discarded after any LED or exposure change
SETTLE_S = 0.15    # let the LED and the driver settle
COOL_S = 0.25      # keep duty low; LED output droops as the junction heats


# --------------------------------------------------------------------------
# LED control
# --------------------------------------------------------------------------

class Leds:
    def __init__(self, port=None, baud=115200, mode="serial"):
        self.mode, self.ser = mode, None
        if mode != "serial":
            return
        if serial is None:
            raise SystemExit("pip install pyserial, or use --manual / --sim")
        self.ser = serial.Serial(port, baud, timeout=2)
        time.sleep(2.0)                      # Arduino resets when the port opens
        self.ser.reset_input_buffer()
        self.set(0)
        print(f"LED controller on {port}")

    def set(self, n):
        """n = 1..4 for a single LED, 0 for all off, -1 for all on."""
        if self.mode == "sim":
            return
        if self.mode == "manual":
            label = "ALL OFF" if n == 0 else ("ALL ON" if n < 0 else f"LED {n} ONLY")
            input(f"    >> switch to {label}, then press Enter...")
            return
        self.ser.write((b"A\n" if n < 0 else f"L{n}\n".encode()))
        self.ser.readline()

    def close(self):
        if self.ser:
            self.set(0)
            self.ser.close()


# --------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------

def open_camera(index, width, height, exposure_us):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {index}")
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)          # helps; not always honoured
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)    # 0.25 = manual on most UVC cams
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    if exposure_us:
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure_us / 1e6)
    for _ in range(15):
        cap.read()

    print("camera settings actually in effect:")
    for k in ["FRAME_WIDTH", "FRAME_HEIGHT", "EXPOSURE", "AUTO_EXPOSURE",
              "GAIN", "AUTO_WB"]:
        print(f"   {k:14s} {cap.get(getattr(cv2, 'CAP_PROP_' + k)):.4f}")
    print("   ^ if AUTO_EXPOSURE did not change, your driver ignored it.\n")
    return cap


def flush(cap, n=FLUSH):
    """Throw away buffered frames exposed under the previous LED state."""
    for _ in range(n):
        cap.read()


def grab(cap):
    for _ in range(3):
        ok, f = cap.read()
        if ok:
            return f
    raise RuntimeError("frame grab failed")


# --------------------------------------------------------------------------
# Capture
# --------------------------------------------------------------------------

def capture_set(cap, leds, outdir, shot_n, save_dark=True, prefix="shot"):
    d = Path(outdir) / f"{prefix}_{shot_n:03d}"
    d.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n  capturing {prefix} {shot_n:03d} -> {d}")

    if save_dark:
        # Any light that is not from your LEDs adds equally to both members of
        # an opposing pair. That inflates the denominator of the pair ratio and
        # your surface comes out flatter than it is. Subtract it.
        leds.set(0)
        time.sleep(SETTLE_S)
        flush(cap)
        cv2.imwrite(str(d / "dark.png"), grab(cap))
        print("    dark   saved")

    means = []
    for i in range(1, N_LEDS + 1):
        leds.set(i)
        time.sleep(SETTLE_S)
        flush(cap)                      # <-- the important line
        img = grab(cap)
        cv2.imwrite(str(d / f"led{i}.png"), img)
        g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sat = float((g >= 250).mean())
        means.append(float(g.mean()))
        warn = "  ** CLIPPING **" if sat > 0.001 else ""
        print(f"    led{i}  saved   mean={means[-1]:6.1f}  "
              f"clipped={sat*100:.2f}%{warn}")
        leds.set(0)
        time.sleep(COOL_S)

    # Opposing pairs should be close on a roughly flat, roughly centred target.
    # A large gap means unmatched LED flux, and flux mismatch turns directly
    # into a systematic tilt across the whole reconstruction.
    p1 = abs(means[0] - means[2]) / max(means[0] + means[2], 1e-9) * 2
    p2 = abs(means[1] - means[3]) / max(means[1] + means[3], 1e-9) * 2
    print(f"    pair imbalance: 1-3 {p1*100:5.1f}%   2-4 {p2*100:5.1f}%")

    # If the four frames are indistinguishable, the LEDs never switched --
    # nothing plugged in, wrong serial port, or stale frames getting through.
    spread = (max(means) - min(means)) / max(np.mean(means), 1e-9)
    if spread < 0.02:
        print("    ** all four frames look identical (spread "
              f"{spread*100:.1f}%) -- LEDs are not switching. **")
        print("       expected if you are testing on a webcam with no LEDs.")

    (d / "notes.txt").write_text(
        f"{stamp}\nmeans {means}\npair imbalance {p1:.4f} {p2:.4f}\n")
    print(f"  saved -> {d.resolve()}\n")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None, help="Arduino serial port")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--out",
                    default=str(Path(__file__).resolve().parent / "shots"))
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--exposure-us", type=float, default=None)
    ap.add_argument("--manual", action="store_true",
                    help="no Arduino: prompts you to switch LEDs by hand")
    ap.add_argument("--sim", action="store_true", help="no LED hardware at all")
    ap.add_argument("--no-dark", action="store_true")
    ap.add_argument("--flatfield", action="store_true",
                    help="capture a flat-field calibration set (dark + 4 LEDs of a "
                         "FLAT MATTE-WHITE card at the bone working distance); saved "
                         "as flatfield_NNN, not shot_NNN")
    a = ap.parse_args()

    mode = "manual" if a.manual else ("sim" if a.sim else "serial")
    if mode == "serial" and not a.port:
        raise SystemExit("--port is required (or use --manual / --sim)")

    cap = open_camera(a.camera, a.width, a.height, a.exposure_us)
    leds = Leds(a.port, mode=mode)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prefix = "flatfield" if a.flatfield else "shot"
    shot_n = 1 + max([int(p.name.split("_")[1]) for p in out.glob(f"{prefix}_*")]
                     or [0])

    print("=" * 56)
    print(f"  saving to: {out.resolve()}")
    if a.flatfield:
        print("  *** FLAT-FIELD MODE ***")
        print("  Fill the frame with a FLAT MATTE-WHITE card at the bone")
        print("  working distance, SAME exposure as your shots. Watch for")
        print("  clipping (tilt/dim slightly if it saturates).")
    print(f"  next: {prefix}_{shot_n:03d}")
    print("  G  capture 4 photos (one per LED)")
    print("  Q  quit")
    print("=" * 56)

    leds.set(-1)            # all on, so you can see what you are aiming at
    time.sleep(SETTLE_S)
    flush(cap)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            view = frame.copy()
            cv2.putText(view,
                        f"G = capture   Q = quit   next: {prefix}_{shot_n:03d}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow("endoscope  [G] capture  [Q] quit", view)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord("g"), ord("G")):
                capture_set(cap, leds, out, shot_n, save_dark=not a.no_dark,
                            prefix=prefix)
                shot_n += 1
                leds.set(-1)              # back to preview
                time.sleep(SETTLE_S)
                flush(cap)
            elif k in (ord("q"), ord("Q"), 27):
                break
    finally:
        leds.close()
        cap.release()
        cv2.destroyAllWindows()
        print(f"saved to {out.resolve()}")


if __name__ == "__main__":
    main()