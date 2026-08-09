"""
LED index->azimuth permutation study (a real methodological result for the paper).

For the two CT-validated near levels, compute far-field photometric depth under all
4! = 24 assignments of azimuths {0,90,180,270} to LED indices {1,2,3,4}, and correlate
the depth sampled at the fiducials with the CT marker depth. Shows that a wrong LED
labelling INVERTS the recovered surface (negative correlation), and that the physical
wiring maximises it.

Run: python led_permutation_study.py
"""
import os, sys, math, json, itertools
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "reconstruction"))
import bone_depth_batch as bd
bd.UNDISTORT_INPUTS = False
import calibrate_photometric_ct as C

cal = json.load(open(bd.project_path("calibration", "calibration.txt")))
K = np.array(cal["camera_matrix"]); DIST = np.array(cal["dist_coeffs"])
ELEV = math.degrees(math.atan2(40.0, 6.05))
AZ_SET = [0.0, 90.0, 180.0, 270.0]
TRUE = {1: 90.0, 2: 270.0, 3: 180.0, 4: 0.0}      # confirmed physical wiring


def load(shot):
    d = bd.project_path("Data_collection", "calib_charuco", shot)
    dark = bd.load_rgb(os.path.join(d, "dark.png"), bd.WORK_LONG_EDGE)
    lums = [bd.solve_luminance(np.clip(bd.load_rgb(os.path.join(d, f"led{i}.png"),
            bd.WORK_LONG_EDGE) - dark, 0, 1)) for i in (1, 2, 3, 4)]
    return bd.balance_exposure(lums)


def ct_depth(corr):
    ids = sorted(corr)
    obj = np.array([corr[i][1] for i in ids], float)
    img = np.array([corr[i][0] for i in ids])
    _, rv, tv = cv2.solvePnP(obj, img, K, DIST, flags=cv2.SOLVEPNP_SQPNP)
    R, _ = cv2.Rodrigues(rv)
    return (R @ obj.T + tv.reshape(3, 1)).T[:, 2]


def disk(z, u, v, r=14):
    H, W = z.shape; yy, xx = np.ogrid[:H, :W]
    return float(np.median(z[(xx - u) ** 2 + (yy - v) ** 2 <= r * r]))


def corr_for(shot, azmap):
    corr = C.CT_CORR[shot]; ids = sorted(corr)
    lums = load(shot)
    Ls = np.array([bd.light_vector(azmap[i], ELEV) for i in (1, 2, 3, 4)])
    n, _ = bd.photometric_stereo(lums, Ls)
    z = bd.normals_to_depth(n)
    zp = np.array([disk(z, *corr[i][0]) for i in ids])
    cd = ct_depth(corr)
    return float(np.corrcoef(zp, cd)[0, 1])


def main():
    shots = ["shot_004", "shot_005"]
    results = []
    for perm in itertools.permutations(AZ_SET):
        azmap = {i + 1: perm[i] for i in range(4)}
        cs = [corr_for(s, azmap) for s in shots]
        results.append((azmap, cs, np.mean(cs)))
    results.sort(key=lambda r: -r[2])
    print("LED index->azimuth permutation study (correlation of far-field depth vs CT):")
    print(f"{'rank':5}{'1':>5}{'2':>5}{'3':>5}{'4':>5}{'r(4)':>8}{'r(5)':>8}{'mean':>8}")
    for k, (az, cs, m) in enumerate(results):
        tag = "  <- TRUE wiring" if az == TRUE else ""
        if k < 3 or k > len(results) - 3 or az == TRUE:
            print(f"{k+1:<5}{az[1]:5.0f}{az[2]:5.0f}{az[3]:5.0f}{az[4]:5.0f}"
                  f"{cs[0]:8.2f}{cs[1]:8.2f}{m:8.2f}{tag}")
    best = results[0]; worst = results[-1]
    print(f"\nbest mean r = {best[2]:+.2f} (wiring {dict(best[0])})")
    print(f"worst mean r = {worst[2]:+.2f} (a mislabelling INVERTS the surface)")


if __name__ == "__main__":
    main()
