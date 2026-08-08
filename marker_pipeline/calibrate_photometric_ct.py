"""
Calibrate the photometric-stereo depth to true DISTANCE FROM LENS using CT.

The photometric solve gives a relative depth (arbitrary scale, offset). The CT
gives the true 3-D marker positions; the 4 markers of one segment give the
CT->camera pose (PnP with the real calibration), so each marker's true camera-
frame depth (mm from the lens) is known. Fit z_lens = a*z_photo + b over the
markers -> the photometric depth is now calibrated to real mm from the lens.

This is the CT used as CALIBRATION (ground control), not as a distance readout.
Extend CT_CORR for more shots. Run: python calibrate_photometric_ct.py
"""

import os
import sys
import math
import json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "reconstruction"))
import bone_depth_batch as bd
bd.UNDISTORT_INPUTS = False              # detect + solve on raw frames

SHOT_ROOT = bd.project_path("Data_collection", "calib_charuco")
# CONFIRMED LED->azimuth mapping (validated against CT on shots 4 & 5: the far-
# field depth correlates +0.97/+0.99, vs -0.73/-0.70 for the old 0/90/180/270).
# LED1=top, LED2=bottom, LED3=left, LED4=right; opposing pairs (1,2) and (3,4).
AZ = {1: 90.0, 2: 270.0, 3: 180.0, 4: 0.0}
A_MM, WD_MM = 6.05, 40.0
ELEV = math.degrees(math.atan2(WD_MM, A_MM))

# per shot: {my_marker_id: (pixel_xy, CT_xyz_mm)}  (correspondence you gave)
CT_CORR = {
    "shot_004": {
        1: ((370.5, 165.2), (-6.62, 22.0, -18.85)),   # M02
        2: ((483.5, 306.0), (3.4, 23.1, -8.59)),       # M04
        3: ((385.0, 371.0), (6.37, 13.59, -15.77)),    # M03
        4: ((312.2, 410.0), (5.88, 3.8, -20.03)),      # M01
    },
    "shot_005": {
        1: ((339.5, 166.8), (-2.1, 22.64, 12.59)),     # M06
        2: ((440.2, 295.2), (5.69, 21.12, 21.81)),     # M08
        3: ((406.8, 361.2), (6.85, 8.31, 18.36)),      # M07
        4: ((311.5, 373.8), (4.09, -1.66, 12.23)),     # M05
    },
    "shot_006": {
        1: ((276.5, 239.0), (1.27, 16.36, 42.93)),     # M09
        2: ((406.0, 297.0), (5.02, 20.0, 54.21)),      # M11
        3: ((426.0, 384.2), (6.93, 6.53, 54.86)),      # M12
        4: ((303.5, 384.8), (2.33, -6.52, 47.09)),     # M10
    },
    "shot_007": {                                       # only 3 markers this shot
        1: ((159.0, 242.2), (0.49, 11.9, 72.14)),      # M13
        2: ((346.0, 309.0), (3.49, 15.04, 87.17)),     # M15
        3: ((150.8, 366.5), (-0.45, -10.73, 76.71)),   # M14
    },
    "shot_008": {
        1: ((458.2, 295.8), (-3.0, 9.0, 121.09)),      # M19
        2: ((257.8, 329.5), (-1.81, 4.74, 106.25)),    # M17
        3: ((374.7, 433.7), (0.38, -6.3, 113.07)),     # M18
        4: ((161.0, 436.3), (-3.33, -15.36, 102.53)),  # M16
    },
}


def photometric_depth(shot):
    d = os.path.join(SHOT_ROOT, shot)
    dark = bd.load_rgb(os.path.join(d, "dark.png"), bd.WORK_LONG_EDGE)
    lums, Ls, ref = [], [], None
    for i in (1, 2, 3, 4):
        rgb = bd.load_rgb(os.path.join(d, f"led{i}.png"), bd.WORK_LONG_EDGE)
        if i == 1:
            ref = rgb
        lums.append(bd.solve_luminance(np.clip(rgb - dark, 0, 1)))
        Ls.append(bd.light_vector(AZ[i], ELEV))
    lums = bd.balance_exposure(lums)
    normals, albedo = bd.photometric_stereo(lums, np.array(Ls))
    z = bd.normals_to_depth(normals)     # relative depth (arbitrary units)
    return z, ref, bd.bone_mask(albedo)


def sample(z, u, v, r=14):
    H, W = z.shape
    yy, xx = np.ogrid[:H, :W]
    disk = (xx - u) ** 2 + (yy - v) ** 2 <= r * r
    return float(np.median(z[disk]))


def main(shot="shot_004"):
    corr = CT_CORR[shot]
    ids = sorted(corr)
    z, ref, mask = photometric_depth(shot)

    # CT -> camera pose from the 4 markers (real calibration + distortion)
    cal = json.load(open(bd.project_path("calibration", "calibration.txt")))
    K = np.array(cal["camera_matrix"]); dist = np.array(cal["dist_coeffs"])
    img = np.array([corr[i][0] for i in ids], float)
    obj = np.array([corr[i][1] for i in ids], float)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_SQPNP)
    R, _ = cv2.Rodrigues(rvec)
    cam = (R @ obj.T + tvec).T
    z_lens_true = cam[:, 2]                              # axial depth from lens (mm)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    reproj = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)

    # calibrate: z_lens = a*z_photo + b   (4 markers -> over-determined)
    z_photo = np.array([sample(z, *corr[i][0]) for i in ids])
    A = np.column_stack([z_photo, np.ones_like(z_photo)])
    (a, b), *_ = np.linalg.lstsq(A, z_lens_true, rcond=None)
    resid = (a * z_photo + b) - z_lens_true
    z_cal = a * z + b                                   # calibrated depth-from-lens map

    # ---- figure ----
    fig = plt.figure(figsize=(13, 5.8), dpi=120)
    fig.patch.set_facecolor("#f5f5f0")
    ax0 = fig.add_subplot(1, 3, 1)
    ax0.imshow(np.clip(ref, 0, 1)); ax0.axis("off")
    ax0.set_title(f"{shot} led1", fontsize=10, fontweight="bold")

    lo, hi = np.percentile(z_cal[mask] if mask.any() else z_cal, [2, 98])
    ax1 = fig.add_subplot(1, 3, 2)
    im = ax1.imshow(z_cal, cmap="turbo_r", vmin=lo, vmax=hi)
    for i in ids:
        u, v = corr[i][0]
        ax1.add_patch(plt.Circle((u, v), 15, fill=False, ec="k", lw=1.6))
        ax1.annotate(f"{a*sample(z,u,v)+b:.0f}|CT{cam[ids.index(i),2]:.0f}",
                     (u, v), (u + 17, v), color="k", fontsize=8, fontweight="bold",
                     va="center", bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                            alpha=0.7, ec="none"))
    ax1.set_title("CT-calibrated depth from lens (mm)\nlabel = est | CT",
                  fontsize=10, fontweight="bold"); ax1.axis("off")
    plt.colorbar(im, ax=ax1, shrink=0.8, pad=0.02).set_label(
        "axial distance from lens (mm)", fontsize=8)

    ax2 = fig.add_subplot(1, 3, 3, projection="3d")
    zsm = gaussian_filter(z_cal, 1.5)
    H, W = z_cal.shape; step = max(1, W // 140)
    yg, xg = np.mgrid[0:H:step, 0:W:step]
    ax2.plot_surface(xg, yg, -zsm[::step, ::step], cmap="turbo",
                     linewidth=0, antialiased=True)
    ax2.set_title("3-D (up = toward lens)", fontsize=10, fontweight="bold")
    ax2.view_init(elev=55, azim=-60)

    fig.suptitle(f"{shot} — photometric stereo CALIBRATED to lens with CT "
                 f"({len(ids)} markers)", fontsize=12, fontweight="bold", y=0.99)
    fig.text(0.5, 0.01, f"z_lens = {a:.2f}*z_photo {b:+.1f} mm.  PnP reproj "
             f"{reproj.mean():.2f} px (fx={K[0,0]:.0f}).  marker fit RMS "
             f"{np.sqrt(np.mean(resid**2)):.2f} mm.", ha="center", fontsize=8,
             style="italic", color="#555")
    out = bd.project_path("depth_outputs", "calib_charuco_slides",
                          f"{shot}_ct_calibrated.png")
    fig.savefig(out, bbox_inches="tight", facecolor="#f5f5f0"); plt.close(fig)

    print(f"{shot}: PnP reproj {reproj.mean():.2f} px (per-marker "
          f"{np.round(reproj,2)})   fx={K[0,0]:.0f}")
    print(f"  calibration: z_lens = {a:.3f}*z_photo {b:+.2f}   marker RMS "
          f"{np.sqrt(np.mean(resid**2)):.2f} mm")
    for i in ids:
        print(f"  marker {i}: photo {sample(z,*corr[i][0]):+.2f} -> "
              f"{a*sample(z,*corr[i][0])+b:5.1f} mm   CT {cam[ids.index(i),2]:5.1f} mm")
    print(f"slide -> {out}")


if __name__ == "__main__":
    for _shot in CT_CORR:
        main(_shot)
