"""
Predict the working-distance envelope of near-coaxial photometric stereo.

Model: the tilt-encoding lateral component of each LED direction is sin(alpha),
alpha = atan(a/WD), a = LED ring radius. The photometric normal solve amplifies
image noise into slope error by ~1/sin(alpha) = sqrt(a^2+WD^2)/a. So the surface
(and registration) error grows ~linearly with WD, and the solve degenerates once
alpha falls below a noise-set critical angle. We measure WD per level (mean camera-
frame marker depth via PnP), compare to near/far-field FRE, and locate alpha_crit.

Run: python analyze_envelope.py
"""
import os, sys, math, json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "reconstruction"))
import bone_depth_batch as bd
bd.UNDISTORT_INPUTS = False
import calibrate_photometric_ct as C

cal = json.load(open(bd.project_path("calibration", "calibration.txt")))
K = np.array(cal["camera_matrix"]); DIST = np.array(cal["dist_coeffs"])
A_MM = 6.05

# FRE (mm) measured earlier; shot_008 near-field degenerate -> NaN
NEAR = {"shot_004": 1.72, "shot_005": 1.32, "shot_006": 1.74, "shot_007": 9.72, "shot_008": np.nan}
FAR  = {"shot_004": 1.54, "shot_005": 1.22, "shot_006": 6.57, "shot_007": 11.14, "shot_008": 5.66}
N_MARK = {"shot_004": 4, "shot_005": 4, "shot_006": 4, "shot_007": 3, "shot_008": 4}


def working_distance(corr):
    ids = sorted(corr)
    obj = np.array([corr[i][1] for i in ids], float)
    img = np.array([corr[i][0] for i in ids])
    _, rv, tv = cv2.solvePnP(obj, img, K, DIST, flags=cv2.SOLVEPNP_SQPNP)
    R, _ = cv2.Rodrigues(rv)
    return float(np.mean((R @ obj.T + tv.reshape(3, 1)).T[:, 2]))


rows = []
print(f"{'level':8}{'WD(mm)':>8}{'alpha(deg)':>11}{'1/sin':>8}{'near FRE':>10}{'far FRE':>9}{'markers':>9}")
for shot, corr in C.CT_CORR.items():
    wd = working_distance(corr)
    alpha = math.degrees(math.atan2(A_MM, wd))
    amp = 1.0 / math.sin(math.radians(alpha))
    rows.append((shot, wd, alpha, amp, NEAR[shot], FAR[shot], N_MARK[shot]))
    fre = "deg." if np.isnan(NEAR[shot]) else f"{NEAR[shot]:.2f}"
    print(f"{shot[-3:]:8}{wd:8.1f}{alpha:11.1f}{amp:8.1f}{fre:>10}{FAR[shot]:9.2f}{N_MARK[shot]:9d}")

wd = np.array([r[1] for r in rows]); alpha = np.array([r[2] for r in rows])
near = np.array([r[4] for r in rows]); nm = np.array([r[6] for r in rows])

# fit FRE ~ k*WD/a + c on the 4-marker, non-degenerate levels (isolate the envelope)
ok = (~np.isnan(near)) & (nm == 4)
x = wd[ok] / A_MM
k, c = np.polyfit(x, near[ok], 1)
# critical angle: midpoint between last success (shot_006) and failure (shot_008)
a_success = alpha[[r[0] for r in rows].index("shot_006")]
a_fail = alpha[[r[0] for r in rows].index("shot_008")]
alpha_crit = 0.5 * (a_success + a_fail)
wd_crit = A_MM / math.tan(math.radians(alpha_crit))
print(f"\nfit: near-field FRE ~= {k:.3f}*(WD/a) + {c:.2f} mm  (4-marker levels)")
print(f"critical angle alpha_crit ~= {alpha_crit:.1f} deg  ->  WD_max ~= {wd_crit:.0f} mm")

# ---- figure: FRE vs working distance + model + envelope ----
fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=130)
fig.patch.set_facecolor("white")
xs = np.linspace(15, 120, 100)
ax.plot(xs, k * xs / A_MM + c, "-", color="#0f6e8c", lw=1.8,
        label=f"noise-limited model  FRE $\\propto$ WD/a")
ax.axvspan(wd_crit, 125, color="#b23b3b", alpha=0.08)
ax.axvline(wd_crit, color="#b23b3b", ls="--", lw=1.4,
           label=f"predicted envelope  WD$_{{max}}$$\\approx${wd_crit:.0f} mm")
for shot, w, al, amp, nf, ff, m in rows:
    good = (m == 4) and not np.isnan(nf)
    if not np.isnan(nf):
        ax.scatter(w, nf, s=90, zorder=5,
                   c=("#1f7a4d" if good else "#c08a2e"),
                   marker=("o" if m == 4 else "^"),
                   edgecolors="k", linewidths=0.8)
        ax.annotate(f"{shot[-3:]}" + ("" if m == 4 else " (3 mk)"),
                    (w, nf), (w + 2, nf + 0.4), fontsize=8)
    else:
        ax.scatter(w, 12, s=120, marker="x", c="#b23b3b", zorder=5)
        ax.annotate(f"{shot[-3:]} degenerate", (w, 12), (w - 30, 12.3),
                    fontsize=8, color="#b23b3b")
ax.set_xlabel("working distance WD (mm)")
ax.set_ylabel("near-field 3-D FRE (mm)")
ax.set_title("Working-distance envelope of near-coaxial photometric stereo\n"
             f"(LED ring a={A_MM} mm; degenerates below $\\alpha\\approx${alpha_crit:.0f}$\\degree$)",
             fontsize=10, fontweight="bold")
ax.set_ylim(0, 13); ax.set_xlim(10, 125)
ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.25)
sec = ax.secondary_xaxis("top", functions=(
    lambda w: np.degrees(np.arctan2(A_MM, np.maximum(w, 1e-6))),
    lambda a: A_MM / np.tan(np.radians(np.maximum(a, 1e-6)))))
sec.set_xlabel("LED half-angle $\\alpha$ (deg)", fontsize=9)
out = bd.project_path("paper", "figures", "fig5_envelope.png")
fig.savefig(out, bbox_inches="tight", facecolor="white"); print(f"\nfigure -> {out}")
