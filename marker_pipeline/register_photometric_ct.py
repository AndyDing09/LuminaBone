"""
Register the PHOTOMETRIC-stereo surface into the CT frame (same pipeline as
register_inverse_square_ct.py, but the depth comes from the albedo-invariant
Woodham normal solve instead of raw 1/sqrt(I)).

Per shot: photometric depth (relative) -> per-shot affine anchor to CT marker
depths -> back-project to camera 3-D -> camera->CT via marker PnP -> PLY + overlay
+ 3-D marker RMS.  Prints a side-by-side table vs the inverse-square result.

Run: python register_photometric_ct.py
"""

import os
import sys
import numpy as np
import cv2
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "reconstruction"))
import bone_depth_batch as bd
bd.UNDISTORT_INPUTS = False
import calibrate_photometric_ct as C
import register_inverse_square_ct as R          # reuse pose/backproject/write_ply/_disk

OUT = bd.project_path("depth_outputs", "ct_registered")
os.makedirs(OUT, exist_ok=True)


def main():
    print("register PHOTOMETRIC surface into CT frame (mm):")
    photo_rms = {}
    for shot, corr in C.CT_CORR.items():
        ids = sorted(corr)
        z, ref, mask = C.photometric_depth(shot)      # relative depth, albedo-invariant
        Rp, t = R.pose(corr)
        obj = np.array([corr[i][1] for i in ids], float)
        ct_d = (Rp @ obj.T + t.reshape(3, 1)).T[:, 2]
        rr = np.array([R._disk(z, *corr[i][0]) for i in ids])
        a, b = np.linalg.lstsq(np.column_stack([rr, np.ones(len(rr))]),
                               ct_d, rcond=None)[0]
        zc_map = a * z + b

        # clip to CT depth band (drop integration drift far from the markers)
        zlo, zhi = ct_d.min() - 15, ct_d.max() + 15
        keep = mask & (zc_map >= zlo) & (zc_map <= zhi)
        vv, uu = np.where(keep)
        cam = R.backproject(uu.astype(float), vv.astype(float), zc_map[vv, uu])
        ct = (Rp.T @ (cam - t).T).T
        col = ref[vv, uu]
        ply = os.path.join(OUT, f"{shot}_surface_photo_ct.ply")
        R.write_ply(ply, ct, col)

        mk_uv = np.array([corr[i][0] for i in ids], float)
        mk_z = np.array([R._disk(zc_map, *corr[i][0]) for i in ids])
        mk_cam = R.backproject(mk_uv[:, 0], mk_uv[:, 1], mk_z)
        mk_ct = (Rp.T @ (mk_cam - t).T).T
        truth = obj
        err = np.linalg.norm(mk_ct - truth, axis=1)
        rms = float(np.sqrt(np.mean(err ** 2)))
        photo_rms[shot] = rms
        print(f"  {shot}: {len(ct):6d} pts   3-D marker RMS {rms:5.2f} mm"
              f"   (per-marker {np.round(err,1)})  -> {os.path.basename(ply)}")

        fig = plt.figure(figsize=(7, 6), dpi=120)
        fig.patch.set_facecolor("#f5f5f0")
        ax = fig.add_subplot(111, projection="3d")
        s = max(1, len(ct) // 6000)
        ax.scatter(ct[::s, 0], ct[::s, 1], ct[::s, 2], c=col[::s], s=2, alpha=0.5)
        ax.scatter(truth[:, 0], truth[:, 1], truth[:, 2], c="k", s=90, marker="X",
                   label="CT markers (truth)", depthshade=False)
        ax.scatter(mk_ct[:, 0], mk_ct[:, 1], mk_ct[:, 2], facecolors="none",
                   edgecolors="r", s=140, linewidths=2, label="reconstructed markers")
        pad = 20
        ax.set_xlim(truth[:, 0].min() - pad, truth[:, 0].max() + pad)
        ax.set_ylim(truth[:, 1].min() - pad, truth[:, 1].max() + pad)
        ax.set_zlim(truth[:, 2].min() - pad, truth[:, 2].max() + pad)
        ax.set_xlabel("CT x (mm)"); ax.set_ylabel("CT y (mm)"); ax.set_zlabel("CT z (mm)")
        ax.set_title(f"{shot} — PHOTOMETRIC surface registered to CT\n"
                     f"3-D marker RMS {rms:.1f} mm", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        fig.savefig(os.path.join(OUT, f"{shot}_ct_overlay_photo.png"),
                    bbox_inches="tight", facecolor="#f5f5f0"); plt.close(fig)

    # side-by-side vs inverse-square
    print("\n3-D marker RMS (mm):  inverse-square  vs  photometric")
    inv = {"shot_004": 6.39, "shot_005": 9.44, "shot_006": 9.55,
           "shot_007": 7.91, "shot_008": 6.40}
    print(f"{'shot':10} {'inverse-sq':>12} {'photometric':>12}")
    for s in C.CT_CORR:
        print(f"{s:10} {inv[s]:12.2f} {photo_rms[s]:12.2f}")
    print(f"{'mean':10} {np.mean(list(inv.values())):12.2f} "
          f"{np.mean(list(photo_rms.values())):12.2f}")
    print(f"\nPLYs + overlays -> {OUT}")


if __name__ == "__main__":
    main()
