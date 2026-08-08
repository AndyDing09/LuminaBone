"""
Near-field photometric solve (per-pixel light directions + 1/r^2), CT-registered.

Far-field treats every LED as a single constant direction; on a rig whose lights
sit ~6 mm off-axis at ~20 mm that is wrong, and the mis-modelled brightness
gradient integrates into a BOWL. Here each pixel gets its own light vector and
1/r^2 attenuation from a point source at the LED's real position, iterated with
the surface depth (bootstrapped from the CT working distance, re-anchored to the
CT markers each pass). Same +x right / +y up / +z-toward-camera frame as
bone_depth_batch.light_vector, so the normals feed normals_to_depth unchanged.

Also trims the mask by photometric residual (drops specular marker holes / edge
drift -> the streaky tail) and keeps the largest connected component.

Run: python nearfield_ct.py            # shot_004 only, verbose
     python nearfield_ct.py all        # every CT shot
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
from scipy.ndimage import (binary_erosion, binary_opening, label,
                           median_filter, gaussian_filter)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "reconstruction"))
import bone_depth_batch as bd
bd.UNDISTORT_INPUTS = False
import calibrate_photometric_ct as C
import register_inverse_square_ct as R   # pose / backproject / write helpers

cal = json.load(open(bd.project_path("calibration", "calibration.txt")))
K = np.array(cal["camera_matrix"]); DIST = np.array(cal["dist_coeffs"])
FX, FY, CX, CY = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
OUT = bd.project_path("depth_outputs", "ct_registered")

A_MM = 6.05                              # LED radius off the optical axis
AZ = {1: 90.0, 2: 270.0, 3: 180.0, 4: 0.0}
# LED positions in the light frame (+x right, +y up, +z toward camera), lens plane
P = {s: np.array([A_MM * math.cos(math.radians(a)),
                  A_MM * math.sin(math.radians(a)), 0.0]) for s, a in AZ.items()}


def load_lums(shot):
    d = bd.project_path("Data_collection", "calib_charuco", shot)
    dark = bd.load_rgb(os.path.join(d, "dark.png"), bd.WORK_LONG_EDGE)
    lums, ref = [], None
    for i in (1, 2, 3, 4):
        rgb = bd.load_rgb(os.path.join(d, f"led{i}.png"), bd.WORK_LONG_EDGE)
        if i == 1:
            ref = rgb
        lums.append(bd.solve_luminance(np.clip(rgb - dark, 0, 1)))
    return bd.balance_exposure(lums), ref


def _geometry(D):
    """per-pixel unit light vectors (H,W,4,3) and 1/r^2 attenuation (H,W,4)."""
    H, W = D.shape
    U, V = np.meshgrid(np.arange(W), np.arange(H))
    Xw = np.stack([(U - CX) / FX * D, -(V - CY) / FY * D, -D], axis=-1)  # +y up, +z->cam
    Lhat = np.empty((H, W, 4, 3)); att = np.empty((H, W, 4))
    for k, s in enumerate((1, 2, 3, 4)):
        w = P[s][None, None, :] - Xw                      # surface -> LED
        d2 = np.sum(w * w, axis=-1)
        Lhat[..., k, :] = w / np.sqrt(d2)[..., None]
        att[..., k] = 1.0 / d2
    return Lhat, att


def nearfield(shot, corr, iters=4):
    ids = sorted(corr)
    lums, ref = load_lums(shot)
    I = np.stack(lums, axis=0)                            # (4,H,W)
    H, W = I.shape[1:]
    M = np.moveaxis(I, 0, -1)                             # (H,W,4)
    base_mask = bd.bone_mask(I.mean(0))

    Rp, t = R.pose(corr)
    obj = np.array([corr[i][1] for i in ids], float)
    ct_d = (Rp @ obj.T + t.reshape(3, 1)).T[:, 2]
    D = np.full((H, W), float(ct_d.mean()))              # bootstrap: flat plane at WD

    g = None
    for _ in range(iters):
        Lhat, att = _geometry(D)
        m = M / att                                      # I_s / att = rho (n·l)
        A = np.einsum("hwks,hwkt->hwst", Lhat, Lhat)
        rhs = np.einsum("hwks,hwk->hws", Lhat, m)
        g = np.linalg.solve(A + 1e-6 * np.eye(3), rhs[..., None])[..., 0]  # rho*n (H,W,3)
        alb = np.linalg.norm(g, axis=-1)
        n = g / np.where(alb > 1e-9, alb, 1.0)[..., None]
        n[n[..., 2] < 0] *= -1.0
        z_rel = bd.normals_to_depth(n)
        rr = np.array([R._disk(z_rel, *corr[i][0]) for i in ids])
        a, b = np.linalg.lstsq(np.column_stack([rr, np.ones(len(rr))]),
                               ct_d, rcond=None)[0]
        D = a * z_rel + b                                # mm depth-from-lens

    # photometric residual -> confidence (drops specular markers / bad edges)
    Lhat, att = _geometry(D)
    Ipred = np.einsum("hws,hwks->hwk", g, Lhat) * att
    resid = np.abs(M - Ipred).sum(-1) / (M.sum(-1) + 1e-6)
    conf = resid < 0.22                                  # moderate: keep real bone
    band = (D > ct_d.min() - 15) & (D < ct_d.max() + 15)
    spike = np.abs(D - median_filter(D, size=5)) < 8.0   # only gross depth spikes
    keep = base_mask & conf & band & spike
    # crop the frame border (FFT integration leaves wrap-around ramps -> the fan)
    H, W = keep.shape
    bm = int(0.09 * min(H, W))
    border = np.zeros_like(keep); border[bm:-bm, bm:-bm] = True
    keep &= border
    # (specular marker glints are already dropped by the residual-confidence gate
    # above; don't punch big holes at the markers -> they must stay visible on the
    # mesh so the registration can be checked.)
    keep = binary_opening(keep, iterations=2)            # kills speckle + thin fans
    lab, num = label(keep)                               # keep all sizeable blobs
    if num:                                              # (marker holes must not split)
        sizes = np.array([(lab == i).sum() for i in range(1, num + 1)])
        big = 1 + np.where(sizes >= max(300, 0.15 * sizes.max()))[0]
        keep = np.isin(lab, big)
    keep = binary_erosion(keep, iterations=1)

    # mask-aware smoothing of the depth used for the mesh (markers stay on raw D)
    w = keep.astype(float)
    Dsm = gaussian_filter(D * w, 2.0) / np.maximum(gaussian_filter(w, 2.0), 1e-6)
    Dsm = np.where(keep, Dsm, D)

    relief = (np.percentile(D[keep], 90) - np.percentile(D[keep], 10)) if keep.any() else 0
    return D, Dsm, ref, keep, Rp, t, obj, ct_d, relief


def grid_faces(idx, zmap, tol=3.0):
    a = idx[:-1, :-1]; b = idx[:-1, 1:]; c = idx[1:, :-1]; d = idx[1:, 1:]
    za = zmap[:-1, :-1]; zb = zmap[:-1, 1:]; zc = zmap[1:, :-1]; zd = zmap[1:, 1:]
    ok = (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
    ok &= (np.maximum.reduce([za, zb, zc, zd]) -
           np.minimum.reduce([za, zb, zc, zd])) <= tol
    a, b, c, d = a[ok], b[ok], c[ok], d[ok]
    return np.vstack([np.column_stack([a, b, d]), np.column_stack([a, d, c])])


def write_ply_mesh(path, xyz, rgb, faces):
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        for (x, y, z), (r, gg, bb) in zip(xyz, rgb):
            f.write(f"{x:.3f} {y:.3f} {z:.3f} {r} {gg} {bb}\n")
        for a, b_, c in faces:
            f.write(f"3 {a} {b_} {c}\n")


def run(shot):
    corr = C.CT_CORR[shot]
    ids = sorted(corr)
    D, Dsm, ref, keep, Rp, t, obj, ct_d, relief = nearfield(shot, corr)
    if keep.sum() < 2000:                       # near-field degenerate (far/coaxial shot)
        print(f"  {shot}: near-field UNSTABLE ({keep.sum()} px survive) — "
              f"use far-field mesh instead. skipping.")
        return None, None

    idx = -np.ones(keep.shape, int)
    vv, uu = np.where(keep)
    idx[vv, uu] = np.arange(len(vv))
    cam = R.backproject(uu.astype(float), vv.astype(float), Dsm[vv, uu])
    ct = (Rp.T @ (cam - t).T).T
    col = ref[vv, uu]
    faces = grid_faces(idx, Dsm)
    ply = os.path.join(OUT, f"{shot}_mesh_nearfield_ct.ply")
    write_ply_mesh(ply, ct, col, faces)

    mk_uv = np.array([corr[i][0] for i in ids], float)
    mk_z = np.array([R._disk(D, *corr[i][0]) for i in ids])
    mk_ct = (Rp.T @ (R.backproject(mk_uv[:, 0], mk_uv[:, 1], mk_z) - t).T).T
    err = np.linalg.norm(mk_ct - obj, axis=1)
    rms = float(np.sqrt(np.mean(err ** 2)))
    print(f"  {shot}: near-field 3-D marker RMS {rms:5.2f} mm  relief {relief:4.1f} mm "
          f"(CT range {ct_d.max()-ct_d.min():.1f})  {len(ct)}v/{len(faces)}f "
          f"-> {os.path.basename(ply)}")

    fig = plt.figure(figsize=(7, 6), dpi=120); fig.patch.set_facecolor("#f5f5f0")
    ax = fig.add_subplot(111, projection="3d")
    s = max(1, len(ct) // 6000)
    ax.scatter(ct[::s, 0], ct[::s, 1], ct[::s, 2], c=col[::s], s=2, alpha=0.5)
    ax.scatter(obj[:, 0], obj[:, 1], obj[:, 2], c="k", s=90, marker="X",
               label="CT markers", depthshade=False)
    ax.scatter(mk_ct[:, 0], mk_ct[:, 1], mk_ct[:, 2], facecolors="none",
               edgecolors="r", s=140, linewidths=2, label="reconstructed")
    pad = 20
    ax.set_xlim(obj[:, 0].min() - pad, obj[:, 0].max() + pad)
    ax.set_ylim(obj[:, 1].min() - pad, obj[:, 1].max() + pad)
    ax.set_zlim(obj[:, 2].min() - pad, obj[:, 2].max() + pad)
    ax.set_xlabel("CT x"); ax.set_ylabel("CT y"); ax.set_zlabel("CT z")
    ax.set_title(f"{shot} — NEAR-FIELD surface registered to CT\n"
                 f"3-D marker RMS {rms:.1f} mm, relief {relief:.1f} mm",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    fig.savefig(os.path.join(OUT, f"{shot}_ct_overlay_nearfield.png"),
                bbox_inches="tight", facecolor="#f5f5f0"); plt.close(fig)
    return rms, relief


if __name__ == "__main__":
    shots = list(C.CT_CORR) if (len(sys.argv) > 1 and sys.argv[1] == "all") else ["shot_004"]
    print("near-field photometric solve, CT-registered:")
    for sh in shots:
        run(sh)
    print(f"\nmeshes + overlays -> {OUT}")
