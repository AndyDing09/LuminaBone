"""Export the CT-free evaluation data for the Octave figure scripts.

Re-runs the CT-free solve (mu* = 8, multi-start, rigid-only evaluation --
identical to paper_ctfree_eval2.py) and writes, per shot,

    octave_figs/data/<shot>_plotdata.mat     (Octave: load('<shot>_plotdata.mat'))
        ct      N x 3   reconstructed surface points in CT (RAS) mm
        col     N x 3   image RGB of those points, 0..1 (LED1 reference frame)
        obj     M x 3   CT fiducials (ground truth)
        mk      M x 3   reconstructed fiducials after the rigid Kabsch alignment
        uv      M x 2   fiducial glint pixels in the 640x480 frame (x, y)
        fre     1 x 1   FRE RMS (mm), fre_each 1 x M
        b, s, mu, residual, keep_px, ids
        mesh_v  V x 3, mesh_c V x 3, mesh_f F x 3 (1-based)   grid-decimated
                mesh of the same surface for patch()-style rendering
        ref     480 x 640 x 3 uint8   LED1 frame (display-stretched)

and also re-creates depth_outputs/ct_registered/<shot>_plotdata_mu8.npz so
paper_figs.py works again.  Run:  python export_octave_plotdata.py [shot ...]
"""
import os, sys, time
import numpy as np
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nearfield_ctfree as NF
from nearfield_ct import grid_faces

OUT = NF.OUT
ODIR = os.path.join(os.path.dirname(HERE), "octave_figs", "data")
os.makedirs(ODIR, exist_ok=True)
MU = 8.0
MESH_STEP = 3            # decimate the pixel grid 3x for the exported mesh


def export(shot):
    t = time.time()
    D, Dsm, ref, keep, scan, hist = NF.reconstruct(shot, mu_fixed=MU,
                                                   multi_start=True, verbose=False)
    b, s, mu, e = hist[-1]
    if keep.sum() < 2000:
        print(f"{shot}: UNSTABLE ({keep.sum()} px) -- skipped", flush=True)
        return
    ev = NF.evaluate(shot, D, keep)
    Rk, tk = ev["Rk"], ev["tk"]

    # --- scatter points (same subsampling as paper_ctfree_eval2.py) ---
    vv, uu = np.where(keep)
    step = max(1, len(vv) // 9000)
    cam = NF.R.backproject(uu[::step].astype(float), vv[::step].astype(float),
                           Dsm[vv, uu][::step])
    ct = (Rk @ cam.T).T + tk
    col = ref[vv, uu][::step]
    mk_ct = (Rk @ ev["mk"].T).T + tk
    np.savez(os.path.join(OUT, f"{shot}_plotdata_mu8.npz"),
             ct=ct, col=col, obj=ev["obj"], mk=mk_ct, fre=ev["fre_rms"])

    # --- decimated mesh ---
    k2 = keep[::MESH_STEP, ::MESH_STEP]
    D2 = Dsm[::MESH_STEP, ::MESH_STEP]
    idx = -np.ones(k2.shape, int)
    v2, u2 = np.where(k2)
    idx[v2, u2] = np.arange(len(v2))
    camv = NF.R.backproject((u2 * MESH_STEP).astype(float),
                            (v2 * MESH_STEP).astype(float), D2[v2, u2])
    mesh_v = (Rk @ camv.T).T + tk
    mesh_c = ref[v2 * MESH_STEP, u2 * MESH_STEP]
    mesh_f = grid_faces(idx, D2, tol=3.0 * MESH_STEP) + 1      # 1-based

    corr = NF.C.CT_CORR[shot]
    uv = np.array([corr[i][0] for i in ev["ids"]], float)
    disp = np.clip((np.clip(ref, 0, 1) ** (1 / 1.9)) * 1.25, 0, 1)   # display only
    sio.savemat(os.path.join(ODIR, f"{shot}_plotdata.mat"), dict(
        ct=ct, col=np.clip(col, 0, 1), obj=ev["obj"], mk=mk_ct, uv=uv,
        fre=float(ev["fre_rms"]), fre_each=ev["fre"], ids=np.array(ev["ids"]),
        b=float(b), s=float(s), mu=float(mu), residual=float(e),
        keep_px=int(keep.sum()), dist_scale=float(np.median(ev["ratio"])),
        sim_scale=float(ev["scale"]),
        mesh_v=mesh_v, mesh_c=np.clip(mesh_c, 0, 1), mesh_f=mesh_f,
        ref=(disp * 255).astype(np.uint8)), do_compression=True)
    print(f"{shot}: b={b:.2f} s={s:.3f} FRE={ev['fre_rms']:.2f} mm "
          f"(each {np.array2string(ev['fre'], precision=2)}), "
          f"{len(ct)} pts, mesh {len(mesh_v)}v/{len(mesh_f)}f, "
          f"{time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    shots = sys.argv[1:] or ["shot_004", "shot_005", "shot_006", "shot_007"]
    for sh in shots:
        export(sh)
    print("done", flush=True)
