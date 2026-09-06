"""Second pass: (a) redo mu=8 runs saving plot npz + brighter overlay figure,
(b) mu=2 (the vote's weak preference) full comparison."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nearfield_ctfree as NF
OUT = NF.OUT
SHOTS = ["shot_004", "shot_005", "shot_006", "shot_007"]

def eval_mu(mu, save_npz=False):
    res = {}
    for shot in SHOTS:
        D, Dsm, ref, keep, scan, hist = NF.reconstruct(shot, mu_fixed=mu,
                                                       multi_start=True, verbose=False)
        b, s, m_, e = hist[-1]
        if keep.sum() < 2000:
            res[shot] = dict(unstable=True); continue
        ev = NF.evaluate(shot, D, keep)
        res[shot] = dict(b=round(b,2), s=round(s,3), residual=round(e,4),
                         dist_scale=round(float(np.median(ev["ratio"])),3),
                         sim_scale=round(ev["scale"],3), fre_rms=round(ev["fre_rms"],2))
        if save_npz:
            vv, uu = np.where(keep)
            step = max(1, len(vv)//9000)
            cam = NF.R.backproject(uu[::step].astype(float), vv[::step].astype(float),
                                   Dsm[vv,uu][::step])
            ct = (ev["Rk"] @ cam.T).T + ev["tk"]
            mk_ct = (ev["Rk"] @ ev["mk"].T).T + ev["tk"]
            np.savez(os.path.join(OUT, f"{shot}_plotdata_mu8.npz"),
                     ct=ct, col=ref[vv,uu][::step], obj=ev["obj"], mk=mk_ct,
                     fre=ev["fre_rms"])
        print(shot, "mu", mu, res[shot], flush=True)
    return res

r8 = eval_mu(8.0, save_npz=True)
r2 = eval_mu(2.0)
json.dump({"mu8": r8, "mu2": r2},
          open(os.path.join(OUT, "ctfree_mu_compare.json"), "w"), indent=1)
print("saved ctfree_mu_compare.json")
