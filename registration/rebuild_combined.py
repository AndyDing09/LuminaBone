#!/usr/bin/env python3
"""
rebuild_combined.py -- write a CLEAN, non-truncated combined model and verify it.

The earlier combined_CT_plus_17.ply was cut off mid-write (time limit), so its
header promised more vertices than were on disk -> MeshLab "Unexpected EOF".
This rebuilds it fully, flushes+fsyncs, then re-opens and checks the byte count
against the header so truncation is impossible to miss.

Includes only the 12 patches that registered plausibly (scale in range); the 5
degenerate collapses are excluded so the model is clean.
"""
import os, glob, importlib.util, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(n, f):
    s = importlib.util.spec_from_file_location(n, os.path.join(HERE, f))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
lr = _load("lr", "landmark_register.py")

OUT = os.path.join(HERE, "reg_out")
OK = ["loc01","loc02","loc03","loc04","loc05","loc06","loc08",
      "loc12","loc13","loc15","loc16","loc17"]          # the 12 plausible ones

# distinct colors
import colorsys
cols = [(np.array(colorsys.hsv_to_rgb(i/len(OK), 0.75, 1.0))*255).astype(np.uint8)
        for i in range(len(OK))]

xyz_all, rgb_all = [], []

# CT (grey) from cache, downsampled
c = np.load(os.path.join(HERE, "ct_ras_cache.npz"))
CT = c["Vfull"]
rng = np.random.default_rng(0)
CTd = CT[rng.choice(len(CT), min(150000, len(CT)), replace=False)]
xyz_all.append(CTd); rgb_all.append(np.tile([170,170,170], (len(CTd),1)))

# each OK patch, downsampled to ~25k
for name, col in zip(OK, cols):
    P,_,_ = lr.read_ply(os.path.join(OUT, f"{name}_reg.ply"))
    if len(P) > 25000:
        P = P[rng.choice(len(P), 25000, replace=False)]
    xyz_all.append(P); rgb_all.append(np.tile(col, (len(P),1)))

XYZ = np.vstack(xyz_all).astype(np.float32)
RGB = np.vstack(rgb_all).astype(np.uint8)
path = os.path.join(OUT, "combined_CT_plus_17.ply")

# write with explicit flush + fsync
hdr = ("ply\nformat binary_little_endian 1.0\n"
       f"element vertex {len(XYZ)}\n"
       "property float x\nproperty float y\nproperty float z\n"
       "property uchar red\nproperty uchar green\nproperty uchar blue\n"
       "end_header\n").encode("ascii")
rec = np.empty(len(XYZ), dtype=[("x","<f4"),("y","<f4"),("z","<f4"),
                                ("r","u1"),("g","u1"),("b","u1")])
rec["x"],rec["y"],rec["z"] = XYZ[:,0],XYZ[:,1],XYZ[:,2]
rec["r"],rec["g"],rec["b"] = RGB[:,0],RGB[:,1],RGB[:,2]
with open(path, "wb") as f:
    f.write(hdr); rec.tofile(f); f.flush(); os.fsync(f.fileno())

# verify: header count vs actual bytes
expected = len(hdr) + len(XYZ)*15
actual = os.path.getsize(path)
print(f"vertices written : {len(XYZ):,}")
print(f"expected bytes   : {expected:,}")
print(f"actual bytes     : {actual:,}")
print("INTEGRITY:", "OK - not truncated" if actual == expected else "!! MISMATCH")
print("wrote", path)
