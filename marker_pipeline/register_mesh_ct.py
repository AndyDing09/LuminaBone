"""
Same CT registration as register_photometric_ct.py, but writes a TRIANGULATED
MESH (faces from the pixel grid) instead of a bare point cloud -- so 3D Slicer
actually renders it (Slicer draws surfaces; a point-only PLY shows nothing).

Neighboring kept pixels form two triangles per quad, skipping quads that straddle
a depth discontinuity (>3 mm) so we don't web across holes/edges.

Output: depth_outputs/ct_registered/<shot>_mesh_photo_ct.ply  (CT mm frame)
Run: python register_mesh_ct.py
"""

import os
import sys
import numpy as np
import matplotlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "reconstruction"))
import bone_depth_batch as bd
bd.UNDISTORT_INPUTS = False
import calibrate_photometric_ct as C
import register_inverse_square_ct as R

OUT = bd.project_path("depth_outputs", "ct_registered")
os.makedirs(OUT, exist_ok=True)
STEP_TOL = 3.0        # mm: don't connect pixels across a bigger depth jump


def write_ply_mesh(path, xyz, rgb, faces):
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(xyz, rgb):
            f.write(f"{x:.3f} {y:.3f} {z:.3f} {r} {g} {b}\n")
        for a, b_, c in faces:
            f.write(f"3 {a} {b_} {c}\n")


def grid_faces(idx, zmap):
    """two triangles per fully-valid quad, skipping depth discontinuities."""
    a = idx[:-1, :-1]; b = idx[:-1, 1:]; c = idx[1:, :-1]; d = idx[1:, 1:]
    za = zmap[:-1, :-1]; zb = zmap[:-1, 1:]; zc = zmap[1:, :-1]; zd = zmap[1:, 1:]
    ok = (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
    span = np.maximum.reduce([za, zb, zc, zd]) - np.minimum.reduce([za, zb, zc, zd])
    ok &= span <= STEP_TOL
    a, b, c, d = a[ok], b[ok], c[ok], d[ok]
    return np.vstack([np.column_stack([a, b, d]),
                      np.column_stack([a, d, c])])


def main():
    print("meshed photometric surface, registered to CT (mm):")
    for shot, corr in C.CT_CORR.items():
        ids = sorted(corr)
        z, ref, mask = C.photometric_depth(shot)
        Rp, t = R.pose(corr)
        obj = np.array([corr[i][1] for i in ids], float)
        ct_d = (Rp @ obj.T + t.reshape(3, 1)).T[:, 2]
        rr = np.array([R._disk(z, *corr[i][0]) for i in ids])
        a, b = np.linalg.lstsq(np.column_stack([rr, np.ones(len(rr))]),
                               ct_d, rcond=None)[0]
        zmap = a * z + b

        zlo, zhi = ct_d.min() - 15, ct_d.max() + 15
        keep = mask & (zmap >= zlo) & (zmap <= zhi)
        idx = -np.ones(keep.shape, int)
        vv, uu = np.where(keep)
        idx[vv, uu] = np.arange(len(vv))
        cam = R.backproject(uu.astype(float), vv.astype(float), zmap[vv, uu])
        ct = (Rp.T @ (cam - t).T).T
        col = ref[vv, uu]
        faces = grid_faces(idx, zmap)
        ply = os.path.join(OUT, f"{shot}_mesh_photo_ct.ply")
        write_ply_mesh(ply, ct, col, faces)
        print(f"  {shot}: {len(ct):6d} verts  {len(faces):6d} faces "
              f"-> {os.path.basename(ply)}")
    print(f"\nmeshes -> {OUT}")
    print("In Slicer: Add Data -> *_mesh_photo_ct.ply (loads as a Model, renders as a surface).")


if __name__ == "__main__":
    main()
