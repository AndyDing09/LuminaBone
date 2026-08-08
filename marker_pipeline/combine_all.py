"""
Merge all per-shot registered meshes into ONE combined mesh (already in CT frame,
each patch registered via its own shot's markers), plus one combined marker set so
the whole thing can be fiducial-registered at once IF the specimen was rigid across
shots (levels don't move). shots 4-7 use the near-field mesh; shot 8 (near-field
degenerate at ~110 mm) uses the far-field mesh.

Outputs (depth_outputs/ct_registered/):
  combined_photometric_ct.ply        the full multi-patch surface
  combined_CT_truth.mrk.json         all CT markers (green, fixed)
  combined_reconstructed_RAS/LPS.mrk.json   all mesh markers (moving)

Run: python combine_all.py
"""

import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "reconstruction"))
import bone_depth_batch as bd

OUT = bd.project_path("depth_outputs", "ct_registered")
MESH = {  # shot -> which registered mesh file to use
    "shot_004": "shot_004_mesh_nearfield_ct.ply",
    "shot_005": "shot_005_mesh_nearfield_ct.ply",
    "shot_006": "shot_006_mesh_nearfield_ct.ply",
    "shot_007": "shot_007_mesh_nearfield_ct.ply",
    "shot_008": "shot_008_mesh_photo_ct.ply",      # near-field failed -> far-field
}


def read_ply(p):
    L = open(p).read().splitlines()
    nv = [int(l.split()[2]) for l in L if l.startswith("element vertex")][0]
    nf = [int(l.split()[2]) for l in L if l.startswith("element face")][0]
    h = L.index("end_header") + 1
    verts = [L[h + i].split() for i in range(nv)]
    faces = [L[h + nv + i].split() for i in range(nf)]
    V = np.array([[float(x) for x in v] for v in verts])          # x y z r g b
    F = np.array([[int(x) for x in f[1:4]] for f in faces])       # tri indices
    return V, F


def write_ply(path, V, F):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(V)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(F)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        for x, y, z, r, g, b in V:
            f.write(f"{x:.3f} {y:.3f} {z:.3f} {int(r)} {int(g)} {int(b)}\n")
        for a, b_, c in F:
            f.write(f"3 {a} {b_} {c}\n")


def merge_markups(paths, shots, color):
    cps = []
    for pth, sh in zip(paths, shots):
        for cp in json.load(open(pth))["markups"][0]["controlPoints"]:
            cp = dict(cp)
            cp["label"] = f"{sh.split('_')[1]}_{cp['label']}"      # 004_m1 ...
            cp["id"] = str(len(cps) + 1)
            cps.append(cp)
    cs = json.load(open(paths[0]))["markups"][0]["coordinateSystem"]
    return {"@schema": json.load(open(paths[0]))["@schema"],
            "markups": [{"type": "Fiducial", "coordinateSystem": cs,
                         "controlPoints": cps,
                         "display": {"color": color, "selectedColor": color,
                                     "glyphScale": 3.0, "textScale": 3.0}}]}


def main():
    shots = list(MESH)
    allV = np.empty((0, 6)); allF = np.empty((0, 3), int); off = 0
    for sh in shots:
        V, F = read_ply(os.path.join(OUT, MESH[sh]))
        allV = np.vstack([allV, V])
        allF = np.vstack([allF, F + off]) if len(F) else allF
        off += len(V)
        print(f"  {sh}: +{len(V)} verts / {len(F)} faces ({MESH[sh]})")
    cpath = os.path.join(OUT, "combined_photometric_ct.ply")
    write_ply(cpath, allV, allF)
    print(f"combined mesh: {len(allV)} verts / {len(allF)} faces -> {os.path.basename(cpath)}")

    truth = [os.path.join(OUT, f"{s}_CT_truth.mrk.json") for s in shots]
    json.dump(merge_markups(truth, shots, [0.2, 1.0, 0.2]),
              open(os.path.join(OUT, "combined_CT_truth.mrk.json"), "w"), indent=1)
    for cs, col in (("RAS", [1.0, 0.3, 0.3]), ("LPS", [0.3, 0.5, 1.0])):
        rec = [os.path.join(OUT, f"{s}_reconstructed_{cs}.mrk.json") for s in shots]
        json.dump(merge_markups(rec, shots, col),
                  open(os.path.join(OUT, f"combined_reconstructed_{cs}.mrk.json"), "w"),
                  indent=1)
    print("combined markers -> combined_CT_truth / combined_reconstructed_RAS / _LPS")


if __name__ == "__main__":
    main()
