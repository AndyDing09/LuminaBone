#!/usr/bin/env python3
"""
landmark_register.py  --  Landmark (fiducial) similarity registration
======================================================================

Aligns an endoscope point cloud / mesh to a CT-derived spine model using a
set of MATCHED landmark pairs, via the closed-form Umeyama solution
(Umeyama 1991; equivalent to Horn's absolute-orientation). Solves for a
similarity transform T = [ sR | t ; 0 0 0 1 ] that maps SOURCE landmarks
onto TARGET landmarks in a least-squares sense.

Why this method (not ICP):
  With >=3 non-collinear matched points the similarity transform is uniquely
  and stably determined in closed form. Unlike surface ICP on a near-planar
  brightness-relief patch, there is no rotational/scale ambiguity, and you get
  an honest fiducial registration error (FRE) in millimetres.

Coordinate systems:
  Slicer markups store a declared coordinate system (LPS or RAS). Both fiducial
  files MUST be in the same system (they will be, if both are picked in the same
  Slicer scene). The script checks the declared systems and converts RAS<->LPS
  if they differ, so the result is always consistent.

Inputs it understands:
  * Slicer markups JSON  (*.mrk.json)          <- current Slicer format
  * Slicer legacy fiducial CSV (*.fcsv)        <- older format
  * plain CSV / TSV  (x,y,z[,label] per row)

Outputs (into --outdir):
  * transform_source_to_target.npy   4x4 matrix (row-major, applies to column vec)
  * transform_source_to_target.txt   same, human readable, + scale/rot/trans
  * transform_itk.tfm                ITK/Slicer-importable AffineTransform (LPS)
  * <cloud>_registered.ply           the full cloud after the transform
  * registration_report.json         scale, FRE (rms/max/per-point), diagnostics

Usage:
  python landmark_register.py \
      --source-fids  source_on_endoscope.mrk.json \
      --target-fids  target_on_CT.mrk.json \
      --cloud        depth_outputs/ply_allon/loc03_allon.ply \
      --outdir       ./reg_out \
      [--rigid]                      # rigid only (no scale), scale forced to 1
"""

import argparse, json, os, re, sys
import numpy as np


# --------------------------------------------------------------------------- #
# Fiducial readers  (all return: labels[list], pts[N,3], coord_system 'RAS'|'LPS')
# --------------------------------------------------------------------------- #
def read_mrk_json(path):
    d = json.load(open(path))
    mk = d["markups"][0]
    cs = mk.get("coordinateSystem", "LPS").upper()
    labels, pts = [], []
    for cp in mk["controlPoints"]:
        labels.append(cp.get("label", ""))
        pts.append(cp["position"])
    return labels, np.asarray(pts, float), cs


def read_fcsv(path):
    cs = "RAS"  # .fcsv default is RAS unless header says otherwise
    labels, pts = [], []
    for line in open(path):
        line = line.strip()
        if line.startswith("#"):
            m = re.search(r"CoordinateSystem\s*=\s*(\w+)", line)
            if m:
                cs = m.group(1).upper()
                if cs in ("0", "RAS"): cs = "RAS"
                if cs in ("1", "LPS"): cs = "LPS"
            continue
        if not line:
            continue
        f = line.split(",")
        # fcsv columns: id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,...
        pts.append([float(f[1]), float(f[2]), float(f[3])])
        labels.append(f[11] if len(f) > 11 else "")
    return labels, np.asarray(pts, float), cs


def read_plain(path):
    labels, pts = [], []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        f = re.split(r"[,\t ]+", line)
        pts.append([float(f[0]), float(f[1]), float(f[2])])
        labels.append(f[3] if len(f) > 3 else "")
    return labels, np.asarray(pts, float), "RAS"


def read_fids(path):
    p = path.lower()
    if p.endswith(".mrk.json") or p.endswith(".json"):
        return read_mrk_json(path)
    if p.endswith(".fcsv"):
        return read_fcsv(path)
    return read_plain(path)


def ras_lps_flip(pts):
    """RAS<->LPS is negation of the first two axes (X,Y)."""
    out = pts.copy()
    out[:, 0] *= -1
    out[:, 1] *= -1
    return out


# --------------------------------------------------------------------------- #
# Umeyama similarity  (source -> target)
# --------------------------------------------------------------------------- #
def umeyama(src, dst, with_scale=True):
    """
    Least-squares similarity transform mapping src onto dst.
    src, dst: (N,3). Returns 4x4 matrix M, scale s, R(3x3), t(3,).
    Reference: Umeyama, IEEE PAMI 1991.
    """
    assert src.shape == dst.shape and src.shape[0] >= 3, "need >=3 matched pts"
    n, dim = src.shape
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    sc = src - mu_s
    dc = dst - mu_d
    # covariance dst<-src
    cov = (dc.T @ sc) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:   # reflection guard
        S[-1, -1] = -1
    R = U @ S @ Vt
    if with_scale:
        var_s = (sc ** 2).sum() / n
        s = (D * np.diag(S)).sum() / var_s
    else:
        s = 1.0
    t = mu_d - s * R @ mu_s
    M = np.eye(4)
    M[:3, :3] = s * R
    M[:3, 3] = t
    return M, s, R, t


def apply_M(M, pts):
    return (M[:3, :3] @ pts.T).T + M[:3, 3]


# --------------------------------------------------------------------------- #
# Minimal PLY IO  (ascii + binary_little_endian; xyz + optional rgb + faces)
# --------------------------------------------------------------------------- #
def read_ply(path):
    f = open(path, "rb")
    hdr = b""
    while b"end_header" not in hdr:
        chunk = f.readline()
        if not chunk:
            break
        hdr += chunk
    lines = hdr.decode("ascii", "ignore").splitlines()
    fmt = "ascii"
    nv = nf = 0
    vprops = []       # list of (name,type)
    section = None
    for l in lines:
        p = l.split()
        if not p:
            continue
        if p[0] == "format":
            fmt = p[1]
        elif p[0] == "element" and p[1] == "vertex":
            nv = int(p[2]); section = "v"
        elif p[0] == "element" and p[1] == "face":
            nf = int(p[2]); section = "f"
        elif p[0] == "property" and section == "v":
            vprops.append((p[-1], p[1]))
    names = [n for n, _ in vprops]
    has_rgb = all(c in names for c in ("red", "green", "blue"))
    if fmt == "ascii":
        verts = np.zeros((nv, len(vprops)))
        for i in range(nv):
            verts[i] = [float(x) for x in f.readline().split()[:len(vprops)]]
        xyz = verts[:, [names.index("x"), names.index("y"), names.index("z")]]
        rgb = (verts[:, [names.index("red"), names.index("green"), names.index("blue")]].astype(np.uint8)
               if has_rgb else None)
        faces = None
        if nf:
            faces = np.array([[int(x) for x in f.readline().split()[1:4]] for _ in range(nf)])
    else:
        np_ty = {"float": "<f4", "float32": "<f4", "double": "<f8",
                 "uchar": "u1", "uint8": "u1", "int": "<i4", "int32": "<i4"}
        dt = np.dtype([(n, np_ty[t]) for n, t in vprops])
        v = np.frombuffer(f.read(nv * dt.itemsize), dtype=dt, count=nv)
        xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
        rgb = (np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.uint8)
               if has_rgb else None)
        faces = None
        if nf:
            # face list: uchar count + int*count ; assume triangles
            rest = f.read()
            faces = np.empty((nf, 3), np.int32)
            off = 0
            for i in range(nf):
                cnt = rest[off]; off += 1
                idx = np.frombuffer(rest, "<i4", count=cnt, offset=off)
                faces[i] = idx[:3]; off += 4 * cnt
    return xyz, rgb, faces


def write_ply(path, xyz, rgb=None, faces=None):
    xyz = np.asarray(xyz, np.float32)
    nv = len(xyz)
    h = ["ply", "format binary_little_endian 1.0", f"element vertex {nv}",
         "property float x", "property float y", "property float z"]
    if rgb is not None:
        h += ["property uchar red", "property uchar green", "property uchar blue"]
    if faces is not None and len(faces):
        h += [f"element face {len(faces)}", "property list uchar int vertex_indices"]
    h += ["end_header"]
    with open(path, "wb") as o:
        o.write(("\n".join(h) + "\n").encode())
        if rgb is not None:
            dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                           ("r", "u1"), ("g", "u1"), ("b", "u1")])
            rec = np.empty(nv, dt)
            rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
            rgb = np.asarray(rgb, np.uint8)
            rec["r"], rec["g"], rec["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
            rec.tofile(o)
        else:
            xyz.tofile(o)
        if faces is not None and len(faces):
            fdt = np.dtype([("n", "u1"), ("a", "<i4"), ("b", "<i4"), ("c", "<i4")])
            fr = np.empty(len(faces), fdt)
            fr["n"] = 3
            fr["a"], fr["b"], fr["c"] = faces[:, 0], faces[:, 1], faces[:, 2]
            fr.tofile(o)


def write_itk_tfm(path, M):
    """
    Write an ITK AffineTransform (.tfm) that Slicer can import as a linear
    transform. ITK works in LPS and stores the transform as it maps points.
    We convert our RAS/target-frame matrix M (source->target) into LPS.
    Parameters line = 9 rotation/scale entries (row-major) then 3 translation.
    """
    F = np.diag([-1, -1, 1]).astype(float)          # RAS->LPS
    Mlps = np.eye(4)
    Mlps[:3, :3] = F @ M[:3, :3] @ F
    Mlps[:3, 3] = F @ M[:3, 3]
    A = Mlps[:3, :3].reshape(-1)
    t = Mlps[:3, 3]
    with open(path, "w") as o:
        o.write("#Insight Transform File V1.0\n")
        o.write("#Transform 0\n")
        o.write("Transform: AffineTransform_double_3_3\n")
        o.write("Parameters: " + " ".join(f"{x:.10g}" for x in list(A) + list(t)) + "\n")
        o.write("FixedParameters: 0 0 0\n")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-fids", required=True)
    ap.add_argument("--target-fids", required=True)
    ap.add_argument("--cloud", required=True, help="PLY to transform (source frame)")
    ap.add_argument("--outdir", default="./reg_out")
    ap.add_argument("--rigid", action="store_true", help="no scaling (s=1)")
    ap.add_argument("--common-system", default="RAS", choices=["RAS", "LPS"])
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    sl, sp, scs = read_fids(args.source_fids)
    tl, tp, tcs = read_fids(args.target_fids)
    print(f"source fids: {len(sp)} ({scs})   target fids: {len(tp)} ({tcs})")
    if len(sp) != len(tp):
        sys.exit(f"ERROR: fiducial counts differ ({len(sp)} vs {len(tp)}). "
                 "They must be matched pairs in the same order.")
    if len(sp) < 3:
        sys.exit("ERROR: need at least 3 matched fiducials.")

    # bring both into the requested common coordinate system
    def to_common(pts, cs):
        return pts if cs == args.common_system else ras_lps_flip(pts)
    sp = to_common(sp, scs)
    tp = to_common(tp, tcs)

    M, s, R, t = umeyama(sp, tp, with_scale=not args.rigid)

    # fiducial registration error
    mapped = apply_M(M, sp)
    resid = np.linalg.norm(mapped - tp, axis=1)
    fre_rms = float(np.sqrt((resid ** 2).mean()))
    fre_max = float(resid.max())

    # collinearity / conditioning warning
    _, sv, _ = np.linalg.svd(sp - sp.mean(0))
    cond = float(sv[-1] / sv[0]) if sv[0] else 0.0

    print(f"\nestimated scale s = {s:.6f}")
    print(f"FRE  rms = {fre_rms:.4f}   max = {fre_max:.4f}  (same units as target fids)")
    print(f"landmark spread conditioning (smallest/largest sing.val) = {cond:.4f}"
          + ("   <-- WARNING: near-collinear/planar landmarks!" if cond < 0.02 else ""))

    # transform the cloud
    xyz, rgb, faces = read_ply(args.cloud)
    xyz_reg = apply_M(M, xyz)
    base = os.path.splitext(os.path.basename(args.cloud))[0]
    reg_ply = os.path.join(args.outdir, f"{base}_registered.ply")
    write_ply(reg_ply, xyz_reg, rgb, faces)

    # save transforms
    np.save(os.path.join(args.outdir, "transform_source_to_target.npy"), M)
    with open(os.path.join(args.outdir, "transform_source_to_target.txt"), "w") as o:
        o.write("# similarity transform  source -> target  (applies to column vector)\n")
        o.write(f"# common coordinate system: {args.common_system}\n")
        o.write(f"# scale = {s:.8f}\n")
        o.write("# 4x4 matrix:\n")
        for row in M:
            o.write("  ".join(f"{v: .8f}" for v in row) + "\n")
    write_itk_tfm(os.path.join(args.outdir, "transform_itk.tfm"), M)

    report = {
        "method": "Umeyama closed-form similarity (rigid+scale)"
                  if not args.rigid else "Umeyama rigid (no scale)",
        "n_landmarks": int(len(sp)),
        "source_labels": sl, "target_labels": tl,
        "common_coordinate_system": args.common_system,
        "scale": s,
        "rotation_matrix": R.tolist(),
        "translation": t.tolist(),
        "transform_4x4": M.tolist(),
        "fre_rms": fre_rms, "fre_max": fre_max,
        "per_landmark_residual": resid.tolist(),
        "landmark_conditioning": cond,
        "near_planar_warning": cond < 0.02,
        "registered_cloud": os.path.abspath(reg_ply),
    }
    json.dump(report, open(os.path.join(args.outdir, "registration_report.json"), "w"),
              indent=2)
    print(f"\nwrote:\n  {reg_ply}\n  {args.outdir}/transform_source_to_target.(npy|txt)"
          f"\n  {args.outdir}/transform_itk.tfm\n  {args.outdir}/registration_report.json")


if __name__ == "__main__":
    main()
