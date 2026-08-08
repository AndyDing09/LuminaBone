"""
Export Slicer markups (.mrk.json) so you can SEE and MEASURE the registration:
  <shot>_CT_truth.mrk.json      -> the CT marker coords you gave (should land on your red spheres)
  <shot>_reconstructed.mrk.json -> where the near-field surface puts those markers

Load BOTH in Slicer. If CT_truth lands on your existing red spheres, the coordinate
frame is right and the gap between truth<->reconstructed IS the registration error
(the ~1.7 mm). If CT_truth is flipped off the spheres, switch RAS<->LPS below.

Run: python export_fiducials.py            # shot_004
     python export_fiducials.py all
"""

import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "reconstruction"))
import bone_depth_batch as bd
bd.UNDISTORT_INPUTS = False
import calibrate_photometric_ct as C
import register_inverse_square_ct as R
import nearfield_ct as NF

COORD = "RAS"          # <-- if truth markers land flipped in Slicer, change to "LPS"
OUT = bd.project_path("depth_outputs", "ct_registered")


def markups(points, labels, color, coord=COORD):
    return {
        "@schema": "https://raw.githubusercontent.com/Slicer/Slicer/main/Modules/"
                   "Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
        "markups": [{
            "type": "Fiducial",
            "coordinateSystem": coord,
            "controlPoints": [
                {"id": str(k + 1), "label": lab,
                 "position": [float(p[0]), float(p[1]), float(p[2])]}
                for k, (p, lab) in enumerate(zip(points, labels))],
            "display": {"color": color, "selectedColor": color,
                        "glyphScale": 3.0, "textScale": 3.0},
        }],
    }


def run(shot):
    corr = C.CT_CORR[shot]
    ids = sorted(corr)
    labels = [f"m{i}" for i in ids]
    truth = np.array([corr[i][1] for i in ids], float)

    # reconstructed marker positions from the near-field solve (same as the mesh)
    D, Dsm, ref, keep, Rp, t, obj, ct_d, relief = NF.nearfield(shot, corr)
    mk_uv = np.array([corr[i][0] for i in ids], float)
    mk_z = np.array([R._disk(D, *corr[i][0]) for i in ids])
    recon = (Rp.T @ (R.backproject(mk_uv[:, 0], mk_uv[:, 1], mk_z) - t).T).T
    err = np.linalg.norm(recon - truth, axis=1)

    p1 = os.path.join(OUT, f"{shot}_CT_truth.mrk.json")
    json.dump(markups(truth, labels, [0.2, 1.0, 0.2]), open(p1, "w"), indent=1)   # green

    # SAME numbers, two convention tags: whichever matches your PLY import lands ON
    # the mesh as Slicer displays it (Slicer often imports .ply as LPS -> flipped).
    for cs, col in (("RAS", [1.0, 0.3, 0.3]), ("LPS", [0.3, 0.5, 1.0])):
        pr = os.path.join(OUT, f"{shot}_reconstructed_{cs}.mrk.json")
        json.dump(markups(recon, [f"{l}*" for l in labels], col, coord=cs),
                  open(pr, "w"), indent=1)
    print(f"{shot}: FRE {np.sqrt(np.mean(err**2)):.2f} mm  per-marker {np.round(err,2)}")
    print(f"   {os.path.basename(p1)}  (CT truth, green)")
    print(f"   {shot}_reconstructed_RAS.mrk.json / _LPS.mrk.json  (mesh markers)")


if __name__ == "__main__":
    shots = list(C.CT_CORR) if (len(sys.argv) > 1 and sys.argv[1] == "all") else ["shot_004"]
    for sh in shots:
        run(sh)
    print("\nLoad both .mrk.json in Slicer (Add Data). Green should sit on your red "
          "CT spheres; green<->red* gap = the registration error.")
