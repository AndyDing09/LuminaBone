#  slicer_register.py  --  run INSIDE 3D Slicer's Python console
#  =============================================================
#  Does the endoscope-cloud -> CT-spine registration entirely inside Slicer.
#  No manual clicking of transforms required. Two modes:
#
#    A) auto_icp()      - fully automatic ICP (similarity: rotation+trans+scale).
#                         One call. May be wrong when the endoscope patch is a
#                         small near-flat piece of the whole vertebra -- check
#                         the printed mean surface error and the 3D view.
#
#    B) from_fiducials()- reliable. You drop >=3 matched points in two markups
#                         lists named 'source' (on endoscope) and 'target'
#                         (on CT), then call this. Uses a similarity landmark
#                         transform, prints the fiducial error (FRE).
#
#  HOW TO RUN
#  ----------
#  1. In Slicer:  View  ->  Python Console   (or the "Python" toolbar icon)
#  2. Paste this whole file into the console and press Enter.
#     (or:  exec(open(r"C:\\Users\\andyd\\OneDrive\\LuminaBone\\registration\\slicer_register.py").read()) )
#  3. It auto-loads the two models and runs auto_icp(). Then read the report.
#  4. If the auto fit is poor, place fiducials and call:  from_fiducials()
#
#  Edit CT_PATH / ENDO_PATH below if you want a different location.

import os, numpy as np, slicer, vtk

REG_DIR   = r"C:\Users\andyd\OneDrive\LuminaBone\registration"
CT_PATH   = os.path.join(REG_DIR, "ct_spine.stl")
# endoscope side: use the *mesh* version so it displays as a surface in Slicer
ENDO_PATH = r"C:\Users\andyd\OneDrive\LuminaBone\depth_outputs\ply_allon_mesh\loc03_allon_mesh.ply"
OUT_PATH  = os.path.join(REG_DIR, "endo_registered_in_slicer.ply")


# --------------------------------------------------------------------------- #
def _load():
    ct   = slicer.util.loadModel(CT_PATH)
    endo = slicer.util.loadModel(ENDO_PATH)
    ct.GetDisplayNode().SetColor(0.9, 0.85, 0.75); ct.GetDisplayNode().SetOpacity(1.0)
    endo.GetDisplayNode().SetColor(0.2, 0.6, 1.0); endo.GetDisplayNode().SetOpacity(0.9)
    return ct, endo


def _mean_surface_error(moving_poly, fixed_poly):
    """Mean distance from moving points to the fixed surface (mm)."""
    d = vtk.vtkImplicitPolyDataDistance(); d.SetInput(fixed_poly)
    pts = moving_poly.GetPoints()
    n = pts.GetNumberOfPoints()
    step = max(1, n // 20000)                     # subsample for speed
    vals = [abs(d.EvaluateFunction(pts.GetPoint(i))) for i in range(0, n, step)]
    return float(np.mean(vals)), float(np.median(vals))


def _apply_matrix_and_harden(model, vtk_matrix):
    tn = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode",
                                            "endo_to_CT")
    tn.SetMatrixTransformToParent(vtk_matrix)
    model.SetAndObserveTransformNodeID(tn.GetID())
    slicer.vtkSlicerTransformLogic().hardenTransform(model)
    return tn


def _get_models():
    ms = slicer.util.getNodesByClass("vtkMRMLModelNode")
    ms = [m for m in ms if m.GetName() not in ("Red Volume Slice", "Green Volume Slice",
                                               "Yellow Volume Slice")]
    return ms


# --------------------------------------------------------------------------- #
def auto_icp():
    """Automatic ICP: endoscope -> CT, similarity (includes uniform scale)."""
    ct, endo = _load()
    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(endo.GetPolyData())             # moving
    icp.SetTarget(ct.GetPolyData())               # fixed
    icp.GetLandmarkTransform().SetModeToSimilarity()   # rotation+translation+scale
    icp.StartByMatchingCentroidsOn()
    icp.SetMaximumNumberOfIterations(200)
    icp.SetMaximumNumberOfLandmarks(4000)
    icp.Modified(); icp.Update()
    M = icp.GetMatrix()
    scale = (np.linalg.norm([M.GetElement(0,0),M.GetElement(1,0),M.GetElement(2,0)]))
    _apply_matrix_and_harden(endo, M)
    mean_e, med_e = _mean_surface_error(endo.GetPolyData(), ct.GetPolyData())
    slicer.util.saveNode(endo, OUT_PATH)
    print("=" * 60)
    print("AUTO ICP (similarity) result")
    print(f"  recovered scale         : {scale:.5f}")
    print(f"  mean surface error (mm) : {mean_e:.3f}")
    print(f"  median surface error(mm): {med_e:.3f}")
    print(f"  saved registered model  : {OUT_PATH}")
    print("  --> Inspect the 3D view. If the blue patch does NOT sit on the")
    print("      correct region of the bone, the auto fit failed (expected for")
    print("      a small flat patch) -- use from_fiducials() instead.")
    print("=" * 60)
    return icp


# --------------------------------------------------------------------------- #
def setup_fiducials():
    """
    Create the two empty point lists named 'source' and 'target' and arm the
    mouse for placing points. Call this ONCE, then:
      * click 4-5 features on the RED endoscope patch      (fills 'source')
      * then run  activate('target')  and click the SAME spots on the BONE
      * then run  from_fiducials()
    """
    for n in ("source", "target"):
        old = slicer.mrmlScene.GetFirstNodeByName(n)
        if old:
            slicer.mrmlScene.RemoveNode(old)
    src = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "source")
    tgt = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "target")
    src.GetDisplayNode().SetSelectedColor(0, 1, 0)   # green on endoscope
    tgt.GetDisplayNode().SetSelectedColor(1, 1, 0)   # yellow on bone
    activate("source")
    print("Created 'source' (green) and 'target' (yellow).")
    print("STEP 1: click 4-5 features on the RED endoscope patch now.")
    print("STEP 2: run  activate('target')  then click the SAME spots on the BONE.")
    print("STEP 3: run  from_fiducials()")
    return src, tgt


def activate(name):
    """Make a point list the active one and turn on click-to-place (persistent)."""
    node = slicer.util.getNode(name)
    sel = slicer.app.applicationLogic().GetSelectionNode()
    sel.SetReferenceActivePlaceNodeClassName("vtkMRMLMarkupsFiducialNode")
    sel.SetActivePlaceNodeID(node.GetID())
    inter = slicer.app.applicationLogic().GetInteractionNode()
    inter.SetPlaceModePersistence(1)
    inter.SetCurrentInteractionMode(inter.Place)
    print(f"'{name}' is now active — click points on the model in a 3D/slice view.")
    print(f"   (currently {node.GetNumberOfControlPoints()} points placed)")


def from_fiducials(source_name="source", target_name="target", with_scale=True):
    """
    Similarity landmark registration from two matched markups lists.
    Create point lists named 'source' (clicked on the endoscope model) and
    'target' (same physical points, same order, on the CT), then call this.
    """
    src = slicer.mrmlScene.GetFirstNodeByName(source_name)
    tgt = slicer.mrmlScene.GetFirstNodeByName(target_name)
    if src is None or tgt is None:
        print("!! No 'source'/'target' point lists yet. Run  setup_fiducials()  first,")
        print("   place your points, then call from_fiducials() again.")
        return
    ns, nt = src.GetNumberOfControlPoints(), tgt.GetNumberOfControlPoints()
    if ns != nt or ns < 3:
        print(f"!! Need >=3 MATCHED points in the same order. Currently source={ns}, "
              f"target={nt}.")
        print("   Add/finish points (use activate('source') / activate('target')) then retry.")
        return

    S = vtk.vtkPoints(); T = vtk.vtkPoints()
    p = [0, 0, 0]
    for i in range(ns):
        src.GetNthControlPointPositionWorld(i, p); S.InsertNextPoint(p)
        tgt.GetNthControlPointPositionWorld(i, p); T.InsertNextPoint(p)
    lm = vtk.vtkLandmarkTransform()
    lm.SetSourceLandmarks(S); lm.SetTargetLandmarks(T)
    lm.SetModeToSimilarity() if with_scale else lm.SetModeToRigidBody()
    lm.Update()
    M = lm.GetMatrix()

    # FRE
    res = []
    for i in range(ns):
        S.GetPoint(i, p); q = M.MultiplyPoint([p[0], p[1], p[2], 1.0])
        t = [0, 0, 0]; T.GetPoint(i, t)
        res.append(np.linalg.norm(np.array(q[:3]) - np.array(t)))
    fre = float(np.sqrt(np.mean(np.square(res))))
    scale = np.linalg.norm([M.GetElement(0,0),M.GetElement(1,0),M.GetElement(2,0)])

    endo = [m for m in _get_models() if "allon" in m.GetName().lower()
            or "loc" in m.GetName().lower()]
    if not endo:
        print("!! couldn't find the endoscope model node; load it first.")
        return
    endo = endo[0]
    _apply_matrix_and_harden(endo, M)
    ct = [m for m in _get_models() if m is not endo]
    if ct:
        mean_e, med_e = _mean_surface_error(endo.GetPolyData(), ct[0].GetPolyData())
    else:
        mean_e = med_e = float("nan")
    slicer.util.saveNode(endo, OUT_PATH)
    print("=" * 60)
    print("LANDMARK (similarity) registration")
    print(f"  n landmarks             : {ns}")
    print(f"  recovered scale         : {scale:.5f}")
    print(f"  FRE rms (mm)            : {fre:.3f}")
    print(f"  per-landmark resid (mm) : {[round(r,2) for r in res]}")
    print(f"  mean surface error (mm) : {mean_e:.3f}")
    print(f"  saved registered model  : {OUT_PATH}")
    print("  Tip: run refine_icp() afterwards to tighten the surface fit.")
    print("=" * 60)


def refine_icp():
    """Point-to-surface ICP starting from the current (landmark) alignment."""
    ms = _get_models()
    endo = [m for m in ms if "allon" in m.GetName().lower() or "loc" in m.GetName().lower()]
    ct   = [m for m in ms if m not in endo]
    if not (endo and ct):
        print("!! need both models loaded"); return
    endo, ct = endo[0], ct[0]
    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(endo.GetPolyData()); icp.SetTarget(ct.GetPolyData())
    icp.GetLandmarkTransform().SetModeToRigidBody()
    icp.StartByMatchingCentroidsOff(); icp.SetMaximumNumberOfIterations(100)
    icp.Modified(); icp.Update()
    _apply_matrix_and_harden(endo, icp.GetMatrix())
    mean_e, med_e = _mean_surface_error(endo.GetPolyData(), ct.GetPolyData())
    slicer.util.saveNode(endo, OUT_PATH)
    print(f"ICP refine -> mean surface error {mean_e:.3f} mm  (saved {OUT_PATH})")


# ---- run automatic path on load -------------------------------------------- #
print("slicer_register loaded. Running auto_icp() ...")
print("If auto fit is poor: place 'source'/'target' fiducials, then call from_fiducials()")
try:
    auto_icp()
except Exception as e:
    print("auto_icp() error:", e)
    print("Load the models manually or fix CT_PATH/ENDO_PATH, then re-run.")
