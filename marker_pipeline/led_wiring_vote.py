"""
CT-FREE selection of the LED index->azimuth wiring, by photometric evidence.

The wiring in use ({1:90, 2:270, 3:180, 4:0}) was originally chosen by
correlating far-field depth against CT marker depth (led_permutation_study.py).
That leaves a ground-truth provenance hole in the "no CT in the reconstruction"
claim, flagged by the 2026-08-10 audit.

This script closes it without CT. The near-field tied-normal objective of
nearfield_ctfree.py measures how well ONE surface, lit from the assumed LED
positions, explains all four images. A wrong index->azimuth assignment puts the
light in the wrong place, so no surface can explain the four images and the
residual rises. Scoring all 4! = 24 permutations therefore selects the wiring
from the images alone.

Each permutation is run through the same CT-free machinery (multi-seed
trajectories at search resolution, rig-level mu) and scored by its best
achievable residual. Scores are normalised per shot before summing so that no
single shot dominates the vote.

Run: python led_wiring_vote.py                # shots 4,5,6, all 24 permutations
"""

import os
import sys
import math
import json
import itertools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "reconstruction"))
import bone_depth_batch as bd
bd.UNDISTORT_INPUTS = False
import calibrate_photometric_ct as C
import nearfield_ctfree as CF
from nearfield_ct import A_MM, load_lums

AZ_SET = [0.0, 90.0, 180.0, 270.0]
TRUE = {1: 90.0, 2: 270.0, 3: 180.0, 4: 0.0}     # the CT-derived wiring in use
NAMES = {0.0: "R", 90.0: "T", 180.0: "L", 270.0: "B"}
MU = 8.0
ITERS = 3
SEEDS = (48.0, 62.0)


def positions(azmap):
    return {s: np.array([A_MM * math.cos(math.radians(a)),
                         A_MM * math.sin(math.radians(a)), 0.0])
            for s, a in azmap.items()}


def score(shot_data, azmap):
    """best CT-free photometric residual achievable under this wiring."""
    CF.P = positions(azmap)                       # patch the forward model
    CF._GRIDS.clear()
    best = np.inf
    for (E, M, mask) in shot_data:
        s_best = min(CF._trajectory(E, E.Ms, E.mask, E.fx, E.fy, E.cx, E.cy,
                                    b0, ITERS, mu_fixed=MU)[0][0]
                     for b0 in SEEDS)
        best = s_best if best is np.inf else best
        yield s_best


def main():
    shots = ["shot_004", "shot_005", "shot_006"]
    prepared = {}
    for sh in shots:
        lums, _ = load_lums(sh)
        mask = bd.bone_mask(np.stack(lums, 0).mean(0))
        prepared[sh] = CF.Objective(lums, mask)
    print(f"CT-free wiring vote over {len(shots)} shots, 24 permutations "
          f"(mu={MU}, {ITERS} iters, seeds {SEEDS})\n")

    table = {}
    for perm in itertools.permutations(AZ_SET):
        azmap = {i + 1: perm[i] for i in range(4)}
        CF.P = positions(azmap)
        CF._GRIDS.clear()
        res = []
        for sh in shots:
            E = prepared[sh]
            res.append(min(CF._trajectory(E, E.Ms, E.mask, E.fx, E.fy,
                                          E.cx, E.cy, b0, ITERS,
                                          mu_fixed=MU)[0][0]
                           for b0 in SEEDS))
        table[tuple(sorted(azmap.items()))] = np.array(res)
        tag = "  <- wiring in use" if azmap == TRUE else ""
        print("  " + " ".join(f"{i}:{NAMES[azmap[i]]}" for i in (1, 2, 3, 4)) +
              "   " + "  ".join(f"{r:.4f}" for r in res) + tag, flush=True)

    # per-shot normalisation, then sum (no shot dominates)
    mat = np.array(list(table.values()))
    norm = mat / mat.min(axis=0, keepdims=True)
    votes = norm.sum(axis=1)
    order = np.argsort(votes)
    keys = list(table.keys())
    print(f"\n{'rank':6}{'wiring':22}{'score':>9}   (1.00 x 3 = perfect)")
    for rank, idx in enumerate(order[:5], 1):
        az = dict(keys[idx])
        tag = "  <- wiring in use" if az == TRUE else ""
        print(f"{rank:<6}" +
              " ".join(f"{i}:{NAMES[az[i]]}" for i in (1, 2, 3, 4)).ljust(22) +
              f"{votes[idx]:9.4f}{tag}")
    win = dict(keys[order[0]])
    true_rank = 1 + int(np.where(order == keys.index(tuple(sorted(TRUE.items()))))[0][0])
    print(f"\nbest by photometric evidence: {win}")
    print(f"wiring in use ranks #{true_rank} of 24")
    print("AGREEMENT: provenance is CT-free." if win == TRUE else
          "DISAGREEMENT: the images prefer a different wiring -- investigate.")
    json.dump({"winner": {str(k): v for k, v in win.items()},
               "in_use_rank": true_rank,
               "shots": shots, "mu": MU},
              open(os.path.join(HERE, "ctfree_wiring_vote.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
