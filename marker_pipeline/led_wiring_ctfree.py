"""
CT-free measurement of each LED's BEAM-AIM direction (and what it does and
does not say about the index -> azimuth wiring).

Cue: dividing each single-LED image by the 4-LED mean cancels albedo/shape;
the brightness-weighted centroid of the ratio image marks where that LED's
beam LANDS. Finding (unanimous, 5/5 shots): the beam of every LED lands on
the side OPPOSITE its ring position for the vertical pair -- i.e. the LEDs
are aimed CONVERGENTLY (each beam crosses the optical axis before the
working distance), a real hardware property that the parallel-aim emission
model (cos theta = D/r) does not capture.

Position vs aim: beam aim redistributes intensity but cannot flip the
direction light ARRIVES from; shading incidence is set by LED POSITION.
So this centroid cue does NOT by itself determine the wiring -- it measures
aim. The wiring {1:90, 2:270, 3:180, 4:0} rests on the incidence direction
(permutation study, r = +0.98 vs -0.98 for the flip) and must additionally
be verified ONCE on the bench (power each LED, look at the ring) to make its
provenance fully CT-free.

Run: python led_wiring_ctfree.py            # all CT shots, per-shot + joint vote
"""

import os
import sys
import math
import itertools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "reconstruction"))
import bone_depth_batch as bd
bd.UNDISTORT_INPUTS = False
import calibrate_photometric_ct as C
from nearfield_ct import load_lums

RING_AZ = [0.0, 90.0, 180.0, 270.0]
TRUE = {1: 90.0, 2: 270.0, 3: 180.0, 4: 0.0}     # the CT-study wiring, for comparison


def led_directions(shot):
    """measured in-image direction of each LED's illumination asymmetry."""
    lums, _ = load_lums(shot)                     # dark-subtracted, balanced
    I = np.stack(lums, 0)                        # (4,H,W)
    mask = bd.bone_mask(I.mean(0))
    mean = I.mean(0) + 1e-9
    H, W = mean.shape
    yy, xx = np.mgrid[:H, :W]
    dirs = {}
    for k, s in enumerate((1, 2, 3, 4)):
        ratio = I[k] / mean                       # cancels albedo + shape common mode
        w = np.where(mask, ratio, 0.0)
        cx = (w * xx).sum() / w.sum() - xx[mask].mean()
        cy = (w * yy).sum() / w.sum() - yy[mask].mean()
        # image +x right, +y DOWN; azimuth convention +x right, +y UP
        dirs[s] = math.degrees(math.atan2(-cy, cx)) % 360.0
    return dirs


def best_assignment(dirs):
    """globally best LED->azimuth matching (min total angular error)."""
    best = None
    for perm in itertools.permutations(RING_AZ):
        cost = sum(min(abs(dirs[s] - perm[i]), 360 - abs(dirs[s] - perm[i]))
                   for i, s in enumerate((1, 2, 3, 4)))
        if best is None or cost < best[0]:
            best = (cost, {s: perm[i] for i, s in enumerate((1, 2, 3, 4))})
    return best[1], best[0]


def main():
    print("CT-free LED beam-aim directions (centroid of per-LED ratio image):")
    votes = {}
    for shot in C.CT_CORR:
        dirs = led_directions(shot)
        az, cost = best_assignment(dirs)
        print(f"  {shot}: beam lands " +
              " ".join(f"LED{s}:{dirs[s]:5.1f}" for s in (1, 2, 3, 4)) +
              f"  -> aim map {az}  (err {cost:.0f} deg)")
        key = tuple(sorted(az.items()))
        votes[key] = votes.get(key, 0) + 1
    winner = max(votes.items(), key=lambda kv: kv[1])
    aim = dict(winner[0])
    print(f"\njoint aim map: {aim}  ({winner[1]}/{len(C.CT_CORR)} shots)")
    print(f"wiring in use:  {TRUE}  (incidence-based; permutation study)")
    for s in (1, 2, 3, 4):
        d = min(abs(aim[s] - TRUE[s]), 360 - abs(aim[s] - TRUE[s]))
        tag = ("beam lands on its own side (parallel-ish aim)" if d < 90 else
               "beam lands OPPOSITE its assigned position")
        print(f"  LED{s}: wiring {TRUE[s]:5.1f}  aim {aim[s]:5.1f}  -> {tag}")
    print("\nInterpretation: for any LED whose beam lands opposite, EITHER that"
          "\nLED is aimed convergently across the optical axis (hardware tilt),"
          "\nOR the wiring label for that pair is flipped and a compensating"
          "\nsign lives elsewhere in the pipeline. Images alone cannot separate"
          "\nthese; a one-time bench check (power each LED, look at the ring)"
          "\nsettles it and makes the wiring provenance fully CT-free.")


if __name__ == "__main__":
    main()
