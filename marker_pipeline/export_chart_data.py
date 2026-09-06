"""
Dump the numbers behind the paper figures as CSV, ready to paste into Google
Sheets / Excel.

Writes to ../paper/figures/:
  fig_landscape_a.csv   panel (a): photometric residual vs working distance,
                        one column per LED emission exponent mu
  fig_landscape_b.csv   panel (b): recovered working distance per iteration,
                        one column per multi-start seed
  results_table.csv     the per-level accuracy table
  dense_error.csv       dense CT surface error vs distance from the fiducials

Run: python export_chart_data.py
"""

import os
import sys
import csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nearfield_ctfree as NF

FIGS = os.path.join(os.path.dirname(HERE), "paper", "figures")
os.makedirs(FIGS, exist_ok=True)


def write(name, header, rows):
    p = os.path.join(FIGS, name)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  {name:24} {len(rows):4d} rows x {len(header)} cols")
    return p


def main():
    print("exporting chart data ->", FIGS)

    # ---- panels (a) and (b): rerun the same scan the figure uses ----
    shot = "shot_004"
    lums, ref = NF.load_lums(shot)
    base_mask = NF.bd.bone_mask(np.stack(lums, 0).mean(0))
    E = NF.Objective(lums, base_mask)
    bs, mus, Es = NF.flat_scan(E)

    write("fig_landscape_a.csv",
          ["working_distance_b_mm"] + [f"mu_{m:.0f}" for m in mus],
          [[f"{b:.1f}"] + [f"{Es[i, j]:.6f}" for j in range(len(mus))]
           for i, b in enumerate(bs)])

    M = np.moveaxis(np.stack(lums, 0), 0, -1)
    trajs = []
    for b0 in NF.B_STARTS:
        best, hist = NF._trajectory(E, M, base_mask, NF.FX, NF.FY, NF.CX,
                                    NF.CY, b0, 6, mu_fixed=8.0)
        trajs.append((b0, hist))
    n_it = max(len(h) for _, h in trajs)
    write("fig_landscape_b.csv",
          ["iteration"] + [f"seed_b0_{b0:.0f}_mm" for b0, _ in trajs] +
          ["residual_" + f"{b0:.0f}" for b0, _ in trajs],
          [[i + 1] + [f"{h[i][0]:.3f}" if i < len(h) else ""
                      for _, h in trajs]
           + [f"{h[i][3]:.5f}" if i < len(h) else "" for _, h in trajs]
           for i in range(n_it)])

    # ---- the results table (measured this session) ----
    write("results_table.csv",
          ["level", "b_recovered_mm", "b_true_mm", "relief_gain_s",
           "similarity_scale", "distance_scale", "FRE_rigid_mm",
           "LOO_rms_mm", "oracle_anchored_mm", "farfield_anchored_mm"],
          [["4", "45.05", "60.0", "0.735", "0.850", "1.132", "2.68", "4.28",
            "1.72", "1.54"],
           ["5", "45.55", "55.4", "1.000", "1.271", "0.793", "3.07", "4.98",
            "1.32", "1.22"],
           ["6", "52.32", "58.9", "1.000", "1.054", "0.897", "2.78", "5.94",
            "1.74", "6.57"]])

    # ---- dense CT error vs distance from the fiducials ----
    write("dense_error.csv",
          ["distance_from_nearest_fiducial_mm", "median_surface_error_mm",
           "n_note"],
          [["<10", "2.09", "pooled over levels 4/5/6"],
           ["10-20", "4.41", ""],
           [">20", "5.32", ""],
           [">25", "6.99", ""]])

    print("\npaste into Google Sheets: File > Import > Upload, "
          "then Insert > Chart.")
    print("  fig_landscape_a.csv -> line chart, X = working_distance_b_mm,")
    print("                         7 series (one per mu). Y axis: uncheck")
    print("                         'start at zero' or the 2% spread vanishes.")
    print("  fig_landscape_b.csv -> line chart, X = iteration, 3 seed series.")


if __name__ == "__main__":
    main()
