#!/usr/bin/env python3
"""Survey TGM endpoints that sit on the readout-window truncation plane.

Background (docs/28_tgm-untagged-cases.md, Case 2): charge deposited near an
anode has ~zero drift time, so it arrives at the detector at ~= the cluster's
t0.  For a cosmic early enough that t0 precedes the readout-window open time,
that anode-side end of the track is never digitized and the reconstruction
stops on the plane

    |x_cut| = A + v * t0            (A = x_anode - v * t_window_open)

An endpoint on this plane is NOT evidence that the track stopped -- charge
beyond it could not exist in the data -- but TaggerCheckTGM has no concept of
the readout boundary, so such an end reads "inside the FV" and CASE A never
fires.  This script measures the plane and lists the affected clusters.

Repro:
    cd sbnd_xin && python3 tgm_readout_cut.py work-mcp10

Inputs are our own pipeline products only: work-mcp10/nusel-table.tsv (bundle
table from nusel_extract.py) and the per-event mabc-pr.zip written by
run_nusel_evt.sh.  Nothing is written.
"""
import json
import os
import sys
import zipfile

import numpy as np

# SBND fiducial x wall, cfg/pgrapher/experiment/sbnd/clus.jsonnet sbnd_pr_fv.
X_FV = 201.05
# Seed model used only to pick which endpoints to fit; the fit re-derives both.
SEED_A, SEED_V = 234.08, 0.1563
# An endpoint counts as "on the cut" within this tolerance (cm).
TOL = 1.5


def load_table(root):
    """nusel-table.tsv is whitespace-aligned, not tab-delimited."""
    rows = [ln.split() for ln in open(os.path.join(root, "nusel-table.tsv")) if ln.strip()]
    hdr = rows[0]
    return [dict(zip(hdr, r)) for r in rows[1:] if r[0] != "run"]


def main(root="work-mcp10"):
    rows = load_table(root)
    cache = {}
    recs = []
    for r in rows:
        evt, cid, t0 = r["event"], int(r["main_id"]), float(r["flash_time_us"])
        zp = os.path.join(root, f"nusel_evt{evt}", "mabc-pr.zip")
        if not os.path.exists(zp):
            continue
        if zp not in cache:
            d = json.loads(zipfile.ZipFile(zp).read("data/0/0-clustering-global.json"))
            cache[zp] = (np.array(d["cluster_id"]), np.array(d["x"]))
        ids, x = cache[zp]
        m = ids == cid
        if not m.any():
            continue
        cut = SEED_A + SEED_V * t0
        # The cut only bites when it falls strictly inside the FV x wall.
        hit = None
        if abs(cut) < X_FV:
            for obs, sgn in ((x[m].max(), +1), (x[m].min(), -1)):
                if abs(obs - sgn * cut) < TOL:
                    hit = (abs(obs), sgn)
        recs.append((evt, cid, t0, float(r["len_main_cm"]), r["tgm"], r["label"],
                     x[m].min(), x[m].max(), hit))

    clipped = [c for c in recs if c[8]]
    print(f"main clusters: {len(recs)}   endpoints on the readout cut: {len(clipped)}")

    # Global fit |x_end| = A + v*t0 over the clipped endpoints.
    T = np.array([c[2] for c in clipped])
    X = np.array([c[8][0] for c in clipped])
    v, A = np.polyfit(T, X, 1)
    res = X - (A + v * T)
    print(f"\nfit  |x_cut| = A + v*t0  ->  A = {A:.3f} cm,  v = {v:.5f} cm/us")
    print(f"     residual rms {res.std():.3f} cm, max |r| {np.abs(res).max():.3f} cm")
    print(f"     drift speed in run_nusel_evt.sh: 0.1563 cm/us  (agreement "
          f"{abs(v - 0.1563) / 0.1563 * 100:.2f}%)")
    print(f"     implied readout open t_start = {(X_FV - A) / v:.1f} us")

    print("\nclipped clusters:")
    print("  evt      clus     t0[us]     len  tgm  label         |x_end|   x_cut  resid")
    for (evt, cid, t0, ln, tgm, lab, _, _, hit), rr in zip(clipped, res):
        print(f"  {evt} {cid:5d} {t0:10.1f} {ln:7.1f}    {tgm}  {lab:<12s} "
              f"{hit[0]:8.2f} {A + v * t0:7.2f} {rr:+6.2f}")

    # Sanity: no cluster may exceed its own cut on either side.
    bad = [c for c in recs if abs(SEED_A + SEED_V * c[2]) < X_FV
           and (c[7] > SEED_A + SEED_V * c[2] + TOL
                or c[6] < -(SEED_A + SEED_V * c[2]) - TOL)]
    print(f"\nenvelope violations (clusters reaching past their own cut): {len(bad)}")

    longnt = [c for c in clipped if c[5] == "not-tagged" and c[3] > 150]
    print(f"long (>150 cm) not-tagged clipped clusters: {len(longnt)}")
    for c in longnt:
        print(f"   evt{c[0]} clus {c[1]}  {c[3]:.0f} cm")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "work-mcp10")
