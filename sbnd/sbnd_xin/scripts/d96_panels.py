#!/usr/bin/env python3
"""doc 96 -- projection panels of the three owner-flagged doc-95 mains.

Left: z-y (the view the over-clustering is visible in).  Right: z-x (drift).
Points are the final production Q/L clustering layer; the in-beam main is drawn
in colour, everything else in the event in light grey for context.  For a main
that the flash merge built out of several pre-merge clusters (n_frag > 1, from
the per-point real_cluster_id the Bee clustering layer carries) each pre-merge
component gets its own colour -- that is the partition ClusteringUnmergeBundle
restores at the head of the PR chain.

Repro:  python3 scripts/d96_panels.py
"""
import json
import os
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "work-dbg25a-ql"
OUT = "docs/96_sep"
os.makedirs(OUT, exist_ok=True)
CASES = [("272-2-30", 30, 5, "TGM"), ("105-23-21", 21, 15, "TGM"), ("105-23-5", 5, 18, "STM")]
COL = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]

for rse, evt, mid, verdict in CASES:
    with zipfile.ZipFile(f"{ROOT}/ql_evt{evt}/mabc-all-apa.zip") as z:
        d = json.loads(z.read("data/0/0-clustering-global.json"))
    cid = np.array(d["cluster_id"], int)
    rid = np.array(d["real_cluster_id"], int)
    P = np.c_[d["x"], d["y"], d["z"]].astype(float)
    m = cid == mid
    comps = sorted(set(rid[m].tolist()), key=lambda r: -(m & (rid == r)).sum())

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))
    for a, (i, j, xl, yl) in zip(ax, [(2, 1, "z [cm]", "y [cm]"), (2, 0, "z [cm]", "x [cm]")]):
        a.scatter(P[~m, i], P[~m, j], s=0.4, c="#cccccc", lw=0, rasterized=True)
        for k, r in enumerate(comps):
            s = m & (rid == r)
            a.scatter(P[s, i], P[s, j], s=0.6, c=COL[k % len(COL)], lw=0, rasterized=True,
                      label=f"pre-merge {r} ({s.sum()} pts)")
        a.set_xlabel(xl); a.set_ylabel(yl); a.grid(alpha=.25, lw=.4)
    ax[1].axhline(0, color="k", lw=.8, ls="--")            # cathode
    ax[0].legend(markerscale=12, fontsize=8, loc="best")
    fig.suptitle(f"{rse}  in-beam main cid={mid}  ({m.sum()} pts, {verdict})  "
                 f"-- grey = rest of the event", fontsize=11)
    fig.tight_layout()
    f = f"{OUT}/d96-{rse}.png"
    fig.savefig(f, dpi=130)
    plt.close(fig)
    print("wrote", f)
