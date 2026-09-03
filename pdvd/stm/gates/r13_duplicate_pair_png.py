#!/usr/bin/env python3
"""doc pdvd/26 item 4: picture of the near-duplicate segment pair that made
examine_partial_identical_segments loop (round 1) and now survives into the
output (the split is skipped).  For each (event, cluster): the cluster's Bee
"clustering" charge points (grey), the candidate's PR segments from the calib
dump's `candidates` list (coloured, labelled by segment id), its vertices, and
the three points named in the log (V = the vertex, A/B = the two far ends).
Three projections + a zoom on the V->A/B trunk.

Usage: python3 stm/gates/r13_duplicate_pair_png.py <arm_tag> <out_dir>
"""
import glob
import json
import math
import os
import sys
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PDVD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAG = sys.argv[1] if len(sys.argv) > 1 else "d25r13fix"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(PDVD, "docs", "pics")

# (event, cluster, V, A, B) in cm -- from the wct_pr log's do_rough_path triple
CASES = [
    ("039349_14", 36, (273.26, -118.90, 86.61), (196.86, -167.76, 151.48), (196.86, -167.47, 150.98)),
    ("039349_53", 53, (210.54, 112.39, 161.17), (343.22, 139.80, 40.20), (314.19, 134.80, 66.72)),
]


def load(ev):
    d = f"{PDVD}/work/{ev}_{TAG}"
    calib = json.load(open(glob.glob(f"{d}/calib-pr-evt*.json")[0]))
    z = zipfile.ZipFile(f"{d}/mabc-pr.zip")
    bee = json.loads(z.read("data/0/0-clustering-global.json"))
    return calib, bee


def main():
    os.makedirs(OUT, exist_ok=True)
    for ev, cid, V, A, B in CASES:
        calib, bee = load(ev)
        cand = None
        for c in calib.get("candidates", []):
            if any(s.get("cluster_id") == cid for s in c.get("segments", [])):
                cand = c
                break
        segs = [s for s in (cand["segments"] if cand else []) if s.get("cluster_id") == cid]
        verts = [v for v in (cand["vertices"] if cand else []) if v.get("cluster_id") == cid]
        # Charge: Bee "clustering" points within 6 cm of any PR fit point of this
        # cluster (the Bee layer's cluster numbering need not equal the PR id --
        # on 039349/53 id 53 is a different cluster 170 cm away in y).
        segpts = [(p["x"], p["y"], p["z"]) for s in segs for p in s["points"]]
        def near_seg(x, y, zz):
            return any(abs(x - a) < 6 and abs(y - b) < 6 and abs(zz - c) < 6 for a, b, c in segpts[::3])
        cx, cy, cz = [], [], []
        for x, y, zz in zip(bee["x"], bee["y"], bee["z"]):
            if near_seg(x, y, zz):
                cx.append(x); cy.append(y); cz.append(zz)
        # Steiner cloud of the PR cluster (calib dump `steiner`, keyed by the PR id)
        st = [t for t in calib.get("steiner", []) if t.get("cluster_id") == cid]
        sx, sy, sz = (st[0]["x"], st[0]["y"], st[0]["z"]) if st else ([], [], [])
        print(f"{ev} cluster {cid}: {len(cx)} charge points near the fits, {len(sx)} steiner points, candidate found={cand is not None}, {len(segs)} segments, {len(verts)} vertices")

        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        projs = [("z", "y", 0, 1), ("z", "x", 0, 2), ("x", "y", 1, 2)]
        pick = {"x": 0, "y": 1, "z": 2}
        def P(pt, k):
            return pt[pick[k]]
        for ax, (h, v, _, _) in zip(axes.flat[:3], projs):
            ax.scatter([p for p in (cx if h == "x" else cy if h == "y" else cz)],
                       [p for p in (cx if v == "x" else cy if v == "y" else cz)],
                       s=2, c="0.75", label="imaged charge near the fits")
            ax.scatter([p for p in (sx if h == "x" else sy if h == "y" else sz)],
                       [p for p in (sx if v == "x" else sy if v == "y" else sz)],
                       s=3, c="k", label=f"steiner points of cluster {cid}")
            for i, s in enumerate(segs):
                pts = s["points"]
                ax.plot([p[h] for p in pts], [p[v] for p in pts], "-", lw=1.5,
                        label=f"seg {s['id']} L={s['length']:.0f} cm" if i < 12 else None)
            for vv in verts:
                f = vv["fit"]
                fx, fy, fz = (f["x"], f["y"], f["z"]) if isinstance(f, dict) else f[:3]
                ax.plot([{"x": fx, "y": fy, "z": fz}[h]], [{"x": fx, "y": fy, "z": fz}[v]], "k+", ms=8)
            for name, pt, mk in (("V", V, "r*"), ("A", A, "bs"), ("B", B, "g^")):
                ax.plot([P(pt, h)], [P(pt, v)], mk, ms=9, label=name)
            ax.set_xlabel(f"{h} [cm]"); ax.set_ylabel(f"{v} [cm]"); ax.grid(alpha=0.3)
        axes.flat[0].legend(fontsize=7, loc="best")
        # zoom: the V -> A/B trunk in the z-y projection, +-15 cm around the line
        ax = axes.flat[3]
        ax.scatter(cz, cy, s=4, c="0.75")
        ax.scatter(sz, sy, s=6, c="k", label="steiner points")
        for s in segs:
            pts = s["points"]
            ax.plot([p["z"] for p in pts], [p["y"] for p in pts], "-", lw=1.5, label=f"seg {s['id']}")
        for name, pt, mk in (("V", V, "r*"), ("A", A, "bs"), ("B", B, "g^")):
            ax.plot([pt[2]], [pt[1]], mk, ms=10, label=name)
        zlo, zhi = sorted((V[2], A[2])); ylo, yhi = sorted((V[1], A[1]))
        ax.set_xlim(zlo - 15, zhi + 15); ax.set_ylim(ylo - 15, yhi + 15)
        ax.set_xlabel("z [cm]"); ax.set_ylabel("y [cm]"); ax.set_title("zoom: the V -> A/B trunk (z-y)"); ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best")
        fig.suptitle(f"PDVD {ev} cluster {cid} ({TAG}): the near-duplicate pair V->A / V->B (doc pdvd/26 item 4)")
        fig.tight_layout()
        out = os.path.join(OUT, f"doc26_{ev}_cluster{cid}_duplicate_pair.png")
        fig.savefig(out, dpi=110); plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
