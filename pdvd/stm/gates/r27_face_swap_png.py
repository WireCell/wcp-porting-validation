#!/usr/bin/env python3
"""doc pdvd/27: 039349/53 -- the same cluster under a stale (v6-sampled) point tree
vs a regenerated (v7-uvwfit) one.  Left: the d25r13fix arm: the input cluster's
own points (Bee 'clustering' layer, grey) and the PR steiner cloud of the same
cluster (black), one face height apart in y.  Right: d27v7 -- both on top of
each other, no hole, no chords.
usage: r27_face_swap_png.py [stale_tag fresh_tag]   (default d25r13fix d27v7)"""
import json, glob, zipfile, sys, os, numpy as np, collections
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
PDVD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EV = "039349_53"; STALE, FRESH = (sys.argv[1:3] if len(sys.argv) > 2 else ("d25r13fix", "d27v7"))
BOX = (200, 350, 215, 340, 20, 175)   # x0 x1 y0 y1 z0 z1 (cm): the A->V track in the input frame
OUT = f"{PDVD}/docs/pics/doc27_{EV}_face_swap.png"

def load(tag):
    d = f"{PDVD}/work/{EV}_{tag}"
    bee = json.loads(zipfile.ZipFile(f"{d}/mabc-pr.zip").read("data/0/0-clustering-global.json"))
    calib = json.load(open(glob.glob(f"{d}/calib-pr-evt*.json")[0]))
    bx, by, bz, bc = map(np.array, (bee["x"], bee["y"], bee["z"], bee["cluster_id"]))
    x0, x1, y0, y1, z0, z1 = BOX
    sel = (bx > x0) & (bx < x1) & (by > y0) & (by < y1) & (bz > z0) & (bz < z1)
    bid = collections.Counter(bc[sel].tolist()).most_common(1)[0][0]
    m = bc == bid
    # the PR steiner cloud that belongs to this input cluster: best match after trying y shifts of 0 / +-1 face
    tree = cKDTree(np.c_[bx[m], by[m], bz[m]]); best = None
    for t in calib["steiner"]:
        sx, sy, sz = map(np.array, (t["x"], t["y"], t["z"]))
        for dy in (0.0, 168.4, -168.4):
            n = int((tree.query(np.c_[sx, sy + dy, sz])[0] < 3).sum())
            if best is None or n > best[0]: best = (n, dy, t["cluster_id"], sx, sy, sz)
    n, dy, cid, sx, sy, sz = best
    segs = []
    for c in calib.get("candidates", []):
        for s in c.get("segments", []):
            if s.get("cluster_id") == cid: segs.append(s)
    return dict(bid=bid, bx=bx[m], by=by[m], bz=bz[m], cid=cid, sx=sx, sy=sy, sz=sz, dy=dy, nmatch=n, segs=segs)

S, F = load(STALE), load(FRESH)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for col, (D, tag) in enumerate(((S, STALE), (F, FRESH))):
    for row, (h, v) in enumerate((("z", "y"), ("z", "x"))):
        ax = axes[row, col]
        P = {"x": (D["bx"], D["sx"]), "y": (D["by"], D["sy"]), "z": (D["bz"], D["sz"])}
        ax.scatter(P[h][0], P[v][0], s=3, c="0.75", label=f"input cluster {D['bid']} (Bee clustering layer, {len(D['bx'])} pts)")
        ax.scatter(P[h][1], P[v][1], s=4, c="k", label=f"PR steiner cloud of cluster {D['cid']} ({len(D['sx'])} pts)")
        for s in D["segs"]:
            pts = s["points"]; xs = [p[h] for p in pts]; ys = [p[v] for p in pts]
            ax.plot(xs, ys, lw=1.2, label=f"seg {s.get('segment_id', s.get('id'))}")
        ax.set_xlabel(f"{h} [cm]"); ax.set_ylabel(f"{v} [cm]"); ax.grid(alpha=.3)
        if row == 0:
            shift = "" if D["dy"] == 0 else f" -- steiner cloud sits {abs(D['dy']):.0f} cm ({'+' if D['dy']<0 else '-'}y) from its own points"
            ax.set_title(f"{tag}: {D['nmatch']}/{len(D['sx'])} steiner pts on the cluster's charge{shift}", fontsize=10)
            ax.legend(fontsize=7, loc="best")
fig.suptitle(f"PDVD {EV}: input cluster vs PR steiner cloud -- stale v6 point tree under a v7-uvwfit PR job (left) vs regenerated (right)  [doc pdvd/27]")
fig.tight_layout(); fig.savefig(OUT, dpi=110)
print(OUT); print("stale:", S["bid"], S["cid"], S["dy"], S["nmatch"], len(S["sx"]), "segs", len(S["segs"])); print("fresh:", F["bid"], F["cid"], F["dy"], F["nmatch"], len(F["sx"]), "segs", len(F["segs"]))
