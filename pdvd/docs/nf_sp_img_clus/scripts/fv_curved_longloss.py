#!/usr/bin/env python3
"""Doc pdvd/41 sec 11 -- what ARE the long TGM losses of sec 10?

For every cluster the curved fiducial drops from TGM (and a control: the ones it
keeps), find the DECIDING end -- the one farther from every boundary surface, the
end that makes or breaks a both-ends-at-a-boundary verdict -- and ask three
questions of it:

  Q1 readout.  Is it at the readout-window edge?  The raw readout x of a point is
      the pctree 3d/x (mm); per drift side the window runs from the anode at tick
      0 (|x_raw| = 341.55 cm) to the late edge at 398.52 cm.  An out-of-time
      cosmic whose charge arrives after the window closes is TRUNCATED there and
      its true end was never recorded -- doc pdvd/25 M5.  The tick-0 edge is not
      truncation: it IS the anode plane.

  Q2 direction.  How far is the wall ALONG the track?  A perpendicular gap of
      10 cm on a track running nearly parallel to the wall is a metre of path,
      i.e. an end that is not "just short of exiting" at all.

  Q3 charge beyond.  Walk the local track direction from the end to the wall and
      count imaged points of OTHER clusters inside a cylinder around it.  Charge
      there means the track DID reach the wall and the end was lost to
      clustering; no charge means the charge itself stops.

Repro:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
  python3 docs/nf_sp_img_clus/scripts/fv_curved_longloss.py \
      /home/xqian/tmp/doc41/ab_verdicts.json --tag d41fvoff --minlen 200 \
      --out /home/xqian/tmp/doc41/longloss
"""
import argparse, io, json, os, re, sys, tarfile, zipfile
from collections import Counter, defaultdict

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fv_curved_load import bee_points, load_pct, read_tlas
from fv_curved_map import XW, YW, ZLO, ZHI, CATH, WALLS, wall_dist

RAW_LATE, RAW_EARLY = 398.52, 341.55      # cm, per side (doc 41 sec 2)
YSEAM, ZSEAM = 168.50, 149.65
CYL_R = 5.0                               # cm, radius of the "charge beyond" cylinder


def surfaces(p):
    """distance from a point to each boundary surface of the tagger volume."""
    d = {w: float(wall_dist(w, p[1], p[2])) for w in WALLS}
    d["anode"] = float(XW - abs(p[0]))
    return d


def event_points(workdir):
    """Bee points (t0-corrected cm) + the matching RAW readout x and drift side."""
    P, Q, C, run, evt = bee_points(workdir)
    tgz = [f for f in os.listdir(workdir) if re.match(r"pctree-evt\d+\.tar\.gz$", f)]
    tgz = os.path.join(workdir, tgz[0])
    d = load_pct(tgz, lambda p: "/namedpcs/3d/arrays/" in p
                 and p.rsplit("/", 1)[-1] in ("x", "x_t0cor", "y", "z", "wpid"))
    g = lambda n: [v for k, v in d.items() if k.endswith("/arrays/" + n)][0]
    xraw, xcor, py, pz = g("x") / 10.0, g("x_t0cor") / 10.0, g("y") / 10.0, g("z") / 10.0
    side = np.where((g("wpid") >> 4) < 4, -1, 1).astype(np.int8)
    phys = np.abs(P[:, 0]) < 1e4
    pp = np.abs(xcor) < 1e4
    tree = cKDTree(np.column_stack([xcor[pp], py[pp], pz[pp]]))
    dist, idx = tree.query(P[phys], k=1)
    xr = np.full(len(P), np.nan); sd = np.zeros(len(P), np.int8)
    xr[phys] = xraw[pp][idx]; sd[phys] = side[pp][idx]
    return dict(P=P, q=Q, cid=C, xraw=xr, side=sd, phys=phys,
                match_max_cm=float(dist.max()) if len(dist) else 0.0)


def analyse_cluster(E, cid, minpts=5):
    m = (E["cid"] == cid) & E["phys"]
    if m.sum() < minpts:
        return None
    Q = E["P"][m]
    c = Q - Q.mean(0)
    axis = np.linalg.svd(c, full_matrices=False)[2][0]
    t = c @ axis
    ends = [Q[int(np.argmax(t))], Q[int(np.argmin(t))]]
    eidx = [int(np.flatnonzero(m)[int(np.argmax(t))]), int(np.flatnonzero(m)[int(np.argmin(t))])]
    ds = [surfaces(e) for e in ends]
    dmin = [min(d.values()) for d in ds]
    iw = int(np.argmax(dmin))                       # the deciding end
    e, dsur, gi = ends[iw], ds[iw], eidx[iw]
    wall = min(dsur, key=dsur.get)

    # --- Q1: the readout window, in the RAW frame
    xr = float(E["xraw"][gi]); sg = int(E["side"][gi])
    late = RAW_LATE if sg < 0 else -RAW_LATE        # the truncating edge
    early = -RAW_EARLY if sg < 0 else RAW_EARLY     # tick 0 == the anode plane
    d_late, d_early = abs(xr - late), abs(xr - early)

    # --- local direction at the deciding end, pointing OUTWARD
    k = min(40, m.sum())
    dd = np.linalg.norm(Q - e, axis=1)
    near = Q[np.argsort(dd)[:k]]
    lc = near - near.mean(0)
    ldir = np.linalg.svd(lc, full_matrices=False)[2][0]
    if np.dot(e - near.mean(0), ldir) < 0:
        ldir = -ldir

    # --- Q2: how far is the surface ALONG the track?
    steps = np.arange(0.0, 400.0, 1.0)
    pathlen = np.nan
    for s in steps[1:]:
        p = e + ldir * s
        if (abs(p[0]) > XW or abs(p[1]) > YW or p[2] < ZLO or p[2] > ZHI):
            pathlen = float(s); break

    # --- Q3: charge of OTHER clusters along that extrapolation
    other = E["phys"] & (E["cid"] != cid)
    O = E["P"][other]
    nbeyond = 0; first_gap = np.nan
    if len(O) and np.isfinite(pathlen):
        v = O - e
        s = v @ ldir
        perp = np.linalg.norm(v - np.outer(s, ldir), axis=1)
        sel = (s > 1.0) & (s < pathlen) & (perp < CYL_R)
        nbeyond = int(sel.sum())
        if nbeyond:
            first_gap = float(np.min(s[sel]))
    return dict(cid=int(cid), n=int(m.sum()), len_cm=round(float(t.max() - t.min()), 1),
                wall=wall, wall_dist_cm=round(dsur[wall], 2),
                other_end_cm=round(dmin[1 - iw], 2),
                x_cm=round(float(e[0]), 1), y_cm=round(float(e[1]), 1), z_cm=round(float(e[2]), 1),
                side=sg, xraw_cm=round(xr, 1),
                d_late_cm=round(d_late, 1), d_early_cm=round(d_early, 1),
                cos_to_wall=round(float(abs(np.dot(ldir, {"y+": [0, 1, 0], "y-": [0, -1, 0],
                                                          "z-": [0, 0, -1], "z+": [0, 0, 1],
                                                          "anode": [np.sign(e[0]), 0, 0]}[wall]))), 3),
                path_to_wall_cm=None if not np.isfinite(pathlen) else round(pathlen, 1),
                n_other_beyond=nbeyond,
                first_other_gap_cm=None if not np.isfinite(first_gap) else round(first_gap, 1),
                near_seam=bool(abs(abs(e[1]) - YSEAM) < 3 or abs(e[1]) < 3 or abs(e[2] - ZSEAM) < 3),
                near_cathode=bool(abs(e[0]) < 10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("verdicts"); ap.add_argument("--tag", default="d41fvoff")
    ap.add_argument("--pdvd", default="/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd")
    ap.add_argument("--minlen", type=float, default=200.0)
    ap.add_argument("--out", default="/home/xqian/tmp/doc41/longloss")
    a = ap.parse_args()

    R = json.load(open(a.verdicts))
    G = {(r["run"], r["idx"], r["cid"]): r for r in R["geometry"]}
    want = defaultdict(dict)
    for k, r in G.items():
        if r["no_t0"] or r["len_cm"] < a.minlen:
            continue
        if r["cat"] in ("tgm_lost", "tgm_kept"):
            want[(k[0], k[1])][k[2]] = r["cat"]
    print(f"events to open: {len(want)}; clusters: "
          f"{sum(len(v) for v in want.values())} "
          f"(lost {sum(1 for v in want.values() for c in v.values() if c=='tgm_lost')})")

    rows = []
    for (run, idx), cids in sorted(want.items()):
        wd = os.path.join(a.pdvd, "work", f"{run}_{idx}_{a.tag}")
        try:
            E = event_points(wd)
        except Exception as ex:
            print("  skip", wd, ex); continue
        for cid, cat in cids.items():
            r = analyse_cluster(E, cid)
            if r is None:
                continue
            r.update(run=run, idx=idx, cat=cat, match_max_cm=E["match_max_cm"])
            rows.append(r)
    json.dump(rows, open(a.out + "_ends.json", "w"), indent=1)
    print("wrote", a.out + "_ends.json", len(rows), "rows")

    for cat in ("tgm_lost", "tgm_kept"):
        R2 = [r for r in rows if r["cat"] == cat]
        if not R2:
            continue
        n = len(R2)
        dl = np.array([r["d_late_cm"] for r in R2])
        de = np.array([r["d_early_cm"] for r in R2])
        wd = np.array([r["wall_dist_cm"] for r in R2])
        pl = np.array([r["path_to_wall_cm"] if r["path_to_wall_cm"] is not None else np.nan for r in R2])
        cw = np.array([r["cos_to_wall"] for r in R2])
        nb = np.array([r["n_other_beyond"] for r in R2])
        print(f"\n=== {cat}: {n} clusters longer than {a.minlen:.0f} cm ===")
        print(f"  deciding end, perpendicular distance to its surface: median {np.median(wd):.1f} cm")
        print(f"  Q1 at the LATE readout edge (<5 cm):  {int((dl < 5).sum()):4d}  ({100*(dl<5).mean():.0f} %)"
              f"   <10 cm: {int((dl<10).sum()):4d}")
        print(f"     at the tick-0 edge = the anode plane: {int((de < 5).sum()):4d}")
        print(f"  Q2 path along the track to the wall:  median {np.nanmedian(pl):.0f} cm;"
              f"  >100 cm: {int(np.nansum(pl > 100)):4d} ({100*np.nanmean(pl>100):.0f} %)")
        print(f"     |cos| between the track and the wall normal: median {np.median(cw):.2f};"
              f"  <0.2 (running along the wall): {int((cw<0.2).sum())} ({100*(cw<0.2).mean():.0f} %)")
        print(f"  Q3 other-cluster charge in the {CYL_R:.0f} cm cylinder out to the wall:"
              f"  {int((nb>0).sum())} of {n} ({100*(nb>0).mean():.0f} %), median {np.median(nb):.0f} points")
        print(f"     near a CRP/CRU seam: {sum(1 for r in R2 if r['near_seam'])};"
              f"  within 10 cm of the cathode: {sum(1 for r in R2 if r['near_cathode'])}")
        print("  deciding surface:", Counter(r["wall"] for r in R2).most_common())


if __name__ == "__main__":
    main()
