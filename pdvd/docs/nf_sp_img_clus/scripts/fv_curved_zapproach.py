#!/usr/bin/env python3
"""Doc pdvd/41 sec 13 -- what decides whether a track REACHES a z wall?

Sec 11.3 showed that most long tracks approaching a z wall in the anode half get
to within 1 cm of it, but ~20 % stop 12-25 cm short, and the owner's Bee scan
found the stop-short ones are through-going by topology.  So the question is not
"is the boundary curved" but "what makes THIS track stop short".

Per long cluster and per z wall: the closest approach, and at that point the
position (x, y), the track direction, and the per-plane LATTICE PHASE RATE
d(pitch)/d(path) -- 0 means the track runs parallel to that plane's strips, the
configuration whose signal is prolonged on one channel and which signal
processing is known to lose.

Repro:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
  python3 docs/nf_sp_img_clus/scripts/fv_curved_zapproach.py --out /home/xqian/tmp/doc41/zapp
"""
import argparse, json, os, re, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from goodpoint_pitch_census import Detector
from fv_curved_longloss import event_points
from fv_curved_map import XW, YW, ZLO, ZHI, WALLS, wall_dist

YSEAM = 168.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdvd", default="/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd")
    ap.add_argument("--tag", default="d41fvoff")
    ap.add_argument("--verdicts", default="/home/xqian/tmp/doc41/ab_verdicts.json")
    ap.add_argument("--minlen", type=float, default=200.0)
    ap.add_argument("--out", default="/home/xqian/tmp/doc41/zapp")
    # doc 43: the 25 cm cap sits on the p90 at the cathode (12 % of z+ bottom
    # approaches are beyond 20 cm); the quantile surface is built at --cap 40.
    ap.add_argument("--cap", type=float, default=25.0)
    a = ap.parse_args()

    G = json.load(open(a.verdicts))["geometry"]
    ev = defaultdict(list)
    for r in G:
        if not r["no_t0"] and r["len_cm"] >= a.minlen:
            ev[(r["run"], r["idx"])].append((r["cid"], r["cat"]))

    rows = []
    for (run, idx), cl in sorted(ev.items()):
        wd = os.path.join(a.pdvd, "work", f"{run}_{idx}_{a.tag}")
        tgz = [f for f in os.listdir(wd) if re.match(r"pctree-evt\d+\.tar\.gz$", f)]
        if not tgz:
            continue
        try:
            D = Detector(os.path.join(wd, tgz[0]))
            E = event_points(wd)
        except Exception as ex:
            print("  skip", wd, ex); continue
        P, cid_all, ph = E["P"], E["cid"], E["phys"]
        for cid, cat in cl:
            m = (cid_all == cid) & ph
            if m.sum() < 20:
                continue
            Q = P[m]
            for w in WALLS:
                d = wall_dist(w, Q[:, 1], Q[:, 2])
                i = int(np.argmin(d))
                if d[i] > a.cap:
                    continue
                p = Q[i]
                dd = np.linalg.norm(Q - p, axis=1)
                near = Q[np.argsort(dd)[:40]]
                if len(near) < 10:
                    continue
                lc = near - near.mean(0)
                ldir = np.linalg.svd(lc, full_matrices=False)[2][0]
                if np.dot(p - near.mean(0), ldir) < 0:
                    ldir = -ldir
                # phase rate at the approach point, from the track's own last points
                _, apa, face, rate = D.measure(near * 10.0)
                rr = np.nanmedian(rate, axis=0)
                nrm = {"y+": np.array([0,1.,0]), "y-": np.array([0,-1.,0]), "z-": np.array([0,0,-1.]), "z+": np.array([0,0,1.])}[w]
                rows.append(dict(run=run, idx=idx, cid=int(cid), cat=cat, wall=w,
                                 dmin=float(d[i]), x=float(p[0]), y=float(p[1]), z=float(p[2]),
                                 half=("cathode" if abs(p[0]) < 170 else "anode"),
                                 cos=float(abs(np.dot(ldir, nrm))),
                                 rate_u=float(rr[0]), rate_v=float(rr[1]), rate_w=float(rr[2]),
                                 rate_min=float(np.nanmin(rr)),
                                 dy_seam=float(abs(abs(p[1]) - YSEAM)),
                                 apa=int(np.bincount(apa).argmax()), face=int(np.bincount(face).argmax()),
                                 npts=int(m.sum())))
    json.dump(rows, open(a.out + "_zapp.json", "w"), indent=1)
    print(f"{len(rows)} (cluster, z wall) approaches within 25 cm\n")

    R = [r for r in rows if r["half"] == "anode"]
    print(f"ANODE HALF ({len(R)} approaches) -- where the mapped inset is ~0-1 cm")
    close = [r for r in R if r["dmin"] < 3]
    far = [r for r in R if r["dmin"] > 8]
    print(f"  reaches the wall (<3 cm): {len(close)}   stops short (>8 cm): {len(far)}")
    for lab, S in (("reaches", close), ("stops short", far)):
        if not S:
            continue
        print(f"  {lab:12s} median |cos| {np.median([r['cos'] for r in S]):.2f}"
              f"  min phase rate {np.median([r['rate_min'] for r in S]):.2f}"
              f"  (frozen <0.15: {100*np.mean([r['rate_min'] < 0.15 for r in S]):4.0f} %)"
              f"  |y| median {np.median([abs(r['y']) for r in S]):5.0f}"
              f"  at the CRU seam (<5 cm): {100*np.mean([r['dy_seam'] < 5 for r in S]):4.0f} %")
    print("\n  stop-short rate vs the minimum phase rate:")
    for lo, hi in ((0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 2)):
        S = [r for r in R if lo <= r["rate_min"] < hi]
        if len(S) < 5:
            continue
        print(f"    min rate {lo:.1f}-{hi:.1f}: n {len(S):4d}   >8 cm short: {100*np.mean([r['dmin']>8 for r in S]):5.1f} %"
              f"   median dmin {np.median([r['dmin'] for r in S]):5.1f} cm")
    print("\n  stop-short rate vs |cos| to the wall normal:")
    for lo, hi in ((0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)):
        S = [r for r in R if lo <= r["cos"] < hi]
        if len(S) < 5:
            continue
        print(f"    |cos| {lo:.1f}-{hi:.1f}: n {len(S):4d}   >8 cm short: {100*np.mean([r['dmin']>8 for r in S]):5.1f} %"
              f"   median dmin {np.median([r['dmin'] for r in S]):5.1f} cm")
    print("\n  stop-short rate vs |y| (the wall's own coordinate):")
    for lo, hi in ((0, 84), (84, 168.5), (168.5, 252), (252, 340)):
        S = [r for r in R if lo <= abs(r["y"]) < hi]
        if len(S) < 5:
            continue
        print(f"    |y| {lo:5.0f}-{hi:5.0f}: n {len(S):4d}   >8 cm short: {100*np.mean([r['dmin']>8 for r in S]):5.1f} %"
              f"   median dmin {np.median([r['dmin'] for r in S]):5.1f} cm")


if __name__ == "__main__":
    main()
