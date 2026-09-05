#!/usr/bin/env python3
"""Doc pdvd/41 sec 12 -- is the short end an INDUCTION-plane signal-processing loss?

A 3-D imaged point needs charge in all three planes.  If the induction (U, V)
signal processing fails near a wall while the collection plane (W) keeps its
signal, the 3-D cloud stops even though the charge does not -- and a fiducial
surface mapped from 3-D points would be measuring the SP failure, not the
detector.  The ctpc carries the per-plane charge map, so the question is
answerable directly: walk the track's own direction BEYOND its last 3-D point
and ask each plane separately whether there is charge there.

  in-track control : the same query on the track's interior, where all three
                     planes must answer yes by construction;
  null control     : the same probe displaced 25 cm perpendicular to the track,
                     which measures how often an unrelated track's charge lands
                     on the projection by accident.

Uses the Detector replica of goodpoint_pitch_census.py (doc 32 round 2): the
ctpc is an exact (drift-step x wire-pitch) lattice and the projection is refit
from the dump's own 2dp{p}_y arrays, so this is the code's own query, offline.

Repro:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
  python3 docs/nf_sp_img_clus/scripts/fv_curved_planes.py \
      --sel /home/xqian/tmp/doc41/class5.json --out /home/xqian/tmp/doc41/planes
"""
import argparse, json, os, re, sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from goodpoint_pitch_census import Detector
from fv_curved_longloss import event_points, surfaces
from fv_curved_map import XW, YW, ZLO, ZHI, WALLS, wall_dist

PLANES = ("U", "V", "W")


def probe_points(end, ldir, upto, step=1.0):
    s = np.arange(step, upto + 1e-9, step)
    return end[None, :] + np.outer(s, ldir), s


def perp(ldir):
    a = np.array([1.0, 0, 0]) if abs(ldir[0]) < 0.9 else np.array([0, 1.0, 0])
    v = np.cross(ldir, a)
    return v / np.linalg.norm(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sel", required=True, help="json list of [run, idx, cid]")
    ap.add_argument("--ends", default="/home/xqian/tmp/doc41/longloss_ends.json")
    ap.add_argument("--pdvd", default="/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd")
    ap.add_argument("--tag", default="d41fvoff")
    ap.add_argument("--thr", type=float, default=10.0, help="ctpc hit threshold, mm")
    ap.add_argument("--reach", type=float, default=30.0, help="cm to probe beyond the end")
    ap.add_argument("--out", default="/home/xqian/tmp/doc41/planes")
    a = ap.parse_args()

    ends = {(r["run"], r["idx"], r["cid"]): r for r in json.load(open(a.ends))}
    sel = [tuple(x) for x in json.load(open(a.sel))]
    byev = defaultdict(list)
    for run, idx, cid in sel:
        byev[(run, idx)].append(cid)

    rows = []
    for (run, idx), cids in sorted(byev.items()):
        wd = os.path.join(a.pdvd, "work", f"{run}_{idx}_{a.tag}")
        tgz = [f for f in os.listdir(wd) if re.match(r"pctree-evt\d+\.tar\.gz$", f)]
        if not tgz:
            print("  no pctree in", wd); continue
        try:
            D = Detector(os.path.join(wd, tgz[0]))
            E = event_points(wd)
        except Exception as ex:
            print("  skip", wd, ex); continue
        P, cid_all, ph = E["P"], E["cid"], E["phys"]
        for cid in cids:
            m = (cid_all == cid) & ph
            if m.sum() < 20:
                continue
            Q = P[m]
            r = ends.get((run, idx, cid))
            if r is None:
                continue
            end = np.array([r["x_cm"], r["y_cm"], r["z_cm"]])
            dd = np.linalg.norm(Q - end, axis=1)
            near = Q[np.argsort(dd)[:40]]
            lc = near - near.mean(0)
            ldir = np.linalg.svd(lc, full_matrices=False)[2][0]
            if np.dot(end - near.mean(0), ldir) < 0:
                ldir = -ldir
            reach = min(a.reach, r["path_to_wall_cm"] or a.reach)
            if reach < 3:
                continue
            out, s = probe_points(end, ldir, reach)
            inn = Q[np.argsort(dd)[:int(min(30, m.sum()))]]          # the last ~30 cm of track
            nul = out + perp(ldir) * 25.0
            res = {}
            for nm, pts in (("beyond", out), ("intrack", inn), ("null", nul)):
                keep = ((np.abs(pts[:, 0]) < XW) & (np.abs(pts[:, 1]) < YW)
                        & (pts[:, 2] > ZLO) & (pts[:, 2] < ZHI))
                if keep.sum() == 0:
                    res[nm] = None; continue
                dist, _, _, _ = D.measure(pts[keep] * 10.0)          # mm
                hit = dist < a.thr
                res[nm] = dict(n=int(keep.sum()),
                               frac=[float(np.nanmean(hit[:, p])) for p in range(3)],
                               w_only=float(np.nanmean(hit[:, 2] & ~(hit[:, 0] & hit[:, 1]))),
                               all3=float(np.nanmean(hit.all(axis=1))))
            rows.append(dict(run=run, idx=idx, cid=int(cid), len_cm=r["len_cm"],
                             wall=r["wall"], gap_cm=r["wall_dist_cm"], reach_cm=float(reach),
                             **{f"{k}_{kk}": (None if res[k] is None else res[k][kk])
                                for k in ("beyond", "intrack", "null") for kk in ("n", "frac", "w_only", "all3")}))
            b = res["beyond"]; c = res["intrack"]; n0 = res["null"]
            print(f"  {run}/{idx} c{cid:<4d} {r['len_cm']:5.0f} cm  {r['wall']:>5s} +{r['wall_dist_cm']:5.1f} "
                  f"| beyond U {b['frac'][0]:.2f} V {b['frac'][1]:.2f} W {b['frac'][2]:.2f} all3 {b['all3']:.2f} "
                  f"| in-track {c['frac'][0]:.2f}/{c['frac'][1]:.2f}/{c['frac'][2]:.2f} "
                  f"| null W {n0['frac'][2]:.2f}" if b and c and n0 else f"  {run}/{idx} c{cid}: incomplete")
    json.dump(rows, open(a.out + "_planes.json", "w"), indent=1)

    def agg(key):
        v = np.array([r[f"{key}_frac"] for r in rows if r[f"{key}_frac"] is not None], float)
        return v.mean(axis=0), np.array([r[f"{key}_all3"] for r in rows if r[f"{key}_all3"] is not None]).mean()
    print(f"\n{len(rows)} clusters, ctpc hit threshold {a.thr:.0f} mm, probing up to {a.reach:.0f} cm beyond the end")
    for key in ("intrack", "beyond", "null"):
        f, a3 = agg(key)
        print(f"  {key:8s} mean per-plane occupancy  U {f[0]:.3f}  V {f[1]:.3f}  W {f[2]:.3f}   all three {a3:.3f}")
    print("wrote", a.out + "_planes.json")


if __name__ == "__main__":
    main()
