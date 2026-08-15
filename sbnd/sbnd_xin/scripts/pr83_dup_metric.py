#!/usr/bin/env python3
"""pr/83 duplicate-trajectory metric.

Scans PR arms for the two duplicate-trajectory pathologies reported by the
owner on mcp1k events 283040 / 59899 / 72586 (doc pr/83):

  (a) duplicate PAIR: two segments in the same cluster whose fitted
      trajectories lie on top of each other -- fraction of the shorter
      member's points within DUP_TOL of the longer member >= DUP_FRAC and
      chord directions within DUP_ANGLE.
  (b) self-RETRACE: one segment whose path doubles back on itself --
      path/chord >= FOLD_RATIO and the fraction of second-half-by-arclength
      points within DUP_TOL of the first half >= FOLD_FRAC.

Segment identity comes from the calib-pr-evt<ID>.json `segments` list (id,
cluster_id, points).  The same constants as NeutrinoGraphAudit op1 are used
(dup_tol 1.4 cm, dup_frac 0.7, dup_angle 20 deg).

Usage:
  pr83_dup_metric.py <arm> [<arm2> ...] [--events 283040,59899,72586]
                     [--tsv out.tsv] [--min-len 10]

Per arm, walks pr_evt*/calib-pr-evt*.json (or the given events only) and
prints one line per pathology plus a per-arm summary.  Exit code 0 always
(reporting tool, not a gate).
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np

DUP_TOL = 1.4      # cm, NeutrinoGraphAudit mvga_dup_tol
DUP_FRAC = 0.7     # NeutrinoGraphAudit mvga_dup_frac
DUP_ANGLE = 20.0   # deg, NeutrinoGraphAudit mvga_dup_angle
FOLD_RATIO = 5.0   # path/chord threshold for self-retrace
FOLD_FRAC = 0.5    # return-leg overlap threshold


def seg_points(seg):
    return np.array([[p["x"], p["y"], p["z"]] for p in seg.get("points", [])])


def frac_within(A, B, tol):
    """Fraction of points in A within tol of any point in B."""
    if len(A) == 0 or len(B) == 0:
        return 0.0
    n = 0
    for a in A:
        if np.linalg.norm(B - a, axis=1).min() < tol:
            n += 1
    return n / len(A)


def chord_angle_deg(P, Q):
    u = P[-1] - P[0]
    v = Q[-1] - Q[0]
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 90.0
    c = abs(float(np.dot(u, v)) / (nu * nv))
    return math.degrees(math.acos(min(1.0, c)))


def analyze_event(path, min_len):
    """Return (dup_pairs, retraces) lists for one calib json."""
    d = json.load(open(path))
    segs = [s for s in d.get("segments", [])
            if isinstance(s, dict) and len(s.get("points", [])) >= 2]
    by_cluster = {}
    for s in segs:
        by_cluster.setdefault(s.get("cluster_id"), []).append(s)

    dup_pairs, retraces = [], []
    for cid, ss in sorted(by_cluster.items(), key=lambda t: (t[0] is None, t[0])):
        pts = {s["id"]: seg_points(s) for s in ss}
        lens = {}
        for s in ss:
            P = pts[s["id"]]
            lens[s["id"]] = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
        ids = sorted(pts)
        # (a) duplicate pairs
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                if min(lens[a], lens[b]) < min_len:
                    continue
                short, lng = (a, b) if lens[a] <= lens[b] else (b, a)
                f = frac_within(pts[short], pts[lng], DUP_TOL)
                if f < DUP_FRAC:
                    continue
                ang = chord_angle_deg(pts[short], pts[lng])
                if ang < DUP_ANGLE or ang > 180 - DUP_ANGLE:
                    dup_pairs.append(dict(cluster=cid, seg_a=short, seg_b=lng,
                                          len_a=lens[short], len_b=lens[lng],
                                          overlap=f, angle=ang))
        # (b) self-retrace
        for s in ss:
            sid = s["id"]
            P = pts[sid]
            L = lens[sid]
            if L < min_len:
                continue
            C = float(np.linalg.norm(P[-1] - P[0]))
            ratio = L / max(C, 1e-6)
            if ratio < FOLD_RATIO:
                continue
            al = np.concatenate([[0], np.cumsum(
                np.linalg.norm(np.diff(P, axis=0), axis=1))])
            half = np.searchsorted(al, al[-1] / 2)
            f = frac_within(P[half:], P[:half], DUP_TOL)
            # 'retrace': return leg lies on the outbound leg (59899 seg 5005).
            # 'fold': path/chord grossly pathological even without tight
            # retrace overlap -- e.g. 72586 seg 17004 whose return leg rides
            # ~5 cm off the outbound via a ghost bridge.  A genuine track
            # never has L/C >= 5 at L >= 20 cm.
            kind = "retrace" if f >= FOLD_FRAC else ("fold" if L >= 20.0 else None)
            if kind:
                retraces.append(dict(cluster=cid, seg=sid, path=L,
                                     chord=C, ratio=ratio, overlap=f,
                                     kind=kind))
    return dup_pairs, retraces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--events", default=None,
                    help="comma-separated event ids; default: all pr_evt*")
    ap.add_argument("--tsv", default=None)
    ap.add_argument("--min-len", type=float, default=10.0,
                    help="ignore segments shorter than this (cm)")
    args = ap.parse_args()

    rows = []
    for arm in args.arms:
        if args.events:
            paths = [os.path.join(arm, f"pr_evt{e}", f"calib-pr-evt{e}.json")
                     for e in args.events.split(",")]
        else:
            paths = sorted(glob.glob(os.path.join(arm, "pr_evt*", "calib-pr-evt*.json")))
        n_dup_evt, n_ret_evt, n_evt = 0, 0, 0
        for p in paths:
            if not os.path.exists(p):
                print(f"MISSING {p}", file=sys.stderr)
                continue
            evt = os.path.basename(os.path.dirname(p)).replace("pr_evt", "")
            n_evt += 1
            dups, rets = analyze_event(p, args.min_len)
            if dups:
                n_dup_evt += 1
            if rets:
                n_ret_evt += 1
            for x in dups:
                print(f"{arm} evt {evt} DUP cluster {x['cluster']} "
                      f"segs {x['seg_a']}/{x['seg_b']} "
                      f"len {x['len_a']:.1f}/{x['len_b']:.1f} cm "
                      f"overlap {x['overlap']:.2f} angle {x['angle']:.1f}")
                rows.append([arm, evt, "dup", x["cluster"],
                             f"{x['seg_a']}/{x['seg_b']}",
                             f"{x['overlap']:.3f}", f"{x['angle']:.1f}",
                             f"{x['len_a']:.1f}", f"{x['len_b']:.1f}"])
            for x in rets:
                print(f"{arm} evt {evt} {x['kind'].upper()} cluster {x['cluster']} "
                      f"seg {x['seg']} path {x['path']:.1f} chord {x['chord']:.1f} "
                      f"ratio {x['ratio']:.1f} overlap {x['overlap']:.2f}")
                rows.append([arm, evt, x["kind"], x["cluster"], str(x["seg"]),
                             f"{x['overlap']:.3f}", f"{x['ratio']:.1f}",
                             f"{x['path']:.1f}", f"{x['chord']:.1f}"])
        print(f"# {arm}: {n_evt} events scanned, "
              f"{n_dup_evt} with dup pairs, {n_ret_evt} with self-retrace")
    if args.tsv:
        with open(args.tsv, "w") as f:
            f.write("arm\tevent\tkind\tcluster\tsegs\toverlap\tangle_or_ratio\tlen_a_or_path\tlen_b_or_chord\n")
            for r in rows:
                f.write("\t".join(str(x) for x in r) + "\n")
        print(f"# wrote {args.tsv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
