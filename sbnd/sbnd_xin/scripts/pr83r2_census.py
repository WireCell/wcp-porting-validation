#!/usr/bin/env python3
"""pr/83 round 2 census -- stacked duplicate trajectories, extracted straight from
`mabc-pr.zip` (no calib-pr-evt*.json, no reruns).

Doc 83 (2026-08-15) root-caused a duplicate/out-and-back trajectory pathology to
`break_segment`'s unoriented-edge slicing and shipped `break_seg_orient` (SBND
production ON).  This script re-runs the SAME duplicate-pair metric as
`pr83_dup_metric.py`, but sourced from the Bee display payload
(`data/0/0-track_fit-global.json` inside `mabc-pr.zip`) instead of
`calib-pr-evt<ID>.json`, because no surviving >=200-event PR arm still carries the
calib json (both retired and superseded by pr_display-less runs -- see
sbnd_xin/docs/work-tags.md and the 2026-08-17 cross-session note).

Validated equivalence (2026-08-17): on the 9 events that still carry BOTH files
(`work-r1qlmc-prod0813`, `work-r2mc-prod0813`), the Bee `real_cluster_id`-keyed
segment set is IDENTICAL to the calib `segments` list -- same ids, same point
counts, 0 bee-only, 0 calib-only, 0 length mismatch > 0.05 cm; feeding the Bee
segments through `pr83_dup_metric.py`'s own `analyze_event()` (reformatted as a
fake calib json) reproduces this script's findings exactly.

Adds one thing `pr83_dup_metric.py` cannot: a crosstab against each event's
`wct_pr_evt<ID>.log` for `mvga: op3 ... carried>=2` on the SAME cluster --
splitting findings into "class A" (co-located with an mvga interposed-splice
carry >= 2, i.e. new since pr/85+86) vs "class B" (no such carry -- either the
pre-existing short near-parallel prong class doc 83 sec 6.2 deferred, or
something else, left un-diagnosed here).

Usage:
  pr83r2_census.py <arm> [<arm2> ...] [--tsv out.tsv]
      [--min-len 10] [--dup-tol 1.4] [--dup-frac 0.7] [--dup-angle 20]

Exit code 0 always (reporting tool, not a gate).
"""
import argparse
import glob
import json
import math
import os
import re
import sys
import zipfile

import numpy as np

CARRY_RE = re.compile(
    r"mvga: op3 (stub-interposed|created-splice) cluster=(\d+)[^\n]*?carried=(\d+)")


def bee_segments(zip_path):
    """{real_cluster_id: (points[N,3], cluster_id)} from mabc-pr.zip's track_fit layer."""
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith("track_fit-global.json")]
        if not names:
            return None
        d = json.loads(z.read(names[0]))
    P = np.array([d["x"], d["y"], d["z"]], dtype=float).T
    rid = np.asarray(d["real_cluster_id"])
    cid = np.asarray(d["cluster_id"])
    out = {}
    for r in set(rid.tolist()):
        if r < 0:  # vertices-layer sentinel, not a segment (see doc 83 r2 sec)
            continue
        m = rid == r
        if m.sum() < 2:
            continue
        out[int(r)] = (P[m], int(cid[m][0]))
    return out


def seg_len(P):
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


def frac_within(A, B, tol):
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


def dup_pairs_for_event(segs, min_len, dup_tol, dup_frac, dup_angle):
    """segs: {real_cluster_id: (points, cluster_id)}. Returns list of finding dicts,
    one per (cluster, dup-pair-member-set) -- callers group by cluster."""
    by_cluster = {}
    for rid, (pts, cid) in segs.items():
        by_cluster.setdefault(cid, []).append((rid, pts))

    findings = []
    for cid, members in sorted(by_cluster.items()):
        ids = sorted(r for r, _ in members)
        pts = {r: p for r, p in members}
        lens = {r: seg_len(p) for r, p in pts.items()}
        pairs = []
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                if min(lens[a], lens[b]) < min_len:
                    continue
                s, l = (a, b) if lens[a] <= lens[b] else (b, a)
                f = frac_within(pts[s], pts[l], dup_tol)
                if f < dup_frac:
                    continue
                ang = chord_angle_deg(pts[s], pts[l])
                if ang < dup_angle or ang > 180 - dup_angle:
                    pairs.append(dict(seg_a=s, seg_b=l, overlap=f, angle=ang,
                                       len_a=lens[s], len_b=lens[l]))
        if pairs:
            members_involved = sorted({p["seg_a"] for p in pairs} | {p["seg_b"] for p in pairs})
            findings.append(dict(cluster=cid, pairs=pairs, members=members_involved,
                                  n_pairs=len(pairs),
                                  sum_len=sum(lens[m] for m in members_involved),
                                  max_len=max(lens[m] for m in members_involved)))
    return findings


def carry_sites(log_path):
    """set of (cluster_id) with an op3 stub-interposed/created-splice carried>=2."""
    if not os.path.exists(log_path):
        return set()
    txt = open(log_path, "rb").read().decode("utf-8", "replace")
    return {int(m.group(2)) for m in CARRY_RE.finditer(txt) if int(m.group(3)) >= 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--tsv", default=None)
    ap.add_argument("--min-len", type=float, default=10.0)
    ap.add_argument("--dup-tol", type=float, default=1.4)
    ap.add_argument("--dup-frac", type=float, default=0.7)
    ap.add_argument("--dup-angle", type=float, default=20.0)
    args = ap.parse_args()

    rows = []
    n_classA = n_classB = 0
    for arm in args.arms:
        zips = sorted(glob.glob(os.path.join(arm, "pr_evt*", "mabc-pr.zip")))
        n_evt = 0
        n_dup_evt = 0
        for zp in zips:
            evtdir = os.path.dirname(zp)
            evt = os.path.basename(evtdir).replace("pr_evt", "")
            segs = bee_segments(zp)
            if segs is None:
                print(f"MISSING track_fit layer: {zp}", file=sys.stderr)
                continue
            n_evt += 1
            findings = dup_pairs_for_event(segs, args.min_len, args.dup_tol,
                                            args.dup_frac, args.dup_angle)
            if not findings:
                continue
            n_dup_evt += 1
            carried = carry_sites(os.path.join(evtdir, f"wct_pr_evt{evt}.log"))
            for f in findings:
                klass = "A" if f["cluster"] in carried else "B"
                if klass == "A":
                    n_classA += 1
                else:
                    n_classB += 1
                print(f"{arm} evt {evt} clus {f['cluster']} class {klass}: "
                      f"{f['n_pairs']} dup pairs, {len(f['members'])} segs, "
                      f"sum_len {f['sum_len']:.0f} cm, longest {f['max_len']:.0f} cm")
                rows.append([arm, evt, str(f["cluster"]), klass,
                             str(f["n_pairs"]), str(len(f["members"])),
                             f"{f['sum_len']:.1f}", f"{f['max_len']:.1f}",
                             ",".join(str(m) for m in f["members"])])
        print(f"# {arm}: {n_evt} events, {n_dup_evt} with >=1 dup-pair finding")

    print(f"# TOTAL: {len(rows)} (event,cluster) findings -- "
          f"class A (mvga-carry co-located) {n_classA}, class B (no carry) {n_classB}")

    if args.tsv:
        with open(args.tsv, "w") as f:
            f.write("arm\tevent\tcluster\tclass\tn_dup_pairs\tn_segs\tsum_len_cm\t"
                     "max_len_cm\tmember_seg_ids\n")
            for r in rows:
                f.write("\t".join(r) + "\n")
        print(f"# wrote {args.tsv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
