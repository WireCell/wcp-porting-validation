#!/usr/bin/env python3
"""doc pdvd/32: how much of a track the STM fit trajectory actually covers.

Event 298595 cluster 109 shows a `stm_fit` trajectory that stops short of BOTH
physical ends of the track.  This script measures that, and separates the two
readings it could have:

  * the Steiner TERMINALS do not reach the ends either -- i.e. doc 31's
    subject, a terminal-coverage problem; or
  * the terminals DO reach the ends and the trajectory did not use them -- i.e.
    a defect in the endpoint choice or in the fit's end handling, downstream of
    everything doc 31 studies.

So every shortfall here is measured against the cluster's own TERMINAL extent,
not against the raw point cloud.  A shortfall of 0 means the fit reached the
outermost terminal; a positive shortfall counts terminals that were available
and unused.

Cluster selection is doc 31's, unchanged, so the population is the same one
`steiner_terminal_geometry.py` reports on:
    >= --minpts steiner points, PCA linearity lam0/sum > --minlin,
    p5-p95 first-axis extent > --minlen cm.

Orientation note: the sign of a PCA axis is arbitrary, so "low end" and "high
end" are not comparable ACROSS clusters.  The per-cluster table prints them in
axis order for traceability, but every population number is built from the
orientation-independent min / max / sum of the two shortfalls.

Optional --axis mode projects onto a named segment instead of the PCA axis, for
the one cluster the owner pointed at:
    --axis 109:343.1,140.1,195.3:221.4,196.8,253.7 --tube 8
which reports the profile along that segment and, if the per-(apa,face) Bee
zips are present next to mabc-pr.zip, which volume each stretch of track is in
(matched on (y,z) only -- the per-face zips carry per-face uncorrected drift x).

Usage:
  python3 stm_trajectory_coverage.py \
      d31r6e2e:/path/pdvd/work/039252_2_d31r6e2e \
      d31r7t500:/path/pdvd/work/039349_14_d31r7t500
"""
import argparse
import glob
import json
import os
import sys
import zipfile

import numpy as np

BEE_CLUSTERING = "data/0/0-clustering-global.json"
BEE_STM_FIT = "data/0/0-stm_fit-global.json"


def load_bee(zpath, member):
    """(Nx3 points, cluster_id, q) from one Bee layer, or (None, None, None)."""
    with zipfile.ZipFile(zpath) as z:
        if member not in z.namelist():
            return None, None, None
        d = json.loads(z.read(member))
    if not d.get("x"):
        return None, None, None
    return (np.array([d["x"], d["y"], d["z"]], dtype=float).T,
            np.asarray(d["cluster_id"], dtype=int),
            np.asarray(d["q"], dtype=float))


def load_steiner(calib_path):
    """{cluster_id: (Nx3 steiner points, terminal mask)} from the PR calib dump.

    PrDisplayDump's `steiner` section carries x/y/z/flag_terminal only -- no
    charge -- so nothing here can depend on the terminal charge ordering.
    """
    with open(calib_path) as fp:
        cal = json.load(fp)
    out = {}
    for e in cal.get("steiner", []):
        P = np.array([e["x"], e["y"], e["z"]], dtype=float).T
        out[int(e["cluster_id"])] = (P, np.asarray(e["flag_terminal"], dtype=int).astype(bool))
    return out


def pca_axis(P):
    """(centroid, first principal axis, eigenvalue fractions) of a point set."""
    mu = P.mean(axis=0)
    X = P - mu
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    lam = s ** 2
    return mu, vt[0], lam


def arm_workdir(spec):
    """'label:/path/to/workdir' -> (label, workdir, mabc-pr.zip, calib json)."""
    label, _, path = spec.partition(":")
    if not path:
        label, path = os.path.basename(spec.rstrip("/")), spec
    zp = os.path.join(path, "mabc-pr.zip")
    cal = sorted(glob.glob(os.path.join(path, "calib-pr-evt*.json")))
    if not os.path.exists(zp) or not cal:
        return label, path, None, None
    return label, path, zp, cal[0]


def coverage_rows(zp, calib, args):
    """One record per cluster passing doc 31's selection."""
    Pc, cc, _ = load_bee(zp, BEE_CLUSTERING)
    Ps, cs, _ = load_bee(zp, BEE_STM_FIT)
    steiner = load_steiner(calib)
    rows = []
    for cid in sorted(steiner):
        SP, term = steiner[cid]
        if len(SP) < args.minpts:
            continue
        mu, ax, lam = pca_axis(SP)
        if lam.sum() <= 0 or lam[0] / lam.sum() <= args.minlin:
            continue
        t_all = (SP - mu) @ ax
        lo, hi = np.percentile(t_all, [5, 95])
        if hi - lo <= args.minlen:
            continue
        rec = {"cid": cid, "L_p5p95": hi - lo, "nsteiner": len(SP),
               "nterm": int(term.sum()), "nraw": int((cc == cid).sum()) if cc is not None else 0}
        # terminal extent: the outermost terminals, NOT the trimmed window
        t_term = t_all[term]
        rec["t_term_lo"], rec["t_term_hi"] = float(t_term.min()), float(t_term.max())
        rec["L_term"] = rec["t_term_hi"] - rec["t_term_lo"]
        F = Ps[cs == cid] if Ps is not None else np.empty((0, 3))
        rec["nfit"] = len(F)
        if len(F) < 5:
            rec["short_lo"] = rec["short_hi"] = float("nan")
            rows.append(rec)
            continue
        tf = (F - mu) @ ax
        rec["L_fit"] = float(tf.max() - tf.min())
        rec["arc"] = float(np.linalg.norm(np.diff(F, axis=0), axis=1).sum())
        rec["short_lo"] = float(tf.min() - rec["t_term_lo"])
        rec["short_hi"] = float(rec["t_term_hi"] - tf.max())
        rows.append(rec)
    return rows


def report_arm(label, path, rows):
    print(f"\n== {label}  ({path})")
    fitted = [r for r in rows if r["short_lo"] == r["short_lo"]]
    print(f"   {len(rows)} long-straight cluster(s); {len(fitted)} of them carry an stm_fit trajectory")
    if not rows:
        return None
    print("    cid  nraw nterm  nfit   L_term    L_fit  short_lo short_hi   short_min short_max")
    for r in rows:
        if r["short_lo"] != r["short_lo"]:
            print(f"   {r['cid']:4d} {r['nraw']:5d} {r['nterm']:5d} {0:5d} {r['L_term']:8.1f}"
                  f"        -         -        -           -         -   (no fit)")
            continue
        smin = min(r["short_lo"], r["short_hi"])
        smax = max(r["short_lo"], r["short_hi"])
        print(f"   {r['cid']:4d} {r['nraw']:5d} {r['nterm']:5d} {r['nfit']:5d} {r['L_term']:8.1f}"
              f" {r['L_fit']:8.1f} {r['short_lo']:9.1f} {r['short_hi']:8.1f}   {smin:9.1f} {smax:9.1f}")
    if not fitted:
        return None
    a = np.array([[min(r["short_lo"], r["short_hi"]), max(r["short_lo"], r["short_hi"])] for r in fitted])
    tot = a.sum(axis=1)
    covered = int((tot < 2.0).sum())
    print(f"   --> n={len(a)}  shortfall vs the TERMINAL extent:"
          f" median min {np.median(a[:, 0]):.1f} cm, median max {np.median(a[:, 1]):.1f} cm,"
          f" median total {np.median(tot):.1f} cm")
    print(f"       both ends covered (total < 2 cm): {covered}/{len(a)}")
    return a


def report_axis(zp, calib, cid, A, B, tube):
    """Profile one cluster along a named A->B segment, plus its (apa,face) split."""
    Pc, cc, _ = load_bee(zp, BEE_CLUSTERING)
    Ps, cs, _ = load_bee(zp, BEE_STM_FIT)
    steiner = load_steiner(calib)
    if cid not in steiner or Pc is None:
        print(f"   cluster {cid}: not present")
        return
    ax = B - A
    L = float(np.linalg.norm(ax))
    ax = ax / L
    fmt = lambda p: "(" + ", ".join(f"{v:g}" for v in p) + ")"
    print(f"\n-- named axis for cluster {cid}: A={fmt(A)} -> B={fmt(B)}, |AB| = {L:.2f} cm, tube {tube} cm")

    def proj(P):
        d = P - A
        t = d @ ax
        return t, np.linalg.norm(d - np.outer(t, ax), axis=1)

    R = Pc[cc == cid]
    tr, pr = proj(R)
    core = pr < tube
    print(f"   raw points {len(R)}, {core.sum()} inside the tube; core extent t = [{tr[core].min():.1f}, {tr[core].max():.1f}] cm")
    SP, term = steiner[cid]
    ts, ps = proj(SP)
    ct = (ps < tube) & term
    print(f"   terminals in the tube {ct.sum()}; extent t = [{ts[ct].min():.1f}, {ts[ct].max():.1f}] cm")
    tt = np.sort(ts[ct])
    gaps = np.diff(tt)
    print(f"   terminal gaps along the axis: median {np.median(gaps):.2f} cm, largest {gaps.max():.2f} cm")
    F = Ps[cs == cid] if Ps is not None else np.empty((0, 3))
    if len(F) >= 2:
        tf, _ = proj(F)
        seg = np.linalg.norm(np.diff(F, axis=0), axis=1)
        print(f"   stm_fit {len(F)} pts, extent t = [{tf.min():.1f}, {tf.max():.1f}] cm,"
              f" arc {seg.sum():.1f} cm, median step {np.median(seg):.2f} cm,"
              f" pieces (step > 2 cm) {int((seg > 2.0).sum()) + 1}")
        print(f"   UNCOVERED: {tf.min() - ts[ct].min():.1f} cm at the A end,"
              f" {ts[ct].max() - tf.max():.1f} cm at the B end")
        print(f"   terminals stranded beyond the fit: {int((ts[ct] < tf.min()).sum())} at A,"
              f" {int((ts[ct] > tf.max()).sum())} at B")
    # (apa,face) split, matched on (y,z): the per-face zips carry per-face
    # uncorrected drift x, so x cannot be used to match.
    from scipy.spatial import cKDTree
    faces = sorted(glob.glob(os.path.join(os.path.dirname(zp), "mabc-anode*-face*.zip")))
    if faces:
        print("   (apa,face) volumes crossed, from the per-face Bee zips:")
        for f in faces:
            with zipfile.ZipFile(f) as z:
                nm = [n for n in z.namelist() if "clustering" in n]
                if not nm:
                    continue
                d = json.loads(z.read(nm[0]))
            if not d.get("y"):
                continue
            dist, _ = cKDTree(np.array([d["y"], d["z"]]).T).query(np.stack([R[core, 1], R[core, 2]], axis=1))
            hit = dist < 0.05
            if not hit.any():
                continue
            th = tr[core][hit]
            print(f"     {os.path.basename(f).replace('mabc-', '').replace('.zip', ''):22s}"
                  f" {int(hit.sum()):5d} pts  t = [{th.min():7.1f}, {th.max():7.1f}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="+", help="LABEL:/path/to/work/<run>_<evt>_<tag>")
    ap.add_argument("--minpts", type=int, default=200)
    ap.add_argument("--minlen", type=float, default=50.0)
    ap.add_argument("--minlin", type=float, default=0.95)
    ap.add_argument("--tube", type=float, default=8.0)
    ap.add_argument("--axis", default=None,
                    help="CID:x,y,z:x,y,z -- profile this cluster along a named segment")
    args = ap.parse_args()

    print(f"selection: >= {args.minpts} steiner points, lam0/sum > {args.minlin},"
          f" p5-p95 first-axis extent > {args.minlen} cm  (doc 31's, unchanged)")
    print("shortfall is measured against the cluster's own TERMINAL extent, not the raw cloud")

    pooled = []
    for spec in args.arms:
        label, path, zp, calib = arm_workdir(spec)
        if zp is None:
            print(f"\n== {label}  ({path})\n   SKIP: no mabc-pr.zip / calib-pr-evt*.json", file=sys.stderr)
            continue
        rows = coverage_rows(zp, calib, args)
        a = report_arm(label, path, rows)
        if a is not None:
            pooled.append(a)
        if args.axis:
            cid, _, rest = args.axis.partition(":")
            sa, _, sb = rest.partition(":")
            report_axis(zp, calib, int(cid),
                        np.array([float(v) for v in sa.split(",")]),
                        np.array([float(v) for v in sb.split(",")]), args.tube)
    if len(pooled) > 1:
        a = np.vstack(pooled)
        tot = a.sum(axis=1)
        print(f"\n== pooled over {len(pooled)} arm(s): n={len(a)}"
              f"  median min {np.median(a[:, 0]):.1f} cm, median max {np.median(a[:, 1]):.1f} cm,"
              f" median total {np.median(tot):.1f} cm,"
              f" both ends covered {int((tot < 2.0).sum())}/{len(a)}")


if __name__ == "__main__":
    main()
