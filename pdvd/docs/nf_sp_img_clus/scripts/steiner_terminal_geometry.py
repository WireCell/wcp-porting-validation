#!/usr/bin/env python3
"""doc pdvd/31 round 8: WHERE the excess terminals sit, and what a cross-blob
suppression pass would actually remove.

Round 7 measured that PDVD marks ~1.5x more terminals per cm of track than SBND
at a matched charge selectivity, and attributed the residual to blob
granularity.  That attribution mixes granularities (a whole-cluster probe
aggregate against a per-corridor median -- section 11.7 caveat 2), and it does
not say which of the two geometric axes carries the excess:

  * ALONG the drift/time axis -- more occupied time slices per cm of track, or
    more terminals per occupied slice; or
  * ACROSS it, WITHIN one slice -- several blobs of the same slice overlapping
    in 3-D (doc 28: 83.6 % of PDVD (U,V) crossings are ambiguous against SBND's
    21.0 %), each contributing its own terminal at nearly the same place.

Those two point at different designs, so this script measures them separately
and then SIMULATES the candidate criteria offline on both detectors, using the
SAME cluster selection as steiner_density_xdet.py (>= --minpts steiner points,
PCA linearity > --minlin, 5th-95th percentile first-axis extent > --minlen cm).

What is measured per surviving cluster
  slice pitch     median spacing of the distinct x levels the cluster occupies.
                  MEASURED, not derived from tick width x drift speed -- round 7
                  falsified a slice-thickness hypothesis built that way.
  per slice       terminals and points per OCCUPIED slice.  This is the number
                  that separates the two axes above; terminals/cm does not.
  in-slice pairs  fraction of terminals having ANOTHER terminal in the same
                  slice within --inslice cm.  That is the ghost signature.
  nn3d            nearest-neighbour distance between terminals in 3-D.  Sets the
                  radius any suppression pass would use.

What is simulated (offline, on the stored terminal set)
  nms R           greedy charge-ordered non-maximum suppression at radius R cm:
                  walk the terminals in some order, keep one if no already-kept
                  terminal lies within R, else drop it.
                  ORDER: the dump carries no charge (PrDisplayDump.cxx:1113
                  stores x/y/z/flag_terminal only), so the true charge order is
                  not reproducible here.  Two deterministic orders bracket it --
                  along the principal axis, and a seeded shuffle.  The COUNT is
                  order-robust (any maximal R-separated subset of a curve of
                  length L has between L/2R and L/R+1 members); WHICH terminals
                  survive is not, so the gap after NMS is reported for both
                  orders and should be read as an indication, not a prediction.
                  The coverage BOUND does not depend on order: every dropped
                  terminal lies within R of a kept one, so the kept set is an
                  R-covering of the original and a terminal-free run of length G
                  can grow to at most G + 2R.
  slice R         the same greedy pass restricted to pairs in the SAME slice.
                  By construction it can never remove the last terminal of a
                  slice, so it cannot open an along-track gap at all.

Usage:
  python3 steiner_terminal_geometry.py \
      PDVD:/path/work/039349_14_d31r7t500/calib-pr-evt19689.json \
      SBND:'/path/work-mcp2k-d97fvpr2/pr_evt*/calib-pr-evt*.json'
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

# x levels within this many cm are the same time slice.  The sampler puts every
# point of a blob at the slice's own x, so the levels are exact in practice;
# the tolerance only guards float round-tripping through JSON.
XTOL = 0.005


def greedy_nms(P, order, R):
    """Indices kept by greedy non-maximum suppression at radius R."""
    tree = cKDTree(P)
    dead = np.zeros(len(P), bool)
    kept = []
    for i in order:
        if dead[i]:
            continue
        kept.append(i)
        for j in tree.query_ball_point(P[i], R):
            if j != i:
                dead[j] = True
    return np.asarray(kept, int)


def slice_levels(x):
    """(level index per point, sorted level centres) for the x coordinate."""
    o = np.argsort(x)
    xs = x[o]
    brk = np.concatenate(([0], np.nonzero(np.diff(xs) > XTOL)[0] + 1, [len(xs)]))
    lvl = np.empty(len(x), int)
    centres = np.empty(len(brk) - 1)
    for k in range(len(brk) - 1):
        seg = o[brk[k]:brk[k + 1]]
        lvl[seg] = k
        centres[k] = xs[brk[k]:brk[k + 1]].mean()
    return lvl, centres


def gap_along(t_term, t_lo, t_hi):
    """Largest terminal-free run along the axis inside the [p5, p95] window."""
    tt = np.sort(t_term[(t_term >= t_lo) & (t_term <= t_hi)])
    edges = np.concatenate(([t_lo], tt, [t_hi]))
    return float(np.diff(edges).max())


def analyse(e, args):
    P = np.stack([e["x"], e["y"], e["z"]], 1).astype(float)
    if len(P) < args.minpts:
        return None
    term = np.asarray(e["flag_terminal"], bool)
    if len(term) != len(P):
        return None

    C = P - P.mean(0)
    lam, vec = np.linalg.eigh(C.T @ C / len(C))
    order = np.argsort(lam)[::-1]
    lam, vec = lam[order], vec[:, order]
    if lam.sum() <= 0 or lam[0] / lam.sum() <= args.minlin:
        return None
    axis = vec[:, 0]
    t_all = C @ axis
    t_lo, t_hi = np.percentile(t_all, 5), np.percentile(t_all, 95)
    L = float(t_hi - t_lo)
    if L < args.minlen:
        return None
    nterm = int(term.sum())
    if nterm < 2:
        return None

    T = P[term]                      # the terminals themselves
    tT = t_all[term]
    lvl, centres = slice_levels(P[:, 0])
    dif = np.diff(centres)
    pitch = float(np.median(dif)) if len(dif) else float("nan")
    nslice = len(centres)
    tlvl = lvl[term]

    # in-slice degeneracy: another terminal in the SAME slice within --inslice
    n_pair = 0
    for k in np.unique(tlvl):
        m = tlvl == k
        if m.sum() < 2:
            continue
        Q = T[m]
        d = np.linalg.norm(Q[:, None, :] - Q[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        n_pair += int((d.min(1) < args.inslice).sum())

    tree = cKDTree(T)
    nn = tree.query(T, k=2)[0][:, 1]

    rec = dict(
        event=os.path.basename(e["_path"]), cluster_id=e["cluster_id"],
        main=bool(e["is_main_cluster"]), L=L, npts=len(P), nterm=nterm,
        term_per_cm=nterm / L, cosx=abs(float(axis[0])),
        pitch=pitch, nslice=nslice, slice_per_cm=nslice / L,
        term_per_slice=nterm / nslice, pts_per_slice=len(P) / nslice,
        term_slice_occ=len(np.unique(tlvl)) / nslice,
        term_per_occ=nterm / len(np.unique(tlvl)),
        inslice_frac=n_pair / nterm,
        nn3d=float(np.median(nn)),
        gap=gap_along(tT, t_lo, t_hi),
    )

    rng = np.random.default_rng(20260903)
    ax_order = np.argsort(tT)
    rnd_order = rng.permutation(nterm)
    for R in args.nms:
        ka = greedy_nms(T, ax_order, R)
        kr = greedy_nms(T, rnd_order, R)
        rec[f"nms{R}_ax"] = len(ka) / L
        rec[f"nms{R}_rnd"] = len(kr) / L
        rec[f"nms{R}_gap"] = gap_along(tT[ka], t_lo, t_hi)
        rec[f"nms{R}_keep"] = len(ka) / nterm
        # same-slice-only variant: greedy within each slice, never across
        keep = []
        for k in np.unique(tlvl):
            idx = np.nonzero(tlvl == k)[0]
            if len(idx) == 1:
                keep.append(idx)
                continue
            sub = greedy_nms(T[idx], np.argsort(tT[idx]), R)
            keep.append(idx[sub])
        ks = np.concatenate(keep)
        rec[f"sl{R}_percm"] = len(ks) / L
        rec[f"sl{R}_keep"] = len(ks) / nterm
        rec[f"sl{R}_gap"] = gap_along(tT[ks], t_lo, t_hi)
    return rec


def collect(paths, args):
    out = []
    for p in paths:
        try:
            d = json.load(open(p))
        except Exception as exc:
            print(f"  skip {os.path.basename(p)}: {exc}", file=sys.stderr)
            continue
        for e in d.get("steiner", []):
            if "flag_terminal" not in e:
                continue
            e["_path"] = p
            r = analyse(e, args)
            if r is not None:
                out.append(r)
    return out


def med(recs, key, fmt="{:5.2f}"):
    v = np.asarray([r[key] for r in recs], float)
    a, b, c = np.percentile(v, 25), np.median(v), np.percentile(v, 75)
    return (fmt + " [" + fmt + "," + fmt + "]").format(b, a, c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+", help="LABEL:glob")
    ap.add_argument("--minpts", type=int, default=200)
    ap.add_argument("--minlen", type=float, default=50.0)
    ap.add_argument("--minlin", type=float, default=0.95)
    ap.add_argument("--inslice", type=float, default=1.0)
    ap.add_argument("--nms", type=float, nargs="+", default=[0.3, 0.5, 0.8, 1.0, 1.5])
    args = ap.parse_args()

    per = {}
    for spec in args.specs:
        label, pattern = spec.split(":", 1)
        paths = sorted(glob.glob(pattern))
        per[label] = collect(paths, args)
        print(f"== {label}: {len(paths)} dump(s), {len(per[label])} long straight cluster(s)")

    def geom(label, recs):
        """The per-slice decomposition of terminals/cm, one arm, one bin."""
        if not recs:
            print(f"  {label:22s} (no cluster passes the cut)")
            return
        print(f"  {label:22s} n={len(recs):4d}  L={med(recs,'L','{:6.1f}')}cm  "
              f"|cos(axis,x)|={med(recs,'cosx','{:5.2f}')}  "
              f"terminals/cm={med(recs,'term_per_cm')}")
        print(f"  {'':22s}   slice pitch={med(recs,'pitch','{:5.3f}')}cm  "
              f"slices/cm={med(recs,'slice_per_cm')}  "
              f"pts/slice={med(recs,'pts_per_slice')}  "
              f"terminals/slice={med(recs,'term_per_slice')}")
        print(f"  {'':22s}   slices with a terminal={med(recs,'term_slice_occ')}  "
              f"terminals per OCCUPIED slice={med(recs,'term_per_occ')}  "
              f"in-slice partner < {args.inslice}cm={med(recs,'inslice_frac')}")
        print(f"  {'':22s}   nn3d={med(recs,'nn3d')}cm  "
              f"largest terminal-free run={med(recs,'gap','{:5.2f}')}cm")

    print("\n-- geometry of the stored terminal set (median [IQR] over clusters)")
    for label, recs in per.items():
        geom(label, recs)

    print("\n-- the same decomposition binned by drift alignment |cos(axis, x)|")
    for lo, hi, name in ((0.0, 0.3, "transverse"), (0.3, 0.7, "oblique"),
                         (0.7, 1.01, "along drift")):
        for label, recs in per.items():
            geom(f"{label} {name}", [r for r in recs if lo <= r["cosx"] < hi])
        print()

    def nms_table(title, pick):
        print(f"\n-- simulated GLOBAL greedy suppression at radius R -- {title}")
        print(f"  {'arm':12s} {'R':>5s} {'axis order':>16s} {'random order':>16s} "
              f"{'kept':>14s} {'gap after':>16s}")
        for label, recs in per.items():
            recs = pick(recs)
            if not recs:
                continue
            for R in args.nms:
                print(f"  {label:12s} {R:5.2f} {med(recs,f'nms{R}_ax'):>16s} "
                      f"{med(recs,f'nms{R}_rnd'):>16s} {med(recs,f'nms{R}_keep'):>14s} "
                      f"{med(recs,f'nms{R}_gap'):>16s}")

    nms_table("all long straight clusters", lambda r: r)
    nms_table("along drift only (|cos(axis,x)| > 0.7), the matched bin",
              lambda r: [x for x in r if x["cosx"] >= 0.7])

    print("\n-- simulated SAME-SLICE-ONLY suppression at radius R (cannot open a gap)")
    print(f"  {'arm':12s} {'R':>5s} {'terminals/cm':>16s} {'kept':>14s} {'gap after':>16s}")
    for label, recs in per.items():
        if not recs:
            continue
        for R in args.nms:
            print(f"  {label:12s} {R:5.2f} {med(recs,f'sl{R}_percm'):>16s} "
                  f"{med(recs,f'sl{R}_keep'):>14s} {med(recs,f'sl{R}_gap'):>16s}")


if __name__ == "__main__":
    main()
