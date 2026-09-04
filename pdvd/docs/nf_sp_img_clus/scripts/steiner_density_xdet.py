#!/usr/bin/env python3
"""doc pdvd/31 round 7: is PDVD's Steiner terminal density high *compared to SBND*?

The owner's round-6 scan said the terminals look too dense and asked whether
that is the 500 e threshold (PDVD) instead of the 4000 e one (SBND, uBooNE, and
the C++ default).  Answering that across detectors needs a metric that is not
confounded by the three things that differ anyway:

  * point spacing   -- SBND 3.00 mm pitch on all planes, PDVD 7.65/7.65/5.10 mm.
                       Coarser sampling should give PDVD FEWER points per cm, so
                       points/cm is always reported next to terminals/cm.  If
                       PDVD is denser in terminals while sparser in points, the
                       result is strong.
  * topology        -- PDVD arms here are cosmics, SBND arms are neutrino MC.
                       Both dumps carry every cluster with `is_main_cluster`, so
                       we select the SAME object on both sides: long, straight,
                       non-main clusters (i.e. cosmic muons in both files).
  * drift alignment -- for a drift-parallel track the blob sequence is set by
                       time slicing, not by wire pitch, so density is reported
                       binned by |cos(axis, x)|.

Selection (identical both detectors): >= `--minpts` steiner points, PCA
linearity lam1/sum(lam) > 0.95, and 5th-95th percentile extent along the first
axis L > `--minlen` cm.  Metrics per surviving cluster: points/L, terminals/L,
terminals/points, and L/terminals ("one terminal per X cm").

`flag_terminal` is the stored `flag_steiner_terminal` attribute of the cluster's
steiner_pc (PrDisplayDump.cxx:1113,1129) -- the FINAL terminal set, after the P2
/ P3 filters, on both detectors.  Same code, same meaning.

Usage:
  python3 steiner_density_xdet.py \
      PDVD:/path/work/039349_14_d31r6e2e/calib-pr-evt19689.json \
      SBND:'/path/work-mcp2k-d97fvpr2/pr_evt*/calib-pr-evt*.json'
"""
import argparse
import glob
import json
import os
import sys

import numpy as np


def clusters(paths, minpts, minlen, minlin):
    """Yield one record per long straight cluster in every calib-pr dump."""
    out = []
    for p in paths:
        try:
            d = json.load(open(p))
        except Exception as exc:                      # a truncated dump is not a datum
            print(f"  skip {os.path.basename(p)}: {exc}", file=sys.stderr)
            continue
        for e in d.get("steiner", []):
            if "flag_terminal" not in e:
                continue
            P = np.stack([e["x"], e["y"], e["z"]], 1).astype(float)
            if len(P) < minpts:
                continue
            term = np.asarray(e["flag_terminal"], bool)
            if len(term) != len(P):
                continue
            C = P - P.mean(0)
            lam, vec = np.linalg.eigh(C.T @ C / len(C))
            order = np.argsort(lam)[::-1]
            lam, vec = lam[order], vec[:, order]
            lin = float(lam[0] / lam.sum()) if lam.sum() > 0 else 0.0
            if lin <= minlin:
                continue
            axis = vec[:, 0]
            t = C @ axis
            L = float(np.percentile(t, 95) - np.percentile(t, 5))
            if L < minlen:
                continue
            nt = int(term.sum())
            if nt == 0:
                continue
            out.append(dict(
                event=os.path.basename(p), cluster_id=e["cluster_id"],
                main=bool(e["is_main_cluster"]), L=L, npts=len(P), nterm=nt,
                pts_per_cm=len(P) / L, term_per_cm=nt / L,
                term_frac=nt / len(P), cm_per_term=L / nt,
                cosx=abs(float(axis[0])),
            ))
    return out


def quartiles(v):
    v = np.asarray(v, float)
    return np.percentile(v, 25), np.median(v), np.percentile(v, 75)


def report(label, recs, main_only=None):
    sel = [r for r in recs if main_only is None or r["main"] == main_only]
    if not sel:
        print(f"  {label:26s} (no cluster passes the cut)")
        return
    def q(k):
        a, b, c = quartiles([r[k] for r in sel])
        return f"{b:6.2f} [{a:5.2f},{c:5.2f}]"
    print(f"  {label:26s} n={len(sel):4d}  L={q('L')}cm  "
          f"points/cm={q('pts_per_cm')}  terminals/cm={q('term_per_cm')}  "
          f"term/point={q('term_frac')}  1 term per {q('cm_per_term')}cm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+", help="LABEL:glob")
    ap.add_argument("--minpts", type=int, default=200)
    ap.add_argument("--minlen", type=float, default=50.0)
    ap.add_argument("--minlin", type=float, default=0.95)
    args = ap.parse_args()

    per = {}
    for spec in args.specs:
        label, pattern = spec.split(":", 1)
        paths = sorted(glob.glob(pattern))
        recs = clusters(paths, args.minpts, args.minlen, args.minlin)
        per[label] = recs
        print(f"== {label}: {len(paths)} dump(s), {len(recs)} long straight cluster(s) "
              f"(>= {args.minpts} pts, linearity > {args.minlin}, L > {args.minlen} cm)")

    print("\n-- all long straight clusters")
    for label, recs in per.items():
        report(label, recs)

    print("\n-- non-main only (cosmic-like on both detectors)")
    for label, recs in per.items():
        report(label, [r for r in recs if not r["main"]])

    print("\n-- binned by drift alignment |cos(axis, x)|")
    for lo, hi, name in ((0.0, 0.3, "transverse to drift"),
                         (0.3, 0.7, "oblique          "),
                         (0.7, 1.01, "along drift      ")):
        for label, recs in per.items():
            report(f"{label} {name}", [r for r in recs if lo <= r["cosx"] < hi])
        print()


if __name__ == "__main__":
    main()
