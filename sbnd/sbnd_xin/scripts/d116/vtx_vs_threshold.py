#!/usr/bin/env python3
"""doc sbnd_xin/116 sec 16: the vertex metric as a function of the match threshold, so the MC
result can be compared with doc pr/150's DATA result on the same construction.

pr/150 scored the vertex on data as "the arm's own candidate vertex nearest the hand-scan click,
within 3 cm" (677 -> 620 of 823, a 6.9 pt fall).  Doc 116 scored it on MC as "the nearest candidate
vertex within 5 cm of the GENIE vertex" (M7/M8, not separable).  The two differ in the reference
point (click vs truth) AND in the threshold (3 vs 5 cm), and sec 11 could only say the data number
"does not transfer".  This separates the two: products/*/truth.tsv carries min_cand_dist_cm -- the
same nearest-candidate distance -- so the MC metric can be recomputed at pr/150's 3 cm, and at 1
and 2 cm, with the same paired exchange and the same exact sign test.

If the MC also falls at 3 cm, the two experiments agree and doc 116's 5 cm choice hid it.  If it
does not, the data fall is a property of the click anchor (pr/150 sec 4.2: 189 of 823 clicks lie
within 0.05 cm of the s0 vertex, so any refit moves away from its own anchor), not of the
trajectory.

Usage: python3 scripts/d116/vtx_vs_threshold.py [--cells ...] > docs/116_figs/116_vtx_threshold.txt
"""
import argparse
import csv
import math
import os
import sys

SX = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
THRESH = [1.0, 2.0, 3.0, 5.0, 10.0]


def in_fv(x, y, z):
    return 5.0 < abs(x) < 190.0 and abs(y) < 190.0 and 10.0 < z < 450.0


def sign_p(k, n):
    """exact two-sided sign test, k = smaller arm of the discordant pairs"""
    if n == 0:
        return 1.0
    c = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2.0 * c / 2.0 ** n)


def load(path):
    """key (run,subrun,event,idx) -> (dist_cm or None) for true interactions with the vertex in FV"""
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if not in_fv(float(r["vx"]), float(r["vy"]), float(r["vz"])):
                continue
            k = (r["run"], r["subrun"], r["event"], r["idx"])
            d = None
            if r.get("event_has_candidate") == "1" and r.get("min_cand_dist_cm") not in ("", None):
                d = float(r["min_cand_dist_cm"])
            out[k] = d
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", nargs="+", default=["s0rep", "cs", "p3bw", "csp3bw", "tfull"])
    ap.add_argument("--samples", nargs="+", default=["cv", "nuecc"])
    a = ap.parse_args()
    print("# doc 116 sec 16 -- the MC vertex metric at pr/150's threshold as well as doc 116's")
    print("# match = the nearest candidate vertex within T cm of the true vertex (the same")
    print("# construction as pr/150's click-anchored metric, with truth in place of the click).")
    for s in a.samples:
        B = load(os.path.join(SX, "products", "d115", s, "truth.tsv"))
        print(f"\n## {s}: {len(B)} true interactions with the vertex in the FV")
        print("| T (cm) | baseline | " + " | ".join(a.cells) + " |")
        print("|---:|---:|" + "---:|" * len(a.cells))
        arms = {}
        for c in a.cells:
            p = os.path.join(SX, "products", "d116", f"{s}-{c}", "truth.tsv")
            if os.path.exists(p):
                arms[c] = load(p)
        for T in THRESH:
            row = [f"| {T:.0f} "]
            kb = sum(1 for d in B.values() if d is not None and d <= T)
            row.append(f"| {kb}/{len(B)} = {100.0*kb/len(B):.1f} % ")
            for c in a.cells:
                A = arms.get(c)
                if A is None:
                    row.append("| n/a ")
                    continue
                ks = sorted(set(A) & set(B))
                ka = sum(1 for k in ks if A[k] is not None and A[k] <= T)
                kbb = sum(1 for k in ks if B[k] is not None and B[k] <= T)
                lost = sum(1 for k in ks if (B[k] is not None and B[k] <= T) and not (A[k] is not None and A[k] <= T))
                gain = sum(1 for k in ks if (A[k] is not None and A[k] <= T) and not (B[k] is not None and B[k] <= T))
                p = sign_p(min(lost, gain), lost + gain)
                row.append(f"| {ka} ({100.0*(ka-kbb)/len(ks):+.1f} pt, -{lost}/+{gain}, p {p:.2g}) ")
            print("".join(row) + "|")
        # median distance among the interactions both arms place within 10 cm
        print("\nmedian nearest-candidate distance (cm), interactions matched within 10 cm in BOTH arms:")
        for c in a.cells:
            A = arms.get(c)
            if A is None:
                continue
            ks = [k for k in sorted(set(A) & set(B))
                  if B[k] is not None and A[k] is not None and B[k] <= 10 and A[k] <= 10]
            db = sorted(B[k] for k in ks)
            da = sorted(A[k] for k in ks)
            n = len(ks)
            closer = sum(1 for k in ks if A[k] < B[k] - 0.01)
            further = sum(1 for k in ks if A[k] > B[k] + 0.01)
            print(f"  {c:8s} n={n:5d}  baseline {db[n//2]:.3f} -> cell {da[n//2]:.3f}   "
                  f"closer/further {closer}/{further}  sign p {sign_p(min(closer, further), closer+further):.3g}")


if __name__ == "__main__":
    main()
