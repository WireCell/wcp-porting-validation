#!/usr/bin/env python3
"""doc pdvd/114 sec 4 -- the case catalogue's figures: representative stretches of every class, drawn with doc 112's
per-case anatomy (three wire planes with the cluster's charge cells, the Steiner vertices by class, the terminals and
the seed; the ridge frame (s, e2) / (s, e3) with the image and the seed; the seed's ridge offset along the arc).

Selection, from d114_support_census.py's <out>_stretches.tsv of both detectors: per detector and class the top-N
stretches by length x max ridge offset, one per (event, cluster), not open-ended, >= MIN_ROWS rows; plus the two
calibration spots of doc 112 (h1, v3).  Each case's click is the stretch's centroid and its window R = max(12 cm,
chord / 2 + 6 cm), so the whole stretch is inside the window.

Outputs: <out>_<case>_{2d,frame,offset}.png, <out>_<case>_walk.tsv, <out>_cases.txt (doc 112 format) and
<out>_cases.tsv (case name -> det event cluster click class sub length maxoff).

Usage: d114_case_figs.py --stretches figs/114_support_pdhd_stretches.tsv figs/114_support_pdvd_stretches.tsv \
           --arm pdhd=d114hbase --arm pdvd=d114vbase --top 3 --out figs/114_case
"""
import argparse, collections, csv, os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import d112_case_anatomy as X
from d114_support_census import CLASS_ORDER

SPOTS = {"h1": ("pdhd", "029107_16", 108, (75.3, 438.3, 450.1), 12.0),
         "v3": ("pdvd", "039349_20", 25, (-270.2, -243.7, 36.0), 12.0)}
MIN_ROWS = 3
MAX_DIMG = 30.0     # cm: a stretch farther than this from every image point is a far bridge, not a case to draw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stretches", nargs="+", required=True)
    ap.add_argument("--arm", action="append", required=True, help="det=arm")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--classes", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    arms = dict(x.split("=") for x in a.arm)
    rows = []
    for path in a.stretches:
        with open(path) as fh:
            rows.extend(csv.DictReader(fh, delimiter="\t"))
    want = set(a.classes.split(",")) if a.classes else None
    cases = []
    seen = set()
    for det in sorted(set(r["det"] for r in rows)):
        for klass in CLASS_ORDER:
            if want and klass not in want:
                continue
            cand = [r for r in rows if r["det"] == det and r["class"] == klass and int(r["open_ended"]) == 0
                    and int(r["nrows"]) >= MIN_ROWS and float(r["max_d_img"]) <= MAX_DIMG]
            cand.sort(key=lambda r: -float(r["length_cm"]) * float(r["max_d_img"]))
            k = 0
            for r in cand:
                key = (det, r["event"], r["cluster"])
                if key in seen:
                    continue
                seen.add(key)
                k += 1
                name = f"{det}_{klass}_{k}"
                click = (float(r["mx"]), float(r["my"]), float(r["mz"]))
                R = max(12.0, float(r["chord_cm"]) / 2 + 6.0)
                cases.append((name, det, r["event"], int(r["cluster"]), click, R, arms[det], klass, r))
                if k >= a.top:
                    break
    for name, (det, ev, cl, click, R) in SPOTS.items():
        cases.append((name, det, ev, cl, click, R, arms[det], "spot", None))
    L = []
    with open(f"{a.out}_cases.tsv", "w") as fh:
        fh.write("case\tdet\tevent\tcluster\tclick\tR\tclass\tnrows\tlength_cm\tmax_d_ridge\tmax_d_img\tseed_vertices\n")
        for name, det, ev, cl, click, R, arm, klass, r in cases:
            fh.write("\t".join(str(x) for x in (name, det, ev, cl, ",".join(f"{c:.1f}" for c in click), f"{R:.1f}", klass,
                                                r["nrows"] if r else "-", r["length_cm"] if r else "-",
                                                r["max_d_ridge"] if r else "-", r["max_d_img"] if r else "-",
                                                r["seed_vertices"] if r else "-")) + "\n")
    for name, det, ev, cl, click, R, arm, klass, r in cases:
        L.append(f"# case {name}: class {klass}" + (f", rows {r['row_a']}-{r['row_b']} ({r['nrows']}), length {r['length_cm']} cm, "
                                                    f"max ridge {r['max_d_ridge']} cm, max image {r['max_d_img']} cm, seed vertices "
                                                    f"{r['seed_vertices']}" if r else ""))
        try:
            X.one_case((name, det, ev, cl, click, R, arm), a.out, L)
        except Exception as e:      # a case that cannot be drawn is reported, not fatal
            L.append(f"## {name}: FAILED {type(e).__name__}: {e}")
        print(name, file=sys.stderr)
    open(f"{a.out}_cases.txt", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
