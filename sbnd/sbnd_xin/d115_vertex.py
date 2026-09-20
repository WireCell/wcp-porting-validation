#!/usr/bin/env python3
"""doc sbnd_xin/115: the neutrino-vertex assessment.

d107_selection.py's Q1 already reports the within-5 cm fraction by true class and Edep bin.
What it does not do is say where the misses go, which is the question the owner asked ("the
neutrino vertex situation"), and which doc 107 sec 5.8 had to answer by hand on 31 events.
Here it is a table, computed from the same two products:

  no candidate in the event        the PR chain produced no T_tagger row at all
  candidate, but none matched      the nearest candidate vertex is 5-20 / 20-50 / >50 cm away
  matched                          a candidate vertex within 5 cm

The distance used per true interaction is truth.tsv's min_cand_dist_cm -- the distance to the
NEAREST candidate vertex in that event -- so the taxonomy is a partition of the denominator,
not overlapping cuts.

Reported for every true interaction with its vertex in the doc-107 FV, split by true class and
by Edep bin, with 68 % Wilson intervals on the matched fraction.

Usage:
  python3 d115_vertex.py products/d115/cv docs/115_vtx --label cv [--edep-min 100]
"""
import argparse
import collections
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MATCH_CM = 5.0
BANDS = [(0.0, 5.0, "matched (<5 cm)"), (5.0, 20.0, "5-20 cm"),
         (20.0, 50.0, "20-50 cm"), (50.0, float("inf"), ">50 cm")]


def in_fv(x, y, z):
    return 5.0 < abs(x) < 190.0 and abs(y) < 190.0 and 10.0 < z < 450.0


def wilson(k, n, z=1.0):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def tclass(flav, ccnc):
    if ccnc == "NC":
        return "NC"
    return {"numu": "numuCC", "nue": "nueCC"}.get(flav, "CC-" + flav)


def band(d, has_cand):
    if not has_cand or d < 0:
        return "no candidate"
    for lo, hi, name in BANDS:
        if lo <= d < hi:
            return name
    return ">50 cm"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("outdir")
    ap.add_argument("--label", required=True)
    ap.add_argument("--edep-min", type=float, default=0.0)
    a = ap.parse_args()
    f = float

    T = list(csv.DictReader(open(os.path.join(a.src, "truth.tsv")), delimiter="\t"))
    sel = []
    for t in T:
        if not in_fv(f(t["vx"]), f(t["vy"]), f(t["vz"])):
            continue
        # d107_selection.py:90 -- with no threshold there is NO Edep cut at all.  Writing this
        # as `Edep <= edep_min` would silently drop the zero-Edep interactions from the
        # denominator and inflate every fraction below.
        if not (a.edep_min <= 0 or f(t["Edep"]) > a.edep_min):
            continue
        t["_cls"] = tclass(t["flav"], t["ccnc"])
        t["_d"] = f(t["min_cand_dist_cm"])
        t["_band"] = band(t["_d"], t["event_has_candidate"] == "1")
        sel.append(t)

    groups = [("all true nu in FV", lambda t: True),
              ("numuCC", lambda t: t["_cls"] == "numuCC"),
              ("nueCC", lambda t: t["_cls"] == "nueCC"),
              ("NC", lambda t: t["_cls"] == "NC"),
              ("Edep < 100 MeV", lambda t: f(t["Edep"]) < 100),
              ("100 <= Edep < 300 MeV", lambda t: 100 <= f(t["Edep"]) < 300),
              ("Edep >= 300 MeV", lambda t: f(t["Edep"]) >= 300)]

    out = ["# doc 115 vertex assessment -- sample %s%s" % (
        a.label, ", Edep > %g MeV" % a.edep_min if a.edep_min else ""),
        "# FV 5<|x|<190, |y|<190, 10<z<450 cm; distance = to the NEAREST candidate vertex",
        "# true interactions with the vertex in the FV: %d" % len(sel), ""]
    hdr = "%-24s %7s  %-27s %8s %8s %8s %8s" % (
        "population", "N", "matched <5 cm", "5-20", "20-50", ">50", "no cand")
    out.append(hdr)
    out.append("-" * len(hdr))
    for gname, pred in groups:
        g = [t for t in sel if pred(t)]
        c = collections.Counter(t["_band"] for t in g)
        n = len(g)
        k = c["matched (<5 cm)"]
        lo, hi = wilson(k, n)
        out.append("%-24s %7d  %5d = %5.1f %% [%4.1f,%4.1f] %8d %8d %8d %8d"
                   % (gname, n, k, 100.0 * k / n if n else float("nan"), 100 * lo, 100 * hi,
                      c["5-20 cm"], c["20-50 cm"], c[">50 cm"], c["no candidate"]))

    # Distance distribution of the matched-or-not population, for the doc's figure.
    d = [min(t["_d"], 199.0) for t in sel if t["_band"] != "no candidate"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.hist(d, bins=[x for x in range(0, 205, 5)], color="#2a78d6", alpha=.85)
    ax.axvline(MATCH_CM, color="#eb6834", lw=1.6, ls="--", label="5 cm match")
    ax.set_xlabel("distance true vertex -> nearest candidate vertex (cm, last bin overflow)")
    ax.set_ylabel("true interactions in FV")
    ax.set_title("%s -- vertex association" % a.label)
    ax.legend(fontsize=8)
    ax.grid(alpha=.25)
    fig.tight_layout()
    os.makedirs(a.outdir, exist_ok=True)
    fig.savefig(os.path.join(a.outdir, "d115_vtx_%s.png" % a.label), dpi=140)

    txt = "\n".join(out) + "\n"
    open(os.path.join(a.outdir, "d115_vtx_%s.txt" % a.label), "w").write(txt)
    print(txt)
    print("-> %s/d115_vtx_%s.{txt,png}" % (a.outdir, a.label))


if __name__ == "__main__":
    main()
