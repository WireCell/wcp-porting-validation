#!/usr/bin/env python3
"""doc sbnd_xin/115: the beam-off (off-beam data) baseline.

The 1000 Run-1 off-beam events carry no truth, so there is no efficiency or purity to quote.
What they measure is the COSMIC BACKGROUND RATE: how often the production chain puts a neutrino
candidate -- and a candidate that passes the selection -- into a beam gate that contains no
neutrino at all.  Every rate here is per beam gate, with a 68 % Wilson interval, so a later
round can be scored against it directly.

No POT or trigger normalisation is applied: the staged sample does not carry the numbers that
would let a rate be turned into a prediction for the MC selection, and inventing one would be
worse than leaving it out.  What is here is a rate per gate and nothing more.

Definitions are doc 107 sec 5.5's, so the beam-off numbers sit in the same frame as the MC ones:
  FV         5 < |x| < 190, |y| < 190, 10 < z < 450 cm
  selection  score cut AND reco vertex in the FV
  cuts       numu_score > 0.9, nue_score > 7.0, nue_score > 4.0

nue_score == -15 is the nue BDT's "did not evaluate" sentinel (UbooneNueBDTScorer.cxx), not a
low score; it is reported as its own fill fraction rather than counted as a failing candidate.

Usage:
  python3 d115_beamoff_rate.py products/d115/off docs/115_off
"""
import collections
import math
import os
import sys

NUE_SENTINEL = -15.0


def in_fv(x, y, z):
    """doc 107 sec 5.5 analysis FV (the owner's choice), cathode excluded."""
    return 5.0 < abs(x) < 190.0 and abs(y) < 190.0 and 10.0 < z < 450.0


def wilson(k, n, z=1.0):
    """68 % Wilson interval, as d107_selection.py computes it."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def frac(k, n):
    lo, hi = wilson(k, n)
    return "%d/%d = %.2f %% [%.2f, %.2f]" % (k, n, 100.0 * k / n if n else float("nan"),
                                             100.0 * lo, 100.0 * hi)


def quantiles(v):
    if not v:
        return "n/a"
    s = sorted(v)
    q = lambda f: s[min(len(s) - 1, int(f * len(s)))]
    return "min %.2f  q25 %.2f  med %.2f  q75 %.2f  max %.2f" % (s[0], q(.25), q(.5), q(.75), s[-1])


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "products/d115/off"
    outdir = sys.argv[2] if len(sys.argv) > 2 else "docs/115_off"
    cpath = os.path.join(src, "candidates.tsv")
    census = os.path.join(src, "file_rse.tsv")
    if not os.path.exists(cpath):
        sys.exit("ERROR: no %s" % cpath)

    # Denominator: every beam gate the chain was GIVEN, not only the ones that produced a
    # candidate, so a lost event counts as a gate with no candidate.  The census is required,
    # never optional: falling back to "events that produced a candidate" would make the
    # denominator a subset of the numerator's population and push every rate toward 100 %.
    # The denominator is the whole measurement here -- it must not be able to degrade quietly.
    if not os.path.exists(census):
        sys.exit("ERROR: no %s -- the beam-gate denominator is required; "
                 "run scripts/d115/rse_census.sh off" % census)
    gates = set()
    with open(census) as fh:
        fh.readline()
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) >= 6:
                gates.add((f[3], f[4], f[5]))
    if not gates:
        sys.exit("ERROR: %s holds no gates" % census)
    rows = []
    with open(cpath) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) >= len(hdr):
                rows.append(dict(zip(hdr, f)))
    ngate = len(gates)

    ev_any, ev_fv = set(), set()
    ev_cut = collections.defaultdict(set)
    n_fill, n_rows = 0, len(rows)
    numu_scores, nue_scores, enu_pass = [], [], collections.defaultdict(list)
    CUTS = [("numu", "numu_score", 0.9), ("nue", "nue_score", 7.0), ("nue", "nue_score", 4.0)]

    for r in rows:
        key = (r["run"], r["subrun"], r["event"])
        ev_any.add(key)
        x, y, z = (float(r[k]) for k in ("nu_x", "nu_y", "nu_z"))
        has_v = r["vertex_default"] == "0"
        fv = has_v and in_fv(x, y, z)
        if fv:
            ev_fv.add(key)
        ns, es = float(r["numu_score"]), float(r["nue_score"])
        numu_scores.append(ns)
        if es != NUE_SENTINEL:
            nue_scores.append(es)
            n_fill += 1
        for name, col, cut in CUTS:
            v = float(r[col])
            if col == "nue_score" and v == NUE_SENTINEL:
                continue
            if fv and v > cut:
                ev_cut[(name, cut)].add(key)
                enu_pass[(name, cut)].append(float(r["reco_Enu"]))

    os.makedirs(outdir, exist_ok=True)
    out = []
    A = out.append
    A("doc 115 -- beam-off (off-beam data) baseline")
    A("source: %s" % cpath)
    A("")
    A("Beam gates processed:                       %d" % ngate)
    A("Candidate rows (T_tagger rows):             %d" % n_rows)
    A("nue BDT evaluated (nue_score != -15):       %s" % frac(n_fill, n_rows))
    A("")
    A("Per-gate rates (68 % Wilson):")
    A("  gate has >= 1 neutrino candidate:         %s" % frac(len(ev_any), ngate))
    A("  ... with the reco vertex in the FV:       %s" % frac(len(ev_fv), ngate))
    for name, col, cut in CUTS:
        A("  ... and %s_score > %-4g (SELECTED):     %s"
          % (name, cut, frac(len(ev_cut[(name, cut)]), ngate)))
    A("")
    A("Score distributions over candidate rows:")
    A("  numu_score            %s" % quantiles(numu_scores))
    A("  nue_score (filled)    %s" % quantiles(nue_scores))
    A("")
    A("kine_reco_Enu (MeV) of the SELECTED candidates:")
    for name, col, cut in CUTS:
        A("  %s > %-4g  n=%d  %s" % (name, cut, len(enu_pass[(name, cut)]),
                                     quantiles(enu_pass[(name, cut)])))
    txt = "\n".join(out) + "\n"
    open(os.path.join(outdir, "d115_beamoff_rate.txt"), "w").write(txt)
    print(txt)
    print("-> %s/d115_beamoff_rate.txt" % outdir)


if __name__ == "__main__":
    main()
