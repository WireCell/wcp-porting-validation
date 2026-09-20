#!/usr/bin/env python3
"""doc sbnd_xin/115: the flash-time <-> true-neutrino-time association, as an independent check
on the truth join.

Doc 108 sec 3.5 measured, on the colleague's LArSoft-chain MC sample (4098 candidate rows with
a vertex within 5 cm), that

    T_tagger.flash_time_us - true nu time  =  +0.136 us

with a 5-95 % range of 0.132-0.140 us and 98.9 % of rows within 0.1 us of the median.

That number was measured on a different chain, a different sample and a different truth source
from this round's, so reproducing it here tests three things at once that nothing else in the
d115 chain tests together: that the events our join pairs really are the same events (a shuffled
join would smear this to the beam-gate width), that MC timing threads correctly through the
standalone chain, and that reality=sim did not move the flash clock.

A median far from +0.136 us is a timing or lineage error to report, not a physics result.

Usage:
  python3 d115_time_assoc.py products/d115/cv docs/115_time --label cv
"""
import argparse
import csv
import os
import statistics

REFERENCE_US = 0.136          # doc 108 sec 3.5
MATCH_CM = 5.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("outdir")
    ap.add_argument("--label", required=True)
    a = ap.parse_args()

    C = list(csv.DictReader(open(os.path.join(a.src, "candidates.tsv")), delimiter="\t"))
    d = []
    for r in C:
        if r["vertex_default"] != "0" or r["t_T"] in ("", "-1"):
            continue
        if not (0 <= float(r["t_dist_cm"]) < MATCH_CM):
            continue                      # matched candidates only: an unmatched one has no time
        d.append(float(r["flash_time_us"]) - float(r["t_T"]))

    out = ["# doc 115 flash-time <-> true-time association -- sample %s" % a.label,
           "# reference: doc 108 sec 3.5, +%.3f us on the LArSoft-chain sample" % REFERENCE_US,
           ""]
    if not d:
        out.append("no matched candidate with a truth time -- nothing to check")
        verdict = "EMPTY"
    else:
        d.sort()
        med = statistics.median(d)
        q = lambda f: d[min(len(d) - 1, int(f * len(d)))]
        near = sum(1 for v in d if abs(v - med) < 0.1)
        out.append("matched candidate rows with a truth time: %d" % len(d))
        out.append("flash_time_us - true T (us):  min %.4f  p05 %.4f  median %.4f  p95 %.4f  max %.4f"
                   % (d[0], q(.05), med, q(.95), d[-1]))
        out.append("within 0.1 us of the median: %d/%d = %.1f %%"
                   % (near, len(d), 100.0 * near / len(d)))
        out.append("offset vs the doc-108 reference: %+.4f us" % (med - REFERENCE_US))
        verdict = "PASS" if abs(med - REFERENCE_US) < 0.01 else "FAIL"
        out.append("")
        out.append("VERDICT %s (|median - reference| %s 0.010 us)"
                   % (verdict, "<" if verdict == "PASS" else ">="))
    txt = "\n".join(out) + "\n"
    os.makedirs(a.outdir, exist_ok=True)
    open(os.path.join(a.outdir, "d115_time_%s.txt" % a.label), "w").write(txt)
    print(txt)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
