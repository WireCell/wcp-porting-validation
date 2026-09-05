#!/usr/bin/env python3
"""Decompose f_off into its two shells, per plane, across arms/detectors.

doc pdhd/stm-tagger-chain sec 8.6.  d42_proj2d_resid.py already splits the
charge that lies OFF the fitted trajectory's footprint into

    f_off_near : 1 < d <= 5  cells (Chebyshev, wires x slices)
    f_off_far  :     d >  5  cells

and writes both per (block, plane) into <out>_blocks.tsv.  Only the NEAR shell
is a statement about the kernel width -- it is the shell a wider transverse
sigma would fill.  The FAR shell is a statement about how much of its own
cluster the fitted trajectory covers.  A single f_off number mixes the two,
which is why the raw f_off column must not be compared across detectors.

This script reports the medians over accepted (status 0) blocks, and the same
split against block length, so the two readings can be told apart.

Repro (paths are the arms of sec 8 / sec 9.1 and doc pdvd/44):

  docs/scripts/d44_foff_shells.py \
     "PDHD stm0=/home/xqian/tmp/pdhdstm/ana2/resid_stm0_blocks.tsv" \
     "PDHD stmw=/home/xqian/tmp/pdhdstm/ana2/resid_stmw_blocks.tsv" \
     "PDVD pre=/home/xqian/tmp/d44/ana/resid_d44ref_blocks.tsv" \
     "PDVD post=/home/xqian/tmp/d44/ana/resid_d44sig_blocks.tsv" \
     "SBND=/home/xqian/tmp/d44/ana/resid_sbnd_d42fit_blocks.tsv"
"""
import argparse
import csv
import statistics as st


def load(path):
    with open(path) as fh:
        lines = [l for l in fh if not l.startswith("#")]
    return [r for r in csv.DictReader(lines, delimiter="\t") if r["status"] == "0"]


def med(rows, key):
    v = [float(r[key]) for r in rows if r[key] not in ("nan", "")]
    return st.median(v) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", help="LABEL=path/to/<out>_blocks.tsv")
    ap.add_argument("--length-plane", default="U",
                    help="plane used for the block-length split (default U)")
    a = ap.parse_args()

    arms = []
    for spec in a.arms:
        label, _, path = spec.partition("=")
        arms.append((label.strip(), path.strip()))

    print("status-0 blocks, medians per plane")
    print("%-11s %-3s %5s %7s %7s %7s" % ("arm", "pl", "n", "f_off", "near", "far"))
    for label, path in arms:
        rows = load(path)
        for pl in ("U", "V", "W"):
            s = [r for r in rows if r["plane"] == pl]
            if not s:
                continue
            print("%-11s %-3s %5d %7.3f %7.3f %7.3f" % (
                label, pl, len(s), med(s, "f_off"),
                med(s, "f_off_near"), med(s, "f_off_far")))

    print()
    print("vs block length, plane %s" % a.length_plane)
    for label, path in arms:
        rows = [r for r in load(path)
                if r["plane"] == a.length_plane and r["f_off"] != "nan"]
        if len(rows) < 8:
            continue
        rows.sort(key=lambda r: float(r["length_cm"]))
        q = len(rows) // 4
        for tag, sl in (("short 25%", rows[:q]), ("mid 50%", rows[q:3 * q]),
                        ("long 25%", rows[3 * q:])):
            if not sl:
                continue
            print("%-11s %-9s n=%4d  %6.1f-%6.1f cm  npts %5.0f  "
                  "f_off %.3f (near %.3f far %.3f)" % (
                      label, tag, len(sl), float(sl[0]["length_cm"]),
                      float(sl[-1]["length_cm"]), med(sl, "npts"),
                      med(sl, "f_off"), med(sl, "f_off_near"), med(sl, "f_off_far")))


if __name__ == "__main__":
    main()
