#!/usr/bin/env python3
"""doc sbnd_xin/116 sec 15.2: is the beam-off move a better selection or a shifted working point?

The single number in sec 8 -- gates passing numu > 0.9, baseline 5/1000 -> 4 (tfull, cs, csp3bw)
or 2 (p3bw) -- cannot tell the two apart: a cell that also loses signal at the same cut has merely
moved along the same curve, and the cut is a free knob.  The score scans already on disk give the
whole curve for both axes of the same cell (the numu BDT score is the same variable in both):

    docs/115_scan/d115_scan_{cv,off}.tsv        the baseline arms
    docs/116_scan/d115_scan_{cv,off}-<cell>.tsv  each doc-116 cell

so for every cut we have (true numuCC efficiency on mc-cv, beam-off gate rate) for the same arm.
This prints, per cell, the efficiency at the cut where the cell's own beam-off count first reaches
the baseline's operating point (5 gates, then 4 and 3), i.e. a comparison at MATCHED cosmic
background, and the beam-off count at the cut where the cell first reaches the baseline's
efficiency, i.e. at MATCHED signal.  With 5 baseline gates the resolution is coarse; the point of
the table is the direction, and that the curves are read from the same 1000 gates.

Usage: python3 scripts/d116/off_roc.py > docs/116_figs/116_off_roc.txt
"""
import csv
import os

SX = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CELLS = ["s0rep", "cs", "p3bw", "csp3bw", "tfull"]
PROD_CUT = 0.9


def scan(path, flav="numu"):
    rows = {}
    with open(path) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["flavour"] != flav:
                continue
            rows[round(float(r["cut"]), 2)] = r
    return rows


def arm(label):
    if label == "baseline":
        return (scan(os.path.join(SX, "docs/115_scan/d115_scan_cv.tsv")),
                scan(os.path.join(SX, "docs/115_scan/d115_scan_off.tsv")))
    return (scan(os.path.join(SX, f"docs/116_scan/d115_scan_cv-{label}.tsv")),
            scan(os.path.join(SX, f"docs/116_scan/d115_scan_off-{label}.tsv")))


def eff_at(cv, cut):
    r = cv[cut]
    return int(r["n_eff_num"]), int(r["n_eff_den"]), 100.0 * float(r["eff"])


def gates_at(off, cut):
    return int(off[cut]["n_pur_num"])


def first_cut_with_gates(off, n):
    """lowest cut (loosest selection) whose beam-off count is <= n"""
    for c in sorted(off):
        if gates_at(off, c) <= n:
            return c
    return None


def first_cut_with_eff(cv, num):
    """highest cut (tightest) that still keeps >= num true numuCC selected"""
    best = None
    for c in sorted(cv):
        if int(cv[c]["n_eff_num"]) >= num:
            best = c
    return best


def main():
    print("# doc 116 sec 15.2 -- numuCC efficiency (mc-cv, 557 signal) against beam-off gate count")
    print("# (1000 Run-1 off-beam gates), both from the same arm's numu BDT score scan.")
    print(f"# production cut = {PROD_CUT}")
    base_cv, base_off = arm("baseline")
    b_sel, b_den, b_eff = eff_at(base_cv, PROD_CUT)
    b_gates = gates_at(base_off, PROD_CUT)
    print(f"\nbaseline at cut {PROD_CUT}: eff {b_sel}/{b_den} = {b_eff:.1f} %, beam-off {b_gates}/1000")
    print("\n## at the production cut, and at cuts matched to the baseline's beam-off count")
    print("| arm | cut 0.9: eff | cut 0.9: gates | cut where gates <= 5 | eff there | "
          "cut where gates <= 4 | eff there | cut where gates <= 3 | eff there |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label in ["baseline"] + CELLS:
        cv, off = arm(label)
        sel, den, eff = eff_at(cv, PROD_CUT)
        cells = [f"| {label} | {sel}/{den} = {eff:.1f} % | {gates_at(off, PROD_CUT)} |"]
        for n in (5, 4, 3):
            c = first_cut_with_gates(off, n)
            if c is None:
                cells.append(" n/a | n/a |")
            else:
                s, d, e = eff_at(cv, c)
                cells.append(f" {c:+.1f} | {s}/{d} = {e:.1f} % |")
        print("".join(cells))
    print("\n## at the cut where each arm first matches the baseline's selected signal count")
    print(f"| arm | cut with >= {b_sel} selected numuCC | eff | beam-off gates there |")
    print("|---|---:|---:|---:|")
    for label in ["baseline"] + CELLS:
        cv, off = arm(label)
        c = first_cut_with_eff(cv, b_sel)
        if c is None:
            print(f"| {label} | never reaches {b_sel} | - | - |")
            continue
        s, d, e = eff_at(cv, c)
        print(f"| {label} | {c:+.1f} | {s}/{d} = {e:.1f} % | {gates_at(off, c)}/1000 |")


if __name__ == "__main__":
    main()
