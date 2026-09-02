#!/usr/bin/env python3
"""doc 95 sec 3 -- check this round against doc 93's published table.

doc 93's numbers are PARSED OUT OF docs/93_*.md, not retyped here: a table
with no script behind it is unverifiable, and retyping is exactly how a
"reproduces exactly" claim goes wrong.

Repro:
  python3 scripts/dbg25_vs_doc93.py
"""
import os, re, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D93 = os.path.join(HERE, "docs", "93_stm-tagger-feedback-8evt.md")
SUM = os.path.join(HERE, "bee", "dbg25", "dbg25-tagger-summary.tsv")

NUM = re.compile(r"^\|\s*\d+\s*\|")


def parse_doc93():
    """rows of RSE -> (verdict, t0, pe, len, npts, nbundles) from doc 93 sec 3."""
    out = {}
    for line in open(D93):
        if not NUM.match(line):
            continue
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) < 12:
            continue
        rse = c[1]
        if not re.match(r"^\d+-\d+-\d+$", rse):
            continue
        verdict = c[2].replace("*", "").strip()
        try:
            out[rse] = (verdict, float(c[3]), float(c[4]), float(c[5]),
                        int(c[6]), int(c[11]))
        except ValueError:
            continue
    return out


def parse_ours():
    out = {}
    for line in open(SUM):
        f = line.rstrip("\n").split("\t")
        if f[0] == "bee_idx" or f[3] == "NO-INBEAM-BUNDLE":
            continue
        if f[3] == "no-bundle":
            continue
        rse = f[2]
        if rse in out:          # keep the first in-beam bundle (the main one)
            continue
        out[rse] = (f[3], float(f[5]), float(f[6]), float(f[7]), int(f[8]),
                    int(f[14]))
    return out


def main():
    d93, ours = parse_doc93(), parse_ours()
    if len(d93) != 8:
        sys.exit(f"ERROR: parsed {len(d93)} rows from doc 93, expected 8")
    names = ("verdict", "t0_us", "flash_pe", "len_cm", "npts", "n_bundle")
    nmatch = ndiff = 0
    print(f"{'RSE':<12} {'field':<9} {'doc 93':>12} {'doc 95':>12}  status")
    for rse in d93:
        if rse not in ours:
            sys.exit(f"ERROR: {rse} present in doc 93, absent here")
        for k, (a, b) in enumerate(zip(d93[rse], ours[rse])):
            if k == 0:
                same = (a == b)
            elif k in (1, 3):                    # t0, length: 1 decimal / 3 dp
                same = abs(a - b) < (0.0005 if k == 1 else 0.05)
            elif k == 2:                         # doc 93 rounded PE to integer
                same = abs(a - b) < 1.0
            else:
                same = (a == b)
            tag = "same" if same else "DIFF"
            if same:
                nmatch += 1
            else:
                ndiff += 1
                print(f"{rse:<12} {names[k]:<9} {a!s:>12} {b!s:>12}  {tag}")
    print()
    print(f"8 events x 6 fields = 48 comparisons: {nmatch} same, {ndiff} differ")
    exp = {"827-27-4", "304-6-28", "146-60-31", "966-2-22"}
    flipped = {r for r in d93 if d93[r][0] != ours[r][0]}
    print(f"verdict changes: {sorted(flipped)}")
    ok = (flipped == exp)
    print("expected changes (the 4 doc-94 guard releases):", sorted(exp))
    print("VERDICT:", "OK" if ok and ndiff == len(flipped) else "REVIEW")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
