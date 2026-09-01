#!/usr/bin/env python3
"""census_ab.py -- diff two fire censuses; classify every tag by arm behaviour.

doc 77 sec 12 (round 3), item 5.  sec 11 measured only the production arm, so a
tag that fires could not be told apart from a tag the CAMPAIGN made fire.  With
the campaign-off arm (empre0901) scanned too, every tag lands in one of four
buckets and only the first is campaign-attributable:

    CAMPAIGN    fires in prod, dark in the off arm  -> the campaign turned it on
    PERTURBED   fires in both, different counts     -> a pre-campaign knob whose
                                                       firing the campaign moved
    IDENTICAL   fires in both, same count           -> untouched baseline
    ZERO        fires in neither                    -> says nothing on its own;
                                                       adjudicate per doc 77 sec 11.3

A ZERO row is NOT evidence a knob is dead: a knob shipped OFF in both arms is
expected to be silent in both, and an untagged knob never appears here at all
(scripts/cfg/tag_coverage.py measures that gap).

Usage:
    scripts/cfg/census_ab.py docs/77-firecensus-prod0901.tsv \
                             docs/77-firecensus-empre0901.tsv [--tsv out.tsv]
"""
import argparse


def load(path):
    out = {}
    for i, line in enumerate(open(path)):
        if i == 0:
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) >= 2:
            out[f[0]] = int(f[1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm_a", help="the ON / production census tsv")
    ap.add_argument("arm_b", help="the OFF / restored census tsv")
    ap.add_argument("--tsv")
    a = ap.parse_args()

    A, B = load(a.arm_a), load(a.arm_b)
    rows = []
    for t in sorted(set(A) | set(B), key=lambda t: (-A.get(t, 0), t)):
        p, e = A.get(t, 0), B.get(t, 0)
        cls = ("ZERO" if p == e == 0 else
               "CAMPAIGN" if e == 0 else
               "IDENTICAL" if p == e else "PERTURBED")
        rows.append((t, p, e, cls))

    w = open(a.tsv, "w") if a.tsv else None
    if w:
        w.write("tag\tarm_a\tarm_b\tdelta\tclass\n")
    for cls in ("CAMPAIGN", "PERTURBED", "IDENTICAL", "ZERO"):
        sel = [r for r in rows if r[3] == cls]
        print("\n== %s (%d) ==" % (cls, len(sel)))
        for t, p, e, _ in sel:
            print("  %-38s A %5d   B %5d   d=%+d" % (t.rstrip(":"), p, e, p - e))
    if w:
        for t, p, e, cls in rows:
            w.write("%s\t%d\t%d\t%d\t%s\n" % (t, p, e, p - e, cls))
        w.close()


if __name__ == "__main__":
    main()
