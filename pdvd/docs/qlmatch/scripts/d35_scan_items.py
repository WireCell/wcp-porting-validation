#!/usr/bin/env python3
"""doc qlmatch/35 amendment 1 sec 1 -- the PRIVATE item list of the blind scan (doc 103 format: key, display_arm, role).

New items = every row of d35/c4_unlabelled.tsv (the C4 candidates with no truth in either cell after the carry), keyed
by their NATIVE key on their own arm (the display reads the arm's own cluster ids).  Calibration = 10 % of that (at
least 2), seed 35, from the q35ctl candidates the record judges (not MESSY / UNCLEAR), shown on q35ctl.  Fork of the
selection of d116_scan_items.py (untouched), which cannot be used as is: it compares native keys of every cell with
the truth, and the ToT arm's native keys are in a different id space (doc 35 sec 4.1).  Refuses an existing --out.

    python3 d35_scan_items.py --out /home/xqian/tmp/p35scan/items/pdvd_items.tsv
"""
import argparse
import csv
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import d35_stm_grade as S      # noqa: E402  (truth = d116 precedence + smx116 + own116v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--unlabelled", default=os.path.join(HERE, "..", "d35", "c4_unlabelled.tsv"))
    ap.add_argument("--ctl", default="q35ctl")
    ap.add_argument("--seed", type=int, default=35)
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit(f"REFUSING: {a.out} exists")
    rows = [l.rstrip("\n").split("\t") for l in open(a.unlabelled) if not l.startswith("#")]
    new = [(native, arm, "new") for arm, key, native, _, _ in rows]
    T, _ = S.G16.truth("pdvd", S.EXTRA, S.OWNER)
    RA = S.G16.U.cell_rows("pdvd", a.ctl)
    judged = sorted(k for k in RA if k in T and T[k][0] not in ("MESSY", "UNCLEAR"))
    ncal = max(2, int(round(0.10 * len(new))))
    cal = [(k, a.ctl, "calibration") for k in random.Random(a.seed).sample(judged, ncal)]
    out = new + cal
    random.Random(a.seed + 1).shuffle(out)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        f.write("# doc qlmatch/35 amendment 1 blind scan item list (pdvd); role for grading only, never shown to a scanner\n")
        f.write("key\tdisplay_arm\trole\n")
        for k, c, r in out:
            f.write(f"{k}\t{c}\t{r}\n")
    by = {}
    for k, c, r in out:
        by[(c, r)] = by.get((c, r), 0) + 1
    print(f"{len(new)} new + {len(cal)} calibration -> {a.out}; by (display_arm, role): {by}")


if __name__ == "__main__":
    main()
