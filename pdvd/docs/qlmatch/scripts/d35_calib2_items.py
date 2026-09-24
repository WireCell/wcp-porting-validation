#!/usr/bin/env python3
"""doc qlmatch/35 amendment 2 sec 1 -- the PRIVATE item list of the larger calibration round (doc 103 format).

26 calibration items, seed 36: q35ctl STM candidates, re-keyed into the record lineage (d35_stm_grade.rekey, carried
with status ok), judged by an owner-derived source (grade-truth source `record` or `owner_review`; not MESSY /
UNCLEAR), none of them a native key of the amendment-1 item list nor one of the 3 post-hoc objects; shown on q35ctl
under the native key.  Plus the one new item of amendment 2, q35ctl 039252_8/102.  Seeded shuffle; refuses an existing
--out.  Never shown to a scanner.

    python3 d35_calib2_items.py --out /home/xqian/tmp/p35scan/items/pdvd_items2.tsv
"""
import argparse
import csv
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import d35_stm_grade as S      # noqa: E402

AMEND1 = "/home/xqian/tmp/p35scan/items/pdvd_items.tsv"
POSTHOC = {"039349_39/46", "039349_39/56", "039349_46/65"}     # key-arm keys, d35/scan_posthoc_agreement.txt
NEW = [("039252_8/102", "q35ctl", "new")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=26)
    ap.add_argument("--seed", type=int, default=36)
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit(f"REFUSING: {a.out} exists")
    G = S.G16
    T, _ = G.truth("pdvd", S.EXTRA, S.OWNER)
    K = set(T) | set(G.U.cell_rows("pdvd", "d116vflip"))
    R, nat, _, _, _, _ = S.rekey("d116vflip", "q35ctl", G.U.cell_rows("pdvd", "q35ctl"), K, 16)
    used = {r["key"] for r in csv.DictReader((l for l in open(AMEND1) if not l.startswith("#")), delimiter="\t")}
    pool = sorted(k for k in R if k in T and T[k][2] in ("record", "owner_review")
                  and T[k][0] not in ("MESSY", "UNCLEAR") and k not in POSTHOC and nat[k] not in used
                  and nat[k] not in {n for n, _, _ in NEW})
    cal = [(nat[k], "q35ctl", "calibration") for k in random.Random(a.seed).sample(pool, a.n)]
    rows = cal + NEW
    random.Random(a.seed + 1).shuffle(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        f.write("# doc qlmatch/35 amendment 2 item list (pdvd); role for grading only, never shown to a scanner\n")
        f.write("key\tdisplay_arm\trole\n")
        for k, c, r in rows:
            f.write(f"{k}\t{c}\t{r}\n")
    print(f"pool {len(pool)}; {len(cal)} calibration + {len(NEW)} new -> {a.out}")


if __name__ == "__main__":
    main()
