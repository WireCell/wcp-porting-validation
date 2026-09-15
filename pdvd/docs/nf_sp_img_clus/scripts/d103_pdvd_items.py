#!/usr/bin/env python3
"""doc pdvd/103 sec 10 -- the private item list of the PDVD production-lineage blind scan (figs/103_pred_amend4.txt sec 3).

    D103_PDVD_CELLS=d103v0,d103v1 D103_PDVD_RECORD=$IMG/pdvd/docs/scan/pdvd_stm_michel_p99rwon_carried_corrected_verdicts.json \
        python3 d103_pdvd_items.py --out /home/xqian/tmp/d103/items/pdvd2_items.tsv

Items: every candidate of A0 or A1 whose key is not in the truth record (role new), plus 20 judged truth items that are
candidates in A0 or A1, drawn with random.Random(103) (role calibration).  display_arm = A1 where the key is a candidate
there, else A0.  The file carries the roles and is PRIVATE (never shown to a scanner).  Refuses an existing --out.
"""
import argparse, json, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d103_union_grade as U


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-calibration", type=int, default=20)
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit(f"REFUSING: {a.out} exists")
    if not os.environ.get("D103_PDVD_CELLS") or "D103_PDVD_RECORD" not in os.environ:
        sys.exit("set D103_PDVD_CELLS and D103_PDVD_RECORD (figs/103_pred_amend4.txt)")
    cells = dict(U.CELLS["pdvd"])
    R = {lab: U.cell_rows("pdvd", arm) for lab, arm in cells.items()}
    rec = {r["key"]: r for r in json.load(open(U.PDVD_RECORD))}
    anyc = set(R["A0"]) | set(R["A1"])
    new = sorted(k for k in anyc if k not in rec)
    pool = sorted(k for k in anyc if k in rec and U.strip(rec[k]["verdict"]) not in ("MESSY", "UNCLEAR"))
    cal = sorted(random.Random(103).sample(pool, a.n_calibration))
    arm = lambda k: cells["A1"] if k in R["A1"] else cells["A0"]
    with open(a.out, "w") as fh:
        fh.write(f"# doc pdvd/103 sec 10 PDVD production-lineage item list (amend4; cells {cells}; truth {os.path.basename(U.PDVD_RECORD)}); "
                 "the role column is for grading only, never shown to the scanner\n")
        fh.write("key\tdisplay_arm\trole\n")
        for k in new:
            fh.write(f"{k}\t{arm(k)}\tnew\n")
        for k in cal:
            fh.write(f"{k}\t{arm(k)}\tcalibration\n")
    print(f"candidates A0 {len(R['A0'])} A1 {len(R['A1'])} union {len(anyc)}; new (unlabelled) {len(new)}; "
          f"calibration {len(cal)} of pool {len(pool)}; display on A1 {sum(1 for k in new + cal if arm(k) == cells['A1'])}")


if __name__ == "__main__":
    main()
