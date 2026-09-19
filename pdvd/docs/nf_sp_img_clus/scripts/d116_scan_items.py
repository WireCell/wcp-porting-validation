#!/usr/bin/env python3
"""doc pdvd/116 sec 5 -- the PRIVATE item list of the blind scan of one detector: every candidate of the named cells
that has no truth of any source (d113_grade precedence, plus --extra-record), with its display arm = the first cell
(in the order given) where it is a candidate, plus a calibration draw of judged items that are candidates in the
first cell (role calibration; the V2 agreement check of d103_scan_record.py).  Format = doc 103's
(key, display_arm, role), read by d103_scan_set.py.  Refuses an existing --out.  Never shown to a scanner.

Usage: d116_scan_items.py --det pdvd --cells d115voff,d115vbwp05,d116vr1,... --out /home/xqian/tmp/d116/items/pdvd_items.tsv [--calib-frac 0.10] [--seed 116] [--extra-record F] [--exclude F ...]
--exclude: earlier item lists of this round (keys already scanned are not drawn again).
"""
import argparse, csv, json, os, random, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d113_grade as G
import d103_union_grade as U


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True, choices=["pdhd", "pdvd"])
    ap.add_argument("--cells", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--calib-frac", type=float, default=0.10); ap.add_argument("--seed", type=int, default=116)
    ap.add_argument("--extra-record", default=None); ap.add_argument("--exclude", nargs="*", default=[])
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit(f"REFUSING: {a.out} exists")
    T, counts = G.truth(a.det)
    if a.extra_record:
        for r in json.load(open(a.extra_record)):
            if r["key"] not in T:
                T[r["key"]] = U.row_truth(r, "smx116")
    done = set()
    for f in a.exclude:
        done |= {r["key"] for r in csv.DictReader((l for l in open(f) if not l.startswith("#")), delimiter="\t")}
    cells = a.cells.split(",")
    R = {c: U.cell_rows(a.det, c) for c in cells}
    new = []
    for c in cells:
        for k in sorted(R[c]):
            if k not in T and k not in done and all(k != n[0] for n in new):
                new.append((k, c, "new"))
    judged = sorted(k for k in R[cells[0]] if k in T and T[k][0] not in ("MESSY", "UNCLEAR") and k not in done)
    ncal = max(2, int(round(a.calib_frac * len(new))))
    cal = [(k, cells[0], "calibration") for k in random.Random(a.seed).sample(judged, min(ncal, len(judged)))]
    rows = new + cal
    random.Random(a.seed + 1).shuffle(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        f.write(f"# doc pdvd/116 blind scan item list ({a.det}; cells {cells}; truth {counts}); role for grading only, never shown to a scanner\n")
        f.write("key\tdisplay_arm\trole\n")
        for k, c, role in rows:
            f.write(f"{k}\t{c}\t{role}\n")
    by = {}
    for k, c, role in rows:
        by[(c, role)] = by.get((c, role), 0) + 1
    print(f"{a.det}: {len(new)} unlabelled candidates + {len(cal)} calibration -> {a.out}; by (display_arm, role): {by}")


if __name__ == "__main__":
    main()
