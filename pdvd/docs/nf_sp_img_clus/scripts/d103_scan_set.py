#!/usr/bin/env python3
"""doc pdvd/103 sec 5 -- build the blind scan set of one detector from the private item list and the display arms'
prep sheets.  Fork of the sheet-writing block of d99_swap_scan_set.py (untouched).

Inputs
  --items  /home/xqian/tmp/d103/items/<det>_items.tsv  (key, display_arm, role new|calibration; PRIVATE)
  --prep-root /home/xqian/tmp/d103/round  (prep_<arm>/smprep-*.json and sheet_<arm>/<det>_stm_michel_scan_sheet.tsv)
Outputs (refuses an existing --out)
  OUT/sheet_<arm>.tsv   the six columns the viewer and shoot.sh read (scan_id tranche event cluster npts muon_len_cm),
                        tranche = 1 on every row, rows in a seeded shuffle, the arm's own scan_id kept
  OUT/items_all.txt     every scannable key, seeded shuffle (the list the waves are cut from; no role, no arm)
  OUT/no_payload.tsv    item keys the prep did not write a payload for (prep filters has_pass, n_profile_pts >= 20,
                        muon_len >= 10 cm) -- reported, never silently dropped
"""
import argparse, csv, os, random, sys
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True, choices=["pdhd", "pdvd"])
    ap.add_argument("--items", required=True)
    ap.add_argument("--prep-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1032)
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit(f"REFUSING: {a.out} exists")
    items = list(csv.DictReader((l for l in open(a.items) if not l.startswith("#")), delimiter="\t"))
    by_arm = {}
    for r in items:
        by_arm.setdefault(r["display_arm"], []).append(r)
    os.makedirs(a.out)
    scannable, missing = [], []
    for arm, rows in sorted(by_arm.items()):
        sheet = f"{a.prep_root}/sheet_{arm}/{a.det}_stm_michel_scan_sheet.tsv"
        S = {"%s/%s" % (r["event"], r["cluster"]): r
             for r in csv.DictReader((l for l in open(sheet) if not l.startswith("#")), delimiter="\t")}
        out = []
        for r in rows:
            k = r["key"]
            ev, cl = k.split("/")
            pay = f"{a.prep_root}/prep_{arm}/smprep-{ev}-c{cl}.json"
            if k not in S or not os.path.exists(pay):
                missing.append((k, arm, r["role"], "not in sheet" if k not in S else "no payload file"))
                continue
            out.append(S[k])
            scannable.append(k)
        random.Random(a.seed).shuffle(out)
        with open(f"{a.out}/sheet_{arm}.tsv", "w") as f:
            f.write(f"# doc pdvd/103 blind scan set ({a.det}), display arm {arm}; tranche = 1 on every row\n")
            f.write("scan_id\ttranche\tevent\tcluster\tnpts\tmuon_len_cm\n")
            for s in out:
                f.write("\t".join([s["scan_id"], "1", s["event"], s["cluster"], s["npts"], s["muon_len_cm"]]) + "\n")
        print(f"  {arm}: {len(out)} rows -> sheet_{arm}.tsv")
    random.Random(a.seed + 1).shuffle(scannable)
    with open(f"{a.out}/items_all.txt", "w") as f:
        f.write("".join(k + "\n" for k in scannable))
    with open(f"{a.out}/no_payload.tsv", "w") as f:
        f.write("key\tdisplay_arm\trole\treason\n" + "".join("\t".join(m) + "\n" for m in missing))
    print(f"[{a.det}] items {len(items)}; scannable {len(scannable)}; without a payload {len(missing)} "
          f"{dict(Counter((m[2], m[3]) for m in missing))}")


if __name__ == "__main__":
    main()
