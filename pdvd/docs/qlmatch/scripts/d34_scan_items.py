#!/usr/bin/env python3
"""doc qlmatch/34 step D -- the round-3 TOP-UP blind-scan items (rules: d34/prereg.md).

The d33_scan_items.py machinery, unchanged (same light-neutral sheet, candidate rule, co-matched intersection,
shuffles, looks, duplicates, calibration rule), with three pre-registered differences:
  - population = movers over the round-2 + round-3 arm set MINUS the 329 round-2 movers (--exclude-key, the
    committed round-2 key);
  - seed 34, cap 300;
  - the scanner INDEX header says "round 3 top-up".
--arms must list the round-2 arms too, so the co-matched intersection covers every frozen arm.

    python3 d34_scan_items.py --exclude-key ../d33/scan_r2/key.json --work-root /home/xqian/tmp/p31/wroot \
        --arms <r2 arms>,<r3 arms> --time-map ../d32/time_map_ctl_to_q32ti.json \
        --sheet-dir /home/xqian/tmp/p34/scan/r3 --key /home/xqian/tmp/p34/scan_key_r3/key.json [--dry-run]
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import d33_scan_items as I       # noqa: E402

I.SEED = 34
I.MAX_MOVERS = 300


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--exclude-key", required=True)
    pre.add_argument("--nwave", type=int, default=I.NWAVE)
    a, rest = pre.parse_known_args()
    I.NWAVE = a.nwave
    ex = {(s["evt"], s["uid"]) for s in json.load(open(a.exclude_key))["sheets"] if s["kind"] == "mover"}
    base_build = I.build

    def build(args):
        arms, movers, pool, co = base_build(args)
        n0 = len(movers)
        movers = [m for m in movers if (m["evt"], m["uid"]) not in ex]
        print(f"top-up: {n0} movers over all arms, {n0 - len(movers)} already scanned in round 2 "
              f"({len(ex)} there), {len(movers)} new")
        return arms, movers, pool, co
    I.build = build
    sys.argv = [sys.argv[0]] + rest
    I.main()
    sd = next((rest[i + 1] for i, x in enumerate(rest) if x == "--sheet-dir"), None)
    for p in glob.glob(os.path.join(sd, "wave*", "INDEX.md")) if sd else []:
        s = open(p).read().replace("doc qlmatch/33 blind scan, round 2,", "doc qlmatch/34 blind scan, round 3 top-up,")
        open(p, "w").write(s)


if __name__ == "__main__":
    main()
