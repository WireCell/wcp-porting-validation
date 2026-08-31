#!/usr/bin/env python3
"""doc pr/125 front 1 -- dQ/dx over the pass3_cone guard's decline set.

Owner (2026-08-29): the four fake electrons (94392/52693/77328/173819) "should
not be ided as electron, but track object, check dQ/dx".  All four are fixed
by pr/124's shower_pass3_cone_guard_len=15 (DEFAULT OFF); the cost row is
415278.  Question here: does a dQ/dx qualifier on the DECLINE test separate
415278's three tracks (pi 36.4 / mu 56.3 / mu 22.1 cm) from the four events'
declined tracks (mu 29.8 / mu 46.8 / mu 34.3 / p 16.5 / p 37.8 cm)?  If yes,
the guard flips refined (K1); if no, the owner pre-decided "flip anyway" at
len=15.

Inputs (no new arms): docs/pr/pr124-pass3-census.tsv (site=pass3_cone rows,
the absorb census from the pr/124 dbg arms) + the final-binary OFF dumps
work-pr124r1-{dbg141v2,dbgv2}-* for per-segment median dQ/dx (points dQ/dx
normalized to the dump's own dqdx_ref muon-plateau tail, same convention as
pr124_gapband_scan.py).

Repro:
  ./scripts/pr125_p3guard_dqdx.py --tsv docs/pr/pr125-p3guard-dqdx.tsv
"""
import argparse
import csv
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(HERE)
TRK_PIDS = (13, 211, 2212)
GUARD_LEN = 15.0
KEY_EVTS = (94392, 52693, 77328, 173819, 415278)
OFF_ROOTS = ("work-pr124r1-dbg141v2-*", "work-pr124r1-dbgv2-*")


def find_dump(ev):
    for g in OFF_ROOTS:
        for r in sorted(glob.glob(os.path.join(SX, g))):
            djs = glob.glob(os.path.join(r, "pr_evt%d" % ev, "calib-pr-evt%d.json" % ev))
            if djs:
                return djs[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", default=os.path.join(SX, "docs/pr/pr124-pass3-census.tsv"))
    ap.add_argument("--tsv")
    args = ap.parse_args()

    cone = [r for r in csv.DictReader(open(args.census), delimiter="\t")
            if r["site"] == "pass3_cone"]
    dumps = {}
    rows = []
    for r in cone:
        ev, seg = int(r["ev"]), int(r["seg"])
        pdg = abs(int(r["pdg"]))
        if pdg not in TRK_PIDS:
            continue
        if ev not in dumps:
            dj = find_dump(ev)
            dumps[ev] = json.load(open(dj)) if dj else None
        j = dumps[ev]
        if j is None:
            continue
        segs = {s["id"]: s for s in j.get("segments", [])}
        s = segs.get(seg)
        if s is None:
            continue
        mip = (j.get("dqdx_ref", {}).get("muon") or [54657.7])[-1]
        dqdx = [abs(p.get("dQ", 0.0)) / p["dx"] / mip
                for p in s.get("points", []) if p.get("dx", 0.0) > 1e-6]
        length = s.get("length", 0.0)
        declined = int(length > GUARD_LEN)
        rows.append(dict(
            ev=ev, seg=seg, pdg=int(r["pdg"]), verdict=r["verdict"],
            kept=int(r["kept"]), len_cm=round(length, 2),
            mdqdx=round(float(np.median(dqdx)), 3) if dqdx else -1.0,
            p90dqdx=round(float(np.percentile(dqdx, 90)), 3) if dqdx else -1.0,
            score=s.get("particle_score", -1.0), declined=declined,
            key=int(ev in KEY_EVTS)))

    cols = ["ev", "seg", "pdg", "verdict", "kept", "len_cm", "mdqdx",
            "p90dqdx", "score", "declined", "key"]
    if args.tsv:
        with open(args.tsv, "w") as f:
            f.write("\t".join(cols) + "\n")
            for r in sorted(rows, key=lambda r: (r["ev"], r["seg"])):
                f.write("\t".join(str(r[c]) for c in cols) + "\n")
        print("wrote %d track-pdg cone-absorb rows -> %s" % (len(rows), args.tsv))

    dec = [r for r in rows if r["declined"]]
    print("\ntrack-pdg pass3_cone absorbs: %d total, %d in the len>15 decline set"
          % (len(rows), len(dec)))

    print("\nKEY events, decline-set rows (the guard's actual fires):")
    for r in sorted(dec, key=lambda r: (r["ev"] not in KEY_EVTS, r["ev"], r["seg"])):
        if r["key"]:
            print("  evt%-7d seg=%-7d pdg=%-6d len=%-7.2f mdqdx=%-6.3f "
                  "p90=%-6.3f score=%-8s verdict=%s"
                  % (r["ev"], r["seg"], r["pdg"], r["len_cm"], r["mdqdx"],
                     r["p90dqdx"], r["score"], r["verdict"]))

    print("\nfull decline set sorted by mdqdx (separator hunt; FIX = the 4 owner"
          "\nevents' segs, COST = 415278's segs, other = collateral exposure):")
    for r in sorted(dec, key=lambda r: r["mdqdx"]):
        cls = ("COST" if r["ev"] == 415278 else
               "FIX" if r["key"] else "other")
        print("  %-5s evt%-7d seg=%-7d pdg=%-6d len=%-7.2f mdqdx=%-6.3f p90=%-6.3f %s"
              % (cls, r["ev"], r["seg"], r["pdg"], r["len_cm"], r["mdqdx"],
                 r["p90dqdx"], r["verdict"]))


if __name__ == "__main__":
    main()
