#!/usr/bin/env python3
"""doc sbnd_xin/123 round 1 -- do the hit flashes restore the partner the cathode rescue had to
supply geometrically?

For every move of the cbr3 census (docs/123_flash/123_rescue_vs_hits.tsv: the near-TPC cluster's
t0, the far-TPC partner's t0 and the reco1 flash situation) look up the r1_census.py table of the
hit-flash arm (one row per hit flash, class match/absorbed/vetoed/dropped/piece/prepulse) and ask,
in the far TPC (the side whose reco1 flash was missing or wrong), whether a hit flash exists within
--win-us of the NEAR cluster's t0 (the time the partner should carry) and what class it has.
'restored' = a hit flash that is NOT a reco1 match (absorbed/vetoed/dropped) sits there; 'reco1 had
it' = the flash there is a reco1 match (the move was a Q/L choice, not a light loss); 'none' = still
no flash there.

usage: r1_rescue_xcheck.py <census.tsv (r1_census --tsv of the mcp arms, concatenated)>
                           [--moves docs/123_flash/123_rescue_vs_hits.tsv] [--win-us 0.3]
"""
import argparse, csv, sys
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("census", nargs="+")
    ap.add_argument("--moves", default="docs/123_flash/123_rescue_vs_hits.tsv")
    ap.add_argument("--win-us", type=float, default=0.3)
    a = ap.parse_args()
    flashes = defaultdict(list)   # (event, tpc) -> [(t_us, pe, class)]
    for path in a.census:
        with open(path) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if r["ours_idx"] == "-1":
                    continue   # reco1 flash with no hit flash
                flashes[(int(r["event"]), int(r["tpc"]))].append((float(r["ours_t_us"]), float(r["ours_pe"]), r["class"]))
    n = defaultdict(int); out = []
    with open(a.moves) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            ev = int(r["event"]); cls = r["class"]
            # two readings of a move (doc 123 sec 6): side A = the far TPC should carry a flash at the
            # NEAR cluster's t0; side B = the near TPC should carry one at the FAR cluster's t0 (the
            # longer-half rule can adopt either flash).  A hit flash on either side that reco1 did not
            # have (absorbed / vetoed / dropped) is the restored partner.
            sides = []
            try:
                sides.append((int(r["far_tpc"]), float(r["near_t0_us"]), "A"))
            except ValueError:
                pass
            try:
                sides.append((int(r["near_tpc"]), float(r["far_t0_us"]), "B"))
            except ValueError:
                pass
            if not sides:
                verdict = "n/a"
            elif all((ev, tpc) not in flashes for tpc, _, _ in sides):
                verdict = "event-not-in-census"
            else:
                found = []
                for tpc, t0, side in sides:
                    for x in flashes.get((ev, tpc), []):
                        if abs(x[0] - t0) <= a.win_us:
                            found.append((x[1], x[2], side, x[0]))
                if not found:
                    verdict = "none"
                else:
                    restored = [f for f in found if f[1] not in ("match", "bugged")]
                    best = max(restored or found, key=lambda f: f[0])
                    verdict = ("restored:%s@%s" % (best[1], best[2])) if restored else "reco1-had-it"
                    r["hit_t_us"] = "%.4f" % best[3]; r["hit_pe"] = "%.1f" % best[0]
            n[(cls, verdict.split(":")[0])] += 1
            out.append((ev, r["sample"], cls, r.get("dt0_us", ""), verdict, r.get("hit_t_us", ""), r.get("hit_pe", "")))
    print("event\tsample\tclass\tdt0_us\tverdict\thit_t_us\thit_pe")
    for o in out:
        print("\t".join(str(x) for x in o))
    print("\n# summary (class, verdict): count", file=sys.stderr)
    for k in sorted(n):
        print("#  %-22s %-22s %d" % (k[0], k[1], n[k]), file=sys.stderr)


if __name__ == "__main__":
    main()
