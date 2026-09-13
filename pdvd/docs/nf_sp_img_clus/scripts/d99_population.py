#!/usr/bin/env python3
"""doc pdvd/99 sec 4.3 / 6.3 -- the STM population without the hand-scan record, and where it moved.

    python3 d99_population.py --pairs p96vprod:p96vprod p96vprod:p98voffq p98voffq:p96vprod p98voffq:p98vonq p98vonq:p98voffq \
        [--json J]

Per arm: STM candidates (T_stm_michel rows), is_stm, michel_found among is_stm, by drift volume (stop_x > 0 = top).
Per pair A:B: every is_stm cluster of A is found in B by geometry (d99_match.match_cluster on the Bee
clustering-global points, the record carry's own matcher and thresholds) and classified by what B says about the matched
cluster: is_stm / candidate but not is_stm / not a candidate / object not matched (non-ok).  A:A is the null control.
Per event: how many of A's is_stm clusters are not is_stm in B (the churn), for joining with frame identity.
"""
import argparse, collections, json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.spatial import cKDTree
import d99_match as M

EVENTS = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/stm/events.txt"


def events():
    out = []
    for line in open(EVENTS):
        p = line.split()
        if len(p) >= 2 and p[0].isdigit():
            out.append("%06d_%s" % (int(p[0]), p[1]))
    return out


def census(job):
    evt, arm = job
    stm, _ = M.load_stm(evt, arm)
    c = collections.Counter()
    for d in stm.values():
        vol = "top" if d["stop_x"] > 0 else "bottom"
        c[("cand", vol)] += 1
        if int(d["is_stm"]) == 1:
            c[("is_stm", vol)] += 1
            if int(d["michel_found"]) == 1:
                c[("is_stm_michel", vol)] += 1
    return c


def pair(job):
    evt, A, B = job
    sa, _ = M.load_stm(evt, A)
    sb, _ = M.load_stm(evt, B)
    ba, bb = M.load_bee(evt, A), M.load_bee(evt, B)
    out = collections.Counter()
    if ba is None or bb is None:
        return evt, out
    tree = cKDTree(bb[0])
    for cid, d in sa.items():
        if int(d["is_stm"]) != 1:
            continue
        vol = "top" if d["stop_x"] > 0 else "bottom"
        pts = ba[0][ba[1] == cid]
        if len(pts) == 0:
            out[("no_points", vol)] += 1
            continue
        m = M.match_cluster(pts, bb[0], bb[1], tree)
        st = M.status_of(m)
        if st != "ok":
            out[("object_" + st, vol)] += 1
            continue
        e = sb.get(m["best"])
        if e is None:
            out[("not_candidate", vol)] += 1
        elif int(e["is_stm"]) != 1:
            out[("candidate_not_stm", vol)] += 1
        else:
            out[("is_stm", vol)] += 1
            if int(d["michel_found"]) == 1:
                out[("michel_kept" if int(e["michel_found"]) == 1 else "michel_lost", vol)] += 1
            elif int(e["michel_found"]) == 1:
                out[("michel_gained", vol)] += 1
    return evt, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    ev = events()
    arms = sorted({x for p in a.pairs for x in p.split(":")})
    res = {"census": {}, "pairs": {}}
    with ProcessPoolExecutor(a.jobs) as ex:
        print(f"events {len(ev)}")
        for arm in arms:
            tot = collections.Counter()
            for c in ex.map(census, [(e, arm) for e in ev]):
                tot.update(c)
            res["census"][arm] = {f"{k[0]}:{k[1]}": v for k, v in tot.items()}
            line = f"  {arm:9s}"
            for k in ("cand", "is_stm", "is_stm_michel"):
                t, b = tot[(k, "top")], tot[(k, "bottom")]
                line += f"  {k} {t + b} (top {t}, bottom {b})"
            print(line)
        for p in a.pairs:
            A, B = p.split(":")
            tot, per = collections.Counter(), {}
            for evt, c in ex.map(pair, [(e, A, B) for e in ev]):
                tot.update(c)
                per[evt] = sum(v for k, v in c.items() if k[0] not in ("is_stm", "michel_kept", "michel_lost", "michel_gained"))
            n = sum(v for k, v in tot.items() if k[0] not in ("michel_kept", "michel_lost", "michel_gained"))
            print(f"  {A} is_stm -> {B}: n {n}")
            for vol in ("top", "bottom", None):
                keys = ("is_stm", "candidate_not_stm", "not_candidate", "object_merged", "object_split", "object_lost",
                        "object_ambiguous", "no_points")
                cnt = {k: (tot[(k, vol)] if vol else tot[(k, "top")] + tot[(k, "bottom")]) for k in keys}
                nn = sum(cnt.values())
                mk = tot[("michel_kept", vol)] if vol else tot[("michel_kept", "top")] + tot[("michel_kept", "bottom")]
                ml = tot[("michel_lost", vol)] if vol else tot[("michel_lost", "top")] + tot[("michel_lost", "bottom")]
                mg = tot[("michel_gained", vol)] if vol else tot[("michel_gained", "top")] + tot[("michel_gained", "bottom")]
                print(f"    {vol or 'all':6s} n {nn:3d}  stays is_stm {cnt['is_stm']} ({cnt['is_stm'] / max(nn, 1):.3f})  "
                      + "  ".join(f"{k} {v}" for k, v in cnt.items() if k != "is_stm" and v)
                      + f"  | among those staying: Michel kept {mk} lost {ml} gained {mg}")
            churn = {e: v for e, v in per.items() if v}
            print(f"    events with churn {len(churn)}/{len(ev)}: " + " ".join(f"{e}:{v}" for e, v in sorted(churn.items())))
            res["pairs"][p] = {"total": {f"{k[0]}:{k[1]}": v for k, v in tot.items()}, "per_event_churn": per}
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
