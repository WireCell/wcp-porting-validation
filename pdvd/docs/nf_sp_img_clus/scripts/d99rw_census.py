#!/usr/bin/env python3
"""doc pdvd/99 sec 4.4 -- the STM population per arm with the readout window each event's frames really have.

    python3 d99rw_census.py --arms p96vprod p99rgprod p98vonq p99rgon [p99rwprod p99rwon] > ../d99/rw_census.txt

Fork by duplication of d99_readout_window.py (its committed output stays as it is).  Per arm and per run group
(run 039349 = 6400-tick frames; runs 039252+039253 = 10000): events, readout_edge_guard firings (the "rejected" Info
lines of wct_pr_*.log), STM candidates (T_stm_michel rows), is_stm by volume (stop_x > 0 = top), michel_found among is_stm,
and readout_window_ticks from pctree-evt*.tlas.
"""
import argparse, collections, glob, re, sys
from concurrent.futures import ProcessPoolExecutor
import d99_match as M
import d99_population as Pp

WORK = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/work"


def one(job):
    evt, arm = job
    d = f"{WORK}/{evt}_{arm}"
    c = collections.Counter()
    for lg in glob.glob(f"{d}/wct_pr_*.log"):
        for line in open(lg, errors="replace"):
            if "readout_edge_guard: cluster" in line and "rejected" in line:
                c["fire"] += 1
    for t in glob.glob(f"{d}/pctree-evt*.tlas"):
        m = re.search(r"^readout_window_ticks=(\d+)", open(t).read(), re.M)
        c["win_" + (m.group(1) if m else "none")] += 1
    stm, _ = M.load_stm(evt, arm)
    for r in stm.values():
        c["cand"] += 1
        if int(r["is_stm"]) == 1:
            c["is_stm_" + ("top" if r["stop_x"] > 0 else "bottom")] += 1
            c["michel"] += int(r["michel_found"]) == 1
    c["events"] += 1
    return evt, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--jobs", type=int, default=24)
    a = ap.parse_args()
    ev = Pp.events()
    print("# doc pdvd/99 sec 4.4 -- STM population per arm, split by frame length (d99rw_census.py)")
    print(f"{'arm':10s} {'runs':16s} {'events':>6s} {'guard firings':>13s} {'candidates':>10s} {'is_stm':>6s} {'top':>4s} "
          f"{'bottom':>6s} {'michel':>6s}  readout_window_ticks")
    for arm in a.arms:
        with ProcessPoolExecutor(a.jobs) as ex:
            res = dict(ex.map(one, [(e, arm) for e in ev]))
        for label, sel in (("039349 (6400)", lambda e: e.startswith("039349")),
                           ("039252+3 (10000)", lambda e: not e.startswith("039349")), ("all", lambda e: True)):
            c = sum((res[e] for e in ev if sel(e)), collections.Counter())
            win = {k[4:]: v for k, v in sorted(c.items()) if k.startswith("win_")}
            print(f"{arm:10s} {label:16s} {c['events']:6d} {c['fire']:13d} {c['cand']:10d} "
                  f"{c['is_stm_top'] + c['is_stm_bottom']:6d} {c['is_stm_top']:4d} {c['is_stm_bottom']:6d} {c['michel']:6d}  {win}")


if __name__ == "__main__":
    sys.exit(main())
