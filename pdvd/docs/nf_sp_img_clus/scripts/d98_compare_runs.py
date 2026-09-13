#!/usr/bin/env python3
"""doc pdvd/98 sec 11 -- the same PDVD Michels in round 1 (production p96vprod: v5-order SP frames, v7 imaging) and in
the latest configuration (doc pdvd/99 p98vonq on the p98von frames: v7 throughout, top SP gain 0.889).

    python3 d98_compare_runs.py [--run latest] > ../../scan/d98/latest/compare_round1.txt

A latest candidate is the same object as a round-1 candidate when its carried-record item names that round-1 key in
`carried_from` (doc pdvd/99 sec 5, geometric match).  On the common in-range Michels (in range on BOTH runs) the
pre-registered readouts of d98_plots.readout are recomputed for each run on identical objects, so the difference is the
reconstruction change alone, not the sample.
"""
import argparse, csv, json, sys
import numpy as np
from scipy import stats
import d98_plots as P

IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
CARRIED = IMG + "/pdvd/docs/scan/pdvd_stm_michel_smx9_carried_p98vonq.json"


def load(path):
    rows = [r for r in csv.DictReader((l for l in open(path) if not l.startswith("#")), delimiter="\t") if r["det"] == "pdvd"]
    for r in rows:
        for k, v in r.items():
            if k not in ("det", "key", "tier", "kind", "source", "d97"):
                r[k] = float(v)
    return {r["key"]: r for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="latest")
    a = ap.parse_args()
    r1 = load(P.SCAN + "/scores.tsv")
    rl = load(f"{P.SCAN}/{a.run}/scores.tsv")
    cf = {r["key"]: r["carried_from"] for r in json.load(open(CARRIED))}
    pairs = [(rl[k], r1[cf[k]]) for k in rl if cf.get(k) in r1]
    print(f"# doc pdvd/98 -- PDVD Michels scored in round 1 (p96vprod) and in run {a.run} (p98vonq); d98_compare_runs.py")
    print(f"round 1 scored {len(r1)}, {a.run} scored {len(rl)}, common objects {len(pairs)}; "
          f"{a.run} only {len(rl) - len(pairs)}, round 1 only {len(r1) - len(pairs)}")
    L = np.array([[p[0]["drift_tick"], p[1]["drift_tick"], p[0]["mu_Z"], p[1]["mu_Z"], p[0]["q_keep"], p[1]["q_keep"],
                   p[0]["frac_lost"], p[1]["frac_lost"], p[0]["inrange"], p[1]["inrange"], p[0]["sig_Z"], p[1]["sig_Z"]]
                  for p in pairs])
    dd, dm = L[:, 0] - L[:, 1], L[:, 2] - L[:, 3]
    print(f"drift label latest - round1: median {np.median(dd):+.2f} cm, p16/p84 {np.quantile(dd, .16):+.2f}/{np.quantile(dd, .84):+.2f}")
    print(f"mu_Z latest - round1: median {np.median(dm):+.1f} cm, p16/p84 {np.quantile(dm, .16):+.1f}/{np.quantile(dm, .84):+.1f}; "
          f"r(mu_Z latest, mu_Z round1) = {stats.pearsonr(L[:, 2], L[:, 3])[0]:+.3f}")
    print(f"q_keep latest/round1: median {np.median(L[:, 4] / L[:, 5]):.3f}; frac_lost median latest {np.median(L[:, 6]):.2f} "
          f"round1 {np.median(L[:, 7]):.2f}")
    both = (L[:, 8] == 1) & (L[:, 9] == 1)
    print(f"in range on both runs: {both.sum()}")
    for name, xi, yi, si in ((a.run, 0, 2, 10), ("round1", 1, 3, 11)):
        rng = np.random.default_rng(P.SEED)
        s = P.readout(L[both, xi], L[both, yi], L[both, si], rng)
        print(f"  common in-range, {name:7s} Z: {P.fmt(s)}")
    # by drift volume (anodes 4-7 top: the x1.125 gain; 0-3 bottom: the wire order only)
    top = np.array([p[0]["apa"] >= 4 for p in pairs])
    for vol, vs in (("top", top), ("bottom", ~top)):
        sel = both & vs
        print(f"  {vol}: common {vs.sum()}, in range on both {sel.sum()}; q_keep latest/round1 median "
              f"{np.median(L[vs, 4] / L[vs, 5]):.3f}; mu_Z latest - round1 median {np.median(dm[vs]):+.1f} cm")
        for name, xi, yi, si in ((a.run, 0, 2, 10), ("round1", 1, 3, 11)):
            rng = np.random.default_rng(P.SEED)
            s = P.readout(L[sel, xi], L[sel, yi], L[sel, si], rng)
            print(f"    {vol:6s} common in-range, {name:11s} Z: {P.fmt(s)}")


if __name__ == "__main__":
    sys.exit(main())
