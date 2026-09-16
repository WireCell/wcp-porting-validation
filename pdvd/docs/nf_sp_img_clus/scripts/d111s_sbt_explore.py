#!/usr/bin/env python3
"""doc pdvd/111 round 2 -- EXPLORATORY, NOT PRE-REGISTERED (found after the frozen ladder had run; it cannot change the
ladder's verdict): charge-weight the same-blob terminal edges.

establish_same_blob_steiner_edges_steiner_graph (SteinerGrapher.cxx) adds every terminal-to-Steiner-point edge of a blob
at its PLAIN length, while every tree edge carries create_enhanced_steiner_graph's charge factor
0.8 + 0.4 * (0.5 Q0/(Qs+Q0) + 0.5 Q0/(Qt+Q0)), Q0 = 10000, i.e. 0.8x-1.2x.  (The base-graph version of the same
function discounts by 0.8 / 0.9; this one has no factor.)  So an `sbt` edge is a fixed-price shortcut across a blob.

Candidates replayed on the same walks as d111s_replay.py (round-1 rough walks):
  base                the dumped weights
  SBT_q               sbt edges re-weighted with the tree edges' charge factor (vertex charges = STGV q, the same
                      calc_charge_wcp values the factor already uses)
  SBT_q+P_paint:2     SBT_q, then the amendment-1 painted-end factor (1 + 2 p/2)
  G_3pl:2, P_paint:2  the ladder's references on the same walks
Metrics and spots as in d111s_replay.py.

Usage: d111s_sbt_explore.py --det pdhd --arm d111hst [--jobs 10] --out figs/111s_explore_sbt_pdhd
"""
import argparse, collections, glob, json, os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool
import numpy as np
import d111s_replay as RP

Q0 = 10000.0
CANDS = [("base", 0), ("SBT_q", 0), ("SBT_q+P_paint", 2), ("G_3pl", 2), ("P_paint", 2)]


def weights_x(sb, fam, k):
    if fam in ("SBT_q", "SBT_q+P_paint"):
        w = sb.E_w.copy()
        m = sb.E_src == 2
        qs = np.clip(sb.V_q[sb.E_s[m]], 0, None); qt = np.clip(sb.V_q[sb.E_t[m]], 0, None)
        w[m] = sb.E_len[m] * (0.8 + 0.4 * (0.5 * Q0 / (qs + Q0) + 0.5 * Q0 / (qt + Q0)))
        if fam == "SBT_q+P_paint":
            painted = RP.painted_vertices(sb)
            p = painted[sb.E_s].astype(float) + painted[sb.E_t].astype(float)
            w = w * (1.0 + k * p / 2.0)
        return w
    return RP.ORIG_WEIGHTS(sb, fam, k)


RP.ORIG_WEIGHTS = RP.weights
RP.weights = weights_x
RP.CANDIDATES = CANDS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    logd = f"/home/xqian/tmp/d111/arm_{a.arm}"
    evs = sorted(os.path.basename(p)[:-len(a.arm) - 1] for p in glob.glob(f"{RP.C.IMG}/{a.det}/work/*_{a.arm}"))
    tot = collections.defaultdict(collections.Counter)
    spots = collections.defaultdict(lambda: collections.defaultdict(list))
    with Pool(a.jobs) as pool:
        for out, sp, miss in pool.imap_unordered(RP.one, [(a.det, a.arm, logd, e) for e in evs]):
            for k, v in out.items():
                tot[k].update(v)
            for name, d in sp.items():
                for key, vals in d.items():
                    spots[name][key].extend(vals)
    base = tot["base:0"]
    sh = lambda acc, n, d: acc[n] / acc[d] if acc[d] else float("nan")
    b_off = sh(base, "off", "L")
    L = [f"# doc pdvd/111 round 2 EXPLORATORY (not pre-registered) same-blob-terminal edge pricing: det={a.det} arm={a.arm} "
         f"events={len(evs)}; walks {base['walks']}",
         "| candidate | off (> 1 cm) | rel. change | far (> 5 cm) | wiggle | cov | seed length (m) | walks changed |",
         "|---|---|---|---|---|---|---|---|"]
    for fam, k in CANDS:
        acc = tot[f"{fam}:{k}"]
        off = sh(acc, "off", "L")
        L.append(f"| {fam}:{k} | {100*off:.2f} % | {100*(off-b_off)/b_off:+.1f} % | {100*sh(acc,'far','L'):.2f} % | "
                 f"{100*sh(acc,'wig','nint'):.2f} % | {100*sh(acc,'cov','qtot'):.2f} % | {acc['L']/100:.1f} | {acc['changed']} |")
    if spots:
        L += ["", "| spot | " + " | ".join(f"{f}:{k}" for f, k in CANDS) + " |", "|---|" + "---|" * len(CANDS)]
        for name in sorted(spots):
            L.append(f"| {name} | " + " | ".join("/".join(f"{v:.2f}" for v in spots[name].get(f"{f}:{k}", [])) or "-"
                                                 for f, k in CANDS) + " |")
    txt = "\n".join(L) + "\n"
    open(a.out + ".txt", "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
