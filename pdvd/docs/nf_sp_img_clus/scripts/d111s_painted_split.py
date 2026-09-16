#!/usr/bin/env python3
"""doc pdvd/111 round 2 -- zero-charge Steiner vertices split into PAINTED and DEAD.

A retiled point with a zero charge on a plane sits either on a cell ImproveCluster_1::hack_activity_improved painted
(activity 1e-3 around the retiler's own Dijkstra paths) or on a dead channel (the same 1e-3 sentinel).  The dump carries,
per Steiner vertex, test_good_point's live/dead plane mask at radius 0.2 cm / ch_range 0 (STGV m02), so for vertices the
two can be told apart: a zero-charge plane whose mask has the dead bit is DEAD, otherwise PAINTED.  (Retiled points carry
no mask; d111s_graph_census.py reports them as painted-or-dead.)

Per detector, for the Steiner vertices of the clusters with a persisted STM record, by ridge offset (<= 1, 1-5, > 5 cm):
share with any painted plane, any dead-only zero plane, and none.

Usage: d111s_painted_split.py --det pdhd --arm d111hst [--jobs 10] > figs/111s_painted_split_<det>.txt
"""
import argparse, collections, glob, os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool
import numpy as np
import uproot
import d111s_common as C
A = C.A


def one(args):
    det, arm, pre = args
    acc = collections.Counter()
    d = f"{C.IMG}/{det}/work/{pre}_{arm}"
    tr = C.parse_trace(f"/home/xqian/tmp/d111/arm_{arm}/evt_{pre}.log.gz")
    have = set(uproot.open(f"{d}/tracking-stm.root")["T_rec_charge"].arrays(["cluster_id"], library="np")["cluster_id"].tolist())
    img = A.bee_layer(f"{d}/mabc-pr.zip", "clustering-global")
    done = set()
    for b in tr["blocks"]:
        cl = b["cl"]
        if cl not in have or cl in done:
            continue
        sb = C.graph_for(tr, cl, b["order"])
        mi = img[1] == cl
        if sb is None or mi.sum() < 5:
            continue
        done.add(cl)
        R = A.Ridge(img[0][mi], img[2][mi])
        if not R.ok:
            continue
        vo = R.offset(sb.V)
        q0 = sb.R_q[sb.V_old] < 1.0
        m = sb.V_m02
        dead = np.c_[[(m >= 0) & (((m >> (3 + p)) & 1) > 0) for p in range(3)]].T
        painted = (q0 & ~dead).any(axis=1)
        deadonly = (q0 & dead).any(axis=1) & ~painted
        for lab, mm in (("on", vo <= 1.0), ("off", (vo > 1.0) & (vo <= 5.0)), ("far", vo > 5.0)):
            acc[("n", lab)] += int(mm.sum())
            acc[("painted", lab)] += int((mm & painted).sum())
            acc[("dead", lab)] += int((mm & deadonly).sum())
            acc[("painted_term", lab)] += int((mm & painted & sb.V_term).sum())
            acc[("n_term", lab)] += int((mm & sb.V_term).sum())
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--jobs", type=int, default=10)
    a = ap.parse_args()
    evs = sorted(os.path.basename(p)[:-len(a.arm) - 1] for p in glob.glob(f"{C.IMG}/{a.det}/work/*_{a.arm}"))
    tot = collections.Counter()
    with Pool(a.jobs) as pool:
        for acc in pool.imap_unordered(one, [(a.det, a.arm, e) for e in evs]):
            tot.update(acc)
    print(f"# doc pdvd/111 round 2 painted vs dead Steiner vertices: det={a.det} arm={a.arm} events={len(evs)}")
    print("| ridge offset | vertices | painted (a zero plane, no dead bit) | dead only | terminals painted |")
    print("|---|---|---|---|---|")
    for lab, name in (("on", "<= 1 cm"), ("off", "1-5 cm"), ("far", "> 5 cm")):
        n = tot[("n", lab)]
        print(f"| {name} | {n} | {100*tot[('painted', lab)]/max(n,1):.1f} % | {100*tot[('dead', lab)]/max(n,1):.1f} % | "
              f"{tot[('painted_term', lab)]} of {tot[('n_term', lab)]} |")


if __name__ == "__main__":
    main()
