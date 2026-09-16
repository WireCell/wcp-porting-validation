#!/usr/bin/env python3
"""doc pdvd/111 round 2 -- EXPLORATORY (not pre-registered, does not change the ladder verdict): is the wiggle that every
re-pricing adds a bow or a lattice staircase?

For each round-1 rough walk (as d111s_replay.py) and each of base / G_3pl:2 / G_2pl:2 / P_paint:2, the replayed seed
polyline is resampled every 0.3 cm and smoothed with a +-2 cm arc-length running mean (ends kept).  Reported, summed over
walks:
  L_raw / L_smooth      jitter ratio (path length the staircase adds on top of the smooth route)
  L_smooth / L_base_sm  the smooth route's length against the base's smooth route (the bow a detour adds)
  jitter rms            rms distance of the resampled seed from its smoothed version (cm)
  off, wig              as in the ladder, for the raw seed and for the SMOOTHED seed
The last pair asks whether re-route-then-smooth keeps the off-ridge gain without the wiggle cost.

Usage: d111s_wiggle_split.py --det pdhd --arm d111hst [--jobs 10] > figs/111s_wiggle_split_<det>.txt
"""
import argparse, collections, glob, os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool
import numpy as np
import uproot
import d111s_common as C
from d111s_replay import weights
A = C.A

CANDS = [("base", 0), ("G_3pl", 2), ("G_2pl", 2), ("P_paint", 2)]


def smooth(P, step=0.3, half=2.0):
    Q = C.resample(P, step)
    if len(Q) < 3:
        return Q, Q
    k = max(1, int(round(half / step)))
    cs = np.vstack([np.zeros(3), np.cumsum(Q, axis=0)])
    n = len(Q)
    lo = np.clip(np.arange(n) - k, 0, n - 1); hi = np.clip(np.arange(n) + k, 0, n - 1)
    S = (cs[hi + 1] - cs[lo]) / (hi - lo + 1)[:, None]
    S[0], S[-1] = Q[0], Q[-1]
    return Q, S


def length(P):
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) if len(P) > 1 else 0.0


def one(args):
    det, arm, pre = args
    acc = collections.defaultdict(collections.Counter)
    d = f"{C.IMG}/{det}/work/{pre}_{arm}"
    tr = C.parse_trace(f"/home/xqian/tmp/d111/arm_{arm}/evt_{pre}.log.gz")
    have = set(uproot.open(f"{d}/tracking-stm.root")["T_rec_charge"].arrays(["cluster_id"], library="np")["cluster_id"].tolist())
    img = A.bee_layer(f"{d}/mabc-pr.zip", "clustering-global")
    seen, ridges = set(), {}
    for order, cl, kind, src, dst, npath in tr["rp"]:
        if kind != "rough" or cl not in have or (cl, src, dst) in seen:
            continue
        seen.add((cl, src, dst))
        sb = C.graph_for(tr, cl, order)
        mi = img[1] == cl
        if sb is None or mi.sum() < 5 or max(src, dst) >= sb.nv:
            continue
        if cl not in ridges:
            ridges[cl] = A.Ridge(img[0][mi], img[2][mi])
        R = ridges[cl]
        if not R.ok:
            continue
        base_sm = None
        for fam, k in CANDS:
            p, _ = sb.walk(src, dst, weights(sb, fam, k))
            if p is None:
                continue
            P = sb.V[p]
            Q, S = smooth(P)
            if fam == "base":
                base_sm = length(S)
            a = acc[f"{fam}:{k}"]
            a["L_raw"] += length(P); a["L_sm"] += length(S); a["L_base_sm"] += base_sm or 0.0
            a["jit2"] += float(np.sum(np.linalg.norm(Q - S, axis=1) ** 2)); a["njit"] += len(Q)
            for lab, X in (("raw", P), ("sm", S)):
                Y, wl = C.polyline_samples(X, 0.3)
                if len(Y):
                    a[f"L_{lab}_s"] += float(wl.sum()); a[f"off_{lab}"] += float(wl[R.offset(Y) > 1.0].sum())
                nw, ni = C.wiggle_share(X)
                a[f"wig_{lab}"] += nw; a[f"nint_{lab}"] += ni
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--jobs", type=int, default=10)
    a = ap.parse_args()
    evs = sorted(os.path.basename(p)[:-len(a.arm) - 1] for p in glob.glob(f"{C.IMG}/{a.det}/work/*_{a.arm}"))
    tot = collections.defaultdict(collections.Counter)
    with Pool(a.jobs) as pool:
        for acc in pool.imap_unordered(one, [(a.det, a.arm, e) for e in evs]):
            for k, v in acc.items():
                tot[k].update(v)
    print(f"# doc pdvd/111 round 2 EXPLORATORY bow/jitter split: det={a.det} arm={a.arm} events={len(evs)}; "
          f"+-2 cm running mean on a 0.3 cm resampling")
    print("| candidate | L_raw/L_smooth | smooth length vs base smooth | jitter rms (cm) | off raw | off smoothed | wiggle raw | wiggle smoothed |")
    print("|---|---|---|---|---|---|---|---|")
    for fam, k in CANDS:
        t = tot[f"{fam}:{k}"]
        print(f"| {fam}:{k} | {t['L_raw']/t['L_sm']:.4f} | {t['L_sm']/t['L_base_sm']:.4f} | {np.sqrt(t['jit2']/t['njit']):.3f} | "
              f"{100*t['off_raw']/t['L_raw_s']:.2f} % | {100*t['off_sm']/t['L_sm_s']:.2f} % | "
              f"{100*t['wig_raw']/t['nint_raw']:.2f} % | {100*t['wig_sm']/t['nint_sm']:.2f} % |")


if __name__ == "__main__":
    main()
