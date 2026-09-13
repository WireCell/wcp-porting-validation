#!/usr/bin/env python3
"""doc pdvd/99 -- per-track stopper plateau table on any PDVD arm, in the format of doc pdhd/27's
dqdx_drift.json, so doc pdhd/29's space-charge fit (d29_space_charge.py --json) runs on it unchanged.

    python3 d99_dqdx_drift.py --arm p96vprod --out /home/xqian/tmp/p98/dq/dqdx_p96vprod.json   # G5b control
    python3 d99_dqdx_drift.py --arm p98von --record <carried record> --out <json>

Fork of pdhd/docs/scan/d27/d27_dqdx_drift.py (which has its arm as a literal and writes the json that doc
pdhd/29 reads): the per-track loop is copied verbatim for PDVD; the printed correlations are dropped.
The PDHD rows are copied from the committed d27/dqdx_drift.json so the output has both detectors.
G5b: on p96vprod with the committed record every PDVD row must equal the committed json's.
"""
import argparse, collections, glob, json, os, sys
import numpy as np, uproot
IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
sys.path.insert(0, IMG + "/pdhd/docs/scan/h25")
import d25_bragg_michel as BM
COMMITTED = IMG + "/pdhd/docs/scan/d27/dqdx_drift.json"
ANODE = {"pdhd": 352.1, "pdvd": 339.91}
DRIFTAX = 0


def ref_table(det, arm):
    js = sorted(glob.glob(f"{IMG}/{det}/work/*_{arm}/calib-pr-evt*.json"))
    r = json.load(open(js[0]))["dqdx_ref"]; g = r["grid"]
    return g["start"] + g["step"] * np.arange(g["n"]), np.asarray(r["muon"], float)


def table(arm, rec):
    det = "pdvd"
    rx, ry = ref_table(det, arm)
    orig = BM.VD_REC
    BM.VD_REC = rec
    try:
        its, _ = BM.items(det, arm, "all")
    finally:
        BM.VD_REC = orig
    good = [x for x in its if x[1] in BM.STOP and int(x[4]["is_stm"]) == 1 and int(x[4]["topology_cleared_bits"]) == 0]
    want = collections.defaultdict(set)
    for x in good:
        ev, c = x[0].rsplit("/", 1); want[ev].add(int(c))
    T = []
    for ev, cs in want.items():
        u = uproot.open(f"{IMG}/{det}/work/{ev}_{arm}/tracking-pr.root")
        p = u["T_stm_michel_pts"].arrays(["cluster_id", "role", "rr", "q", "x", "y", "z"], library="np")
        t = u["T_stm_michel"].arrays(["cluster_id", "t0_us", "muon_len"], library="np")
        C = u["T_stm_michel_2d"].arrays(["cluster_id", "apa", "plane", "role"], library="np") if "T_stm_michel_2d" in u else None
        for c in cs:
            m = (p["cluster_id"] == c) & (p["role"] == 1) & (p["rr"] >= 0) & (p["q"] > 0)
            rr, q, X, Y, Z = (p[k][m] for k in ("rr", "q", "x", "y", "z"))
            _, ix = np.unique(np.round(np.c_[rr, q], 6), axis=0, return_index=True); ix = np.sort(ix)
            rr, q, X, Y, Z = rr[ix], q[ix], X[ix], Y[ix], Z[ix]
            ratio = q / np.interp(rr, rx, ry)
            drift = ANODE[det] - np.abs(X)
            ti = np.where(t["cluster_id"] == c)[0][0]
            d = dict(key=f"{ev}/{c}", t0=float(t["t0_us"][ti]), side="x>0" if np.median(X) > 0 else "x<0")
            if C is not None:
                w = (C["cluster_id"] == c) & (C["plane"] == 2) & (C["role"] == 1)
                d["apa"] = collections.Counter(C["apa"][w].tolist()).most_common(1)[0][0] if w.any() else -1
            pl = (rr >= 40) & (rr < 60)
            if pl.sum() < 10: continue
            d["plat"] = float(np.median(ratio[pl])); d["drift"] = float(np.mean(drift[pl]))
            body = (rr >= 20) & (rr <= 100)
            if body.sum() >= 5:
                P3 = np.c_[X[body], Y[body], Z[body]]; P3 = P3 - P3.mean(0)
                d["cosx"] = abs(float(np.linalg.svd(P3, full_matrices=False)[2][0][DRIFTAX]))
            else: d["cosx"] = np.nan
            d["span"] = float(np.ptp(drift[body])) if body.sum() else 0.0
            T.append(d)
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--record", default=BM.VD_REC)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    T = table(a.arm, a.record)
    C = json.load(open(COMMITTED))
    out = {"pdvd": T, "pdhd": C["pdhd"]}
    json.dump(out, open(a.out, "w"), default=float)
    for side in ("x<0", "x>0"):
        pl = [d["plat"] for d in T if d["side"] == side]
        print(f"{a.arm} {side} n {len(pl)} plateau median {np.median(pl):.4f}")
    if a.arm == "p96vprod" and a.record == BM.VD_REC:
        ref = {d["key"]: d for d in C["pdvd"]}
        mine = {d["key"]: d for d in T}
        same = sum(1 for k in ref if k in mine and all(
            (ref[k].get(f) == mine[k].get(f)) or (isinstance(ref[k].get(f), float) and np.isclose(ref[k][f], mine[k][f], rtol=0, atol=1e-12))
            for f in ("plat", "drift", "cosx", "side", "apa", "t0")))
        ok = same == len(ref) == len(mine)
        print(f"GATE G5b p96vprod vs committed d27/dqdx_drift.json: rows {len(mine)} vs {len(ref)}, identical {same} -> {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
