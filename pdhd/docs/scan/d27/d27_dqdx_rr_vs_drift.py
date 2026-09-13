#!/usr/bin/env python3
"""doc pdhd/27 sec 5 -- (1) separate the residual-range shape from a drift effect with the cross-track regression
s_rr = alpha + beta * (d drift / d rr); (2) per-plane measured / predicted charge on the muon footprint vs drift.

    STM_SCAN_RECORD=<smx27 record> python3 d27_dqdx_rr_vs_drift.py > dqdx_rr_vs_drift.txt   (writes dqdx_rr_vs_drift.json)

Along one track, ratio(rr) = 1 + a*rr + b*drift with drift = d0 + k*rr, so the per-track slope in rr is a + b*k:
alpha (intercept) = the residual-range shape, beta (slope in k) = the drift effect, both per 100 cm.  A synthetic
check of that convention is run first and printed.  Same selection and plateau as d26_dqdx_rr.py."""
import glob, json, os, sys, collections
import numpy as np, uproot
from scipy import stats
IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
sys.path.insert(0, IMG + "/pdhd/docs/scan/h25")
import d25_bragg_michel as BM
ANODE = {"pdhd": 352.1, "pdvd": 339.91}
rng = np.random.default_rng(28)
DUMP = {}


def synthetic(a, b, n=150, seed=1):
    """the regression on tracks with a known rr shape a and drift slope b (per 100 cm), x>0-like leverage in k"""
    r = np.random.default_rng(seed); S = []; K = []
    for _ in range(n):
        k = r.choice([-0.9, -0.6, -0.3, 0.0, 0.05]); rr = np.linspace(20, 100, 130); dr = 150 + k * (rr - 60)
        y = (1 + a * rr / 100 + b * dr / 100) * (1 + r.normal(0, 0.1, len(rr))); y = y / np.median(y)
        S.append(stats.theilslopes(y, rr)[0] * 100); K.append(k)
    return np.linalg.lstsq(np.c_[np.ones(n), K], np.array(S), rcond=None)[0]


print("SYNTHETIC convention check (alpha must follow a, beta must follow b):")
for a_, b_ in ((-0.10, 0.0), (0.0, 0.10), (-0.05, 0.10)):
    c_ = synthetic(a_, b_)
    print(f"  input a {a_:+.2f} b {b_:+.2f} -> alpha {c_[0]:+.3f} beta {c_[1]:+.3f}")


def ref_table(det, arm):
    js = sorted(glob.glob(f"{IMG}/{det}/work/*_{arm}/calib-pr-evt*.json"))
    r = json.load(open(js[0]))["dqdx_ref"]; g = r["grid"]
    return g["start"] + g["step"] * np.arange(g["n"]), np.asarray(r["muon"], float)


for det, arm, pop in (("pdvd", "p96vprod", "all"), ("pdhd", "h26q2dprod", "strict")):
    rx, ry = ref_table(det, arm)
    its, _ = BM.items(det, arm, pop)
    good = [x for x in its if x[1] in BM.STOP and int(x[4]["is_stm"]) == 1 and int(x[4]["topology_cleared_bits"]) == 0]
    want = collections.defaultdict(set)
    for x in good:
        ev, c = x[0].rsplit("/", 1); want[ev].add(int(c))
    T = []
    for ev, cs in want.items():
        u = uproot.open(f"{IMG}/{det}/work/{ev}_{arm}/tracking-pr.root")
        p = u["T_stm_michel_pts"].arrays(["cluster_id", "role", "rr", "q", "x"], library="np")
        C = u["T_stm_michel_2d"].arrays(["cluster_id", "plane", "role", "charge", "pred_mu", "xshared", "flag"], library="np")
        for c in cs:
            m = (p["cluster_id"] == c) & (p["role"] == 1) & (p["rr"] >= 0) & (p["q"] > 0)
            rr, q, X = p["rr"][m], p["q"][m], p["x"][m]
            _, ix = np.unique(np.round(np.c_[rr, q], 6), axis=0, return_index=True); ix = np.sort(ix)
            rr, q, X = rr[ix], q[ix], X[ix]
            ratio = q / np.interp(rr, rx, ry); drift = ANODE[det] - np.abs(X)
            pl = (rr >= 40) & (rr < 60)
            if pl.sum() < 10: continue
            d = dict(key=f"{ev}/{c}", side="x>0" if np.median(X) > 0 else "x<0", plat=float(np.median(ratio[pl])), drift=float(np.mean(drift[pl])))
            b = (rr >= 20) & (rr <= 100)
            if b.sum() >= 30 and np.ptp(rr[b]) >= 40:
                y = ratio[b] / np.median(ratio[b])
                d["s_rr"] = float(stats.theilslopes(y, rr[b])[0] * 100)                 # per 100 cm of rr
                d["ddrr"] = float(np.polyfit(rr[b], drift[b], 1)[0])                      # cm drift per cm rr
            cm = (C["cluster_id"] == c) & (C["role"] == 1) & (C["xshared"] == 0)
            for pp, nm in enumerate("UVW"):
                s = cm & (C["plane"] == pp)
                pr = C["pred_mu"][s].sum()
                d[f"mp{nm}"] = float(C["charge"][s].sum() / pr) if pr > 0 else np.nan
            T.append(d)
    DUMP[det] = T
    print(f"\n=== {det} {arm}: tracks {len(T)}")
    S = [d for d in T if "s_rr" in d]
    for side in ("x<0", "x>0", "both"):
        G = [d for d in S if side == "both" or d["side"] == side]
        if len(G) < 5: continue
        s = np.array([d["s_rr"] for d in G]); k = np.array([d["ddrr"] for d in G])
        A = np.c_[np.ones(len(G)), k]
        coef, *_ = np.linalg.lstsq(A, s, rcond=None)
        bs = np.array([np.linalg.lstsq(A[j], s[j], rcond=None)[0] for j in (rng.integers(0, len(G), len(G)) for _ in range(1000))])
        print(f"  [{side}] n {len(G)}: d drift/d rr p10/p50/p90 {np.percentile(k,10):+.2f}/{np.median(k):+.2f}/{np.percentile(k,90):+.2f} | s_rr median {np.median(s):+.3f} per 100 cm rr"
              f" | fit s_rr = alpha + beta*(d drift/d rr): alpha {coef[0]:+.3f} [{np.percentile(bs[:,0],16):+.3f},{np.percentile(bs[:,0],84):+.3f}]"
              f"  beta {coef[1]:+.3f} [{np.percentile(bs[:,1],16):+.3f},{np.percentile(bs[:,1],84):+.3f}]  (alpha = rr shape per 100 cm rr, beta = drift slope per 100 cm drift)")
    for side in ("x<0", "x>0"):
        G = [d for d in T if d["side"] == side]
        if len(G) < 5: continue
        dr = np.array([d["drift"] for d in G]); pl = np.array([d["plat"] for d in G])
        line = f"  [{side}] n {len(G)} per-plane measured/predicted on the muon footprint:"
        for nm in "UVW":
            v = np.array([d[f"mp{nm}"] for d in G]); ok = np.isfinite(v)
            bins = [f"{np.median(v[ok & (dr >= a) & (dr < bb)]):.3f}({(ok & (dr >= a) & (dr < bb)).sum()})" if (ok & (dr >= a) & (dr < bb)).sum() >= 3 else "-"
                    for a, bb in ((0, 100), (100, 200), (200, 400))]
            line += f"\n      {nm}: median {np.median(v[ok]):.3f}  spearman vs drift {stats.spearmanr(v[ok], dr[ok])[0]:+.2f} (p {stats.spearmanr(v[ok], dr[ok])[1]:.3f})  drift 0-100/100-200/200+: {' '.join(bins)}"
        vW = np.array([d["mpW"] for d in G]); ok = np.isfinite(vW)
        cw = pl * vW
        bins = [f"{np.median(cw[ok & (dr >= a) & (dr < bb)]):.3f}" if (ok & (dr >= a) & (dr < bb)).sum() >= 3 else "-" for a, bb in ((0, 100), (100, 200), (200, 400))]
        bins0 = [f"{np.median(pl[(dr >= a) & (dr < bb)]):.3f}" if ((dr >= a) & (dr < bb)).sum() >= 3 else "-" for a, bb in ((0, 100), (100, 200), (200, 400))]
        line += f"\n      plateau ratio drift 0-100/100-200/200+: {' '.join(bins0)} | collection-implied (plateau x W meas/pred): {' '.join(bins)}  spearman {stats.spearmanr(cw[ok], dr[ok])[0]:+.2f} (p {stats.spearmanr(cw[ok], dr[ok])[1]:.3f})"
        print(line)
json.dump(DUMP, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dqdx_rr_vs_drift.json"), "w"), default=float)
