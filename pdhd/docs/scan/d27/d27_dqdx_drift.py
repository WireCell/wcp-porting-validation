#!/usr/bin/env python3
"""doc pdhd/27 sec 5 -- is the PDVD x>0 near-anode dQ/dx deficit a position effect or a track-population effect?
Same selection and plateau definition as d26_dqdx_rr.py.  PDHD is the control.

    STM_SCAN_RECORD=<smx27 record> python3 d27_dqdx_drift.py > dqdx_drift.txt     (writes dqdx_drift.json)

NOTE: the printed "WITHIN-TRACK slope ... per 100 cm drift" is a raw Theil-Sen slope in drift along each track; along a
downward stopper drift and rr move together, so it mixes the rr shape in.  d27_dqdx_rr_vs_drift.py separates them."""
import glob, json, os, sys, collections
import numpy as np, uproot
from scipy import stats
IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
sys.path.insert(0, IMG + "/pdhd/docs/scan/h25")
import d25_bragg_michel as BM
ANODE = {"pdhd": 352.1, "pdvd": 339.91}
DRIFTAX = 0   # x is the drift axis on both detectors
rng = np.random.default_rng(27)
DUMP = {}


def ref_table(det, arm):
    js = sorted(glob.glob(f"{IMG}/{det}/work/*_{arm}/calib-pr-evt*.json"))
    r = json.load(open(js[0]))["dqdx_ref"]; g = r["grid"]
    return g["start"] + g["step"] * np.arange(g["n"]), np.asarray(r["muon"], float)


def bmed(v, n=2000):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 3: return (np.nan, np.nan, np.nan, len(v))
    m = np.median(v[rng.integers(0, len(v), size=(n, len(v)))], axis=1)
    return (float(np.median(v)), float(np.percentile(m, 16)), float(np.percentile(m, 84)), len(v))


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
            # within-track: ratio normalised to the track's own median over rr 20-100, against drift, Theil-Sen
            d["span"] = float(np.ptp(drift[body])) if body.sum() else 0.0
            if body.sum() >= 30 and d["span"] >= 30:
                y = ratio[body] / np.median(ratio[body])
                d["slope100"] = float(stats.theilslopes(y, drift[body])[0] * 100)
            else: d["slope100"] = np.nan
            d["pts_drift"] = drift[body]; d["pts_norm"] = ratio[body] / np.median(ratio[body]) if body.sum() else np.array([])
            T.append(d)
    DUMP[det] = [{k: v for k, v in d.items() if not isinstance(v, np.ndarray)} for d in T]
    print(f"\n=== {det} {arm}: good stoppers with a plateau {len(T)}")
    for side in ("x<0", "x>0"):
        S = [d for d in T if d["side"] == side]
        if len(S) < 5: continue
        pl = np.array([d["plat"] for d in S]); dr = np.array([d["drift"] for d in S]); cx = np.array([d["cosx"] for d in S]); t0 = np.array([d["t0"] for d in S])
        ok = np.isfinite(cx)
        print(f"  [{side}] n {len(S)} | plateau median {np.median(pl):.3f} | t0_us: zero {int((t0 == 0).sum())}, p10/p50/p90 {np.percentile(t0,10):.0f}/{np.median(t0):.0f}/{np.percentile(t0,90):.0f}")
        print(f"     corr(plateau, drift) spearman {stats.spearmanr(pl, dr)[0]:+.2f} p {stats.spearmanr(pl, dr)[1]:.3f} | corr(plateau, |cos_x|) {stats.spearmanr(pl[ok], cx[ok])[0]:+.2f} p {stats.spearmanr(pl[ok], cx[ok])[1]:.3f} | corr(drift, |cos_x|) {stats.spearmanr(dr[ok], cx[ok])[0]:+.2f}")
        A = np.c_[np.ones(ok.sum()), dr[ok] / 100, cx[ok]]
        coef, *_ = np.linalg.lstsq(A, pl[ok], rcond=None)
        bs = []
        for _ in range(1000):
            j = rng.integers(0, ok.sum(), ok.sum()); cc, *_ = np.linalg.lstsq(A[j], pl[ok][j], rcond=None); bs.append(cc)
        bs = np.array(bs)
        print(f"     plateau ~ a + b*drift/100cm + c*|cos_x|: a {coef[0]:.3f}  b {coef[1]:+.3f} [{np.percentile(bs[:,1],16):+.3f},{np.percentile(bs[:,1],84):+.3f}]  c {coef[2]:+.3f} [{np.percentile(bs[:,2],16):+.3f},{np.percentile(bs[:,2],84):+.3f}]")
        for lo, hi in ((0, 0.5), (0.5, 0.85), (0.85, 1.01)):
            s = ok & (cx >= lo) & (cx < hi)
            if s.sum() < 3: continue
            nb = [f"{np.median(pl[s & (dr >= a) & (dr < b)]):.2f}({(s & (dr >= a) & (dr < b)).sum()})" if (s & (dr >= a) & (dr < b)).sum() >= 2 else "-" for a, b in ((0, 100), (100, 200), (200, 400))]
            print(f"     |cos_x| {lo:.2f}-{hi:.2f}: n {s.sum():3d} plateau {np.median(pl[s]):.3f} drift p50 {np.median(dr[s]):5.0f} | drift 0-100/100-200/200+: {' '.join(nb)}")
        sl = bmed([d["slope100"] for d in S])
        print(f"     WITHIN-TRACK slope of normalised dQ/dx per 100 cm drift (rr 20-100, span >= 30 cm): median {sl[0]:+.3f} [{sl[1]:+.3f},{sl[2]:+.3f}] n {sl[3]}")
        pdr = np.concatenate([d["pts_drift"] for d in S]); pn = np.concatenate([d["pts_norm"] for d in S])
        prof = []
        for a, b in ((0, 25), (25, 50), (50, 100), (100, 150), (150, 200), (200, 250), (250, 300), (300, 360)):
            s = (pdr >= a) & (pdr < b)
            prof.append(f"{a}-{b}:{np.median(pn[s]):.3f}({s.sum()})" if s.sum() >= 50 else f"{a}-{b}:-")
        print("     pooled track-normalised points vs drift:", " ".join(prof))
        if det == "pdvd":
            by = collections.defaultdict(list)
            for d in S: by[d.get("apa", -1)].append(d["plat"])
            print("     plateau by anode:", {k: (round(float(np.median(v)), 3), len(v)) for k, v in sorted(by.items())})
json.dump(DUMP, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dqdx_drift.json"), "w"), default=float)
