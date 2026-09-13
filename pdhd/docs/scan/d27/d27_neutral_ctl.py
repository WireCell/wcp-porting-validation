#!/usr/bin/env python3
"""doc pdhd/27 sec 4 -- a detector-neutral body control (W only, no cell that is also within R of the stop), and the
region's composition by role, PDHD vs PDVD on hand Michel items; split by muon steepness and, on PDHD, by whether the
U/V distances were measured on the muon's face (d27_wrap_face.py).

    STM_SCAN_RECORD=<smx27 record> python3 d27_neutral_ctl.py > neutral_ctl.txt     (writes neutral_ctl.json)

|cos_vert| is the muon direction around the control (role-1 points rr 25-45 cm, principal axis) against the vertical:
PDHD y (the collection wires' direction), PDVD x (the drift direction)."""
import glob, os, sys, collections, json
import numpy as np, uproot
IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
sys.path.insert(0, IMG + "/pdhd/docs/scan/h25")
import d25_bragg_michel as BM
R = 10.0
KMEV = {"pdhd": 41.3035e-6, "pdvd": 42.7058e-6}
VERT = {"pdhd": 1, "pdvd": 0}
CB = ["cluster_id", "plane", "face", "charge", "pred_mu", "pred_all", "xshared", "own_blob", "d_stop_cm", "d_ctl_cm", "role"]
TB = ["cluster_id", "is_stm", "michel_found", "michel_ke_q2d_ctl", "michel_ke_q2d_region"]
out = {}
for det, arm, pop in (("pdhd", "h26q2dprod", "strict"), ("pdvd", "p96vprod", "all")):
    its, _ = BM.items(det, arm, pop); judged = {x[0]: x for x in its}
    rows = []
    for f in sorted(glob.glob(f"{IMG}/{det}/work/*_{arm}/tracking-pr.root")):
        ev = os.path.basename(os.path.dirname(f))[: -len(arm) - 1]
        u = uproot.open(f)
        if "T_stm_michel_2d" not in u: continue
        t = u["T_stm_michel"].arrays(TB, library="np"); C = u["T_stm_michel_2d"].arrays(CB, library="np")
        P = u["T_stm_michel_pts"].arrays(["cluster_id", "role", "rr", "x", "y", "z"], library="np")
        for i in range(len(t["cluster_id"])):
            c = int(t["cluster_id"][i]); key = f"{ev}/{c}"; x = judged.get(key)
            if not x or not (int(t["is_stm"][i]) == 1 and int(t["michel_found"][i]) == 1 and x[1] in BM.STOP and x[2] in BM.MICHEL_KINDS): continue
            m = C["cluster_id"] == c
            w1 = m & (C["plane"] == 2) & (C["role"] == 1)
            face = collections.Counter(C["face"][w1].tolist()).most_common(1)[0][0]
            uv = m & (C["plane"] < 2) & (C["role"] == 1)
            wrong = (uv & (C["face"] != face)).sum() / max(1, uv.sum())
            own = C["own_blob"] != 0
            pmu = np.maximum(C["pred_mu"], 0); xs = C["xshared"] == 1
            con = np.where(xs, np.maximum(C["pred_all"] - pmu, 0), C["charge"] - pmu)
            inR = (C["d_stop_cm"] >= 0) & (C["d_stop_cm"] <= R); inC = (C["d_ctl_cm"] >= 0) & (C["d_ctl_cm"] <= R)
            W = C["plane"] == 2
            cw = m & own & inC & W; cw_clean = cw & ~inR
            d = dict(key=key, wrong=wrong >= 0.5, ctl=float(t["michel_ke_q2d_ctl"][i]), reg=float(t["michel_ke_q2d_region"][i]),
                     ctlW=float(con[cw].sum()) * KMEV[det], ctlW_clean=float(con[cw_clean].sum()) * KMEV[det],
                     ctlW_mu=float(pmu[cw].sum()) * KMEV[det], ctlW_clean_mu=float(pmu[cw_clean].sum()) * KMEV[det],
                     ctlW_clean_meas=float(C["charge"][cw_clean].sum()) * KMEV[det], overlapW=int((cw & inR).sum()))
            # region composition, W plane only (the plane every item has on both detectors), by role
            rw = m & own & inR & W
            for nm, rm in (("r_mich", (C["role"] == 3) | (C["role"] == 4)), ("r_mu", C["role"] == 1), ("r_none", C["role"] == 0)):
                d[nm] = float(con[rw & rm].sum()) * KMEV[det]
            d["r_mu_pred"] = float(pmu[rw].sum()) * KMEV[det]
            pm = (P["cluster_id"] == c) & (P["role"] == 1) & (P["rr"] >= 25) & (P["rr"] <= 45)
            if pm.sum() >= 5:
                X = np.stack([P["x"][pm], P["y"][pm], P["z"][pm]], 1); X = X - X.mean(0)
                d["cosv"] = abs(float(np.linalg.svd(X, full_matrices=False)[2][0][VERT[det]]))
            else: d["cosv"] = np.nan
            rows.append(d)
    out[det] = rows
    A = lambda k, s=None: np.array([r[k] for r in rows if s is None or s(r)], float)
    print(f"\n=== {det} {arm}: hand Michel {len(rows)}")
    groups = [("all", None)]
    if det == "pdhd": groups += [("right face", lambda r: not r["wrong"]), ("WRONG face", lambda r: r["wrong"])]
    for g, s in groups:
        print(f"  [{g}] n {len(A('ctl', s))}: chain ctl {np.median(A('ctl', s)):5.1f} | W-only ctl {np.median(A('ctlW', s)):5.1f} (W mu {np.median(A('ctlW_mu', s)):5.1f})"
              f" | W-only ctl, stop-overlap cells removed {np.median(A('ctlW_clean', s)):5.1f} (mu {np.median(A('ctlW_clean_mu', s)):5.1f},"
              f" (meas-mu)/mu {np.median((A('ctlW_clean_meas', s) - A('ctlW_clean_mu', s)) / np.maximum(A('ctlW_clean_mu', s), 1e-9)):+.3f}) | items with W overlap {int((A('overlapW', s) > 0).sum())}")
        print(f"      region W by role, median MeV-eq: Michel-claimed {np.median(A('r_mich', s)):5.1f} | muon footprint (role 1) {np.median(A('r_mu', s)):5.1f} | unclaimed (role 0) {np.median(A('r_none', s)):5.1f}"
              f" | muon prediction inside the region {np.median(A('r_mu_pred', s)):5.1f}")
        cv = A("cosv", s)
        for lo, hi in ((0, 0.7), (0.7, 0.9), (0.9, 1.01)):
            ss = np.isfinite(cv) & (cv >= lo) & (cv < hi)
            if ss.sum():
                print(f"        |cos_vert| {lo:.1f}-{hi:.2f} n {ss.sum():3d}: chain ctl {np.median(A('ctl', s)[ss]):5.1f} | W-only clean ctl {np.median(A('ctlW_clean', s)[ss]):5.1f}"
                      f" (mu {np.median(A('ctlW_clean_mu', s)[ss]):5.1f}) | W overlap on {int((A('overlapW', s)[ss] > 0).sum())} | region W role-1 {np.median(A('r_mu', s)[ss]):5.1f}"
                      f" role-0 {np.median(A('r_none', s)[ss]):5.1f} mu pred in region {np.median(A('r_mu_pred', s)[ss]):5.1f} | chain region {np.median(A('reg', s)[ss]):5.1f}")
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "neutral_ctl.json"), "w"), default=float)
from scipy import stats
h = [r["ctlW_clean"] for r in out["pdhd"]]; v = [r["ctlW_clean"] for r in out["pdvd"]]
print(f"\nW-only clean control PDHD vs PDVD: medians {np.median(h):.2f} vs {np.median(v):.2f}; Mann-Whitney p {stats.mannwhitneyu(h, v).pvalue:.3f}")
h = [r["ctlW_clean"] / r["ctlW_clean_mu"] for r in out["pdhd"] if r["ctlW_clean_mu"] > 0]; v = [r["ctlW_clean"] / r["ctlW_clean_mu"] for r in out["pdvd"] if r["ctlW_clean_mu"] > 0]
print(f"W-only clean control / its muon prediction: PDHD {np.median(h):+.3f} (n {len(h)}) vs PDVD {np.median(v):+.3f} (n {len(v)}); Mann-Whitney p {stats.mannwhitneyu(h, v).pvalue:.3f}")
