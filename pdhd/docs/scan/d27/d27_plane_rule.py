#!/usr/bin/env python3
"""doc pdhd/27 sec 2 -- PDHD other-face U/V cells by APA / drift side; control and region through the chain's plane rule
with all planes, W only and U/V only (a re-derivation of stm_michel_combine_planes, twinned against michel_q2d_ctl /
michel_q2d_region); W residual and muon charge vs steepness.

    STM_SCAN_RECORD=<smx27 record> python3 d27_plane_rule.py > plane_rule.txt"""
import glob, os, sys, collections
import numpy as np, uproot
IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
sys.path.insert(0, IMG + "/pdhd/docs/scan/h25")
import d25_bragg_michel as BM
R = 10.0; W8 = np.array([0.25, 0.25, 1.0]); SW = 0.04
KMEV = {"pdhd": 41.3035e-6, "pdvd": 42.7058e-6}
VERT = {"pdhd": 1, "pdvd": 0}


def combine(s, n):
    s = np.asarray(s, float); w = W8.copy(); w[np.asarray(n) == 0] = 0
    mn = int(np.argmin(s)); mx = int(np.argmax(s))
    if mn != mx: md = [i for i in range(3) if i not in (mn, mx)][0]
    else: mn, md, mx = 0, 1, 2
    ov = (w * s).sum() / w.sum() if w.sum() > 0 else 0.0
    asy = abs(s[md] - s[mx]) / (s[md] + s[mx]) if s[md] + s[mx] > 0 else 0.0
    drop = -1
    if asy > SW and w[md] + w[mn] > 0:
        ov = (w[md] * s[md] + w[mn] * s[mn]) / (w[md] + w[mn]); drop = mx
    return ov, drop


CB = ["cluster_id", "apa", "plane", "face", "charge", "pred_mu", "pred_all", "xshared", "own_blob", "d_stop_cm", "d_ctl_cm", "role"]
TB = ["cluster_id", "is_stm", "michel_found", "michel_ke_q2d_ctl", "michel_ke_q2d_region", "michel_q2d_ctl", "michel_q2d_region", "stop_x", "michel_ke_best"]
for det, arm, pop in (("pdhd", "h26q2dprod", "strict"), ("pdvd", "p96vprod", "all")):
    its, _ = BM.items(det, arm, pop)
    judged = {x[0]: x for x in its}
    rows = []; twin_bad = 0
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
            mu_face = collections.Counter(C["face"][w1].tolist()).most_common(1)[0][0]
            mu_apa = collections.Counter(C["apa"][w1].tolist()).most_common(1)[0][0]
            uv1 = m & (C["plane"] < 2) & (C["role"] == 1)
            wrong = float((uv1 & (C["face"] != mu_face)).sum()) / max(1, uv1.sum())
            d = dict(key=key, apa=mu_apa, face=mu_face, wrong=wrong, stop_x=float(t["stop_x"][i]), ctl=float(t["michel_ke_q2d_ctl"][i]),
                     reg=float(t["michel_ke_q2d_region"][i]), best=float(t["michel_ke_best"][i]))
            for nm, dc in (("ctl", "d_ctl_cm"), ("reg", "d_stop_cm")):
                s = m & (C[dc] >= 0) & (C[dc] <= R) & (C["own_blob"] != 0)
                pmu = np.maximum(C["pred_mu"][s], 0); xs = C["xshared"][s] == 1
                con = np.where(xs, np.maximum(C["pred_all"][s] - pmu, 0), C["charge"][s] - pmu); pl = C["plane"][s]
                S = [con[pl == p].sum() for p in range(3)]; N = [(pl == p).sum() for p in range(3)]
                full, drop = combine(S, N)
                wonly, _ = combine(S, [0, 0, N[2]])
                ref = float(t[f"michel_q2d_{'ctl' if nm == 'ctl' else 'region'}"][i])
                if abs(full - ref) > 1e-6 * max(1, abs(ref)): twin_bad += 1
                d[f"{nm}_full"] = max(full, 0) * KMEV[det]; d[f"{nm}_wonly"] = max(wonly, 0) * KMEV[det]
                d[f"{nm}_uvonly"] = max(combine(S, [N[0], N[1], 0])[0], 0) * KMEV[det] if (N[0] or N[1]) else np.nan
                d[f"{nm}_drop"] = drop; d[f"{nm}_N"] = N
                d[f"{nm}_S"] = [v * KMEV[det] for v in S]
                mu = [pmu[pl == p].sum() for p in range(3)]; meas = [C["charge"][s][pl == p].sum() for p in range(3)]
                d[f"{nm}_resW"] = (meas[2] - mu[2]) / mu[2] if mu[2] > 0 else np.nan
                d[f"{nm}_muW"] = mu[2] * KMEV[det]
            pm = (P["cluster_id"] == c) & (P["role"] == 1) & (P["rr"] >= 25) & (P["rr"] <= 45)
            if pm.sum() >= 5:
                X = np.stack([P["x"][pm], P["y"][pm], P["z"][pm]], 1); X = X - X.mean(0)
                d["cosv"] = abs(float(np.linalg.svd(X, full_matrices=False)[2][0][VERT[det]]))
            else: d["cosv"] = np.nan
            rows.append(d)
    print(f"\n=== {det} {arm}: hand Michel {len(rows)}; twin of the combined ctl/region charge: {2*len(rows)-twin_bad}/{2*len(rows)}")
    A = lambda k: np.array([r[k] for r in rows], float)
    wr = A("wrong"); cv = A("cosv")
    if det == "pdhd":
        print("  U/V role-1 cells on a face != the muon's face, per item: all-or-nothing?", collections.Counter(np.round(wr, 2)))
        tab = collections.Counter((r["apa"], r["face"], "x<0" if r["stop_x"] < 0 else "x>0", "WRONG" if r["wrong"] > 0.5 else "right") for r in rows)
        for k in sorted(tab): print("    apa %d muon face %d %s U/V %s: %d" % (k + (tab[k],)))
    grp = {"all": np.ones(len(rows), bool)}
    if det == "pdhd":
        grp["right face"] = wr < 0.5; grp["WRONG face"] = wr >= 0.5
    for gname, g in grp.items():
        print(f"  [{gname}] n {g.sum()}")
        for nm in ("ctl", "reg"):
            print(f"    {nm}: chain median {np.median(A(nm)[g]):5.1f} | offline full {np.median(A(nm+'_full')[g]):5.1f} | W-only {np.median(A(nm+'_wonly')[g]):5.1f}"
                  f" | U/V-only {np.nanmedian(A(nm+'_uvonly')[g]):5.1f} | W dropped by the switch on {sum(1 for r, gg in zip(rows, g) if gg and r[nm+'_drop']==2)}"
                  f" | per-plane S p50 U/V/W {np.median([r[nm+'_S'][0] for r, gg in zip(rows, g) if gg]):5.1f}/{np.median([r[nm+'_S'][1] for r, gg in zip(rows, g) if gg]):5.1f}/{np.median([r[nm+'_S'][2] for r, gg in zip(rows, g) if gg]):5.1f}"
                  f" | W (meas-mu)/mu p50 {np.nanmedian(A(nm+'_resW')[g]):+.3f} | W mu p50 {np.median(A(nm+'_muW')[g]):5.1f} MeV-eq")
        for lo, hi in ((0, 0.7), (0.7, 0.9), (0.9, 1.01)):
            s = g & np.isfinite(cv) & (cv >= lo) & (cv < hi)
            if s.sum():
                print(f"      |cos_vert| {lo:.1f}-{hi:.2f} n {s.sum():3d}: ctl chain {np.median(A('ctl')[s]):5.1f} W-only {np.median(A('ctl_wonly')[s]):5.1f}"
                      f" U/V-only {np.nanmedian(A('ctl_uvonly')[s]):5.1f} | ctl W res {np.nanmedian(A('ctl_resW')[s]):+.3f} W mu {np.median(A('ctl_muW')[s]):5.1f}"
                      f" | region chain {np.median(A('reg')[s]):5.1f} W-only {np.median(A('reg_wonly')[s]):5.1f} U/V-only {np.nanmedian(A('reg_uvonly')[s]):5.1f}")
