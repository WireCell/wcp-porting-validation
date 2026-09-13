#!/usr/bin/env python3
"""doc pdhd/27 sec 1-2 -- (1) where PDHD's U/V cells go in the region / control (face of the role-1 cells against the
muon's W face, cells within radius, the tree's region/control cell counts); (2) muon steepness vs the control reading.

    STM_SCAN_RECORD=<smx27 record> python3 d27_ctl_census.py > ctl_census.txt"""
import glob, os, sys, collections
import numpy as np, uproot
IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
sys.path.insert(0, IMG + "/pdhd/docs/scan/h25")
import d25_bragg_michel as BM
R = 10.0
CB = ["cluster_id", "plane", "face", "wire", "channel", "time", "charge", "pred_mu", "xshared", "own_blob", "d_stop_cm", "d_ctl_cm", "role"]
TB = ["cluster_id", "is_stm", "michel_found", "michel_ke_q2d_ctl", "michel_ke_q2d_region", "stop_x", "stop_y", "stop_z",
      "michel_q2d_region_n_u", "michel_q2d_region_n_v", "michel_q2d_region_n_w", "michel_q2d_ctl_n_u", "michel_q2d_ctl_n_v", "michel_q2d_ctl_n_w"]
VERT = {"pdhd": 1, "pdvd": 0}   # index of the vertical axis: PDHD y, PDVD x (drift)

for det, arm, pop in (("pdhd", "h26q2dprod", "strict"), ("pdvd", "p96vprod", "all")):
    its, _ = BM.items(det, arm, pop)
    judged = {x[0]: x for x in its}
    rows = []
    for f in sorted(glob.glob(f"{IMG}/{det}/work/*_{arm}/tracking-pr.root")):
        ev = os.path.basename(os.path.dirname(f))[: -len(arm) - 1]
        u = uproot.open(f)
        if "T_stm_michel_2d" not in u: continue
        t = u["T_stm_michel"].arrays(TB, library="np"); C = u["T_stm_michel_2d"].arrays(CB, library="np")
        P = u["T_stm_michel_pts"].arrays(["cluster_id", "role", "rr", "x", "y", "z"], library="np")
        for i in range(len(t["cluster_id"])):
            c = int(t["cluster_id"][i]); key = f"{ev}/{c}"
            x = judged.get(key)
            if not x or not (int(t["is_stm"][i]) == 1 and int(t["michel_found"][i]) == 1 and x[1] in BM.STOP and x[2] in BM.MICHEL_KINDS): continue
            m = C["cluster_id"] == c
            d = dict(key=key, ctl=float(t["michel_ke_q2d_ctl"][i]), reg=float(t["michel_ke_q2d_region"][i]))
            for p, nm in enumerate("uvw"):
                d[f"reg_n_{nm}"] = int(t[f"michel_q2d_region_n_{nm}"][i]); d[f"ctl_n_{nm}"] = int(t[f"michel_q2d_ctl_n_{nm}"][i])
            # the muon's face = majority face of W role-1 cells (collection channels map to one face)
            w1 = m & (C["plane"] == 2) & (C["role"] == 1)
            mu_face = collections.Counter(C["face"][w1].tolist()).most_common(1)[0][0] if w1.any() else -9
            d["mu_face"] = mu_face
            for p in range(3):
                r1 = m & (C["plane"] == p) & (C["role"] == 1)
                d[f"r1_{p}"] = int(r1.sum())
                d[f"r1_wrongface_{p}"] = int((r1 & (C["face"] != mu_face)).sum())
                d[f"r1_ctl_{p}"] = int((r1 & (C["d_ctl_cm"] >= 0) & (C["d_ctl_cm"] <= R)).sum())
                d[f"r1_stop_{p}"] = int((r1 & (C["d_stop_cm"] >= 0) & (C["d_stop_cm"] <= R)).sum())
                d[f"r1_rightface_ctl_{p}"] = int((r1 & (C["face"] == mu_face) & (C["d_ctl_cm"] >= 0) & (C["d_ctl_cm"] <= R)).sum())
                d[f"r1_wrongface_ctl_{p}"] = int((r1 & (C["face"] != mu_face) & (C["d_ctl_cm"] >= 0) & (C["d_ctl_cm"] <= R)).sum())
                d[f"dctl_min_{p}"] = float(C["d_ctl_cm"][r1].min()) if r1.any() else np.nan
                # W-plane: control cells that are ALSO in the stop region
                cc = m & (C["plane"] == p) & (C["own_blob"] != 0) & (C["d_ctl_cm"] >= 0) & (C["d_ctl_cm"] <= R)
                d[f"ctl_cells_{p}"] = int(cc.sum())
                d[f"ctl_also_stop_{p}"] = int((cc & (C["d_stop_cm"] >= 0) & (C["d_stop_cm"] <= R)).sum())
            # muon direction around the control: role-1 points rr 25..45 cm, principal axis
            pm = (P["cluster_id"] == c) & (P["role"] == 1) & (P["rr"] >= 25) & (P["rr"] <= 45)
            if pm.sum() >= 5:
                X = np.stack([P["x"][pm], P["y"][pm], P["z"][pm]], 1); X = X - X.mean(0)
                v = np.linalg.svd(X, full_matrices=False)[2][0]
                d["cos_vert"] = abs(float(v[VERT[det]])); d["dir"] = v
            else:
                d["cos_vert"] = np.nan
            rows.append(d)
    print(f"\n=== {det} {arm}: hand Michel {len(rows)}")
    for p, nm in enumerate("UVW"):
        g = lambda k: np.array([r[f"{k}{p}"] for r in rows], float)
        print(f"  {nm}: role-1 cells p50 {np.median(g('r1_')):.0f}; on a face != muon face {g('r1_wrongface_').sum()/max(1,g('r1_').sum()):.3f} of cells"
              f" | role-1 in ctl p50 {np.median(g('r1_ctl_')):.0f} (right face {g('r1_rightface_ctl_').sum():.0f}, wrong face {g('r1_wrongface_ctl_').sum():.0f})"
              f" | role-1 in stop region p50 {np.median(g('r1_stop_')):.0f} | nearest role-1 d_ctl p50 {np.nanmedian(g('dctl_min_')):.1f}"
              f" | ctl own cells p50 {np.median(g('ctl_cells_')):.0f}, also in stop region: items {int((g('ctl_also_stop_')>0).sum())}")
    for nm in "uvw":
        print(f"  tree {nm}: region n p50 {np.median([r['reg_n_'+nm] for r in rows]):.0f} (zero on {sum(r['reg_n_'+nm]==0 for r in rows)}), "
              f"ctl n p50 {np.median([r['ctl_n_'+nm] for r in rows]):.0f} (zero on {sum(r['ctl_n_'+nm]==0 for r in rows)})")
    cv = np.array([r["cos_vert"] for r in rows]); ctl = np.array([r["ctl"] for r in rows]); ov = np.array([r["ctl_also_stop_2"] > 0 for r in rows])
    ok = np.isfinite(cv)
    print(f"  |cos(vertical)| at the control p10/p50/p90 {np.nanpercentile(cv,10):.2f}/{np.nanmedian(cv):.2f}/{np.nanpercentile(cv,90):.2f} (n {ok.sum()})")
    for lo, hi in ((0, 0.7), (0.7, 0.9), (0.9, 1.01)):
        s = ok & (cv >= lo) & (cv < hi)
        if s.any():
            print(f"    |cos_vert| {lo:.1f}-{hi:.1f}: n {s.sum():3d} ctl median {np.median(ctl[s]):5.1f}  W ctl overlaps stop region {ov[s].mean():.2f}")
    if det == "pdhd":
        print("  per item:", "; ".join(f"{r['key']} ctl {r['ctl']:.1f} cosv {r['cos_vert']:.2f} ovW {r['ctl_also_stop_2']}/{r['ctl_cells_2']} U/V ctl {r['ctl_n_u']}/{r['ctl_n_v']} reg U/V {r['reg_n_u']}/{r['reg_n_v']} wrongfaceU {r['r1_wrongface_0']}/{r['r1_0']}" for r in sorted(rows, key=lambda r: -r['ctl'])))
