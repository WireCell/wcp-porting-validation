#!/usr/bin/env python3
"""doc pdhd/28 sec 4 -- per-plane region sums on hand Michel items: is the higher corrected reading leakage or the
induction planes' own surplus?

    STM_SCAN_RECORD=<smx27 record> python3 d28_planes.py > planes.txt

PDHD split by the APA the muon's W footprint is on (APA2 = the side where the first-listed wire was mostly right,
APA1/3 = the far-face side), knob off (h28off) vs on (h28wl); PDVD production (p96vprod) and PDVD knob on (p97vwl) as
the reference.  The region sum per plane is the chain's michel_q2d_region_{u,v,w}; the U+V sum is split offline by the
cell's role (Michel-claimed 3/4, muon footprint 1, unclaimed 0), by cross-shared, and on PDHD by whether the channel
has two wires on the APA's active face.  Same rule as the C++ (0 <= d_stop_cm <= 10, own_blob != 0, pmu clamped at 0).
"""
import bz2, collections, glob, json, os, sys
import numpy as np, uproot
IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
sys.path.insert(0, IMG + "/pdhd/docs/scan/h25")
import d25_bragg_michel as BM
R = 10.0; ACT = {0: 0, 1: 1, 2: 0, 3: 1}
g = json.load(bz2.open("/nfs/data/1/xqian/toolkit-dev/wire-cell-data/protodunehd-wires-larsoft-v1.json.bz2"))["Store"]
nact = collections.Counter()
for an in g["anodes"]:
    an = an["Anode"]
    for fi in an["faces"]:
        f = g["faces"][fi]["Face"]
        for pi in f["planes"]:
            for wi in g["planes"][pi]["Plane"]["wires"]:
                if f["ident"] == ACT[an["ident"]]: nact[(an["ident"], g["wires"][wi]["Wire"]["channel"])] += 1


def per_plane(det, arm, pop, K):
    its, _ = BM.items(det, arm, pop); J = {x[0]: x for x in its}
    out = {}
    for fn in sorted(glob.glob(f"{IMG}/{det}/work/*_{arm}/tracking-pr.root")):
        ev = os.path.basename(os.path.dirname(fn))[: -len(arm) - 1]; u = uproot.open(fn)
        if "T_stm_michel" not in u or "T_stm_michel_2d" not in u: continue
        t = u["T_stm_michel"].arrays(["cluster_id", "is_stm", "michel_found"] + ["michel_q2d_region_%s" % p for p in "uvw"], library="np")
        C = u["T_stm_michel_2d"].arrays(["cluster_id", "apa", "plane", "channel", "role", "xshared", "own_blob", "d_stop_cm", "charge", "pred_mu", "pred_all"], library="np")
        for i in range(len(t["cluster_id"])):
            k = f"{ev}/{int(t['cluster_id'][i])}"; x = J.get(k)
            if not (x and x[1] in BM.STOP and x[2] in BM.MICHEL_KINDS and int(t["is_stm"][i]) == 1 and int(t["michel_found"][i]) == 1): continue
            m = C["cluster_id"] == t["cluster_id"][i]
            w1 = m & (C["plane"] == 2) & (C["role"] == 1)
            apa = collections.Counter(C["apa"][w1].tolist()).most_common(1)[0][0] if w1.any() else -1
            s = m & (C["d_stop_cm"] >= 0) & (C["d_stop_cm"] <= R) & (C["own_blob"] != 0)
            pmu = np.maximum(C["pred_mu"], 0); con = np.where(C["xshared"] == 1, np.maximum(C["pred_all"] - pmu, 0), C["charge"] - pmu)
            d = dict(apa=apa, S=[float(t["michel_q2d_region_%s" % p][i]) * K for p in "uvw"])
            uv = s & (C["plane"] < 2)
            d["claimed"] = float(con[uv & ((C["role"] == 3) | (C["role"] == 4))].sum()) * K
            d["footprint"] = float(con[uv & (C["role"] == 1)].sum()) * K
            d["unclaimed"] = float(con[uv & (C["role"] == 0)].sum()) * K
            d["xshared"] = float(con[uv & (C["xshared"] == 1)].sum()) * K
            if det == "pdhd":
                two = np.array([nact.get((int(a), int(ch)), 0) >= 2 for a, ch in zip(C["apa"], C["channel"])])
                d["two"] = float(con[uv & two].sum()) * K
            out[k] = d
    return out


for det, arm, pop, K in (("pdhd", "h28off", "strict", 41.3035e-6), ("pdhd", "h28wl", "strict", 41.3035e-6),
                         ("pdvd", "p96vprod", "all", 42.7058e-6), ("pdvd", "p97vwl", "all", 42.7058e-6)):
    D = per_plane(det, arm, pop, K)
    groups = {"all": list(D)} if det == "pdvd" else {"APA2": [k for k in D if D[k]["apa"] == 2], "APA1/3": [k for k in D if D[k]["apa"] in (1, 3)]}
    for gname, kk in groups.items():
        S = np.array([D[k]["S"] for k in kk]); uvw = np.median(S, axis=0)
        ratio = lambda p: np.median([s[p] / s[2] for s in S if s[2] > 0])
        line = (f"{det} {arm:9s} {gname:6s} n {len(kk):3d}: region sum MeV-eq median U/V/W {uvw[0]:5.1f}/{uvw[1]:5.1f}/{uvw[2]:5.1f}"
                f" | per-item U/W {ratio(0):.2f} V/W {ratio(1):.2f} | U+V by role: claimed %.1f, footprint %.1f, unclaimed %.1f, cross-shared %.1f"
                % tuple(np.median([D[k][x] for k in kk]) for x in ("claimed", "footprint", "unclaimed", "xshared")))
        if det == "pdhd": line += ", on two-active-wire channels %.1f" % np.median([D[k]["two"] for k in kk])
        print(line)
