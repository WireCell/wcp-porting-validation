#!/usr/bin/env python3
"""doc pdhd/27 sec 1 -- decompose michel_ke_q2d_ctl per plane and per cell class on hand Michel items, PDHD h26q2dprod
vs PDVD p96vprod, with a twin of michel_q2d_ctl_{u,v,w} and the charge -> MeV constant read back from the tree.

    STM_SCAN_RECORD=<smx27 record> python3 d27_ctl_planes.py > ctl_planes.txt"""
import glob, os, sys, collections
import numpy as np, uproot
IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
sys.path.insert(0, IMG + "/pdhd/docs/scan/h25")
import d25_bragg_michel as BM
R = 10.0
CB = ["cluster_id", "plane", "charge", "pred_mu", "pred_all", "xshared", "own_blob", "d_stop_cm", "d_ctl_cm", "role", "flag", "wire", "time"]
TB = ["cluster_id", "is_stm", "michel_found", "michel_ke_q2d_ctl", "michel_q2d_ctl", "michel_q2d_ctl_u", "michel_q2d_ctl_v", "michel_q2d_ctl_w",
      "michel_q2d_ctl_n_u", "michel_q2d_ctl_n_v", "michel_q2d_ctl_n_w", "michel_q2d_ctl_dropped_plane", "michel_q2d_ctl_valid", "t0_us", "muon_len"]

for det, arm, pop in (("pdhd", "h26q2dprod", "strict"), ("pdvd", "p96vprod", "all")):
    its, _ = BM.items(det, arm, pop)
    judged = {x[0]: x for x in its}
    rows = []
    ok = bad = 0
    for f in sorted(glob.glob(f"{IMG}/{det}/work/*_{arm}/tracking-pr.root")):
        ev = os.path.basename(os.path.dirname(f))[: -len(arm) - 1]
        u = uproot.open(f)
        if "T_stm_michel_2d" not in u: continue
        t = u["T_stm_michel"].arrays(TB, library="np"); C = u["T_stm_michel_2d"].arrays(CB, library="np")
        for i in range(len(t["cluster_id"])):
            c = int(t["cluster_id"][i]); key = f"{ev}/{c}"
            if key not in judged: continue
            x = judged[key]
            if not (int(t["is_stm"][i]) == 1 and int(t["michel_found"][i]) == 1 and x[1] in BM.STOP and x[2] in BM.MICHEL_KINDS): continue
            s = (C["cluster_id"] == c) & (C["d_ctl_cm"] >= 0) & (C["d_ctl_cm"] <= R) & (C["own_blob"] != 0)
            pmu = np.maximum(C["pred_mu"][s], 0.0); pall = C["pred_all"][s]; q = C["charge"][s]
            xs = C["xshared"][s] == 1; pl = C["plane"][s]
            contrib = np.where(xs, np.maximum(pall - pmu, 0.0), q - pmu)
            cpp = [float(t[f"michel_q2d_ctl_{p}"][i]) for p in "uvw"]
            off = [float(contrib[pl == p].sum()) for p in range(3)]
            if max(abs(a - b) / max(1, abs(b)) for a, b in zip(off, cpp)) < 1e-6: ok += 1
            else: bad += 1
            cls = {
                "xshared": xs,
                "resid_pmu>0": (~xs) & (pmu > 0),
                "unpred_pmu0": (~xs) & (pmu <= 0),
            }
            also_stop = (C["d_stop_cm"][s] >= 0) & (C["d_stop_cm"][s] <= R)
            role = C["role"][s]
            d = dict(key=key, ke=float(t["michel_ke_q2d_ctl"][i]), q=float(t["michel_q2d_ctl"][i]), drop=int(t["michel_q2d_ctl_dropped_plane"][i]),
                     valid=int(t["michel_q2d_ctl_valid"][i]), t0=float(t["t0_us"][i]), mulen=float(t["muon_len"][i]))
            for p in range(3):
                m = pl == p
                d[f"n{p}"] = int(m.sum()); d[f"q{p}"] = off[p]; d[f"mu{p}"] = float(pmu[m].sum()); d[f"meas{p}"] = float(q[m].sum())
                for k, cm in cls.items():
                    d[f"{k}{p}"] = float(contrib[m & cm].sum())
                d[f"stop{p}"] = float(contrib[m & also_stop].sum())
                d[f"r34_{p}"] = float(contrib[m & ((role == 3) | (role == 4))].sum())
                d[f"neg{p}"] = float(contrib[m & (contrib < 0)].sum())
            rows.append(d)
    print(f"\n=== {det} {arm}: hand Michel items {len(rows)}; twin ctl u/v/w {ok}/{ok+bad}")
    kq = np.array([r["ke"] / r["q"] for r in rows if r["q"] > 0])
    print(f"  MeV per 1e6 e (ke/q): median {np.median(kq)*1e6:.4f}  spread {kq.min()*1e6:.4f}..{kq.max()*1e6:.4f}")
    print(f"  ctl MeV median {np.median([r['ke'] for r in rows]):.2f}  valid {collections.Counter(r['valid'] for r in rows)}  dropped {collections.Counter(r['drop'] for r in rows)}")
    K = np.median(kq)
    for p, nm in enumerate("UVW"):
        g = lambda k: np.array([r[f"{k}{p}"] for r in rows])
        print(f"  plane {nm}: cells p50 {np.median(g('n')):5.0f} | sum MeV-eq p50 {np.median(g('q'))*K:6.2f} mean {np.mean(g('q'))*K:6.2f}"
              f" | pred_mu p50 {np.median(g('mu'))*K:6.1f} meas p50 {np.median(g('meas'))*K:6.1f}  (meas-mu)/mu p50 {np.median((g('meas')-g('mu'))/np.maximum(g('mu'),1)):+.3f}")
        for k in ("xshared", "resid_pmu>0", "unpred_pmu0", "stop", "r34_", "neg"):
            v = g(k) * K
            print(f"      {k:12s} MeV-eq p50 {np.median(v):6.2f} mean {np.mean(v):6.2f}  nonzero {int((v != 0).sum())}")
    print("  items (sorted by ctl):", ", ".join(f"{r['key']}:{r['ke']:.1f}" for r in sorted(rows, key=lambda r: -r['ke'])[:12]))
