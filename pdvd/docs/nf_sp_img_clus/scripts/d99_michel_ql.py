#!/usr/bin/env python3
"""doc pdvd/99 -- what the top gain constant does downstream of SP on the Michel energy and the charge-light match.

    python3 d99_michel_ql.py [--arms p96vprod:REC p98voffq:CARRIED_OFF p98vonq:CARRIED_ON]

1. Michel region energy by drift volume on each arm (doc pdhd/29 sec 9 layout): michel_ke_q2d_region / _ctl /
   michel_ke_best medians with bootstrap intervals, top (stop_x > 0, TDE) and bottom (stop_x < 0, BDE), for the hand
   Michel class (hand stopper with michel_kind attached/both, is_stm 1, michel_found 1) and for every chain STM+Michel.
2. Per item, ON / OFF: the same carried record item on both arms (joined by its ORIGINAL key), michel_found on both,
   ratio of michel_ke_q2d_region and _ctl, by volume.  Charge-linear estimators predict top ~ 1/0.889 = 1.125, bottom 1;
   the Michel energy CONSTANT (C, MeV/e) is pooled and unchanged, so this is the size of the shift a flip would bring.
3. Charge-light: for items carried onto both arms, the matched cluster's flash_id and cluster_t0_us (T_cluster) OFF vs
   ON, by volume.  Q/L uses charge; its QtoL scale is pooled top+bottom and was not refit.
"""
import argparse, collections, json, os, sys
import numpy as np
import uproot

IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
sys.path.insert(0, IMG + "/pdhd/docs/scan/d26")
import d26_michel_energy as M
BM = M.BM
if "stop_x" not in M.BR:
    M.BR.append("stop_x")
SCAN = IMG + "/pdvd/docs/scan"
EST = ("michel_ke_q2d_region", "michel_ke_q2d_ctl", "michel_ke_best")


def bmed(v, seed, n=2000):
    v = np.asarray(v, float)
    if len(v) < 3:
        return (float(np.median(v)) if len(v) else np.nan, np.nan, np.nan)
    r = np.random.default_rng(seed)
    m = np.median(v[r.integers(0, len(v), size=(n, len(v)))], axis=1)
    return float(np.median(v)), float(np.percentile(m, 16)), float(np.percentile(m, 84))


def fm(t):
    return f"{t[0]:6.2f} [{t[1]:6.2f},{t[2]:6.2f}]"


def arm_items(arm, rec):
    orig = BM.VD_REC
    BM.VD_REC = rec
    try:
        its, _ = BM.items("pdvd", arm, "all")
    finally:
        BM.VD_REC = orig
    R = {r["key"]: r.get("carried_from", r["key"]) for r in json.load(open(rec))}
    return its, R


def t_cluster(arm, evt):
    f = f"{IMG}/pdvd/work/{evt}_{arm}/tracking-pr.root"
    if not os.path.exists(f):
        return {}
    t = uproot.open(f)["T_cluster"].arrays(["cluster_id", "flash_id", "cluster_t0_us"], library="np")
    return {int(c): (int(fl), float(t0)) for c, fl, t0 in zip(t["cluster_id"], t["flash_id"], t["cluster_t0_us"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=[
        f"p96vprod:{BM.VD_REC}",
        f"p98voffq:{SCAN}/pdvd_stm_michel_smx9_carried_p98voffq.json",
        f"p98vonq:{SCAN}/pdvd_stm_michel_smx9_carried_p98vonq.json"])
    a = ap.parse_args()
    ITEM = {}
    for spec in a.arms:
        arm, rec = spec.split(":", 1)
        if not os.path.exists(rec):
            print(f"[skip] {arm}: no record {rec}")
            continue
        DV = M.read("pdvd", arm)
        its, R = arm_items(arm, rec)
        judged = {x[0]: x for x in its}
        hm = [(k, d) for k, d in DV.items() if k in judged and int(d["is_stm"]) == 1
              and judged[k][1] in BM.STOP and judged[k][2] in BM.MICHEL_KINDS]
        allsel = [(k, d) for k, d in DV.items() if int(d["is_stm"]) == 1]
        print(f"\n=== {arm}: Michel energy by drift volume, median MeV [p16,p84]  (record {os.path.basename(rec)})")
        for lab, pop in (("hand Michel", hm), ("all chain STM+Michel", allsel)):
            for est in EST:
                tp = [float(d[est]) for _, d in pop if float(d["stop_x"]) > 0]
                bt = [float(d[est]) for _, d in pop if float(d["stop_x"]) < 0]
                mt, mb = bmed(tp, 41), bmed(bt, 43)
                print(f"  {lab:21s} {est:21s} top n {len(tp):3d} {fm(mt)}  bottom n {len(bt):3d} {fm(mb)}  "
                      f"top/bottom {mt[0] / mb[0] if mb[0] else float('nan'):.3f}")
        # every candidate (michel_found or not) for the per-item join
        ITEM[arm] = {R.get(k, k): (k, judged[k], DV.get(k)) for k in judged}

    if "p98voffq" in ITEM and "p98vonq" in ITEM:
        off, on = ITEM["p98voffq"], ITEM["p98vonq"]
        common = sorted(set(off) & set(on))
        print(f"\n=== per item ON / OFF on {len(common)} items carried onto both arms")
        for vol in ("top", "bottom"):
            rows = collections.defaultdict(list); mf = collections.Counter()
            for k in common:
                (kf, xf, df), (kn, xn, dn) = off[k], on[k]
                if (float(xn[4]["stop_x"]) > 0) != (vol == "top"):
                    continue
                mf[(int(xf[4]["michel_found"]), int(xn[4]["michel_found"]))] += 1
                if df is not None and dn is not None:
                    for est in EST[:2]:
                        if float(df[est]) > 1.0:
                            rows[est].append(float(dn[est]) / float(df[est]))
            print(f"  {vol:6s} michel_found (OFF,ON): {dict(sorted(mf.items()))}")
            for est in EST[:2]:
                r = bmed(rows[est], 47)
                print(f"    {est:21s} ON/OFF per item (both found, OFF > 1 MeV) n {len(rows[est]):3d} median {fm(r)}")
        print("\n=== charge-light: matched cluster flash_id / cluster_t0_us, OFF vs ON")
        cache = {}
        for vol in ("top", "bottom"):
            c = collections.Counter(); dts = []
            for k in common:
                (kf, xf, _), (kn, xn, _) = off[k], on[k]
                if (float(xn[4]["stop_x"]) > 0) != (vol == "top"):
                    continue
                ef, cf = kf.rsplit("/", 1); en, cn = kn.rsplit("/", 1)
                tf = cache.setdefault(("p98voffq", ef), t_cluster("p98voffq", ef)).get(int(cf))
                tn = cache.setdefault(("p98vonq", en), t_cluster("p98vonq", en)).get(int(cn))
                if tf is None or tn is None:
                    c["no T_cluster row"] += 1
                    continue
                c["same flash" if tf[0] == tn[0] else "flash changed"] += 1
                dts.append(tn[1] - tf[1])
            dts = np.abs(np.asarray(dts))
            print(f"  {vol:6s} {dict(c)}  |dt0| > 1 us: {int((dts > 1).sum())} of {len(dts)}  (1 us = 0.148 cm of x)")


if __name__ == "__main__":
    main()
