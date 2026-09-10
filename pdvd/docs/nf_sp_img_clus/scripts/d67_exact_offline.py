#!/usr/bin/env python3
"""Exact offline re-verdicts for two single-consumer knobs (no arm needed).

ks_margin: read ONLY at CheckSTM_Michel.cxx:1850 (sets R_SHAPE_FLAT = bit 3);
reject_bits is read afterwards only for is_stm = (reject_bits == 0) and the
count.  So on any arm: new is_stm = (bits & ~bit3) == 0 and (bit3 unset, or
ks_flat - ks_mu > m).  Gate: m = 0 must reproduce every published is_stm.

michel_range_energy_dis_cm: read ONLY at :2715 (the veto), michel_found =
conn_type > 0 at :2728.  On an arm run at 5 cm, the items a 3 cm veto adds are
conn_type in (2,3), 3 < dis <= 5, ke_best < 10.  Gate: at 5 cm the rule finds 0.

doc pdvd/67.  Usage: d67_exact_offline.py <production prep> <anchor-3cm prep>
"""
import sys, collections
sys.path.insert(0, "/home/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan")
import census_lib as C
rec = C.load_record()
SF = 1 << 3

def load(prep):
    P, _ = C.load_payloads(prep, rec)
    return {k: P[k]["verdict"] for k in rec if k in P and C.judged(rec[k])}

def sc(sel, truth, keys):
    tp = sum(1 for k in keys if sel[k] and truth[k]); fp = sum(1 for k in keys if sel[k] and not truth[k])
    fn = sum(1 for k in keys if not sel[k] and truth[k])
    pu = tp / max(tp + fp, 1); ef = tp / max(tp + fn, 1)
    return tp, fp, fn, pu, ef, 2 * pu * ef / max(pu + ef, 1e-9)

def pred_stm(v, m):
    b = int(v["reject_bits"] or 0)
    if b & ~SF: return 0
    if b & SF: return 1 if (v["ks_flat"] - v["ks_mu"]) > m else 0
    return 1

def tag(k):
    r = rec[k]; return "%-14s %-10s %-6s" % (k, r["verdict"], r["confidence"])

for name, prep in (("production", sys.argv[1]), ("anchor 3 cm", sys.argv[2])):
    V = load(prep); keys = sorted(V)
    T = {k: C.is_stopper(rec[k]) for k in keys}
    pub = {k: int(V[k]["is_stm"]) for k in keys}
    bad = [k for k in keys if pred_stm(V[k], 0.0) != pub[k]]
    print("\n##### %s: %d judged items; gate m=0 reproduces published is_stm: %d mismatches" % (name, len(keys), len(bad)))
    only_sf = [k for k in keys if int(V[k]["reject_bits"]) == SF]
    print("items whose ONLY reject is shape_flat: %d (stoppers %d)" % (len(only_sf), sum(T[k] for k in only_sf)))
    runs = {"039349": [k for k in keys if k.startswith("039349")], "039252+039253": [k for k in keys if not k.startswith("039349")]}
    for m in (0.0, -0.01, -0.02, -0.03, -0.05):
        sel = {k: pred_stm(V[k], m) for k in keys}
        near = sum(1 for k in only_sf if abs((V[k]["ks_flat"] - V[k]["ks_mu"]) - m) < 1e-6)
        line = "m=%+.2f  all: TP %3d FP %2d FN %3d pur %.3f eff %.3f F1 %.3f" % ((m,) + sc(sel, T, keys))
        for rn, rk in runs.items():
            line += "   | %s TP %3d FP %2d F1 %.3f" % ((rn,) + tuple(sc(sel, T, rk)[i] for i in (0, 1, 5)))
        print(line + ("  (borderline %d)" % near if near else ""))
        if m in (-0.02, -0.05):
            gained = [k for k in keys if sel[k] and not pub[k]]
            print("   gained %d: %d TP, %d FP" % (len(gained), sum(T[k] for k in gained), sum(not T[k] for k in gained)))
            for k in gained:
                if not T[k]: print("     new FP  %s  ks_flat-ks_mu %+.4f  %s" % (tag(k), V[k]["ks_flat"] - V[k]["ks_mu"], rec[k]["evidence"][:150]))
            if m == -0.02:
                print("   new TPs: " + " ".join(k for k in gained if T[k]))
            conf = collections.Counter(rec[k]["confidence"] for k in gained if T[k])
            print("   confidence of new TPs: %s" % dict(conf))

print("\n##### michel_range_energy_dis_cm 3 on production")
V = load(sys.argv[1]); keys = sorted(V)
M = {k: C.is_michel(rec[k]) for k in keys}
mf = {k: int(V[k]["michel_found"]) for k in keys}
def veto(v, dis): return v["michel_conn_type"] in (2, 3) and v["michel_dis_cm"] > dis and v["michel_ke_best"] < 10.0
print("gate: items a 5 cm veto would still find on the 5 cm arm: %d" % sum(1 for k in keys if mf[k] and veto(V[k], 5.0)))
for dis in (5.0, 4.0, 3.0, 2.0):
    sel = {k: mf[k] and not veto(V[k], dis) for k in keys}
    print("dis %.0f cm: TP %3d FP %2d FN %2d pur %.3f eff %.3f F1 %.3f" % ((dis,) + sc(sel, M, keys)))
    if dis in (3.0, 2.0):
        for k in keys:
            if mf[k] and veto(V[k], dis) and not veto(V[k], dis + 1.0):
                v = V[k]; print("    vetoed: %s conn %d dis %.2f ke %.2f  %s" % (tag(k), v["michel_conn_type"], v["michel_dis_cm"], v["michel_ke_best"], "TP LOST" if M[k] else "FP removed"))
