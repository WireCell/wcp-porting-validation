#!/usr/bin/env python3
"""doc pdvd/67.  Exact ks_margin re-verdict (single consumer, CheckSTM_Michel.cxx:1850) on any arm's
payloads, scored against the scan and diffed by name against a baseline arm.
Usage: ks_sweep.py <baseline prep> <arm prep> [<arm prep> ...]"""
import sys
sys.path.insert(0, "/home/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan")
import census_lib as C
rec = C.load_record(); SF = 1 << 3
def load(p):
    P, _ = C.load_payloads(p, rec); return {k: P[k]["verdict"] for k in rec if k in P and C.judged(rec[k])}
def pred(v, m):
    b = int(v["reject_bits"] or 0)
    return 0 if b & ~SF else (1 if not (b & SF) or (v["ks_flat"] - v["ks_mu"]) > m else 0)
B = load(sys.argv[1])
for p in sys.argv[2:]:
    V = load(p); keys = sorted(k for k in V if k in B); T = {k: C.is_stopper(rec[k]) for k in keys}
    gate = sum(pred(V[k], 0.0) != int(V[k]["is_stm"]) for k in keys)
    print("\n##### %s  (%d items, m=0 gate mismatches %d)" % (p.split("/")[-1], len(keys), gate))
    for m in (0.0, -0.01, -0.02, -0.03, -0.05):
        s = {k: pred(V[k], m) for k in keys}; b0 = {k: int(B[k]["is_stm"]) for k in keys}
        tp = sum(s[k] and T[k] for k in keys); fp = sum(s[k] and not T[k] for k in keys); fn = sum(not s[k] and T[k] for k in keys)
        pu = tp / max(tp + fp, 1); ef = tp / max(tp + fn, 1)
        newfp = [k for k in keys if s[k] and not b0[k] and not T[k]]; lost = [k for k in keys if b0[k] and not s[k] and T[k]]
        rmfp = [k for k in keys if b0[k] and not s[k] and not T[k]]; gain = [k for k in keys if s[k] and not b0[k] and T[k]]
        print("m=%+.2f TP %3d FP %2d FN %3d pur %.3f eff %.3f F1 %.3f | vs baseline: +%d TP, -%d TP, +%d FP, -%d FP" % (m, tp, fp, fn, pu, ef, 2*pu*ef/(pu+ef), len(gain), len(lost), len(newfp), len(rmfp)))
        if m in (0.0, -0.02):
            print("    new FP:  " + ", ".join("%s(%s)" % (k, rec[k]["confidence"]) for k in newfp))
            print("    lost TP: " + ", ".join("%s(%s,%s)" % (k, rec[k]["verdict"], rec[k]["confidence"]) for k in lost))
            print("    FP removed: " + ", ".join(rmfp))
