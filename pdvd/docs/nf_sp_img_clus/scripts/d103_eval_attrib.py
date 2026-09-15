#!/usr/bin/env python3
"""doc pdvd/103 sec 4 -- which clause of TaggerCheckSTM::eval_stm_core decides a status 0 <-> 3 flip.  Read-only.

Input: tracking-stm.root of two arms on the same pctrees.
  T_stm_pass  one row per recorded pass (cluster_id, pass, status, kink_num, left_L, exit_L, ...)
  T_stm_eval  one row per eval_stm_core call (cluster_id, pass, strong, verdict, peak_range, offset_length, com_range,
              ks1, ks2, ratio1, ratio2, res_length, ave_res_dqdx) -- TaggerCheckSTM.cxx:2966-2974, verdict :3047

The clauses of eval_stm_core_impl (TaggerCheckSTM.cxx:2980-3036, accept_guards off in both drivers) are re-evaluated
from the recorded values, with D = ks1 - ks2 + (|ratio1-1| - |ratio2-1|)/1.5*0.3:
  R1  ks1 - ks2 >= 0                                        (:2987)
  R2  hypot(ks2/0.06, (ratio2-1)/0.06) < 1.4 and D > -0.02 (:2988)
  R3  residual straightness res_dis1/res_length1 > 0.99    (:2991)  -- NOT recorded; shows up as "predicted accept,
                                                              recorded reject" in the validation line
  R4  the Michel-residual clauses                          (:3005-3022; michel_res_length_cut 65 mm in both drivers)
  R5  no accept clause fires                               (:3025-3035)
Validation: predicted accept vs recorded verdict over every call (a recorded accept that is predicted to reject
would mean the re-evaluation is wrong).

Reports: the rejecting clause on the call (same peak_range/offset/com_range/strong) that accepted in the other arm,
whether any rejecting-arm call accepted, the change in left_L (the eval ladder, :3915-3961, is chosen by left_L and
flag_fix_end), the base population's best D and best (ks1-ks2) by lowest-pass status, and the arm-to-arm change of the
best (ks1-ks2) on clusters whose status did not change.

Usage: d103_eval_attrib.py --det pdhd --base d101hnew --arm d102hcs
"""
import argparse, collections, glob, os
import numpy as np
import uproot

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
MIP = {"pdhd": 56000.0, "pdvd": 55000.0}   # TaggerCheckSTM mip_dqdx, compiled PR config of both drivers
RES_CUT = 65.0                              # michel_res_length_cut (mm)
CM = 10.0


def dval(r):
    return r["ks1"] - r["ks2"] + (abs(r["ratio1"] - 1) - abs(r["ratio2"] - 1)) / 1.5 * 0.3


def clause(r, mip):
    ks1, ks2, r1, r2, rl, av = r["ks1"], r["ks2"], r["ratio1"], r["ratio2"], r["res_length"], r["ave_res_dqdx"]
    D = dval(r)
    if ks1 - ks2 >= 0:
        return "R1 ks1-ks2>=0"
    if np.hypot(ks2 / 0.06, (r2 - 1) / 0.06) < 1.4 and D > -0.02:
        return "R2 near-flat"
    q = av / mip
    if ((rl > 20 * CM and q > 1.2 and D > -0.02) or (rl > 16 * CM and q > 1.45) or (rl > 10 * CM and q > 1.45 and D > -0.05)
            or (rl > 10 * CM and q > 1.7) or (rl > RES_CUT and q > 1.85) or (rl > RES_CUT and q > 1.45 and D > -0.05)
            or (rl > 4 * CM and q > 1.4 and D > 0.02) or (rl > 2 * CM and q > 4.5)):
        return "R4 michel-residual"
    if r["strong"]:
        ok = ((ks1 - ks2 < -0.02 and (ks2 > 0.09 or r2 > 1.5) and ks1 < 0.05 and abs(r1 - 1) < 0.1)
              or (D < 0 and ks1 < 0.05 and abs(r1 - 1) < 0.1))
    else:
        ok = (ks1 - ks2 < -0.02 and ((ks2 > 0.09 and abs(r2 - 1) > 0.1) or r2 > 1.5 or ks2 > 0.2)) or D < 0
    return "accept (predicted)" if ok else "R5 no accept clause"


def load(det, arm):
    E, P = collections.defaultdict(list), collections.defaultdict(list)
    for d in glob.glob(f"{IMG}/{det}/work/*_{arm}"):
        ev = os.path.basename(d)[:-len(arm) - 1]
        try:
            f = uproot.open(f"{d}/tracking-stm.root")
            e = f["T_stm_eval"].arrays(library="np")
            p = f["T_stm_pass"].arrays(library="np")
        except Exception:
            continue
        for i in range(len(e["cluster_id"])):
            E[(ev, int(e["cluster_id"][i]))].append({k: float(e[k][i]) for k in e.keys()})
        for i in range(len(p["cluster_id"])):
            P[(ev, int(p["cluster_id"][i]))].append((int(p["pass"][i]), int(p["status"][i]), float(p["left_L"][i])))
    return E, P


def calls(E, k, ps):
    return [r for r in E.get(k, []) if int(r["pass"]) == ps]


def sig(r):
    return (r["peak_range"], r["offset_length"], r["com_range"], r["strong"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True, choices=["pdhd", "pdvd"])
    ap.add_argument("--base", required=True)
    ap.add_argument("--arm", required=True)
    a = ap.parse_args()
    mip = MIP[a.det]
    (EB, PB), (EA, PA) = load(a.det, a.base), load(a.det, a.arm)
    print(f"[{a.det}] {a.base} -> {a.arm}: clusters with a recorded pass {len(PB)} / {len(PA)}")

    val = collections.Counter()
    for E in (EB, EA):
        for cl in E.values():
            for r in cl:
                val[(clause(r, mip) == "accept (predicted)", int(r["verdict"]))] += 1
    print(f"validation over {sum(val.values())} calls: predicted accept & recorded accept {val[(True, 1)]}, "
          f"predicted reject & recorded reject {val[(False, 0)]}, predicted accept & recorded reject {val[(True, 0)]} "
          f"(R3 straightness, not recorded), predicted reject & recorded accept {val[(False, 1)]} (must be 0)")

    low = lambda P, k: min(P[k])
    attrib, ladder, dleft = collections.Counter(), collections.Counter(), []
    Dacc, Drej = [], []
    for k in set(PB) & set(PA):
        pb, pa = low(PB, k), low(PA, k)
        if {pb[1], pa[1]} != {0, 3}:
            continue
        lab = "0->3" if pb[1] == 0 else "3->0"
        (Eacc, pacc), (Erej, prej) = ((EB, pb), (EA, pa)) if pb[1] == 0 else ((EA, pa), (EB, pb))
        dleft.append(abs(pa[2] - pb[2]))
        ladder[(lab, "a rejecting-arm call accepted" if any(r["verdict"] == 1 for r in calls(Erej, k, prej[0]))
                else "no rejecting-arm call accepted")] += 1
        good = [r for r in calls(Eacc, k, pacc[0]) if r["verdict"] == 1]
        if not good:
            attrib[(lab, "no accepted call recorded")] += 1
            continue
        same = [r for r in calls(Erej, k, prej[0]) if sig(r) == sig(good[0])]
        if not same:
            attrib[(lab, "accepting call's rung not reached in the rejecting arm")] += 1
            continue
        attrib[(lab, clause(same[0], mip))] += 1
        Dacc.append(dval(good[0]))
        Drej.append(dval(same[0]))
    print(f"\nstatus 0<->3 flips: {sum(ladder.values())}; |change in left_L| median {np.median(dleft) if dleft else float('nan'):.1f} mm")
    for (lab, s), n in sorted(ladder.items()):
        print(f"  {lab}: {s}: {n}")
    print("rejecting clause on the accepting call's rung:")
    for (lab, s), n in sorted(attrib.items()):
        print(f"  {lab}: {s}: {n}")
    if Dacc:
        print(f"  D on that call: accepting arm median {np.median(Dacc):+.3f}, rejecting arm median {np.median(Drej):+.3f}")

    print("\nbase population, lowest pass, best (most muon-like) value over its eval calls:")
    for s in (0, 3):
        best_d, best_k = [], []
        for k, pl in PB.items():
            p0 = low(PB, k)
            if p0[1] != s or not calls(EB, k, p0[0]):
                continue
            cl = calls(EB, k, p0[0])
            best_d.append(min(dval(r) for r in cl))
            best_k.append(min(r["ks1"] - r["ks2"] for r in cl))
        best_d, best_k = np.array(best_d), np.array(best_k)
        print(f"  status {s}: n {len(best_d)}; best D median {np.median(best_d):+.3f}, share |D| < 0.06 {np.mean(np.abs(best_d) < 0.06):.2f}; "
              f"best (ks1-ks2) median {np.median(best_k):+.3f}, share |.| < 0.06 {np.mean(np.abs(best_k) < 0.06):.2f}")

    dd = []
    for k in set(PB) & set(PA):
        pb, pa = low(PB, k), low(PA, k)
        if pb[1] != pa[1] or pb[1] not in (0, 3):
            continue
        cb, ca = calls(EB, k, pb[0]), calls(EA, k, pa[0])
        if cb and ca:
            dd.append(min(r["ks1"] - r["ks2"] for r in ca) - min(r["ks1"] - r["ks2"] for r in cb))
    dd = np.array(dd)
    print(f"\nrefit noise: clusters with the same status 0/3 in both arms n {len(dd)}; change of best (ks1-ks2) "
          f"median {np.median(dd):+.3f}, p68 |.| {np.percentile(np.abs(dd), 68):.3f}, p90 |.| {np.percentile(np.abs(dd), 90):.3f}")


if __name__ == "__main__":
    main()
