#!/usr/bin/env python3
"""doc pdvd/100 round 3 -- prereg_round3.md Q3: the STM chain on the QtoL arm against production, graded on the owner-corrected
record (round 2's carry_r2), with round 2's bar applied to BOTH purities.

    python3 d100r3_grade.py --arm p100flip:/home/xqian/tmp/p100/carry_r2/latest_on_p99rwon_corrected.json \
        --arm p101q:/home/xqian/tmp/p101/carry/corrected_on_p101q.json > ../d100/qtol_grade_p100flip_p101q.txt

Per arm: d99_grade's items on the arm's record + doc pdhd/25's census (the same calls d100_michel_scan_score.py sec 4 makes),
is_stm and Michel TP/FP/FN/TN, purity +- binomial error, efficiency, split by volume (stop_x > 0 = top).
Criterion (prereg_round3.md sec 3 Q3): the second arm's Michel purity AND is_stm purity each within 1.5 sigma (errors in
quadrature) of the first arm's, or higher.  Movers are listed by d99_grade.py --movers, not here.
"""
import argparse, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import d99_grade as G


def purity(tp, fp):
    n = tp + fp
    p = tp / n if n else float("nan")
    return p, (math.sqrt(p * (1 - p) / n) if n else float("nan")), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="ARM:RECORD, base arm first")
    a = ap.parse_args()
    if len(a.arm) != 2:
        sys.exit("give exactly two --arm ARM:RECORD (base first)")
    res, names = {}, []
    print("# doc pdvd/100 round 3 -- Q3 grade on the owner-corrected record (prereg_round3.md sec 3)")
    for spec in a.arm:
        arm, rec = spec.split(":", 1)
        names.append(arm)
        its, miss, _, n = G.items_on(arm, rec)
        (TP, FP, FN, TN), _, (mTP, mFP, mFN, mTN) = G.BM.census(its)
        res[arm] = dict(stm=purity(TP, FP), mich=purity(mTP, mFP))
        s, m = res[arm]["stm"], res[arm]["mich"]
        print(f"  {arm:8s} is_stm {TP}/{FP}/{FN}/{TN} purity {s[0]:.3f} +- {s[1]:.3f} eff {TP / max(1, TP + FN):.3f}   "
              f"michel {mTP}/{mFP}/{mFN}/{mTN} purity {m[0]:.3f} +- {m[1]:.3f} eff {mTP / max(1, mTP + mFN):.3f}   "
              f"(record {os.path.basename(rec)}: {n} items, no candidate {miss})")
        for vol in ("top", "bottom"):
            sub = [x for x in its if (float(x[4]["stop_x"]) > 0) == (vol == "top")]
            (TP, FP, FN, TN), _, (mTP, mFP, mFN, mTN) = G.BM.census(sub)
            print(f"      {vol:6s} is_stm {TP}/{FP}/{FN}/{TN} purity {TP / max(1, TP + FP):.3f}   "
                  f"michel {mTP}/{mFP}/{mFN}/{mTN} purity {mTP / max(1, mTP + mFP):.3f} eff {mTP / max(1, mTP + mFN):.3f}")
    b, c = names
    ok = True
    for what, name in (("mich", "Michel purity"), ("stm", "is_stm purity")):
        (pb, sb, _), (pc, sc, _) = res[b][what], res[c][what]
        z = (pc - pb) / math.sqrt(sb ** 2 + sc ** 2)
        passed = abs(z) <= 1.5 or pc >= pb
        ok &= passed
        print(f"  {name}: {b} {pb:.3f} -> {c} {pc:.3f}, difference {pc - pb:+.3f} = {z:+.2f} sigma  {'PASS' if passed else 'FAIL'}")
    print(f"  Q3: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
