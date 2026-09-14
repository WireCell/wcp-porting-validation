#!/usr/bin/env python3
"""doc pdvd/100 round 2 -- the selection check on the own100m re-grade (d100_michel_scan_score.py sec 4).

    python3 d100_michel_scan_subsets.py > ../d100/michel_scan_subsets.txt

own100m corrected exactly the objects where p99wflip and p100c disagree, so the corrected-record PASS could in principle be
carried by where the corrections were made.  This grades both arms on the same corrected records, split into:
  untouched  -- items no owner scan of this round corrected (corrected_by absent)
  touched    -- items own100m / own100 corrected
and, independently, into
  common     -- record objects that are candidates on BOTH arms (same original key)
  arm-only   -- candidates on one arm only (the candidate-set churn; never corrected by own100m)
On common untouched items both chains answer alike by construction, so any purity difference in `untouched` comes from the
arm-only items, still graded on the carried (production-display) verdicts.
"""
import math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import json
import d99_grade as G

C = "/home/xqian/tmp/p100/carry_r2"
REC = {"p99wflip": C + "/latest_on_p99wflip_corrected.json", "p100c": C + "/latest_on_p99rwon_corrected.json"}


def line(name, its):
    (TP, FP, FN, TN), _, (mTP, mFP, mFN, mTN) = G.BM.census(its)
    p = lambda a, b: (a / (a + b), math.sqrt(a * b / (a + b) ** 3)) if a + b else (float("nan"), float("nan"))
    sp, ss = p(TP, FP); mp, ms = p(mTP, mFP)
    return (f"    {name:22s} n {len(its):3d}  is_stm {TP}/{FP}/{FN}/{TN} purity {sp:.3f} +- {ss:.3f} eff {TP / max(1, TP + FN):.3f}   "
            f"michel {mTP}/{mFP}/{mFN}/{mTN} purity {mp:.3f} +- {ms:.3f} eff {mTP / max(1, mTP + mFN):.3f}"), (mp, ms), (sp, ss)


def main():
    by, touched = {}, {}
    for arm, path in REC.items():
        rec = json.load(open(path))
        touched[arm] = {r["key"] for r in rec if r.get("corrected_by")}
        its, miss, orig, n = G.items_on(arm, path)
        by[arm] = (its, orig)
    common = set(by["p99wflip"][1][k] for k, *_ in by["p99wflip"][0]) & set(by["p100c"][1][k] for k, *_ in by["p100c"][0])
    print("# doc pdvd/100 round 2 -- selection check on the corrected-record re-grade; d100_michel_scan_subsets.py")
    print(f"# records {REC}")
    res = {}
    for arm in ("p99wflip", "p100c"):
        its, orig = by[arm]
        sets = {
            "all": its,
            "untouched": [x for x in its if x[0] not in touched[arm]],
            "touched": [x for x in its if x[0] in touched[arm]],
            "common": [x for x in its if orig[x[0]] in common],
            "arm-only": [x for x in its if orig[x[0]] not in common],
            "common untouched": [x for x in its if orig[x[0]] in common and x[0] not in touched[arm]],
            "arm-only untouched": [x for x in its if orig[x[0]] not in common and x[0] not in touched[arm]],
        }
        print(f"\n== {arm} (corrected items with a candidate: {len(sets['touched'])})")
        for name, s in sets.items():
            txt, m, st = line(name, s)
            res[(arm, name)] = (m, st)
            print(txt)
    print("\n== p100c - p99wflip, in sigma (both arms in quadrature)")
    for name in ("all", "untouched", "common", "arm-only", "common untouched", "arm-only untouched"):
        (mw, sw), (swp, sws) = res[("p99wflip", name)]
        (mc, sc), (scp, scs) = res[("p100c", name)]
        zm = (mc - mw) / math.sqrt(sw ** 2 + sc ** 2) if sw == sw and sc == sc and (sw or sc) else float("nan")
        zs = (scp - swp) / math.sqrt(sws ** 2 + scs ** 2) if sws == sws and scs == scs and (sws or scs) else float("nan")
        print(f"  {name:20s} Michel purity {mw:.3f} -> {mc:.3f} ({zm:+.2f} sigma)   is_stm purity {swp:.3f} -> {scp:.3f} ({zs:+.2f} sigma)")


if __name__ == "__main__":
    main()
