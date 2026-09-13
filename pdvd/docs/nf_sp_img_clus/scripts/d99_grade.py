#!/usr/bin/env python3
"""doc pdvd/99 -- grade STM / Michel on an arm against a (carried) hand-scan record.

    python3 d99_grade.py                                          # G5a + the three arms, default records
    python3 d99_grade.py --arms p96vprod:REC p98voffq:CARRIED_OFF p98vonq:CARRIED_ON [--ok-only]

Reuses doc pdhd/25's census and report (pdhd/docs/scan/h25/d25_bragg_michel.py) unchanged: its
items() reads the module global VD_REC at call time, so the record is switched by assignment, not by
forking the grader.  The join is still A.get(key); a carried record's keys are the NEW arm's
"run_evt/cluster_id" (d99_match.py --carried), so the join is correct by construction.

G5a (positive control of this path): the committed record on p96vprod must reproduce doc pdvd/93/96's
is_stm (242,8,34,262) and michel (144,12,20) before any other number is printed.
Movers: an item is followed across arms by its ORIGINAL key (carried_from), and split by volume
(stop_x sign on the arm, top = x > 0).
"""
import argparse, collections, json, os, sys, tempfile

IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
sys.path.insert(0, IMG + "/pdhd/docs/scan/h25")
import d25_bragg_michel as BM

ORIG = BM.VD_REC
SCAN = IMG + "/pdvd/docs/scan"


def items_on(arm, rec_path, ok_only=False):
    rec = json.load(open(rec_path))
    if ok_only:
        rec = [r for r in rec if (r.get("match") or {}).get("status", "ok") == "ok"]
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, dir="/home/xqian/tmp")
        json.dump(rec, tmp); tmp.close(); rec_path = tmp.name
    BM.VD_REC = rec_path
    try:
        its, miss = BM.items("pdvd", arm, "all")
    finally:
        BM.VD_REC = ORIG
    orig = {r["key"]: r.get("carried_from", r["key"]) for r in rec}
    return its, miss, orig, len(rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=[
        f"p96vprod:{ORIG}",
        f"p98voffq:{SCAN}/pdvd_stm_michel_smx9_carried_p98voffq.json",
        f"p98vonq:{SCAN}/pdvd_stm_michel_smx9_carried_p98vonq.json"])
    ap.add_argument("--ok-only", action="store_true", help="grade only items whose match status is ok")
    ap.add_argument("--movers", default="p98voffq,p98vonq", help="pair of arms to list movers between")
    a = ap.parse_args()

    its, _, _, _ = items_on("p96vprod", ORIG)
    got, _, gotm2 = BM.census(its)
    ok = got == (242, 8, 34, 262) and gotm2[:3] == (144, 12, 20)
    print(f"GATE G5a p96vprod on {os.path.basename(ORIG)}: is_stm {got} want (242, 8, 34, 262); "
          f"michel(all judged) {gotm2[:3]} want (144, 12, 20) -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        sys.exit("G5a FAILED -- refusing to print any number")

    by = {}
    for spec in a.arms:
        arm, path = spec.split(":", 1)
        if not os.path.exists(path):
            print(f"\n[skip] {arm}: no record {path}")
            continue
        its, miss, orig, n = items_on(arm, path, a.ok_only)
        by[arm] = {orig[k]: (v, kind, d) for k, v, kind, src, d in its}
        (TP, FP, FN, TN), _, (mTP, mFP, mFN, mTN) = BM.census(its)
        print(f"\n##### {arm}  record {os.path.basename(path)} ({n} items{', ok-only' if a.ok_only else ''}; "
              f"items with no candidate in the arm: {miss})")
        print(f"  is_stm {TP}/{FP}/{FN}/{TN}  purity {TP / max(1, TP + FP):.3f} eff {TP / max(1, TP + FN):.3f}   "
              f"michel(all judged) {mTP}/{mFP}/{mFN}/{mTN}  purity {mTP / max(1, mTP + mFP):.3f} eff {mTP / max(1, mTP + mFN):.3f}")
        for vol in ("top", "bottom"):
            sub = [x for x in its if (float(x[4]["stop_x"]) > 0) == (vol == "top")]
            (TP, FP, FN, TN), _, (mTP, mFP, mFN, mTN) = BM.census(sub)
            print(f"    {vol:6s} n {len(sub):3d}  is_stm {TP}/{FP}/{FN}/{TN}  eff {TP / max(1, TP + FN):.3f}  "
                  f"michel {mTP}/{mFP}/{mFN}/{mTN}  eff {mTP / max(1, mTP + mFN):.3f}")
        BM.report(f"PDVD {arm}", its)

    x, y = a.movers.split(",")
    if x in by and y in by:
        print(f"\n##### movers {x} -> {y} (by original record key)")
        common = sorted(set(by[x]) & set(by[y]))
        print(f"  common items {len(common)}; only in {x}: {len(set(by[x]) - set(by[y]))}; only in {y}: {len(set(by[y]) - set(by[x]))}")
        for fld in ("is_stm", "michel_found"):
            mv = collections.Counter()
            names = collections.defaultdict(list)
            for k in common:
                v, kind, dx = by[x][k]; _, _, dy = by[y][k]
                a0, a1 = int(dx[fld]), int(dy[fld])
                if a0 != a1:
                    hand = ("stopper" if v in BM.STOP else v) if fld == "is_stm" else ("michel" if v == "STM_MICHEL" else v)
                    vol = "top" if float(dy["stop_x"]) > 0 else "bottom"
                    tag = f"{vol} {hand} {a0}->{a1}"
                    mv[tag] += 1; names[tag].append(k)
            print(f"  {fld}: {sum(mv.values())} moved")
            for t, c in sorted(mv.items()):
                print(f"    {t:32s} {c:3d}  {' '.join(names[t][:12])}{' ...' if len(names[t]) > 12 else ''}")


if __name__ == "__main__":
    main()
