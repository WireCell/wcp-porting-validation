#!/usr/bin/env python3
"""doc qlmatch/35 amendment 2 sec 3 -- the combined V2 calibration check over the two blind-scan records.

For every calibration row of the given records (smx35: amendment 1's 4; smx35c: amendment 2's 26), stopper-or-not of
the blind verdict vs the grade truth the items were drawn from (d35_scan_record.old_truth: the d116 union with own116v
and smx116), with d103_scan_record's classes: STM_MICHEL / STM_ONLY (and FRAG_) = stop, THRU / FRAG_THRU = non,
anything else excluded from the ratio.  PASS if disagreements / judged-on-both-sides <= 0.25.  Calibration keys are
native q35ctl keys that equal their record keys (d35_calib2_items.py draws only keys carried ok; the amendment-1 four
map to themselves, doc 35 sec 8), so the truth lookup is direct -- asserted below.

    python3 d35_v2_combined.py ../../scan/pdvd_stm_michel_smx35_verdicts.json ../../scan/pdvd_stm_michel_smx35c_verdicts.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import d35_scan_record as SR     # noqa: E402
import d103_scan_record as R     # noqa: E402


def main():
    old = SR.old_truth("pdvd")
    judged = dis = 0
    print("# doc qlmatch/35 amendment 2 -- combined V2 (stopper-or-not, blind vs the grade truth)")
    for path in sys.argv[1:]:
        rows = [r for r in json.load(open(path)) if r.get("calibration")]
        j = d = 0
        print(f"\n## {os.path.basename(path)}: {len(rows)} calibration rows")
        for r in sorted(rows, key=lambda x: x["key"]):
            assert r["display_arm"] == "q35ctl" and r["key"] in old, r["key"]
            tv, src = old[r["key"]]
            a, b = R.cls(r["verdict"]), R.cls(R.strip(tv))
            if "excl" in (a, b):
                st = "excluded"
            else:
                j += 1
                st = "agree" if a == b else "DISAGREE"
                d += st == "DISAGREE"
            print(f"   {r['key']:<16} truth {tv:<11} ({src:<12}) blind {r['verdict']:<14} ({r['confidence']:<6}) {st}")
        print(f"   judged {j}, disagreements {d}")
        judged += j
        dis += d
    frac = dis / max(1, judged)
    print(f"\nCOMBINED: judged on both sides {judged}, disagreements {dis} ({frac:.1%}) -> "
          f"{'PASS' if frac <= 0.25 else 'FAIL'} (bar <= 25 %)")


if __name__ == "__main__":
    main()
