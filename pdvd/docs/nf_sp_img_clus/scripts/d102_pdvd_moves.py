#!/usr/bin/env python3
"""doc pdvd/102 round 2 -- PDVD per-item STM / Michel verdict moves between two arms, on the hand-scan record.

Fork by duplication of d101_stm_grade_pdvd.py (untouched): same merged record (STM_SCAN_RECORD), same census_lib
truth (judged / is_stopper / is_michel), same payload join by (event, cluster_id), population = judged items
with a payload in BASE.  Instead of counts it reports, per chain (is_stm, michel_found):
  lost_absent    hand positive, chain 1 in BASE, the candidate is absent from ARM
  lost_verdict   hand positive, chain 1 in BASE, present in ARM with chain 0
  gained_verdict / gained_absent   hand positive, chain 0 (or absent) in BASE, 1 in ARM
  new_fp_verdict / new_fp_absent   hand negative, chain 0 (or absent) in BASE, 1 in ARM
  gone_fp        hand negative, chain 1 in BASE, 0 or absent in ARM
and lists the Michel new false positives (the items the owner needs to adjudicate a purity change).

Usage (STM_SCAN_RECORD set as for census_score.py):
  d102_pdvd_moves.py BASE=<prep_dir> ARM=<prep_dir>
"""
import collections, sys

sys.path.insert(0, "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan")
import census_lib as C  # noqa: E402

rec = C.load_record()
(bl, bp), (al, ap) = [s.split("=", 1) for s in sys.argv[1:3]]
B = C.load_payloads(bp, rec)[0]
A = C.load_payloads(ap, rec)[0]
POP = [k for k in rec if k in B and C.judged(rec[k])]
print(f"population = judged items with a payload in '{bl}': {len(POP)}; arm '{al}'")

for title, truth, chain in (("is_stm", C.is_stopper, "is_stm"), ("michel_found", C.is_michel, "michel_found")):
    c = collections.Counter(); newfp = []
    for k in POP:
        t = bool(truth(rec[k]))
        gb = bool(B[k]["verdict"][chain])
        ga = bool(A[k]["verdict"][chain]) if k in A else False
        if gb == ga:
            continue
        where = "absent" if k not in A else "verdict"
        if t:
            c[("lost_" if gb else "gained_") + where] += 1
        else:
            if ga:
                c["new_fp_" + where] += 1; newfp.append(k)
            else:
                c["gone_fp_" + where] += 1
    print(f"\n=== {title}: {dict(sorted(c.items()))}")
    if title == "michel_found":
        print("new Michel false positives (hand non-Michel, michel_found 1 only in the arm):")
        for k in sorted(newfp, key=str):
            print(f"  {k}")
