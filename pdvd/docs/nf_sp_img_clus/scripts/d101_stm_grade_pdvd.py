#!/usr/bin/env python3
"""doc pdvd/101 sec 6.5 -- PDVD STM/Michel grade of the knob arms on a FIXED denominator.

Fork by duplication (CLAUDE.md M10) of the sec-A tables of pdhd/stm_michel_scan/census_score.py
(untouched): same record, same census_lib truth (judged / is_stopper / is_michel), same payload join
by (event, cluster_id).  The one difference: census_score.py scores only the records that have a
payload in the arm, so an arm whose CheckSTM_Michel loses candidates loses denominator.  Here the
population is fixed by the FIRST arm (judged records with a payload there); a record absent from a
later arm is scored chain-negative -- an absent hand stopper is an is_stm miss (FN), an absent hand
Michel a michel_found miss.  Candidates an arm GAINS have no hand verdict and are counted, not scored.

Usage (STM_SCAN_RECORD = the merged smx1a..smx9 record, as for census_score.py):
  d101_stm_grade_pdvd.py <label>=<prep_dir> [<label>=<prep_dir> ...]   # first = population arm
"""
import os, sys

sys.path.insert(0, "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan")
import census_lib as C  # noqa: E402

rec = C.load_record()
arms = [s.split("=", 1) for s in sys.argv[1:]]
pays = [(lab, C.load_payloads(prep, rec)[0]) for lab, prep in arms]
POP = [k for k in rec if k in pays[0][1] and C.judged(rec[k])]
print(f"record {len(rec)} items; population = judged items with a payload in '{arms[0][0]}': {len(POP)}")


def grade(P, truth, chain):
    c = dict(tp=0, fp=0, fn=0, tn=0, absent_pos=0, absent_neg=0)
    for k in POP:
        t = truth(rec[k])
        if k not in P:
            c["absent_pos" if t else "absent_neg"] += 1
            c["fn" if t else "tn"] += 1
            continue
        g = bool(P[k]["verdict"][chain])
        c["tp" if (t and g) else "fn" if t else "fp" if g else "tn"] += 1
    c["purity"] = c["tp"] / max(1, c["tp"] + c["fp"])
    c["efficiency"] = c["tp"] / max(1, c["tp"] + c["fn"])
    return c


for title, truth, chain in (("is_stm", C.is_stopper, "is_stm"), ("michel_found", C.is_michel, "michel_found")):
    print(f"\n=== {title} (fixed denominator {len(POP)}) ===")
    for lab, P in pays:
        c = grade(P, truth, chain)
        print(f"  {lab:10s} TP {c['tp']:3d} FP {c['fp']:3d} FN {c['fn']:3d} TN {c['tn']:3d} | purity {c['purity']:.3f} "
              f"efficiency {c['efficiency']:.3f} | absent from the arm: {c['absent_pos']} hand positives, {c['absent_neg']} others")
for lab, P in pays[1:]:
    gained = [k for k in P if k not in pays[0][1]]
    print(f"\n{lab}: payloads with a record key but none in '{arms[0][0]}' (unscored here): {len(gained)}")
