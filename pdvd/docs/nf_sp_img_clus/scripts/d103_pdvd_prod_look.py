#!/usr/bin/env python3
"""doc pdvd/103 round 2 -- FIRST LOOK (not the pre-registered reading): PDVD production-lineage cells d103v0 (production,
== p100flip 119/119) and d103v1 (fit knobs + charge_stepped), graded on the doc-100 carried + owner-corrected record
(/home/xqian/tmp/p100/carry_r2/latest_on_p99rwon_corrected.json, flat verdict; STM truth stopper, Michel truth STM_MICHEL
as d101_stm_grade_pdvd.py).  Population = judged record items that are a candidate in either cell; unlabelled candidates
are counted, not graded.  103_pred.txt excluded carried records as truth; a PDVD amendment must decide before any reading."""
import sys, os, glob, json, collections
sys.path.insert(0, "/home/xqian/toolkit-dev/wcp-porting-img/pdvd/docs/nf_sp_img_clus/scripts")
import d103_union_grade as U
REC = "/home/xqian/tmp/p100/carry_r2/latest_on_p99rwon_corrected.json"
T = {r["key"]: (U.strip(r["verdict"]), r.get("michel_kind"), r.get("latest_source") or r.get("source")) for r in json.load(open(REC))}
C = [("A0", "d103v0"), ("A1", "d103v1")]
R = {lab: U.cell_rows("pdvd", arm) for lab, arm in C}
anyc = set(R["A0"]) | set(R["A1"])
judged = {k for k in anyc if k in T and T[k][0] not in ("MESSY", "UNCLEAR")}
unl = {k for k in anyc if k not in T}
print(f"# {REC}\n# candidates A0 {len(R['A0'])} A1 {len(R['A1'])} union {len(anyc)}; judged {len(judged)}; unjudged {len(anyc & set(T)) - len(judged)}; unlabelled {len(unl)}")
for lab in ("A0", "A1"):
    print(f"  {lab} unlabelled candidates tagged is_stm {sum(1 for k in unl if R[lab].get(k, (0, 0))[0])}, michel_found {sum(1 for k in unl if R[lab].get(k, (0, 0))[1])}")
pop = sorted(judged)
res = {}
for title, idx, truth in (("is_stm", 0, lambda k: T[k][0] in ("STM_MICHEL", "STM_ONLY")), ("michel_found", 1, lambda k: T[k][0] == "STM_MICHEL")):
    TR = {k: truth(k) for k in pop}
    print(f"== {title} (population {len(pop)})")
    for lab, arm in C:
        chain = {k: v[idx] for k, v in R[lab].items()}
        c = U.counts(pop, TR, chain); p, e = U.pe(c); (plo, phi), (elo, ehi) = U.boot(pop, TR, chain)
        res[(title, lab)] = (p, e)
        print(f"  {lab} {arm} TP {c['tp']} FP {c['fp']} FN {c['fn']} TN {c['tn']} | purity {p:.3f} [{plo:.3f}, {phi:.3f}] efficiency {e:.3f} [{elo:.3f}, {ehi:.3f}]")
    print(f"  A1 - A0: purity {res[(title, 'A1')][0] - res[(title, 'A0')][0]:+.3f}, efficiency {res[(title, 'A1')][1] - res[(title, 'A0')][1]:+.3f}")
    fps = sorted(k for k in pop if R["A1"].get(k, (0, 0))[idx] and not TR[k] and not R["A0"].get(k, (0, 0))[idx])
    print(f"  A1-only FPs {len(fps)} by source {dict(collections.Counter(T[k][2] for k in fps))}: {fps}")


# bound (as d103_bounds.py for round 1): every unlabelled candidate enters the population, labelled best / worst for A1
print("\n== bound over the unlabelled candidates (best case for A1 / worst case for A1)")
for title, idx, truth in (("is_stm", 0, lambda k: T[k][0] in ("STM_MICHEL", "STM_ONLY")), ("michel_found", 1, lambda k: T[k][0] == "STM_MICHEL")):
    for case in ("best", "worst"):
        TR = {k: truth(k) for k in pop}
        for k in unl:
            g0, g1 = R["A0"].get(k, (0, 0))[idx], R["A1"].get(k, (0, 0))[idx]
            # best for A1: a positive wherever A1 tags, else a negative wherever A0 tags; worst: the reverse
            TR[k] = bool(g1) if case == "best" else (bool(g0) and not g1) or (not g1 and not g0)
            if case == "worst" and g1 and not g0:
                TR[k] = False
            if case == "best" and g0 and not g1:
                TR[k] = False
        P = sorted(TR)
        out = []
        for lab, arm in C:
            chain = {k: v[idx] for k, v in R[lab].items()}
            c = U.counts(P, TR, chain); p, e = U.pe(c); out.append((p, e))
        print(f"  {title} {case:5s}: A0 purity {out[0][0]:.3f} eff {out[0][1]:.3f} | A1 purity {out[1][0]:.3f} eff {out[1][1]:.3f} | "
              f"A1 - A0 purity {out[1][0] - out[0][0]:+.3f} eff {out[1][1] - out[0][1]:+.3f}")
