#!/usr/bin/env python3
"""doc pdvd/100 -- (a) does plateau_mip_hi bind on p100c at all; (b) the Michel admission quantities of the new top Michel
false positives (production -> p100c) against the top hand-Michel true positives.

    python3 d100_plateau_michel_diag.py > ../d100/twin_plateau_binding_and_michel_fp.txt

(a) answers whether d100_twin.py's flat plateau column is a result or an inert knob: if (almost) no candidate reaches the
upper edge, "keep 2.0" is vacuous.  (b) lists the items of grade_p99wflip_p100c_latest_record.txt's "top STM_ONLY 0->1"
and "top THRU 0->1" michel_found movers from the p100c payloads.
"""
import json, os, collections
import numpy as np

P = "/home/xqian/tmp/p100/prep_p100c"
REC = "/home/xqian/tmp/p100/carry/latest_on_p99rwon.json"
PM = 1 << 10
print("# doc pdvd/100 -- (a) does plateau_mip_hi bind on p100c? (b) Michel admission quantities of the new top Michel FPs vs the top hand-Michel TPs")
print("# payloads: prep_stm_michel_scan.py --det pdvd --arm p100c (scratch); record: latest carried record on p99rwon keys (d100_carry_verdicts.py)")
n = collections.Counter(); r = []
for f in sorted(os.listdir(P)):
    if not f.startswith("smprep-"): continue
    v = json.load(open(os.path.join(P, f)))["verdict"]; n["candidates"] += 1
    if v.get("bragg_valid"): n["bragg_valid"] += 1
    pm = v.get("plateau_med")
    if pm is not None and v.get("bragg_valid"):
        x = pm / 55000.0; r.append(x)
        for t in (1.3, 1.6, 1.8, 2.0): n[f"plateau/MIP > {t}"] += x > t
        n["plateau/MIP < 0.6 (lower edge)"] += x < 0.6
    n["plateau_off_mip bit in reject_bits"] += bool(int(v.get("reject_bits") or 0) & PM)
print("\n(a)", dict(n))
print("    plateau/MIP p1 / p50 / p99 / max:", np.round(np.percentile(r, [1, 50, 99, 100]), 3))
print("    reading: the upper edge (plateau_mip_hi) binds on no record item at any grid value 1.6-2.5; every plateau rejection is the lower edge 0.6 (not in this round's scope)")
R = {x["key"]: x for x in json.load(open(REC))}
fp = ("039253_11/113 039349_12/52 039349_17/54 039349_18/47 039349_33/67 039349_47/71 039349_48/59 039349_50/49 039349_66/80 "
      "039252_17/95 039253_1/107 039253_11/84 039349_22/40 039349_32/63 039349_62/69 039349_63/44 039349_77/45").split()


def row(k):
    p = f"{P}/smprep-{k.replace('/', '-c')}.json"
    return json.load(open(p))["verdict"] if os.path.exists(p) else None


F = ("michel_conn_type", "michel_mip", "michel_ke_best", "michel_len", "michel_dis_cm", "stop_x")
print("\n(b) top items michel_found 0 on p99wflip -> 1 on p100c whose hand verdict is not STM_MICHEL (grade_p99wflip_p100c_latest_record.txt movers):")
V = []
for k in fp:
    v = row(k); V.append(v)
    print(f"    {k:15s} hand {R[k]['verdict']:9s} " + " ".join(f"{f}={round(v[f], 2) if isinstance(v[f], float) else v[f]}" for f in F))
tp = [row(k) for k, x in R.items() if x.get("verdict") == "STM_MICHEL" and row(k) and row(k)["michel_found"] and row(k)["stop_x"] > 0]
q = lambda a: np.round(np.percentile(np.array(a, float), [16, 50, 84]), 2)
for name, S in (("new top FPs", V), ("top hand-Michel TPs", tp)):
    print(f"    {name:20s} n {len(S):3d}: michel_ke_best p16/50/84 {q([v['michel_ke_best'] for v in S])}; michel_len {q([v['michel_len'] for v in S])}; michel_mip {q([v.get('michel_mip') or 0 for v in S])}; detached (conn 2) {sum(v['michel_conn_type'] == 2 for v in S)}")
