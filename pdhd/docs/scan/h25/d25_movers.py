#!/usr/bin/env python3
"""doc pdhd/25 sec 1 -- item-level movers between two PDHD arms on smx23, by name.

    python3 d25_movers.py <before_arm> <after_arm> [<after_arm> ...]

For each population (APA0 strict / majority / all) lists, by item: gained TP, new FP, lost TP, and the
Michel movers (hand stoppers, michel_kind attached/both), with the reject bits before -> after and the
fields that say WHY (topology_cleared_bits, n_stub_absorb, stop moved, ks gap).  APA assignment is taken
from the BEFORE arm's fit so both arms bucket an item identically.  Truth = d25_bragg_michel's.
"""
import sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d25_bragg_michel as Q
from d25_misses import names

def main():
    b0 = sys.argv[1]
    B = Q.read_arm("pdhd", b0)
    rec = {r["key"]: r for r in __import__("json").load(open(Q.HD_REC))}
    import csv
    POP = {"%s/%s" % (r["event"], r["cluster"]) for r in
           csv.DictReader([l for l in open(Q.HD_KEY) if not l.startswith("#")], delimiter="\t")}
    for a1 in sys.argv[2:]:
        A = Q.read_arm("pdhd", a1)
        print(f"==== {b0} -> {a1}   candidates {len(B)} -> {len(A)}; only in {b0}: {sorted(set(B)-set(A))}; only in {a1}: {sorted(set(A)-set(B))}")
        for pop in ("strict", "majority", "all"):
            rows = collections.defaultdict(list)
            for k in sorted(POP):
                if k not in rec: continue
                v, kind, src = Q.hd_truth(rec[k])
                if v in ("MESSY", "UNCLEAR"): continue
                d0 = B.get(k); d1 = A.get(k)
                if d0 is None:
                    continue
                if pop == "strict" and d0["apa_any0"]: continue
                if pop == "majority" and d0["apa_major"] == 0: continue
                if d1 is None:
                    rows["LEFT the candidate pool"].append((k, v, d0, None)); continue
                hs = v in Q.STOP; s0 = int(d0["is_stm"]); s1 = int(d1["is_stm"])
                if s1 and not s0: rows["gained TP" if hs else "NEW FP"].append((k, v, d0, d1))
                if s0 and not s1: rows["lost TP" if hs else "FP removed"].append((k, v, d0, d1))
                if hs and not (src == "owner_review" and kind is None):
                    hm = kind in Q.MICHEL_KINDS; m0 = int(d0["michel_found"]); m1 = int(d1["michel_found"])
                    if m1 != m0:
                        tag = ("Michel gained" if m1 else "Michel lost") if hm else ("Michel FP added" if m1 else "Michel FP removed")
                        rows[tag].append((k, v, d0, d1))
            print(f"  --- {pop} ---")
            for tag in ("gained TP", "lost TP", "NEW FP", "FP removed", "LEFT the candidate pool",
                        "Michel gained", "Michel lost", "Michel FP added", "Michel FP removed"):
                L = rows.get(tag, [])
                print(f"    {tag:24s} {len(L)}")
                if pop == "all" or tag in ("NEW FP", "lost TP", "LEFT the candidate pool"):
                    for k, v, d0, d1 in L:
                        if d1 is None:
                            print(f"        {k:14s} {v:11s} apa {int(d0['apa_major'])}"); continue
                        moved = abs(float(d0["stop_x"]) - float(d1["stop_x"])) + abs(float(d0["stop_y"]) - float(d1["stop_y"])) + abs(float(d0["stop_z"]) - float(d1["stop_z"]))
                        print(f"        {k:14s} {v:11s} apa {int(d0['apa_major'])}  bits {names(d0['reject_bits'])} -> {names(d1['reject_bits'])}"
                              f"  topo_clr {int(d1['topology_cleared_bits'])}  ks_gap {float(d0['ks_flat'])-float(d0['ks_mu']):+.3f} -> {float(d1['ks_flat'])-float(d1['ks_mu']):+.3f}"
                              f"  mf {int(d0['michel_found'])}->{int(d1['michel_found'])} ke {float(d1['michel_ke_best']):.1f} len {float(d1['michel_len']):.1f}"
                              f"  stop moved {moved:.1f} cm  muon_len {float(d0['muon_len']):.0f}->{float(d1['muon_len']):.0f}")

if __name__ == "__main__":
    main()
