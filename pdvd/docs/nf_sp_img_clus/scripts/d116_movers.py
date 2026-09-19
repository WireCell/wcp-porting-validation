#!/usr/bin/env python3
"""doc pdvd/116 sec 2 -- the STM / Michel tag movers between two arms with their truth, and the unlabelled candidates
(the blind-scan items).  Read-only.  Fork by duplication of the mover block of d113_nearfar_movers.py (untouched),
generalised to any (base, arm) pair; the truth and population are d113_grade.truth / d103_union_grade.population, so
the four-way counts equal the grade's "candidates on one side only" split.

Per metric (is_stm, michel_found) and direction: tp_lost (hand-positive, tagged in A not B), tp_new, fp_new
(hand-negative, tagged in B not A), fp_gone; each key with its truth (verdict / michel_kind / source) and whether the
cluster is a CANDIDATE (a T_stm_michel row) in each arm -- a lost tag on a cluster that is no longer a candidate is a
candidacy change (doc 107), a lost tag on a candidate is a verdict change.

Usage: d116_movers.py --det pdhd --a d115hoff --b d115hp3bwp05 --out figs/116_movers_pdhd [--extra-record F]
Writes <out>.txt (tables), <out>.tsv (one row per mover), <out>_unlabelled.tsv (candidates without truth, by cell).
"""
import argparse, os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d113_grade as G
import d103_union_grade as U


def load(det, a, b, extra_record=None):
    T, counts = G.truth(det)
    if extra_record:
        import json
        n = 0
        for r in json.load(open(extra_record)):
            if r["key"] not in T:
                T[r["key"]] = U.row_truth(r, "smx116"); n += 1
        counts.append((os.path.basename(extra_record), n))
    R = {"A0": U.cell_rows(det, a), "A1": U.cell_rows(det, b)}
    P = U.population(det, T, R, False, stm_only_unset_negative=(det == "pdhd"))
    return T, counts, R, P


def movers(T, R, P):
    out = {}
    for metric, pop, TR, idx in (("is_stm", P["pop"], P["stm_truth"], 0), ("michel_found", P["mpop"], P["m_truth"], 1)):
        g0 = lambda k: R["A0"].get(k, (0, 0))[idx]
        g1 = lambda k: R["A1"].get(k, (0, 0))[idx]
        out[metric] = dict(
            fp_new=[k for k in pop if not TR[k] and g1(k) and not g0(k)],
            fp_gone=[k for k in pop if not TR[k] and g0(k) and not g1(k)],
            tp_lost=[k for k in pop if TR[k] and g0(k) and not g1(k)],
            tp_new=[k for k in pop if TR[k] and g1(k) and not g0(k)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True, choices=["pdhd", "pdvd"])
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra-record", default=None)
    a = ap.parse_args()
    T, counts, R, P = load(a.det, a.a, a.b, a.extra_record)
    M = movers(T, R, P)
    L = [f"# doc pdvd/116 movers {a.det}: A0={a.a} A1={a.b}; truth sources {counts}",
         f"population {len(P['pop'])} (Michel population {len(P['mpop'])}); candidates A0 {len(R['A0'])}, A1 {len(R['A1'])}, "
         f"common {len(set(R['A0']) & set(R['A1']))}, A0-only {len(set(R['A0']) - set(R['A1']))}, A1-only {len(set(R['A1']) - set(R['A0']))}; "
         f"unlabelled candidates {len(P['unlabelled'])}"]
    rows = []
    for metric in ("is_stm", "michel_found"):
        L.append(f"\n== {metric}")
        for cls in ("tp_lost", "tp_new", "fp_new", "fp_gone"):
            ks = M[metric][cls]
            cand = {"both": 0, "A0 only": 0, "A1 only": 0}
            for k in ks:
                cand["both" if k in R["A0"] and k in R["A1"] else "A0 only" if k in R["A0"] else "A1 only"] += 1
            src = {}
            for k in ks:
                src[T[k][2]] = src.get(T[k][2], 0) + 1
            L.append(f"  {cls:8s} {len(ks):3d}  candidate in: {cand}  truth source: {src}")
            for k in ks:
                v, mk, s = T[k]
                c0, c1 = R["A0"].get(k), R["A1"].get(k)
                L.append(f"      {k:16s} truth {v}/{mk} [{s}]  A0 {c0}  A1 {c1}")
                rows.append((metric, cls, k, v, mk, s, c0, c1))
    open(a.out + ".txt", "w").write("\n".join(L) + "\n")
    with open(a.out + ".tsv", "w") as f:
        f.write("metric\tclass\tkey\tverdict\tmichel_kind\tsource\tA0_is_stm\tA0_michel\tA1_is_stm\tA1_michel\n")
        for metric, cls, k, v, mk, s, c0, c1 in rows:
            c0 = c0 or ("-", "-"); c1 = c1 or ("-", "-")
            f.write(f"{metric}\t{cls}\t{k}\t{v}\t{mk}\t{s}\t{c0[0]}\t{c0[1]}\t{c1[0]}\t{c1[1]}\n")
    with open(a.out + "_unlabelled.tsv", "w") as f:
        f.write("key\tin_A0\tin_A1\tA0_is_stm\tA0_michel\tA1_is_stm\tA1_michel\n")
        for k in P["unlabelled"]:
            c0 = R["A0"].get(k, ("-", "-")); c1 = R["A1"].get(k, ("-", "-"))
            f.write(f"{k}\t{int(k in R['A0'])}\t{int(k in R['A1'])}\t{c0[0]}\t{c0[1]}\t{c1[0]}\t{c1[1]}\n")
    print("\n".join(L[:2] + [l for l in L if l.startswith("  ") or l.startswith("\n==")]))


if __name__ == "__main__":
    main()
