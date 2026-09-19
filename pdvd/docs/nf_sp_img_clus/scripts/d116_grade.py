#!/usr/bin/env python3
"""doc pdvd/116 -- the STM / Michel grade of figs/116_pred.txt.  Fork by duplication of d113_grade.py (untouched):
the same truth precedence, population, NEG / POS bounds and -0.020 bar, plus
  --extra-record F   a record folded at the LOWEST precedence (the doc-116 blind scan smx116) -- an item already in
                     a higher source keeps that source;
  --movers-out P     per non-base cell the four-way movers (tp_lost / tp_new / fp_new / fp_gone) of both metrics
                     with truth and source, one tsv per cell (P_<cell>.tsv);
  --split-half       the split-half guard of the rule: the grade repeated on the odd- and even-indexed halves of
                     the sorted event list, deltas vs the base per half (bar -0.040, reported per metric);
  --neg-bar X        the NEG-bound bar of the rule (default -0.030), printed next to the point reading.
--self-test reproduces d113_grade.py's self-test (same numbers, no extra record).

Usage: d116_grade.py --det pdhd --cells A0=d115hoff,T=d115hp3bwp05,R1=d116hr1,R2=d116hr2,R3=d116hr3 \
           [--extra-record pdhd/docs/scan/pdhd_stm_michel_smx116_verdicts.json] [--movers-out figs/116_movers_pdhd] [--split-half]
"""
import argparse, json, os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d113_grade as D                      # noqa: E402  (sets the record env before importing d103_union_grade)
import d103_union_grade as U                # noqa: E402

LIMIT = D.LIMIT


def truth(det, extra):
    T, counts = D.truth(det)
    if extra:
        n = 0
        for r in json.load(open(extra)):
            if r["key"] not in T:
                T[r["key"]] = U.row_truth(r, "smx116"); n += 1
        counts.append((os.path.basename(extra), n))
    return T, counts


def reading(det, T, R, cells, base):
    G, out = D.grade(det, T, R)
    scen = {}
    for name, row in (("NEG", ("THRU", None, "hypo")), ("POS", ("STM_MICHEL", "attached", "hypo"))):
        T2 = dict(T)
        for k in G["unlabelled"]:
            T2[k] = row
        scen[name] = D.grade(det, T2, R)[1]
    return G, out, scen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", choices=["pdhd", "pdvd"])
    ap.add_argument("--cells", help="LAB=arm,... ; the first cell is the base")
    ap.add_argument("--extra-record", default=None)
    ap.add_argument("--movers-out", default=None)
    ap.add_argument("--split-half", action="store_true")
    ap.add_argument("--neg-bar", type=float, default=-0.030)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.argv = [sys.argv[0], "--self-test"]
        return D.main()
    cells = [c.split("=") for c in a.cells.split(",")]
    base = cells[0][0]
    T, src = truth(a.det, a.extra_record)
    R = {lab: U.cell_rows(a.det, arm) for lab, arm in cells}
    G, out, scen = reading(a.det, T, R, cells, base)
    print(f"# doc pdvd/116 grade ({a.det}); truth sources in precedence order (items taken): {src}")
    print(f"# cells: {', '.join(f'{l}={x}' for l, x in cells)}; base {base}; bar A_x - {base} >= {LIMIT} on the point value, NEG bound >= {a.neg_bar}")
    print(f"population {len(G['pop'])} (fixed {len(G['fixed'])}); Michel population {len(G['mpop'])} (excluded {G['mexcl']}); "
          f"unlabelled candidates of any cell {len(G['unlabelled'])}")
    for lab in R:
        u = [k for k in G["unlabelled"] if k in R[lab]]
        print(f"  unlabelled candidates of {lab}: {len(u)}; is_stm 1 {sum(R[lab][k][0] for k in u)}, michel_found 1 {sum(R[lab][k][1] for k in u)}")
    for title in ("is_stm", "michel_found"):
        print(f"\n== {title}")
        for lab, arm in cells:
            c, (p, e), P, TR, chain = out[(title, lab)]
            (plo, phi), (elo, ehi) = U.boot(P, TR, chain)
            print(f"  {lab:4s} {arm:13s} TP {c['tp']:3d} FP {c['fp']:3d} FN {c['fn']:3d} TN {c['tn']:4d} | purity {p:.3f} "
                  f"[{plo:.3f}, {phi:.3f}] efficiency {e:.3f} [{elo:.3f}, {ehi:.3f}]")
    print(f"\n== reading per cell vs {base} (point value; NEG / POS bounds for the unlabelled); rule: point >= {LIMIT} AND NEG >= {a.neg_bar}")
    summary = {}
    for lab, arm in cells[1:]:
        verdicts = []
        for title in ("is_stm", "michel_found"):
            for i, name in ((0, "purity"), (1, "efficiency")):
                d = out[(title, lab)][1][i] - out[(title, base)][1][i]
                dn = scen["NEG"][(title, lab)][1][i] - scen["NEG"][(title, base)][1][i]
                dp = scen["POS"][(title, lab)][1][i] - scen["POS"][(title, base)][1][i]
                v = "PASS" if (d >= LIMIT and dn >= a.neg_bar) else "FAIL"
                v113 = "PASS" if min(dn, dp) >= LIMIT else "FAIL" if max(dn, dp) < LIMIT else "UNDECIDED"
                verdicts.append(v)
                print(f"  {lab:4s} {title:12s} {name:10s}: {d:+.3f}  (NEG {dn:+.3f}, POS {dp:+.3f})  {v}   [doc-113 reading {v113}]")
        summary[lab] = "PASS" if all(v == "PASS" for v in verdicts) else "FAIL"
        print(f"  {lab:4s} T -> {summary[lab]}")
    if a.split_half:
        print(f"\n== split-half guard: the grade on the odd / even halves of the sorted event list (bar {LIMIT * 2:+.3f} per metric)")
        events = sorted({k.split("/")[0] for lab in R for k in R[lab]})
        for hname, half in (("odd", set(events[1::2])), ("even", set(events[0::2]))):
            Rh = {lab: {k: v for k, v in R[lab].items() if k.split("/")[0] in half} for lab in R}
            Gh, outh = D.grade(a.det, T, Rh)
            for lab, arm in cells[1:]:
                ds = []
                for title in ("is_stm", "michel_found"):
                    for i, name in ((0, "pur"), (1, "eff")):
                        ds.append((f"{title[:6]}_{name}", outh[(title, lab)][1][i] - outh[(title, base)][1][i]))
                worst = min(d for _, d in ds)
                print(f"  {hname:4s} half ({len(half)} events, pop {len(Gh['pop'])}) {lab:4s}: " + ", ".join(f"{n} {d:+.3f}" for n, d in ds)
                      + f"  -> {'PASS' if worst >= LIMIT * 2 else 'FAIL'}")
    if a.movers_out:
        for lab, arm in cells[1:]:
            rows = []
            for metric, pop, TR, idx in (("is_stm", G["pop"], G["stm_truth"], 0), ("michel_found", G["mpop"], G["m_truth"], 1)):
                g0 = lambda k: R[base].get(k, (0, 0))[idx]
                g1 = lambda k: R[lab].get(k, (0, 0))[idx]
                for k in pop:
                    cls = ("tp_lost" if TR[k] and g0(k) and not g1(k) else "tp_new" if TR[k] and g1(k) and not g0(k) else
                           "fp_new" if not TR[k] and g1(k) and not g0(k) else "fp_gone" if not TR[k] and g0(k) and not g1(k) else None)
                    if cls:
                        rows.append((metric, cls, k, T[k][0], T[k][1], T[k][2], R[base].get(k), R[lab].get(k)))
            with open(f"{a.movers_out}_{lab}.tsv", "w") as f:
                f.write("metric\tclass\tkey\tverdict\tmichel_kind\tsource\tA0_is_stm\tA0_michel\tA1_is_stm\tA1_michel\n")
                for metric, cls, k, v, mk, s, c0, c1 in rows:
                    c0 = c0 or ("-", "-"); c1 = c1 or ("-", "-")
                    f.write(f"{metric}\t{cls}\t{k}\t{v}\t{mk}\t{s}\t{c0[0]}\t{c0[1]}\t{c1[0]}\t{c1[1]}\n")
            cnt = {}
            for metric, cls, *_ in rows:
                cnt[(metric, cls)] = cnt.get((metric, cls), 0) + 1
            print(f"\n== movers {lab} vs {base}: " + ", ".join(f"{m} {c} {n}" for (m, c), n in sorted(cnt.items())) + f"  -> {a.movers_out}_{lab}.tsv")
    print(f"\n== candidates on one side only, vs {base}, by hand class")
    for lab, arm in cells[1:]:
        for side, X, Y in ((f"{lab} only", R[lab], R[base]), (f"{base} only (vs {lab})", R[base], R[lab])):
            only = [k for k in X if k not in Y]
            cl = {"stopper": 0, "non-stopper": 0, "unjudged": 0, "unlabelled": 0}
            for k in only:
                if k not in T:
                    cl["unlabelled"] += 1
                elif k not in G["judged"]:
                    cl["unjudged"] += 1
                else:
                    cl["stopper" if T[k][0] in ("STM_MICHEL", "STM_ONLY") else "non-stopper"] += 1
            print(f"  {side}: {len(only)} {cl}")


if __name__ == "__main__":
    main()
