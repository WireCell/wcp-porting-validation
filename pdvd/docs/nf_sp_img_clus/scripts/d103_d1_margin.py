#!/usr/bin/env python3
"""doc pdvd/103 sec 11 -- how many hand labels decide the D1 / D2 reading (figs/103_pred_amend2.txt sec 5 and
amendment 4 sec 6: "if D1 depends on 2 or fewer items, serve a seeded 20-item sample of the TP-movers before stating D1").
Read-only.

Truth, population and cells exactly as d103_union_grade.py (imported).  Each population item falls in one of 8 classes
(hand truth, A0 tag, A1 tag); flipping one item's hand truth moves the four counts of both cells in a fixed way.  For
each metric the script searches exhaustively over flip counts per class for the fewest flips that carry A1 - A0 across
-0.020 from the side it is on, and prints the flips that do it.  A flip is a hypothetical relabel, not a claim that a
label is wrong.

    D103_PDVD_CELLS=d103v0,d103v1 D103_PDVD_RECORD=<carried> python3 d103_d1_margin.py --det pdvd --new-record <smx11> \
        --owner-record <own103v>
"""
import argparse, itertools, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d103_union_grade as U

LIMIT = -0.020
KMAX = 12


def pe(tp, fp, fn):
    return tp / max(1, tp + fp), tp / max(1, tp + fn)


def cell_counts(classes, n, lab):
    """counts of one cell from class sizes n[(t, g0, g1)]"""
    tp = fp = fn = 0
    for (t, g0, g1), m in n.items():
        g = g0 if lab == "A0" else g1
        tp += m * (t and g); fp += m * ((not t) and g); fn += m * (t and not g)
    return tp, fp, fn


def delta(n, which):
    a0 = pe(*cell_counts(None, n, "A0"))[which]
    a1 = pe(*cell_counts(None, n, "A1"))[which]
    return a1 - a0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True, choices=["pdhd", "pdvd"])
    ap.add_argument("--new-record", required=True)
    ap.add_argument("--owner-record", default=None)
    ap.add_argument("--stm-only-unset-negative", action="store_true")
    a = ap.parse_args()
    T, _ = U.load_truth(a.det, a.new_record, a.owner_record)
    R = {lab: U.cell_rows(a.det, arm) for lab, arm in U.CELLS[a.det]}
    judged = {k for k, (v, mk, s) in T.items() if v not in ("MESSY", "UNCLEAR")}
    if a.det == "pdhd":
        import csv
        X = f"{U.IMG}/pdhd/docs/scan"
        fixed = {"%s/%s" % (r["event"], r["cluster"]) for r in csv.DictReader(
            [l for l in open(f"{X}/smx18/pdhd_stm_michel_scan_key_p82bhoff.tsv") if not l.startswith("#")], delimiter="\t")}
        fixed &= judged
    else:
        import json
        rec_keys = {r["key"] for r in json.load(open(U.PDVD_RECORD))}
        fixed = {k for k in judged if k in rec_keys and k in R["A0"]}
    anycand = set().union(*[set(v) for v in R.values()])
    pop = sorted(fixed | (anycand & judged))
    stm_truth = {k: T[k][0] in ("STM_MICHEL", "STM_ONLY") for k in pop}
    if a.det == "pdhd":
        mpop = [k for k in pop if stm_truth[k] and not (T[k][1] is None and T[k][2] == "owner_review"
                                                        and not (a.stm_only_unset_negative and T[k][0] == "STM_ONLY"))]
        m_truth = {k: T[k][1] in ("attached", "both") for k in mpop}
    else:
        mpop = list(pop)
        m_truth = {k: T[k][0] == "STM_MICHEL" for k in pop}

    print(f"# doc pdvd/103 D1/D2 label margin ({a.det}; cells {U.CELLS[a.det][0][1]} / {U.CELLS[a.det][-1][1]}; "
          f"owner record {os.path.basename(a.owner_record) if a.owner_record else '-'}); limit A1 - A0 >= {LIMIT}")
    for title, P, TR, idx in (("is_stm", pop, stm_truth, 0), ("michel_found", mpop, m_truth, 1)):
        n = {}
        for k in P:
            c = (bool(TR[k]), bool(R["A0"].get(k, (0, 0))[idx]), bool(R["A1"].get(k, (0, 0))[idx]))
            n[c] = n.get(c, 0) + 1
        for which, name in ((0, "purity"), (1, "efficiency")):
            d0 = delta(n, which)
            passing = d0 >= LIMIT
            best = None
            keys = sorted(n)
            for k in range(1, KMAX + 1):
                for combo in itertools.combinations_with_replacement(keys, k):
                    use = {c: combo.count(c) for c in set(combo)}
                    if any(use[c] > n[c] for c in use):
                        continue
                    m = dict(n)
                    for (t, g0, g1), u in use.items():
                        m[(t, g0, g1)] -= u
                        m[(not t, g0, g1)] = m.get((not t, g0, g1), 0) + u
                    d = delta(m, which)
                    if (d < LIMIT) if passing else (d >= LIMIT):
                        best = (k, use, d)
                        break
                if best:
                    break
            lab = lambda c: f"{'pos' if c[0] else 'neg'}->{'neg' if c[0] else 'pos'} (A0 {int(c[1])}, A1 {int(c[2])})"
            if best:
                k, use, d = best
                how = ", ".join(f"{u} x {lab(c)}" for c, u in sorted(use.items()))
                print(f"  {title:12s} {name:10s} A1 - A0 {d0:+.3f} ({'pass' if passing else 'fail'}); fewest relabels that "
                      f"{'fail' if passing else 'pass'} it: {k} -> {d:+.3f} [{how}]")
            else:
                print(f"  {title:12s} {name:10s} A1 - A0 {d0:+.3f} ({'pass' if passing else 'fail'}); more than {KMAX} relabels needed")


if __name__ == "__main__":
    main()
