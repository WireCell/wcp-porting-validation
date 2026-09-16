#!/usr/bin/env python3
"""doc pdvd/113 -- STM / Michel grade of retile_mode arms against their production base, on the folded truth records.

Reuses d103_union_grade.py (imported, untouched) for the truth rows, the population and the counts.  Truth, in
precedence order -- the records the latest folded grades of docs 103 (PDVD, round 5) and 104/108 (PDHD) were read on:
  pdhd: own103h2 > own103h > smx27 > smx28, with --stm-only-unset-negative (amendment 5);
  pdvd: own103v2 > own103v > p99rwon_carried_corrected > smx11.
Population: d103_union_grade.population() over the given cells (fixed items + every judged candidate of any cell).

Unlabelled candidates (a candidate of some cell with no label anywhere) are outside the population.  Their possible
effect is bounded by re-grading twice with every unlabelled candidate added as a non-stopper (NEG) and as a stopper
with an attached Michel (POS).  A metric PASSES only if A_x - A0 >= -0.020 in both scenarios, FAILS if below in both,
and is UNDECIDED otherwise (figs/113_pred.txt).

--self-test reproduces the two folded grades before any doc-113 number is read:
  pdhd cells A0 d101hnew, K d101hkf, S d102hocs, A1 d102hcs  -> figs/104_own103h2_fold.txt sec 4;
  pdvd cells A0 d103v0, A1 d103v1                             -> figs/103_own103v2_pdvd.txt sec 4.

Usage:
  d113_grade.py --self-test
  d113_grade.py --det pdhd --cells A0=d113hbase,L3=d113hnone,L2=d113hfoot,L1=d113hnopaint [--pairwise]
"""
import argparse, json, os, sys

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
SCAN = {"pdhd": f"{IMG}/pdhd/docs/scan", "pdvd": f"{IMG}/pdvd/docs/scan"}
TRUTH = {
    "pdhd": dict(record=f"{SCAN['pdhd']}/pdhd_stm_michel_smx27_verdicts.json",
                 new=f"{SCAN['pdhd']}/pdhd_stm_michel_smx28_verdicts.json",
                 owner=f"{SCAN['pdhd']}/pdhd_stm_michel_own103h_verdicts.json",
                 fold=f"{SCAN['pdhd']}/pdhd_stm_michel_own103h2_verdicts.json", fold_tag="own103h2"),
    "pdvd": dict(record=f"{SCAN['pdvd']}/pdvd_stm_michel_p99rwon_carried_corrected_verdicts.json",
                 new=f"{SCAN['pdvd']}/pdvd_stm_michel_smx11_verdicts.json",
                 owner=f"{SCAN['pdvd']}/pdvd_stm_michel_own103v_verdicts.json",
                 fold=f"{SCAN['pdvd']}/pdvd_stm_michel_own103v2_verdicts.json", fold_tag="own103v2"),
}
# d103_union_grade reads its record paths from the environment at import time
os.environ["D103_PDHD_RECORD"] = TRUTH["pdhd"]["record"]
os.environ["D103_PDVD_RECORD"] = TRUTH["pdvd"]["record"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d103_union_grade as U  # noqa: E402

LIMIT = -0.020
SELF_TEST = {
    "pdhd": (["A0=d101hnew", "K=d101hkf", "S=d102hocs", "A1=d102hcs"],
             {("is_stm", "A0"): (119, 3, 71), ("is_stm", "A1"): (117, 5, 73),
              ("michel_found", "A0"): (64, 6, 42), ("michel_found", "A1"): (73, 10, 33)}),
    "pdvd": (["A0=d103v0", "A1=d103v1"],
             {("is_stm", "A0"): (239, 5, 152), ("is_stm", "A1"): (266, 5, 125),
              ("michel_found", "A0"): (157, 13, 82), ("michel_found", "A1"): (171, 11, 68)}),
}


def truth(det):
    t = TRUTH[det]
    T, counts = U.load_truth(det, t["new"], t["owner"])
    n = 0
    for r in json.load(open(t["fold"])):
        T[r["key"]] = U.row_truth(r, t["fold_tag"])
        n += 1
    return T, [(os.path.basename(t["fold"]), n)] + counts


def grade(det, T, R):
    G = U.population(det, T, R, False, stm_only_unset_negative=(det == "pdhd"))
    out = {}
    for title, P, TR, idx in (("is_stm", G["pop"], G["stm_truth"], 0), ("michel_found", G["mpop"], G["m_truth"], 1)):
        for lab in R:
            chain = {k: v[idx] for k, v in R[lab].items()}
            c = U.counts(P, TR, chain)
            out[(title, lab)] = (c, U.pe(c), P, TR, chain)
    return G, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", choices=["pdhd", "pdvd"])
    ap.add_argument("--cells", help="LAB=arm,... ; the first cell is the base")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        bad = 0
        for det, (cells, want) in SELF_TEST.items():
            T, _ = truth(det)
            R = {c.split("=")[0]: U.cell_rows(det, c.split("=")[1]) for c in cells}
            _, out = grade(det, T, R)
            for key, (tp, fp, fn) in want.items():
                c = out[key][0]
                ok = (c["tp"], c["fp"], c["fn"]) == (tp, fp, fn)
                bad += not ok
                print(f"self-test {det} {key[0]:12s} {key[1]}: TP {c['tp']} FP {c['fp']} FN {c['fn']} "
                      f"(expected {tp} {fp} {fn}) {'ok' if ok else 'MISMATCH'}")
        print("SELF-TEST", "PASS" if not bad else f"FAIL ({bad})")
        sys.exit(1 if bad else 0)

    cells = [c.split("=") for c in a.cells.split(",")]
    base = cells[0][0]
    T, src = truth(a.det)
    R = {lab: U.cell_rows(a.det, arm) for lab, arm in cells}
    G, out = grade(a.det, T, R)
    print(f"# doc pdvd/113 grade ({a.det}); truth sources in precedence order (items taken): {src}")
    print(f"# cells: {', '.join(f'{l}={x}' for l, x in cells)}; base {base}; bar A_x - {base} >= {LIMIT}")
    print(f"population {len(G['pop'])} (fixed {len(G['fixed'])}); Michel population {len(G['mpop'])} "
          f"(excluded {G['mexcl']}); unlabelled candidates of any cell {len(G['unlabelled'])}")
    for lab in R:
        u = [k for k in G["unlabelled"] if k in R[lab]]
        print(f"  unlabelled candidates of {lab}: {len(u)}; is_stm 1 {sum(R[lab][k][0] for k in u)}, "
              f"michel_found 1 {sum(R[lab][k][1] for k in u)}")

    # the two bounding scenarios for the unlabelled candidates
    scen = {}
    for name, row in (("NEG", ("THRU", None, "hypo")), ("POS", ("STM_MICHEL", "attached", "hypo"))):
        T2 = dict(T)
        for k in G["unlabelled"]:
            T2[k] = row
        scen[name] = grade(a.det, T2, R)[1]

    for title in ("is_stm", "michel_found"):
        print(f"\n== {title}")
        for lab, arm in cells:
            c, (p, e), P, TR, chain = out[(title, lab)]
            (plo, phi), (elo, ehi) = U.boot(P, TR, chain)
            print(f"  {lab:3s} {arm:13s} TP {c['tp']:3d} FP {c['fp']:3d} FN {c['fn']:3d} TN {c['tn']:4d} | purity {p:.3f} "
                  f"[{plo:.3f}, {phi:.3f}] efficiency {e:.3f} [{elo:.3f}, {ehi:.3f}]")

    print(f"\n== reading per cell vs {base} (labelled; NEG / POS bounds for the unlabelled)")
    for lab, arm in cells[1:]:
        verdicts = []
        for title in ("is_stm", "michel_found"):
            for i, name in ((0, "purity"), (1, "efficiency")):
                d = out[(title, lab)][1][i] - out[(title, base)][1][i]
                dn = scen["NEG"][(title, lab)][1][i] - scen["NEG"][(title, base)][1][i]
                dp = scen["POS"][(title, lab)][1][i] - scen["POS"][(title, base)][1][i]
                v = "PASS" if min(dn, dp) >= LIMIT else "FAIL" if max(dn, dp) < LIMIT else "UNDECIDED"
                verdicts.append(v)
                print(f"  {lab:3s} {title:12s} {name:10s}: {d:+.3f}  (NEG {dn:+.3f}, POS {dp:+.3f})  {v}")
        overall = "FAIL" if "FAIL" in verdicts else "UNDECIDED" if "UNDECIDED" in verdicts else "PASS"
        print(f"  {lab:3s} T -> {overall}")

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
