#!/usr/bin/env python3
"""doc pdvd/103 sec 5 -- STM / Michel grade of the four cells on the union record, by figs/103_pred.txt.

Fork by duplication of d101_stm_grade_pdhd.py / d101_stm_grade_pdvd.py (untouched).  Differences:
  * truth comes from a precedence list of records (pred.txt LABELS):
      pdhd: smx22 (owner_review > owner_smx1 > agent), then --new-record (the blind smx28; owner_review > agent)
      pdvd: the merged smx1a..smx9 record, then own100 / own100m / own100x (owner), then --new-record (smx10)
  * population: the graders' fixed population (pdhd: smx18 p82bhoff key items; pdvd: judged record items that are
    candidates in d101vnew), plus -- unless --fixed-only -- every judged item that is a candidate in ANY cell;
  * --fixed-only with no --new-record must reproduce V1 (the doc 101/102 numbers);
  * bootstrap 68 % intervals over the population (1000, seed 103);
  * the gained / lost side of A1 vs A0 by hand class, and the D1 / D2 reading of pred.txt.

Chain verdicts: T_stm_michel is_stm / michel_found per (event, cluster_id); a population item that is not a row in a
cell is chain-negative there.
Michel truth: pdhd -- hand stoppers only, positive when michel_kind in (attached, both); an owner_review stopper with no
michel_kind is excluded (d21_michel_census precedence).  pdvd -- verdict STM_MICHEL positive, every other judged item
negative (census_lib.is_michel).

Usage: d103_union_grade.py --det pdhd [--new-record F] [--fixed-only]
"""
import argparse, csv, glob, json, os
import numpy as np
import uproot

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
# round-1 figures: smx22 (the default).  figs/103_pred_amend3.txt: the PDHD truth record is smx27 -> set this env
PDHD_RECORD = os.environ.get("D103_PDHD_RECORD", f"{IMG}/pdhd/docs/scan/pdhd_stm_michel_smx22_verdicts.json")
CELLS = {"pdhd": [("A0", "d101hnew"), ("K", "d101hkf"), ("S", "d102hocs"), ("A1", "d102hcs")],
         "pdvd": [("A0", "d101vnew"), ("K", "d101vkf"), ("S", "d102vocs"), ("A1", "d102vcsall")]}
# sec 10 / figs/103_pred_amend4.txt: the PDVD production lineage (D103_PDVD_CELLS="d103v0,d103v1") graded on the
# carried + owner-corrected record (D103_PDVD_RECORD).  Both unset => the round-1 cells and record (committed figures).
PDVD_RECORD = os.environ.get("D103_PDVD_RECORD",
                             f"{IMG}/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_smx5_smx6_smx7_smx8_smx9_verdicts.json")
if os.environ.get("D103_PDVD_CELLS"):
    CELLS["pdvd"] = list(zip(("A0", "A1"), os.environ["D103_PDVD_CELLS"].split(",")))


def strip(v):
    return v[5:] if v and v.startswith("FRAG_") else v


def row_truth(r, precedence):
    """(verdict, michel_kind, source) with owner fields first.  The PDVD merged record already folds each owner scan
    into the flat `verdict` (its owner_review, where present, is a free-text note), so census_lib's flat reading is
    kept there, exactly as d101_stm_grade_pdvd.py."""
    if precedence == "record":
        return strip(r["verdict"]), r.get("michel_kind"), "record"
    if isinstance(r.get("owner_review"), dict):
        return strip(r["owner_review"]["verdict"]), r["owner_review"].get("michel_kind"), "owner_review"
    if r.get("owner_smx1"):
        o = r["owner_smx1"]
        return strip(o.get("choice") or o.get("label")), o.get("michel_kind"), "owner"
    return strip(r["verdict"]), r.get("michel_kind"), precedence


def load_truth(det, new_record, owner_record=None):
    T = {}
    def add(path, src):
        n = 0
        for r in json.load(open(path)):
            if r["key"] not in T:
                T[r["key"]] = row_truth(r, src)
                n += 1
        return n
    if det == "pdhd":
        # figs/103_pred_amend3.txt: D103_PDHD_RECORD=<smx27> is the PDHD truth record (smx22 + 20 owner rulings)
        srcs = [(PDHD_RECORD, os.path.basename(PDHD_RECORD).split("_")[3])]
    else:
        S = f"{IMG}/pdvd/docs/scan"
        # figs/103_pred_amend1.txt: own100 / own100m / own100x are keyed on other pctree lineages -> not truth here
        srcs = [(PDVD_RECORD, "record")]
    if owner_record:                       # figs/103_pred_amend2.txt sec 3: own103h above every other source
        srcs.insert(0, (owner_record, "own103h"))
    if new_record:
        srcs.append((new_record, "new_agent"))
    counts = [(os.path.basename(p), add(p, s)) for p, s in srcs]
    return T, counts


def cell_rows(det, arm):
    out = {}
    for d in glob.glob(f"{IMG}/{det}/work/*_{arm}"):
        ev = os.path.basename(d)[:-len(arm) - 1]
        try:
            t = uproot.open(f"{d}/tracking-pr.root")["T_stm_michel"].arrays(["cluster_id", "is_stm", "michel_found"], library="np")
        except Exception:
            continue
        for c, s, m in zip(t["cluster_id"], t["is_stm"], t["michel_found"]):
            out[f"{ev}/{int(c)}"] = (int(s), int(m))
    return out


def counts(pop, truth, chain):
    c = dict(tp=0, fp=0, fn=0, tn=0)
    for k in pop:
        t, g = truth[k], chain.get(k, 0)
        c["tp" if t and g else "fn" if t else "fp" if g else "tn"] += 1
    return c


def pe(c):
    return c["tp"] / max(1, c["tp"] + c["fp"]), c["tp"] / max(1, c["tp"] + c["fn"])


def boot(pop, truth, chain, n=1000, seed=103):
    rng = np.random.default_rng(seed)
    t = np.array([truth[k] for k in pop], bool)
    g = np.array([bool(chain.get(k, 0)) for k in pop], bool)
    P, E = [], []
    for _ in range(n):
        i = rng.integers(0, len(pop), len(pop))
        tp = np.sum(t[i] & g[i]); fp = np.sum(~t[i] & g[i]); fn = np.sum(t[i] & ~g[i])
        P.append(tp / max(1, tp + fp)); E.append(tp / max(1, tp + fn))
    return np.percentile(P, [16, 84]), np.percentile(E, [16, 84])


def population(det, T, R, fixed_only=False, stm_only_unset_negative=False):
    """The grade's population and truth (sec 5, amendments 2-5).  R holds the rows of EVERY cell of the lineage.
    Shared by d103_d1_margin.py and d103_michel_floor_sizing.py so the definition lives in one place."""
    judged = {k for k, (v, mk, s) in T.items() if v not in ("MESSY", "UNCLEAR")}
    if det == "pdhd":
        X = f"{IMG}/pdhd/docs/scan"
        fixed = {"%s/%s" % (r["event"], r["cluster"]) for r in csv.DictReader(
            [l for l in open(f"{X}/smx18/pdhd_stm_michel_scan_key_p82bhoff.tsv") if not l.startswith("#")], delimiter="\t")}
        fixed &= judged
    else:
        rec_keys = {r["key"] for r in json.load(open(PDVD_RECORD))}
        fixed = {k for k in judged if k in rec_keys and k in R["A0"]}
    anycand = set().union(*[set(v) for v in R.values()])
    pop = sorted(fixed if fixed_only else fixed | (anycand & judged))
    unlabelled = sorted(k for k in anycand if k not in T)
    stm_truth = {k: T[k][0] in ("STM_MICHEL", "STM_ONLY") for k in pop}
    if det == "pdhd":
        mpop = [k for k in pop if stm_truth[k] and not (T[k][1] is None and T[k][2] == "owner_review"
                                                        and not (stm_only_unset_negative and T[k][0] == "STM_ONLY"))]
        m_truth = {k: T[k][1] in ("attached", "both") for k in mpop}
        mexcl = sum(1 for k in pop if stm_truth[k]) - len(mpop)
    else:
        mpop = list(pop)
        m_truth = {k: T[k][0] == "STM_MICHEL" for k in pop}
        mexcl = 0
    return dict(judged=judged, fixed=fixed, anycand=anycand, pop=pop, unlabelled=unlabelled,
                stm_truth=stm_truth, mpop=mpop, m_truth=m_truth, mexcl=mexcl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True, choices=["pdhd", "pdvd"])
    ap.add_argument("--new-record", default=None)
    ap.add_argument("--fixed-only", action="store_true")
    ap.add_argument("--owner-record", default=None)   # amendment 2: owner adjudication record, first in precedence
    ap.add_argument("--stm-only-unset-negative", action="store_true")   # sensitivity: an owner STM_ONLY with no kind = no Michel
    a = ap.parse_args()
    T, src_counts = load_truth(a.det, a.new_record, a.owner_record)
    R = {lab: cell_rows(a.det, arm) for lab, arm in CELLS[a.det]}
    G = population(a.det, T, R, a.fixed_only, a.stm_only_unset_negative)
    judged, fixed, anycand, pop, unlabelled = G["judged"], G["fixed"], G["anycand"], G["pop"], G["unlabelled"]
    stm_truth, mpop, m_truth, mexcl = G["stm_truth"], G["mpop"], G["m_truth"], G["mexcl"]
    print(f"# doc pdvd/103 union grade ({a.det}); truth sources (items taken from each, in precedence order): {src_counts}")
    print(f"population {'fixed only' if a.fixed_only else 'fixed + judged candidates of any cell'}: {len(pop)} "
          f"(fixed {len(fixed)}); candidates of any cell with no label anywhere: {len(unlabelled)}; "
          f"unjudged (MESSY/UNCLEAR) candidates: {len([k for k in anycand if k in T and k not in judged])}")
    src = {}
    for k in pop:
        src[T[k][2]] = src.get(T[k][2], 0) + 1
    print(f"population by label source: {src}")

    res = {}
    for title, P, TR, idx in (("is_stm", pop, stm_truth, 0), ("michel_found", mpop, m_truth, 1)):
        print(f"\n== {title} (population {len(P)}{', excluded ' + str(mexcl) if title == 'michel_found' and mexcl else ''})")
        for lab, arm in CELLS[a.det]:
            chain = {k: v[idx] for k, v in R[lab].items()}
            c = counts(P, TR, chain)
            p, e = pe(c)
            (plo, phi), (elo, ehi) = boot(P, TR, chain)
            res[(title, lab)] = (p, e)
            print(f"  {lab:2s} {arm:11s} TP {c['tp']:3d} FP {c['fp']:3d} FN {c['fn']:3d} TN {c['tn']:3d} | purity {p:.3f} "
                  f"[{plo:.3f}, {phi:.3f}] efficiency {e:.3f} [{elo:.3f}, {ehi:.3f}]")

    print("\n== A1 vs A0: candidates on one side only, by hand class (judged; unlabelled shown separately)")
    for side, X, Y in (("A1 only", R["A1"], R["A0"]), ("A0 only", R["A0"], R["A1"])):
        only = [k for k in X if k not in Y]
        cl = {"stopper": 0, "non-stopper": 0, "unjudged": 0, "unlabelled": 0}
        tagged = {"stopper": 0, "non-stopper": 0}
        for k in only:
            if k not in T:
                cl["unlabelled"] += 1
            elif k not in judged:
                cl["unjudged"] += 1
            else:
                h = "stopper" if T[k][0] in ("STM_MICHEL", "STM_ONLY") else "non-stopper"
                cl[h] += 1
                tagged[h] += X[k][0]
        print(f"  {side}: {len(only)} candidates {cl}; of them is_stm 1 on that side: {tagged}")

    print("\n== decision (figs/103_pred.txt): every metric of A1 >= A0 - 0.02 -> D1, else D2")
    worst = []
    for title in ("is_stm", "michel_found"):
        for i, name in ((0, "purity"), (1, "efficiency")):
            d = res[(title, "A1")][i] - res[(title, "A0")][i]
            worst.append(d)
            print(f"  {title} {name}: A1 - A0 = {d:+.3f}")
    print(f"  -> {'D1 (record conditioning)' if min(worst) >= -0.02 else 'D2 (real cost)'}"
          f"{'  [fixed-only / old records: NOT the pre-registered reading]' if a.fixed_only or not a.new_record else ''}")

    if a.owner_record:                     # figs/103_pred_amend2.txt sec 4: (b) untouched, (c) common vs arm-only
        OWN = {r["key"] for r in json.load(open(a.owner_record))}
        print(f"\n== amendment 2 splits (owner record {os.path.basename(a.owner_record)}, {len(OWN)} keys); (a) is above")
        for title, P, TR, idx in (("is_stm", pop, stm_truth, 0), ("michel_found", mpop, m_truth, 1)):
            UT = [k for k in P if k not in OWN]
            print(f"  (b) {title}, untouched population {len(UT)} of {len(P)}")
            d = {}
            for lab in ("A0", "A1"):
                chain = {k: v[idx] for k, v in R[lab].items()}
                c = counts(UT, TR, chain)
                p, e = pe(c)
                (plo, phi), (elo, ehi) = boot(UT, TR, chain)
                d[lab] = (p, e)
                print(f"      {lab} TP {c['tp']:3d} FP {c['fp']:3d} FN {c['fn']:3d} TN {c['tn']:3d} | purity {p:.3f} "
                      f"[{plo:.3f}, {phi:.3f}] efficiency {e:.3f} [{elo:.3f}, {ehi:.3f}]")
            print(f"      A1 - A0: purity {d['A1'][0] - d['A0'][0]:+.3f}, efficiency {d['A1'][1] - d['A0'][1]:+.3f}")
            g0 = {k for k in P if R["A0"].get(k, (0, 0))[idx]}
            g1 = {k for k in P if R["A1"].get(k, (0, 0))[idx]}
            for name, S in (("common tags (A0 and A1)", g0 & g1), ("A0-only tags", g0 - g1), ("A1-only tags", g1 - g0)):
                tp = sum(1 for k in S if TR[k])
                print(f"  (c) {title} {name}: n {len(S)}, TP {tp}, FP {len(S) - tp} (owner-judged {len(S & OWN)})")


if __name__ == "__main__":
    main()
