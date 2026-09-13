#!/usr/bin/env python3
"""doc pdhd/26 sec 2 -- STM and STM+Michel efficiency and purity, PDHD production against PDVD production.

    STM_SCAN_RECORD=<smx27 record> python3 d26_compare.py [--pdhd h26q2dprod] [--pdvd p96vprod] [--figdir ...]

Everything is scored with doc pdhd/25's own functions (d25_bragg_michel.items / census), so the populations and
truth are exactly doc 25's: PDHD = owner_review > owner_smx1 > agent on the record in STM_SCAN_RECORD, the 303-item
key, APA0 strict / majority / all; PDVD = the merged smx1a..smx9 record's flat verdict on items with a candidate.
MESSY / UNCLEAR are unscored on both.

THREE SELECTIONS, each scored against its own truth:
  STM          chain is_stm == 1                          vs hand stopper (STM_MICHEL / STM_ONLY)
  Michel       chain michel_found == 1, on hand stoppers  vs michel_kind attached/both  (PDHD graders' definition)
               chain michel_found == 1, on all judged     vs verdict STM_MICHEL          (PDVD census_score 14.2)
  STM+Michel   chain is_stm == 1 AND michel_found == 1    vs hand Michel item = hand stopper with michel_kind attached/both
               (owner stoppers with no michel_kind are dropped from this truth, the d21_michel_census rule)
  GOLDEN       STM+Michel with the Bragg-path accept (topology_cleared_bits == 0), doc 25 sec 2

PDVD's all-judged denominator adds the record's hand stoppers that have no candidate as misses (doc pdvd/89 sec 1),
printed beside the with-a-candidate figure; PDHD's key has a candidate for every item.
Errors are binomial sqrt(p(1-p)/n).
"""
import argparse, collections, glob, json, math, os, sys
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
sys.path.insert(0, IMG + "/pdhd/docs/scan/h25")
import d25_bragg_michel as BM


def fr(a, n):
    return (a / n if n else float("nan"), math.sqrt((a / n) * (1 - a / n) / n) if n else float("nan"), a, n)


def is_hm(x):
    return x[1] in BM.STOP and x[2] in BM.MICHEL_KINDS


def michel_truth_known(x):
    return not (x[1] in BM.STOP and x[3] == "owner_review" and x[2] is None)


def metrics(its):
    acc = lambda x: int(x[4]["is_stm"]) == 1
    mf = lambda x: int(x[4]["michel_found"]) == 1
    bragg = lambda x: acc(x) and int(x[4]["topology_cleared_bits"]) == 0
    S = [x for x in its if x[1] in BM.STOP]
    out = collections.OrderedDict()
    out["STM purity"] = fr(sum(1 for x in its if acc(x) and x[1] in BM.STOP), sum(1 for x in its if acc(x)))
    out["STM efficiency"] = fr(sum(1 for x in S if acc(x)), len(S))
    SK = [x for x in S if michel_truth_known(x)]
    out["Michel purity (on hand stoppers)"] = fr(sum(1 for x in SK if mf(x) and is_hm(x)), sum(1 for x in SK if mf(x)))
    out["Michel efficiency (on hand stoppers)"] = fr(sum(1 for x in SK if mf(x) and is_hm(x)), sum(1 for x in SK if is_hm(x)))
    out["Michel purity (all judged vs STM_MICHEL)"] = fr(sum(1 for x in its if mf(x) and x[1] == "STM_MICHEL"), sum(1 for x in its if mf(x)))
    out["Michel efficiency (all judged vs STM_MICHEL)"] = fr(sum(1 for x in its if mf(x) and x[1] == "STM_MICHEL"), sum(1 for x in its if x[1] == "STM_MICHEL"))
    J = [x for x in its if michel_truth_known(x)]
    sel = [x for x in J if acc(x) and mf(x)]
    HM = [x for x in J if is_hm(x)]
    out["STM+Michel purity"] = fr(sum(1 for x in sel if is_hm(x)), len(sel))
    out["STM+Michel efficiency"] = fr(sum(1 for x in HM if acc(x) and mf(x)), len(HM))
    gsel = [x for x in J if bragg(x) and mf(x)]
    out["GOLDEN purity (chain selection vs hand Michel)"] = fr(sum(1 for x in gsel if is_hm(x)), len(gsel))
    out["GOLDEN efficiency (over hand Michel)"] = fr(sum(1 for x in HM if bragg(x) and mf(x)), len(HM))
    out["Bragg-path accept | hand Michel"] = fr(sum(1 for x in HM if bragg(x)), len(HM))
    out["Michel found | Bragg-path accept, hand Michel"] = fr(sum(1 for x in HM if bragg(x) and mf(x)), sum(1 for x in HM if bragg(x)))
    out["_sel_composition"] = dict(collections.Counter("hand Michel" if is_hm(x) else x[1] for x in sel))
    out["_gsel_composition"] = dict(collections.Counter("hand Michel" if is_hm(x) else x[1] for x in gsel))
    return out


def pdvd_volumes(arm, its):
    """-> {key: 'x<0' | 'x>0'} from the sign of the median role-1 x (the drift volume the muon sits in)"""
    want = collections.defaultdict(set)
    for x in its:
        ev, c = x[0].rsplit("/", 1); want[ev].add(int(c))
    out = {}
    for ev, cs in want.items():
        p = uproot.open(f"{IMG}/pdvd/work/{ev}_{arm}/tracking-pr.root")["T_stm_michel_pts"].arrays(["cluster_id", "role", "x"], library="np")
        for c in cs:
            m = (p["cluster_id"] == c) & (p["role"] == 1)
            out[f"{ev}/{c}"] = "x<0" if m.any() and np.median(p["x"][m]) < 0 else "x>0"
    return out


def pr(label, M):
    print(f"\n=== {label}")
    for k, v in M.items():
        if k.startswith("_"):
            print(f"  {k[1:]:48s} {v}")
        else:
            print(f"  {k:48s} {v[2]:4d}/{v[3]:4d} = {v[0]:.3f} +- {v[1]:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdhd", default="h26q2dprod")
    ap.add_argument("--pdvd", default="p96vprod")
    ap.add_argument("--figdir", default=IMG + "/pdhd/docs/figs")
    a = ap.parse_args()
    print("PDHD record:", BM.HD_REC)
    print("PDVD record:", BM.VD_REC)
    G = collections.OrderedDict()
    for pop in ("strict", "majority", "all"):
        its, miss = BM.items("pdhd", a.pdhd, pop)
        G[f"PDHD APA0 {pop}" if pop != "all" else "PDHD all APAs"] = its
        c = BM.census(its)
        print(f"PDHD {a.pdhd} {pop:8s}: judged {len(its)}, no candidate {miss}; is_stm {c[0]}, michel(stoppers) {c[1]}, michel(all judged) {c[2]}")
    vits, vmiss = BM.items("pdvd", a.pdvd, "all")
    G["PDVD"] = vits
    c = BM.census(vits)
    print(f"PDVD {a.pdvd}: judged with a candidate {len(vits)}; is_stm {c[0]}, michel(stoppers) {c[1]}, michel(all judged) {c[2]}")
    # PDVD all-judged denominator: record stoppers with no candidate are misses
    A = BM.read_arm("pdvd", a.pdvd)
    nocand = [r for r in json.load(open(BM.VD_REC)) if r["key"] not in A]
    ncs = [r for r in nocand if BM.base(r["verdict"]) in BM.STOP]
    ncm = [r for r in nocand if BM.base(r["verdict"]) == "STM_MICHEL"]
    TP, FP, FN, TN = c[0]
    print(f"PDVD record items with no candidate {len(nocand)} (hand stoppers {len(ncs)}, STM_MICHEL {len(ncm)}, "
          f"verdicts {dict(collections.Counter(BM.base(r['verdict']) for r in nocand))})")
    print(f"PDVD all judged: is_stm {TP}/{FP}/{FN + len(ncs)} eff {TP/(TP+FN+len(ncs)):.3f}; "
          f"michel(all judged) {c[2][0]}/{c[2][1]}/{c[2][2] + len(ncm)} eff {c[2][0]/(c[2][0]+c[2][2]+len(ncm)):.3f}")
    M = collections.OrderedDict((k, metrics(v)) for k, v in G.items())
    for k, v in M.items():
        pr(k, v)

    # ---- split by hardware unit -------------------------------------------------------------
    print("\n=== BY HARDWARE UNIT (PDHD: the candidate's majority APA over all four; PDVD: drift volume of the median role-1 x)")
    hall = G["PDHD all APAs"]
    U = collections.OrderedDict()
    for apa in range(4):
        U[f"PDHD APA{apa}"] = [x for x in hall if x[4]["apa_major"] == apa]
    vol = pdvd_volumes(a.pdvd, vits)
    for v in ("x<0", "x>0"):
        U[f"PDVD {v}"] = [x for x in vits if vol[x[0]] == v]
    MU = collections.OrderedDict((k, metrics(v)) for k, v in U.items())
    keys = ["STM purity", "STM efficiency", "STM+Michel purity", "STM+Michel efficiency", "GOLDEN efficiency (over hand Michel)"]
    print("  %-12s %5s  " % ("unit", "n") + "  ".join("%-26s" % k[:26] for k in keys))
    for u, m in MU.items():
        print("  %-12s %5d  " % (u, len(U[u])) + "  ".join("%3d/%3d %.3f+-%.3f      " % (m[k][2], m[k][3], m[k][0], m[k][1]) for k in keys))

    # ---- figures ------------------------------------------------------------------------------
    os.makedirs(a.figdir, exist_ok=True)
    show = [("STM purity", "STM\npurity"), ("STM efficiency", "STM\nefficiency"),
            ("Michel purity (on hand stoppers)", "Michel purity\n(hand stoppers)"), ("Michel efficiency (on hand stoppers)", "Michel eff.\n(hand stoppers)"),
            ("STM+Michel purity", "STM+Michel\npurity"), ("STM+Michel efficiency", "STM+Michel\nefficiency"),
            ("Bragg-path accept | hand Michel", "Bragg-path\naccept | Michel"), ("Michel found | Bragg-path accept, hand Michel", "Michel found\n| Bragg-path"),
            ("GOLDEN efficiency (over hand Michel)", "GOLDEN\n(over Michel)")]
    groups = list(M)
    col = {"PDHD APA0 strict": "C0", "PDHD APA0 majority": "#6baed6", "PDHD all APAs": "#c6dbef", "PDVD": "C3"}
    fig, ax = plt.subplots(figsize=(13, 5.2))
    w = 0.8 / len(groups)
    for gi, g in enumerate(groups):
        xs = np.arange(len(show)) + (gi - (len(groups) - 1) / 2) * w
        vals = [M[g][k][0] for k, _ in show]; errs = [M[g][k][1] for k, _ in show]
        ax.bar(xs, vals, w, yerr=errs, color=col[g], edgecolor="k", lw=0.4, capsize=2,
               label="%s (%d judged)" % (g + (" " + a.pdhd if g.startswith("PDHD") else " " + a.pdvd), len(G[g])))
        if g in ("PDHD APA0 strict", "PDVD"):
            for x0, v, e in zip(xs, vals, errs):
                ax.text(x0, v + e + 0.012, "%.2f" % v, ha="center", va="bottom", fontsize=6.5, rotation=90)
    ax.set_xticks(np.arange(len(show))); ax.set_xticklabels([s for _, s in show], fontsize=8)
    ax.set_ylim(0.0, 1.18); ax.set_ylabel("fraction (binomial error)"); ax.grid(axis="y", alpha=0.3)
    ax.set_title("STM and STM+Michel on the chain's candidate pool: PDHD production vs PDVD production (hand-scan truth)")
    ax.legend(fontsize=8, ncol=2, loc="lower left")
    fig.tight_layout(); f1 = os.path.join(a.figdir, "26_eff_purity.png"); fig.savefig(f1, dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.6))
    units = list(MU)
    k4 = [("STM purity", "STM purity"), ("STM efficiency", "STM efficiency"), ("STM+Michel purity", "STM+Michel purity"),
          ("STM+Michel efficiency", "STM+Michel efficiency")]
    w = 0.8 / len(k4)
    for ki, (k, lab) in enumerate(k4):
        xs = np.arange(len(units)) + (ki - (len(k4) - 1) / 2) * w
        ax.bar(xs, [MU[u][k][0] for u in units], w, yerr=[MU[u][k][1] for u in units], capsize=2, label=lab,
               color=["C0", "#9ecae1", "C2", "#a1d99b"][ki], edgecolor="k", lw=0.4)
    ax.set_xticks(np.arange(len(units)))
    ax.set_xticklabels(["%s\n(%d judged)%s" % (u, len(U[u]), "\nnot in headline" if u == "PDHD APA0" else "") for u in units], fontsize=8)
    ax.set_ylim(0, 1.15); ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8, ncol=4, loc="upper center")
    ax.set_title("By hardware unit: PDHD APA (majority of role-1 points), PDVD drift volume")
    fig.tight_layout(); f2 = os.path.join(a.figdir, "26_eff_purity_units.png"); fig.savefig(f2, dpi=130); plt.close(fig)
    print("\nfigures: %s %s" % (f1, f2))


if __name__ == "__main__":
    main()
