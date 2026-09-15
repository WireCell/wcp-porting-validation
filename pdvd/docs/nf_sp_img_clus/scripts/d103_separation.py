#!/usr/bin/env python3
"""doc pdvd/103 sec 2 -- does the better trajectory separate stoppers from non-stoppers better?  Read-only.

On hand-labelled clusters present in both arms:
  TaggerCheckSTM   best (lowest) D and best (ks1-ks2) over the lowest recorded pass's eval_stm_core calls
                   (tracking-stm.root T_stm_eval / T_stm_pass; D as in d103_eval_attrib.py)
  CheckSTM_Michel  ks_mu - ks_flat and contrast / contrast_expected (T_stm_michel; candidates in both arms)
AUC = P(a hand stopper is more muon-like than a hand non-stopper), ties counted half.
Also the selection table: of the labelled stoppers / non-stoppers that production's TaggerCheckSTM accepted (lowest
pass status 0), how many the arm still accepts, and how many production-rejected ones the arm accepts.

Truth
  pdhd  pdhd/docs/scan/pdhd_stm_michel_smx22_verdicts.json, precedence owner_review > owner_smx1 > agent (d21_grade.py),
        verdict-blind scan (doc pdhd/18)
  pdvd  $STM_SCAN_RECORD (the merged smx1a..smx9 record) through census_lib (judged / is_stopper), scanned with the
        chain's verdict visible (doc pdvd/55 sec 16.3)
  --extra-record FILE (repeatable) adds records (key -> verdict, same precedence) for items the main record lacks, and
  --only-extra restricts the AUC to those items (the blind round-1 labels).

Usage: d103_separation.py --det pdhd --base d101hnew --arm d102hcs [--extra-record F ...] [--only-extra]
"""
import argparse, collections, glob, json, os, sys
import numpy as np
import uproot

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
sys.path.insert(0, IMG + "/pdhd/stm_michel_scan")


def strip(v):
    return v[5:] if v and v.startswith("FRAG_") else v


def hand(r):
    if r.get("owner_review"):
        return strip(r["owner_review"]["verdict"])
    if r.get("owner_smx1"):
        o = r["owner_smx1"]
        return strip(o.get("choice") or o.get("label"))
    return strip(r["verdict"])


def truth(det, extra):
    T = {}
    if det == "pdhd":
        for r in json.load(open(f"{IMG}/pdhd/docs/scan/pdhd_stm_michel_smx22_verdicts.json")):
            T[r["key"]] = hand(r)
    else:
        import census_lib as C
        for k, r in C.load_record().items():
            T[k] = strip(r["verdict"])
    X = set()
    for f in extra:
        for r in json.load(open(f)):
            if r["key"] not in T:
                T[r["key"]] = hand(r)
                X.add(r["key"])
    T = {k: v in ("STM_MICHEL", "STM_ONLY") for k, v in T.items() if v not in ("MESSY", "UNCLEAR")}
    return T, X


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    x = np.concatenate([pos, neg])
    order = x.argsort(kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    for val in np.unique(x):          # average ranks over ties
        m = x == val
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def dval(r):
    return r["ks1"] - r["ks2"] + (abs(r["ratio1"] - 1) - abs(r["ratio2"] - 1)) / 1.5 * 0.3


def tagger(det, arm):
    out = {}
    for d in glob.glob(f"{IMG}/{det}/work/*_{arm}"):
        ev = os.path.basename(d)[:-len(arm) - 1]
        try:
            f = uproot.open(f"{d}/tracking-stm.root")
            e = f["T_stm_eval"].arrays(library="np")
            p = f["T_stm_pass"].arrays(library="np")
        except Exception:
            continue
        low = {}
        for c, ps, st in zip(p["cluster_id"], p["pass"], p["status"]):
            low[int(c)] = min(low.get(int(c), (99, 0)), (int(ps), int(st)))
        best = {}
        for i in range(len(e["cluster_id"])):
            c = int(e["cluster_id"][i])
            if c not in low or int(e["pass"][i]) != low[c][0]:
                continue
            r = {k: float(e[k][i]) for k in ("ks1", "ks2", "ratio1", "ratio2")}
            b = best.get(c, (np.inf, np.inf))
            best[c] = (min(b[0], dval(r)), min(b[1], r["ks1"] - r["ks2"]))
        for c, (bd, bk) in best.items():
            out[f"{ev}/{c}"] = (bd, bk, low[c][1])
    return out


def michel(det, arm):
    out = {}
    for d in glob.glob(f"{IMG}/{det}/work/*_{arm}"):
        ev = os.path.basename(d)[:-len(arm) - 1]
        try:
            t = uproot.open(f"{d}/tracking-pr.root")["T_stm_michel"].arrays(
                ["cluster_id", "ks_mu", "ks_flat", "contrast", "contrast_expected"], library="np")
        except Exception:
            continue
        for i in range(len(t["cluster_id"])):
            if t["ks_mu"][i] > 0 or t["ks_flat"][i] > 0:
                out[f"{ev}/{int(t['cluster_id'][i])}"] = (float(t["ks_mu"][i] - t["ks_flat"][i]),
                                                         float(t["contrast"][i] / max(1e-9, t["contrast_expected"][i])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True, choices=["pdhd", "pdvd"])
    ap.add_argument("--base", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--extra-record", action="append", default=[])
    ap.add_argument("--only-extra", action="store_true")
    a = ap.parse_args()
    T, X = truth(a.det, a.extra_record)
    if a.only_extra:
        T = {k: v for k, v in T.items() if k in X}
    G0, G1 = tagger(a.det, a.base), tagger(a.det, a.arm)
    K = [k for k in T if k in G0 and k in G1]
    pos, neg = [k for k in K if T[k]], [k for k in K if not T[k]]
    print(f"[{a.det}] {a.base} -> {a.arm}; judged labelled clusters with a TaggerCheckSTM eval in both arms: {len(K)} "
          f"({len(pos)} stoppers / {len(neg)} non-stoppers){' [extra records only]' if a.only_extra else ''}")
    for nm, G in ((a.base, G0), (a.arm, G1)):
        print(f"  {nm:11s} TaggerCheckSTM best D AUC {auc([-G[k][0] for k in pos], [-G[k][0] for k in neg]):.3f}; "
              f"best (ks1-ks2) AUC {auc([-G[k][1] for k in pos], [-G[k][1] for k in neg]):.3f}; "
              f"accepted (status 0): stoppers {np.mean([G[k][2] == 0 for k in pos]):.3f}, non-stoppers {np.mean([G[k][2] == 0 for k in neg]):.3f}")
    M0, M1 = michel(a.det, a.base), michel(a.det, a.arm)
    K2 = [k for k in T if k in M0 and k in M1]
    p2, n2 = [k for k in K2 if T[k]], [k for k in K2 if not T[k]]
    print(f"  CheckSTM_Michel candidates in both arms with a shape test: {len(K2)} ({len(p2)} / {len(n2)})")
    for nm, M in ((a.base, M0), (a.arm, M1)):
        print(f"  {nm:11s} ks_flat - ks_mu AUC {auc([-M[k][0] for k in p2], [-M[k][0] for k in n2]):.3f}; "
              f"contrast / expected AUC {auc([M[k][1] for k in p2], [M[k][1] for k in n2]):.3f}")
    print("  selection by production's TaggerCheckSTM:")
    for lab, S in (("stoppers", pos), ("non-stoppers", neg)):
        acc = [k for k in S if G0[k][2] == 0]
        rej = [k for k in S if G0[k][2] != 0]
        print(f"    {lab:12s} accepted in {a.base} {len(acc):4d} -> still accepted in {a.arm} {sum(G1[k][2] == 0 for k in acc):4d}; "
              f"not accepted in {a.base} {len(rej):4d} -> accepted in {a.arm} {sum(G1[k][2] == 0 for k in rej):4d}")


if __name__ == "__main__":
    main()
