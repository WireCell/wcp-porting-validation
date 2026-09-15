#!/usr/bin/env python3
"""doc pdvd/103 sec 3.3 -- how far can labels on the candidates the record lacks move the grade?  Read-only.

The fixed-denominator grades of doc 101/102 (d101_stm_grade_{pdhd,pdvd}.py) score only record items.  A candidate
that exists only in the arm (it became an STM candidate with the new trajectory) has no label, and it is ALSO a
cluster production missed: if it is a real stopper it is an FN for production too.  This script
  1. re-derives the graders' is_stm TP/FP/FN/TN from T_stm_michel (must equal the graders' printed numbers),
  2. counts the candidates of either arm that no record in pdhd|pdvd/docs/scan labels, split by is_stm in each arm,
  3. prints the best and worst union-record efficiency / purity for both arms (all unlabelled tagged candidates real
     stoppers and all untagged ones not, and the reverse).

Populations (as the graders):
  pdhd  record smx22, items in the smx18 p82bhoff key (303), truth owner_review > owner_smx1 > agent; absent = miss
  pdvd  $STM_SCAN_RECORD (merged smx1a..smx9) judged items that are candidates in the base arm; absent = miss

Usage: d103_bounds.py --det pdhd --base d101hnew --arm d102hcs
"""
import argparse, csv, glob, json, os, sys
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


def rows(det, arm):
    out = {}
    for d in glob.glob(f"{IMG}/{det}/work/*_{arm}"):
        ev = os.path.basename(d)[:-len(arm) - 1]
        try:
            t = uproot.open(f"{d}/tracking-pr.root")["T_stm_michel"].arrays(["cluster_id", "is_stm"], library="np")
        except Exception:
            continue
        for c, s in zip(t["cluster_id"], t["is_stm"]):
            out[f"{ev}/{int(c)}"] = int(s)
    return out


def all_record_keys(det):
    """Keys labelled by a record that figs/103_pred.txt admits as truth: carried-over records (labels copied from
    another arm) and the doc-99 swap scan sw99 are not truth there, so their items count as unlabelled here too."""
    keys = set()
    for f in glob.glob(f"{IMG}/{det}/docs/scan/*verdicts*.json"):
        if "carried" in f or "sw99" in f:
            continue
        J = json.load(open(f))
        keys |= {r["key"] for r in J} if isinstance(J, list) else set(J)
    return keys


def grade(tp, fp, fn):
    return tp / max(1, tp + fp), tp / max(1, tp + fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True, choices=["pdhd", "pdvd"])
    ap.add_argument("--base", required=True)
    ap.add_argument("--arm", required=True)
    a = ap.parse_args()
    B, A = rows(a.det, a.base), rows(a.det, a.arm)
    if a.det == "pdhd":
        X = f"{IMG}/pdhd/docs/scan"
        rec = {r["key"]: hand(r) for r in json.load(open(f"{X}/pdhd_stm_michel_smx22_verdicts.json"))}
        pop = {"%s/%s" % (r["event"], r["cluster"]) for r in csv.DictReader(
            [l for l in open(f"{X}/smx18/pdhd_stm_michel_scan_key_p82bhoff.tsv") if not l.startswith("#")], delimiter="\t")}
        T = {k: v in ("STM_MICHEL", "STM_ONLY") for k, v in rec.items() if k in pop and v not in ("MESSY", "UNCLEAR")}
    else:
        import census_lib as C
        rec = C.load_record()
        T = {k: C.is_stopper(r) for k, r in rec.items() if C.judged(r) and k in B}
    labelled = all_record_keys(a.det)
    print(f"[{a.det}] {a.base} vs {a.arm}; graded population {len(T)}; keys labelled in any record {len(labelled)}")
    cnt = {}
    for nm, R in ((a.base, B), (a.arm, A)):
        tp = sum(1 for k, t in T.items() if t and R.get(k) == 1)
        fp = sum(1 for k, t in T.items() if not t and R.get(k) == 1)
        fn = sum(1 for k, t in T.items() if t and R.get(k) != 1)
        tn = len(T) - tp - fp - fn
        cnt[nm] = (tp, fp, fn)
        p, e = grade(tp, fp, fn)
        print(f"  fixed denominator  {nm:11s} TP {tp:3d} FP {fp:3d} FN {fn:3d} TN {tn:3d}  purity {p:.3f} efficiency {e:.3f}")
    for nm, R, other in ((a.arm, A, a.base), (a.base, B, a.arm)):
        un = [k for k in R if k not in labelled and k not in T]
        tag = sum(R[k] for k in un)
        print(f"  unlabelled candidates of {nm} only or shared: {len(un)} (is_stm 1: {tag}, is_stm 0: {len(un) - tag}); "
              f"of them candidates in {other}: {sum(1 for k in un if k in (B if R is A else A))}")
    # union bound: the unlabelled candidates of both arms, each classified by its is_stm in each arm
    un = {k for k in set(A) | set(B) if k not in labelled and k not in T}
    print(f"  union of unlabelled candidates {len(un)}")
    for case in ("best for the arm", "worst for the arm"):
        c = {nm: list(cnt[nm]) for nm in cnt}
        for k in un:
            sa, sb = A.get(k, 0), B.get(k, 0)
            stopper = (sa == 1) if case == "best for the arm" else (sa == 0)
            for nm, s in ((a.arm, sa), (a.base, sb)):
                if stopper and s == 1:
                    c[nm][0] += 1
                elif stopper:
                    c[nm][2] += 1
                elif s == 1:
                    c[nm][1] += 1
        pa, ea = grade(*c[a.arm])
        pb, eb = grade(*c[a.base])
        print(f"  {case:18s}: {a.base} purity {pb:.3f} eff {eb:.3f} | {a.arm} purity {pa:.3f} eff {ea:.3f} | "
              f"efficiency change {ea - eb:+.3f}, purity change {pa - pb:+.3f}")


if __name__ == "__main__":
    main()
