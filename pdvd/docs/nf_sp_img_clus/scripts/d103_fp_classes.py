#!/usr/bin/env python3
"""doc pdvd/103 sec 5.3 -- what the purity cost of the both-on cell is made of.  Read-only.

For is_stm and michel_found, the false positives of A0 (production) and A1 (both on) on the union record, split by
  label source   owner (owner_review / owner_smx1 / an owner-confidence PDVD record row) | existing agent | new blind agent
  confidence     of the label
  hand class     is_stm: the hand verdict; michel_found: hand verdict + michel_kind (e.g. STM_ONLY/detached dots)
  shared         FP in both cells, only in A1, only in A0
Truth, population and chain exactly as d103_union_grade.py (imported).

Usage: d103_fp_classes.py --det pdhd --new-record F
"""
import argparse, collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d103_union_grade as U


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True, choices=["pdhd", "pdvd"])
    ap.add_argument("--new-record", required=True)
    a = ap.parse_args()
    T, _ = U.load_truth(a.det, a.new_record)
    R = {lab: U.cell_rows(a.det, arm) for lab, arm in U.CELLS[a.det]}
    conf = {}
    base = (U.PDHD_RECORD if a.det == "pdhd" else
            f"{U.IMG}/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_smx5_smx6_smx7_smx8_smx9_verdicts.json")
    for r in json.load(open(base)):
        conf[r["key"]] = r.get("confidence", "")
    for r in json.load(open(a.new_record)):
        conf.setdefault(r["key"], r.get("confidence", ""))

    def source(k):
        s = T[k][2]
        if s in ("owner_review", "owner") or conf.get(k) == "owner":
            return "owner"
        return "new blind agent" if s == "new_agent" else "existing agent"

    judged = lambda k: k in T and T[k][0] not in ("MESSY", "UNCLEAR")
    stopper = lambda k: T[k][0] in ("STM_MICHEL", "STM_ONLY")
    cells = dict(U.CELLS[a.det])       # sec 10: the production lineage has only A0 / A1
    print(f"# doc pdvd/103 false-positive classes ({a.det}); union record, cells A0 {cells['A0']} and A1 {cells['A1']}")
    for title, idx in (("is_stm", 0), ("michel_found", 1)):
        def is_fp(k, lab):
            if not judged(k) or not R[lab].get(k, (0, 0))[idx]:
                return False
            if title == "is_stm":
                return not stopper(k)
            if a.det == "pdhd":
                return stopper(k) and T[k][1] not in ("attached", "both") and not (T[k][1] is None and T[k][2] == "owner_review")
            return T[k][0] != "STM_MICHEL"
        keys = set(R["A0"]) | set(R["A1"])
        fp0 = {k for k in keys if is_fp(k, "A0")}
        fp1 = {k for k in keys if is_fp(k, "A1")}
        print(f"\n== {title}: FP A0 {len(fp0)}, A1 {len(fp1)} (shared {len(fp0 & fp1)}, A1 only {len(fp1 - fp0)}, A0 only {len(fp0 - fp1)})")
        for lab, S in (("A1 only", fp1 - fp0), ("shared", fp0 & fp1), ("A0 only", fp0 - fp1)):
            src = collections.Counter(source(k) for k in S)
            cf = collections.Counter(conf.get(k, "") for k in S)
            cls = collections.Counter(f"{T[k][0]}/{T[k][1]}" if title == "michel_found" else T[k][0] for k in S)
            # sorted: Counter order over a set follows PYTHONHASHSEED (round-1 figures differ only in this order)
            print(f"  {lab:8s} n {len(S):3d} | source {dict(sorted(src.items()))} | confidence {dict(sorted(cf.items()))} | "
                  f"hand class {dict(sorted(cls.items(), key=lambda kv: (-kv[1], kv[0])))}")


if __name__ == "__main__":
    main()
