#!/usr/bin/env python3
"""doc pdvd/103 sec 10.4 -- the PDHD Michel false positives of A0 / A1 after the owner's own103h adjudication, with the
chain's Michel geometry.  Truth and the Michel definition as d103_union_grade.py --stm-only-unset-negative
(figs/103_pred_amend5.txt; own103h > smx27 > smx28; positive = a hand stopper with michel_kind attached / both; an owner
STM_ONLY with no kind is negative, an owner STM_MICHEL with no kind is excluded).  Read-only.

    D103_PDHD_RECORD=$IMG/pdhd/docs/scan/pdhd_stm_michel_smx27_verdicts.json \
        python3 d103_michel_fp_list_own.py > figs/103_michel_fp_pdhd_own103h.txt
"""
import glob, json, os, sys
import uproot
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d103_union_grade as U

IMG = U.IMG
N28 = f"{IMG}/pdhd/docs/scan/pdhd_stm_michel_smx28_verdicts.json"
OWN = f"{IMG}/pdhd/docs/scan/pdhd_stm_michel_own103h_verdicts.json"
COLS = ["cluster_id", "is_stm", "michel_found", "michel_len", "michel_conn_type", "michel_n_pieces", "michel_dis_cm",
        "n_dots", "michel_ke_best", "michel_kink_deg"]


def main():
    T, _ = U.load_truth("pdhd", N28, OWN)
    own = {r["key"] for r in json.load(open(OWN))}
    print("# doc pdvd/103 round 2: PDHD Michel false positives after own103h (truth own103h > smx27 > smx28; grader definition)")
    for lab, arm in (("A0", "d101hnew"), ("A1", "d102hcs")):
        R = {}
        for d in glob.glob(f"{IMG}/pdhd/work/*_{arm}"):
            ev = os.path.basename(d)[:-len(arm) - 1]
            t = uproot.open(f"{d}/tracking-pr.root")["T_stm_michel"].arrays(COLS, library="np")
            for i in range(len(t["cluster_id"])):
                R[f"{ev}/{int(t['cluster_id'][i])}"] = {c: float(t[c][i]) for c in COLS}
        # figs/103_pred_amend5.txt (a): an owner STM_ONLY with no kind is Michel-negative (not excluded)
        fps = sorted(k for k, r in R.items() if k in T and r["michel_found"] and T[k][0] in ("STM_MICHEL", "STM_ONLY")
                     and not (T[k][1] is None and T[k][2] == "owner_review" and T[k][0] == "STM_MICHEL")
                     and T[k][1] not in ("attached", "both"))
        print(f"== {lab} {arm}: Michel FPs {len(fps)}")
        for k in fps:
            r = R[k]
            v, mk, s = T[k]
            print(f"  {k:14s} truth {v}/{mk} ({s}{', own103h' if k in own else ''}) | michel_len {r['michel_len']:.1f} cm "
                  f"conn {int(r['michel_conn_type'])} pieces {int(r['michel_n_pieces'])} dis {r['michel_dis_cm']:.1f} "
                  f"n_dots {int(r['n_dots'])} ke {r['michel_ke_best']:.1f} MeV kink {r['michel_kink_deg']:.0f} is_stm {int(r['is_stm'])}")


if __name__ == "__main__":
    main()
