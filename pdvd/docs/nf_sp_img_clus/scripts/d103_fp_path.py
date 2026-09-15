#!/usr/bin/env python3
"""doc pdvd/103 sec 7 -- how the both-on cell's new false positives got through.  Read-only.

For each false positive of A1 (both on) that is not one in A0 (production), on the union record:
  is_stm FP        in A0: not a candidate (TaggerCheckSTM lowest-pass status from the job log) or a candidate rejected by
                   CheckSTM_Michel (its reject bits in A0)
  michel_found FP  in A0: michel_found 0 on a candidate, or not a candidate; in A1: michel_conn_type, michel_len,
                   michel_kink_deg
So round 2 can be aimed at the stage and the test that admit them.

Usage: d103_fp_path.py --det pdhd --new-record F
"""
import argparse, collections, glob, json, os, re, sys
import uproot
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d103_union_grade as U

BITS = ["no_chain", "stop_unmatched", "no_bragg", "shape_flat", "not_muon_pid", "continuation", "stop_near_boundary",
        "vertex_hadron", "short", "profile_sparse", "plateau_off_mip", "stop_into_dead", "cluster_not_track",
        "profile_geometry"]
R_PASS = re.compile(r"persist_stm_fit: cluster (\d+) stmfit pass=(\d+) status=(-?\d+)")


def rows(det, arm):
    out = {}
    for d in glob.glob(f"{U.IMG}/{det}/work/*_{arm}"):
        ev = os.path.basename(d)[:-len(arm) - 1]
        try:
            t = uproot.open(f"{d}/tracking-pr.root")["T_stm_michel"].arrays(
                ["cluster_id", "is_stm", "michel_found", "reject_bits", "michel_conn_type", "michel_len", "michel_kink_deg"], library="np")
        except Exception:
            continue
        for i in range(len(t["cluster_id"])):
            out[f"{ev}/{int(t['cluster_id'][i])}"] = {k: float(t[k][i]) for k in t.keys()}
    return out


def status(det, arm):
    out = {}
    for d in glob.glob(f"{U.IMG}/{det}/work/*_{arm}"):
        ev = os.path.basename(d)[:-len(arm) - 1]
        for lg in glob.glob(f"{d}/wct_pr_*.log")[:1]:
            for line in open(lg, errors="replace"):
                m = R_PASS.search(line)
                if m:
                    k = f"{ev}/{m.group(1)}"
                    out[k] = min(out.get(k, (99, 0)), (int(m.group(2)), int(m.group(3))))
    return {k: v[1] for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True, choices=["pdhd", "pdvd"])
    ap.add_argument("--new-record", required=True)
    a = ap.parse_args()
    T, _ = U.load_truth(a.det, a.new_record)
    a0, a1 = U.CELLS[a.det][0][1], U.CELLS[a.det][3][1]
    B, A, SB = rows(a.det, a0), rows(a.det, a1), status(a.det, a0)
    judged = lambda k: k in T and T[k][0] not in ("MESSY", "UNCLEAR")
    stopper = lambda k: T[k][0] in ("STM_MICHEL", "STM_ONLY")
    names = lambda rb: [n for i, n in enumerate(BITS) if int(rb) >> i & 1]
    print(f"# doc pdvd/103: the path of A1's new false positives ({a.det}; A0 {a0}, A1 {a1})")
    for title in ("is_stm", "michel_found"):
        def fp(k, R):
            r = R.get(k)
            if r is None or not judged(k) or not r[title]:
                return False
            if title == "is_stm":
                return not stopper(k)
            if a.det == "pdhd":
                return stopper(k) and T[k][1] not in ("attached", "both") and not (T[k][1] is None and T[k][2] == "owner_review")
            return T[k][0] != "STM_MICHEL"
        new = sorted(k for k in A if fp(k, A) and not fp(k, B))
        path = collections.Counter(); bits = collections.Counter(); conn = collections.Counter(); lens = []
        for k in new:
            if k not in B:
                path[f"not an A0 candidate (A0 TaggerCheckSTM status {SB.get(k, 'none')})"] += 1
            else:
                if title == "is_stm":
                    path["A0 candidate, CheckSTM_Michel rejected"] += 1
                    for n in names(B[k]["reject_bits"]) or ["(none)"]:
                        bits[n] += 1
                else:
                    path[f"A0 candidate, michel_found 0 (A0 is_stm {int(B[k]['is_stm'])})"] += 1
            if title == "michel_found":
                conn[int(A[k]["michel_conn_type"])] += 1
                lens.append(A[k]["michel_len"])
        print(f"\n== {title}: A1-only false positives {len(new)}")
        for p, n in path.most_common():
            print(f"  {p}: {n}")
        if bits:
            print(f"  A0 reject bits of the rejected ones: {dict(bits.most_common())}")
        if conn:
            import numpy as np
            print(f"  A1 michel_conn_type (1 attached, 2 bridged, 3 charge-only): {dict(conn)}; michel_len median {np.median(lens):.1f} cm")
        print("  keys: " + ", ".join(new))


if __name__ == "__main__":
    main()
