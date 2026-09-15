#!/usr/bin/env python3
"""doc pdvd/103 sec 3 -- record-free census of what a trajectory change does to the STM/Michel taggers.  Read-only.

For one detector, a base arm and one or more arms run on the SAME pctrees (only the PR fit / retile differs):
  A  record-free totals on the events readable in every arm: CheckSTM_Michel candidates (= T_stm_michel rows),
     is_stm, michel_found
  B  the candidate swap base -> arm: lost / gained ids, the distance from each to the nearest other-arm candidate of
     the same event (max of entry and stop distance, either orientation), and whether the swapped clusters' T_cluster
     rows are identical between the arms (same npoints, flash, length -- i.e. the same cluster, a different verdict)
  C  TaggerCheckSTM verdicts from the job log (`visit: TaggerCheckSTM: cluster N -> STM=s` and `persist_stm_fit ...
     status=`): clusters evaluated, STM=1, flips, and the lowest pass's status before -> after for the flips
     (status codes TaggerCheckSTM.cxx:922-935: 0 accepted, 2 long leftover, 3 eval_stm failed, 4 other tracks,
     5 proton, 7 guard)
  D  kept candidates (a row in both arms): is_stm transitions and the CheckSTM_Michel reject bits that appear on a
     1->0 flip / disappear on a 0->1 flip (bit order census_lib.STM_BITS = StmMichelFunctions.h:555-571)

Usage: d103_churn_census.py --det pdhd --base d101hnew --arms d101hkf,d102hocs,d102hcs
"""
import argparse, collections, glob, os, re
import numpy as np
import uproot

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
BITS = ["no_chain", "stop_unmatched", "no_bragg", "shape_flat", "not_muon_pid", "continuation", "stop_near_boundary",
        "vertex_hadron", "short", "profile_sparse", "plateau_off_mip", "stop_into_dead", "cluster_not_track",
        "profile_geometry"]
BR = ["cluster_id", "is_stm", "michel_found", "reject_bits", "entry_x", "entry_y", "entry_z", "stop_x", "stop_y", "stop_z"]
R_STM = re.compile(r"TaggerCheckSTM: cluster (\d+) \S+ STM=(\d)")
R_PASS = re.compile(r"persist_stm_fit: cluster (\d+) stmfit pass=(\d+) status=(-?\d+)")


def evdirs(det, arm):
    return {os.path.basename(d)[:-len(arm) - 1]: d for d in glob.glob(f"{IMG}/{det}/work/*_{arm}")}


def load_rows(det, arm):
    rows, tcl = {}, {}
    for ev, d in evdirs(det, arm).items():
        try:
            f = uproot.open(f"{d}/tracking-pr.root")
            t = f["T_stm_michel"].arrays(BR, library="np")
            c = f["T_cluster"].arrays(library="np")
        except Exception:
            continue
        for i in range(len(t["cluster_id"])):
            rows[(ev, int(t["cluster_id"][i]))] = {b: float(t[b][i]) for b in BR}
        cols = sorted(k for k in c.keys() if k != "cluster_id")
        for i in range(len(c["cluster_id"])):
            tcl[(ev, int(c["cluster_id"][i]))] = tuple(round(float(c[k][i]), 3) for k in cols)
    return rows, tcl


def load_log(det, arm):
    out = {}
    for ev, d in evdirs(det, arm).items():
        logs = glob.glob(f"{d}/wct_pr_*.log")
        if not logs:
            continue
        for line in open(logs[0], errors="replace"):
            m = R_STM.search(line)
            if m:
                out.setdefault((ev, int(m.group(1))), {"passes": []})["stm"] = int(m.group(2))
                continue
            m = R_PASS.search(line)
            if m:
                out.setdefault((ev, int(m.group(1))), {"passes": []})["passes"].append((int(m.group(2)), int(m.group(3))))
    return out


def pos(r, w):
    return np.array([r[w + "_x"], r[w + "_y"], r[w + "_z"]])


def nearest(r, others):
    best = np.inf
    for r2 in others:
        d1 = max(np.linalg.norm(pos(r, "entry") - pos(r2, "entry")), np.linalg.norm(pos(r, "stop") - pos(r2, "stop")))
        d2 = max(np.linalg.norm(pos(r, "entry") - pos(r2, "stop")), np.linalg.norm(pos(r, "stop") - pos(r2, "entry")))
        best = min(best, d1, d2)
    return best


def names(rb):
    return {n for i, n in enumerate(BITS) if int(rb) >> i & 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True, choices=["pdhd", "pdvd"])
    ap.add_argument("--base", required=True)
    ap.add_argument("--arms", required=True)
    a = ap.parse_args()
    arms = a.arms.split(",")
    data = {x: load_rows(a.det, x) for x in [a.base] + arms}
    evs = set.intersection(*[{k[0] for k in data[x][1]} for x in data])
    print(f"[{a.det}] base {a.base}; events with T_stm_michel readable in every arm: {len(evs)}")

    print("\nA. record-free totals")
    for x in data:
        R = {k: v for k, v in data[x][0].items() if k[0] in evs}
        print(f"  {x:11s} candidates {len(R):5d}  is_stm {sum(int(v['is_stm']) for v in R.values()):4d}  "
              f"michel_found {sum(int(v['michel_found']) for v in R.values()):4d}")

    B, TB = data[a.base]
    B = {k: v for k, v in B.items() if k[0] in evs}
    logB = load_log(a.det, a.base)
    for x in arms:
        A, TA = data[x]
        A = {k: v for k, v in A.items() if k[0] in evs}
        print(f"\n==== {a.base} -> {x}")
        print("B. candidate swap")
        for lab, X, Y, TX, TY in (("lost (base only)", B, A, TB, TA), ("gained (arm only)", A, B, TA, TB)):
            only = [k for k in X if k not in Y]
            h = collections.Counter()
            same_tc = 0
            for k in only:
                d = nearest(X[k], [r2 for k2, r2 in Y.items() if k2[0] == k[0]])
                h["<5 cm" if d < 5 else "5-20 cm" if d < 20 else "20-60 cm" if d < 60 else ">60 cm / none"] += 1
                same_tc += int(k in TX and k in TY and TX[k] == TY[k])
            is_stm = sum(int(X[k]["is_stm"]) for k in only)
            print(f"  {lab:18s} {len(only):4d} (is_stm 1 on its own side: {is_stm}); nearest other-arm candidate: "
                  f"{dict(sorted(h.items()))}; identical T_cluster row in both arms: {same_tc}")

        print("C. TaggerCheckSTM verdicts (job log)")
        logA = load_log(a.det, x)
        keys = [k for k in logB if k[0] in evs]
        flips, seq = collections.Counter(), collections.Counter()
        for k in keys:
            rb, ra = logB[k], logA.get(k)
            if ra is None:
                flips["not evaluated in arm"] += 1
                continue
            if rb.get("stm") != ra.get("stm"):
                flips[f"STM {rb.get('stm')}->{ra.get('stm')}"] += 1
                sb = min(rb["passes"])[1] if rb["passes"] else None
                sa = min(ra["passes"])[1] if ra["passes"] else None
                seq[(f"STM {rb.get('stm')}->{ra.get('stm')}", sb, sa)] += 1
        n1 = lambda L: sum(v.get("stm", 0) for k, v in L.items() if k[0] in evs)
        print(f"  evaluated {len(keys)}; STM=1 base {n1(logB)} arm {n1(logA)}; flips {dict(flips)}")
        for (lab, sb, sa), n in seq.most_common(12):
            print(f"    {lab}: lowest-pass status {sb} -> {sa}: {n}")

        print("D. kept candidates")
        kept = [k for k in B if k in A]
        tr, newb, goneb = collections.Counter(), collections.Counter(), collections.Counter()
        for k in kept:
            sb, sa = int(B[k]["is_stm"]), int(A[k]["is_stm"])
            tr[f"{sb}->{sa}"] += 1
            nb, na = names(B[k]["reject_bits"]), names(A[k]["reject_bits"])
            if (sb, sa) == (1, 0):
                for n in (na - nb) or {"(no new bit)"}:
                    newb[n] += 1
            if (sb, sa) == (0, 1):
                for n in (nb - na) or {"(no cleared bit)"}:
                    goneb[n] += 1
        print(f"  kept {len(kept)}: is_stm {dict(sorted(tr.items()))}")
        print(f"    1->0 bits that appear: {dict(newb.most_common())}")
        print(f"    0->1 bits that clear:  {dict(goneb.most_common())}")


if __name__ == "__main__":
    main()
