#!/usr/bin/env python3
"""doc pdvd/104 sec 3 -- where does A1's "Michel" charge come from?  (read-only, no chain run)

Sec 2 shows no cut on the admission geometry separates the A1-only Michel false positives from the true Michels.
This script asks the prior question: are those false positives a NEW object the better trajectory found, or the
muon's OWN charge that the trajectory stopped short of and handed to the Michel builder?

THE TEST.  A0 and A1 read byte-identical pctrees (the arms are staged by d102_run_arms.sh with readlink -f onto the
same *_d51hclus tarballs), so the two arms' 3-D points live in the same frame and are directly comparable.  For each
candidate, take A1's Michel-member points -- T_stm_michel_pts rows with role == 3 (CheckSTM_Michel.cxx add_points,
role 3 = Michel object member) -- and measure their distance to A0's MUON CHAIN points (role == 1) for the same
cluster.  If A1's "Michel" sits ON A0's muon chain, it is charge A0 fitted as muon and A1 re-labelled as Michel.

CONTROL, and why it is needed.  A true Michel is attached at the stop, so it is NEAR the muon chain by construction;
"near" proves nothing.  The control is the set of true Michels BOTH arms tag: if containment discriminated, they
would be far from A0's chain.  Read the two distributions together, never the false positives alone
(feedback_split_by_class_before_reading_a_control).

WHAT THE TEST CANNOT BE.  A0's chain is not available when A1 runs, so containment is a DIAGNOSTIC, not a candidate
veto -- and the control shows it would not work as one anyway.

seg_id encodes the cluster as cluster_id*1000 + graph_index (CheckSTM_Michel.cxx persist), so rows are selected by
seg_id // 1000 == cluster_id.  The 0.6 cm containment radius is one PDHD wire pitch (0.4792 cm) rounded up; it is
reported alongside the full distance distribution so the conclusion does not rest on it.

    python3 d104_michel_provenance.py > figs/104_michel_provenance.txt
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import uproot
import d104_michel_features as F

NEAR_CM = 0.6
PTCOLS = ["x", "y", "z", "role", "seg_id"]


def pts(det, arm, ev):
    d = f"{F.IMG}/{det}/work/{ev}_{arm}"
    try:
        return uproot.open(f"{d}/tracking-pr.root")["T_stm_michel_pts"].arrays(PTCOLS, library="np")
    except Exception:
        return None


def sel(t, cid, role):
    if t is None:
        return np.empty((0, 3))
    m = (t["role"] == role) & ((t["seg_id"] // 1000) == cid)
    return np.stack([t["x"][m], t["y"][m], t["z"][m]], axis=1)


def one(det, cells, key):
    ev, cid = key.split("/")
    cid = int(cid)
    a, b = pts(det, cells[0], ev), pts(det, cells[1], ev)
    mich1, chain0, chain1 = sel(b, cid, 3), sel(a, cid, 1), sel(b, cid, 1)
    if len(mich1) == 0:
        return None, "A1 Michel object has no fitted points (conn 3, charge only)"
    if len(chain0) == 0:
        return None, "not a candidate in A0 (no muon chain to compare against)"
    d0 = np.sqrt(((mich1[:, None, :] - chain0[None, :, :]) ** 2).sum(-1)).min(1)
    d1 = np.sqrt(((mich1[:, None, :] - chain1[None, :, :]) ** 2).sum(-1)).min(1)
    return dict(n=len(mich1), med0=float(np.median(d0)), max0=float(d0.max()),
                frac=float((d0 < NEAR_CM).mean()), med1=float(np.median(d1)),
                n0=len(chain0), n1=len(chain1)), None


def show(det, cells, keys, A0, A1, title):
    print(f"\n  -- {title}")
    print(f"     {'key':16s} {'nMich':>5s} {'A0chain':>7s} {'medD(A0)':>8s} {'maxD':>6s} {'<0.6cm':>7s} "
          f"{'medD(A1own)':>11s} | muon_len A0 -> A1")
    stat = []
    for k in keys:
        r, why = one(det, cells, k)
        ml0 = A0.get(k, {}).get("muon_len", float("nan"))
        ml1 = A1.get(k, {}).get("muon_len", float("nan"))
        if r is None:
            print(f"     {k:16s} -- {why}")
            continue
        stat.append(r["frac"])
        print(f"     {k:16s} {r['n']:5d} {r['n0']:7d} {r['med0']:8.2f} {r['max0']:6.2f} {r['frac']:6.0%} "
              f"{r['med1']:11.2f} | {ml0:7.1f} -> {ml1:7.1f}")
    if stat:
        print(f"     -> containment fraction: median {np.median(stat):.0%}, "
              f"{sum(1 for s in stat if s >= 0.5)} of {len(stat)} items at 50 % or more")


def main():
    print("# doc pdvd/104 sec 3: is A1's Michel charge the muon's own charge?")
    print(f"# A1 Michel-member points (role 3) vs A0 muon-chain points (role 1), same cluster, same pctree.")
    print(f"# medD(A0) = median distance to A0's chain; <0.6cm = fraction inside one wire pitch of it;")
    print("# medD(A1own) = median distance to A1's OWN chain, for scale.")
    for det in ("pdhd",):
        D = F.load(det)
        tp, fp = F.split(D)
        A0, A1 = D["A0"], D["A1"]
        only = sorted(k for k, v in fp if not A0.get(k, {}).get("michel_found", 0))
        both = sorted(k for k, v in tp if A0.get(k, {}).get("michel_found", 0))
        print(f"\n===== {det}: A0 {D['cells'][0]} -> A1 {D['cells'][1]} =====")
        show(det, D["cells"], only, A0, A1,
             f"the {len(only)} A1-only Michel false positives (doc 103 sec 10.4's class)")
        show(det, D["cells"], both, A0, A1,
             f"CONTROL: the {len(both)} true Michels BOTH arms tag")


if __name__ == "__main__":
    main()
