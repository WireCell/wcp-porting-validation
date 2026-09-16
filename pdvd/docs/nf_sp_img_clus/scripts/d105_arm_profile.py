#!/usr/bin/env python3
"""doc pdvd/105 sec 2 -- the Bragg-position discriminator, sized offline (read-only, no chain run).

Doc 104 sec 7 left ONE candidate discriminator unrefuted: a stopping muon deposits its Bragg peak at the stop, so
if the muon really stopped where the tagger says, the charge attached there is a Michel; if the trajectory stopped
SHORT, the attached piece is muon and the rise sits further along it.  Doc 104's L4 tested only "is there a Bragg
rise at the claimed stop" (figs/104_lever_sizing.txt) and failed; this script tests the POSITION comparison it is
the degenerate case of.

WHAT MAKES THIS MEASURABLE WITHOUT A NEW ARM.  T_stm_michel_pts persists q (dQ/dx, e/cm) and x/y/z for the Michel
object's own points (role 3) as well as for the chain (role 1).  rr and L are -1 on role-3 rows -- the Michel is
not on the chain's arclength -- so the arm's internal profile is rebuilt GEOMETRICALLY here, ordering the role-3
points by distance from the muon stop (T_stm_michel stop_x/y/z).  q_sup (segment median dQ/dx / chain plateau
median) is persisted per row and used as the cross-check on the per-muon normalisation.

THE FEATURES, each normalised to the muon's OWN plateau (plateau_med) rather than to the fixed 43000 e/cm that
michel_mip uses, so a muon-by-muon gain or angle difference cannot fake the comparison:
  q_near   median q of the third of the arm nearest the stop,      / plateau_med
  q_far    median q of the third of the arm at its far end,        / plateau_med
  slope    q_far / q_near  -- rising toward the far end is the truncated-muon signature
  contrast the tagger's own Bragg contrast at the claimed stop,    / contrast_expected (this is L4)

Restricted to conn_type 1: conn 2 (bridged) and 3 (charge only) have no attached arm, so "along the arm" is not
defined for them.  They carry 3 of PDHD's 9 false positives and 21 of its 77 true Michels and are reported apart.

Separation is read as AUC over the same labelled population the grade uses (d103_union_grade.population through
d104_michel_features.load), TP = hand Michel, FP = the arm's Michel false positives.  A feature that cannot beat
AUC 0.5 by more than its own bootstrap spread is not a discriminator, whatever a hand-picked cut does.

    python3 d105_arm_profile.py > figs/105_arm_profile.txt
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import uproot
import d104_michel_features as F

HEAD = ["cluster_id", "stop_x", "stop_y", "stop_z", "plateau_med", "tail_med", "contrast", "contrast_expected",
        "michel_len", "michel_mip", "michel_conn_type", "michel_found"]
PTS = ["x", "y", "z", "q", "q_sup", "role", "seg_id"]
MIN_PTS = 6            # below this an outer/inner third is one or two points and the profile is noise


def arm_features(det, arm):
    """{key: dict of arm-profile features} for every candidate with an attached Michel object and enough points."""
    out = {}
    for d in __import__("glob").glob(f"{F.IMG}/{det}/work/*_{arm}"):
        ev = os.path.basename(d)[:-len(arm) - 1]
        try:
            f = uproot.open(f"{d}/tracking-pr.root")
            H = f["T_stm_michel"].arrays(HEAD, library="np")
            T = f["T_stm_michel_pts"].arrays(PTS, library="np")
        except Exception:
            continue
        cl = T["seg_id"] // 1000
        for i, cid in enumerate(H["cluster_id"]):
            cid = int(cid)
            if not H["michel_found"][i] or H["michel_conn_type"][i] != 1:
                continue
            pl = float(H["plateau_med"][i])
            s = (cl == cid) & (T["role"] == 3) & (T["q"] > 0)
            n = int(s.sum())
            if n < MIN_PTS or pl <= 0:
                continue
            stop = np.array([H["stop_x"][i], H["stop_y"][i], H["stop_z"][i]], dtype=float)
            P = np.stack([T["x"][s], T["y"][s], T["z"][s]], axis=1).astype(float)
            q = T["q"][s].astype(float)
            o = np.argsort(np.linalg.norm(P - stop, axis=1))
            q = q[o]
            k = max(2, n // 3)
            q_near, q_far = float(np.median(q[:k])), float(np.median(q[-k:]))
            ce = float(H["contrast_expected"][i])
            out[f"{ev}/{cid}"] = dict(n=n, q_near=q_near / pl, q_far=q_far / pl,
                                      slope=q_far / max(1.0, q_near),
                                      contrast=(float(H["contrast"][i]) / ce) if ce > 0 else -1.0,
                                      qsup=float(np.median(T["q_sup"][s])), len=float(H["michel_len"][i]))
    return out


def auc(pos, neg):
    """rank AUC, ties at 0.5; returns (auc, 68 % bootstrap interval)"""
    if not pos or not neg:
        return float("nan"), (float("nan"), float("nan"))
    def one(a, b):
        a, b = np.asarray(a), np.asarray(b)
        return float((np.sum(a[:, None] > b[None, :]) + 0.5 * np.sum(a[:, None] == b[None, :])) / (len(a) * len(b)))
    rng = np.random.default_rng(105)
    bs = [one(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(400)]
    return one(pos, neg), (float(np.percentile(bs, 16)), float(np.percentile(bs, 84)))


def main():
    print("# doc pdvd/105 sec 2: the Bragg-position discriminator -- is the rise at the claimed stop or along the arm?")
    print("# Arm profile rebuilt geometrically from role-3 points (rr/L are -1 there); q normalised to the muon's")
    print(f"# OWN plateau_med.  conn_type 1 only, at least {MIN_PTS} live Michel points.")
    print("# AUC > 0.5 means the feature is LARGER on true Michels; a discriminator needs |AUC - 0.5| >> its spread.")
    for det in ("pdhd", "pdvd"):
        D = F.load(det)
        tp, fp = F.split(D)
        A = arm_features(det, D["cells"][1])
        tpk = [k for k, _ in tp if k in A]
        fpk = [k for k, _ in fp if k in A]
        print(f"\n===== {det}: A1 {D['cells'][1]} | attached Michels with a usable profile: TP {len(tpk)} of "
              f"{len(tp)}, FP {len(fpk)} of {len(fp)} =====")
        if not fpk:
            print("   no false positive has an attached arm with enough points -- nothing to separate here")
            continue
        for name in ("q_near", "q_far", "slope", "contrast", "qsup"):
            P = [A[k][name] for k in tpk]
            N = [A[k][name] for k in fpk]
            a, (lo, hi) = auc(P, N)
            qp = np.percentile(P, [25, 50, 75])
            print(f"   {name:9s} TP q1/med/q3 {qp[0]:6.2f} {qp[1]:6.2f} {qp[2]:6.2f} | FP "
                  + " ".join(f"{x:5.2f}" for x in sorted(N))
                  + f" | AUC {a:.3f} [{lo:.3f}, {hi:.3f}]")
        print("   false positives in detail:")
        for k in sorted(fpk):
            v = A[k]
            print(f"     {k:16s} n {v['n']:3d} len {v['len']:5.1f} q_near {v['q_near']:5.2f} q_far {v['q_far']:5.2f} "
                  f"slope {v['slope']:5.2f} contrast {v['contrast']:5.2f}")


if __name__ == "__main__":
    main()
