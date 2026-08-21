#!/usr/bin/env python3
"""doc sbnd_xin/docs/pr/108 -- stage-by-stage diff of the trajectory rounds
between two WCT_TRAJ_DUMP / WCP_TRAJ_DUMP files (same record layout):

  <call> call excl=<0|1> cluster=<id>
  <call> map<stage> <seg> <i> x y z nU0 nV0 nW0 nU1 nV1 nW1 qU qV qW kept dis_cut nU2 nV2 nW2
  (n0 = cells associated, n1 = after update_association (exclusion), n2 = after examine_point_association)
  <call> fit<stage> <seg> <i> x y z dQ dx

For each file the "call of interest" is the LAST do_multi_tracking call whose
fit3 (final) trajectory has a point within --rj cm of the given junction (x,y,z).
Points of the two sides are matched by position at each stage (nearest, <= --match cm).
Reports, per stage and for points within --rj of the junction: number of points,
matched, association counts before/after exclusion (U/V/W sums), cells stripped by
exclusion, quantity (live-plane fraction), dropped points, and fit displacement
between rounds.  Usage:
  pr108_stage_diff.py A.dump B.dump --j x y z [--label-a WCP --label-b WCT] [--rj 3] [--call-a N --call-b M]
"""
import argparse, collections
import numpy as np


def parse(path):
    calls = collections.OrderedDict()
    for line in open(path):
        f = line.split()
        if len(f) < 2: continue
        c = int(f[0])
        if f[1] == "call":
            calls[c] = dict(excl=int(f[2].split("=")[1]), cluster=int(f[3].split("=")[1]), map={}, fit={})
            continue
        rec = calls.setdefault(c, dict(excl=-1, cluster=-1, map={}, fit={}))
        if f[1].startswith("map"):
            st = int(f[1][3:]); v = [float(x) for x in f[2:]]
            rec["map"].setdefault(st, []).append(v)
        elif f[1].startswith("fit"):
            st = int(f[1][3:]); v = [float(x) for x in f[2:]]
            rec["fit"].setdefault(st, []).append(v)
    for rec in calls.values():
        for d in (rec["map"], rec["fit"]):
            for k in d: d[k] = np.array(d[k])
    return calls


def pick_call(calls, J, rj, forced=None):
    if forced is not None: return forced
    best = None
    for c, rec in calls.items():
        f = rec["fit"].get(3)
        if f is None or len(f) == 0: continue
        if np.linalg.norm(f[:, 2:5] - J, axis=1).min() <= rj: best = c
    return best


def near(arr, J, rj):
    return np.linalg.norm(arr[:, 2:5] - J, axis=1) <= rj


def match(A, B, tol):
    idx = np.full(len(A), -1)
    if len(B) == 0: return idx
    for i in range(len(A)):
        d = np.linalg.norm(B[:, 2:5] - A[i, 2:5], axis=1); k = d.argmin()
        if d[k] <= tol: idx[i] = k
    return idx


def summ_map(M, sel):
    m = M[sel]
    if len(m) == 0: return "n=0"
    n0 = m[:, 5:8].sum(0); n1 = m[:, 8:11].sum(0); q = m[:, 11:14].mean(0); kept = int(m[:, 14].sum())
    n2 = m[:, 16:19].sum(0) if m.shape[1] >= 19 else n1
    return (f"n={len(m)} kept={kept} assoc U/V/W={int(n0[0])}/{int(n0[1])}/{int(n0[2])} "
            f"after-excl={int(n1[0])}/{int(n1[1])}/{int(n1[2])} (stripped {int((n0-n1).sum())}) "
            f"after-examine={int(n2[0])}/{int(n2[1])}/{int(n2[2])} <q>={q[0]:.2f}/{q[1]:.2f}/{q[2]:.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--j", type=float, nargs=3, required=True)
    ap.add_argument("--label-a", default="A"); ap.add_argument("--label-b", default="B")
    ap.add_argument("--rj", type=float, default=3.0); ap.add_argument("--match", type=float, default=0.5)
    ap.add_argument("--call-a", type=int); ap.add_argument("--call-b", type=int)
    a = ap.parse_args(); J = np.array(a.j)
    CA, CB = parse(a.a), parse(a.b)
    ca, cb = pick_call(CA, J, a.rj, a.call_a), pick_call(CB, J, a.rj, a.call_b)
    print(f"junction {J}: {a.label_a} call {ca} (excl={CA[ca]['excl'] if ca else None}, {len(CA)} calls) ; "
          f"{a.label_b} call {cb} (excl={CB[cb]['excl'] if cb else None}, {len(CB)} calls)")
    if ca is None or cb is None: return
    RA, RB = CA[ca], CB[cb]
    for st in (1, 2, 3):
        MA, MB = RA["map"].get(st), RB["map"].get(st)
        if MA is None or MB is None: print(f" map{st}: missing on one side"); continue
        sa, sb = near(MA, J, a.rj), near(MB, J, a.rj)
        print(f" map{st} near-J  {a.label_a}: {summ_map(MA, sa)}")
        print(f"              {a.label_b}: {summ_map(MB, sb)}")
        idx = match(MA[sa], MB[sb], a.match)
        ok = idx >= 0
        if ok.any():
            A_, B_ = MA[sa][ok], MB[sb][idx[ok]]
            d0 = (B_[:, 5:8] - A_[:, 5:8]); d1 = (B_[:, 8:11] - A_[:, 8:11])
            print(f"   matched {ok.sum()}/{sa.sum()}: per-point assoc0 diff U/V/W median {np.median(d0,0).round(1)}  after-excl diff median {np.median(d1,0).round(1)}  kept A/B {int(A_[:,14].sum())}/{int(B_[:,14].sum())}  dis_cut med A/B {np.median(A_[:,15]):.2f}/{np.median(B_[:,15]):.2f}")
    for st in (1, 2, 3):
        FA, FB = RA["fit"].get(st), RB["fit"].get(st)
        if FA is None or FB is None: print(f" fit{st}: missing on one side"); continue
        sa, sb = near(FA, J, a.rj), near(FB, J, a.rj)
        idx = match(FA[sa], FB[sb], a.match); ok = idx >= 0
        dd = np.linalg.norm(FB[sb][idx[ok], 2:5] - FA[sa][ok, 2:5], axis=1) if ok.any() else np.array([])
        qa = FA[sa][:, 5].sum(); qb = FB[sb][:, 5].sum()
        print(f" fit{st} near-J: n {a.label_a}={sa.sum()} {a.label_b}={sb.sum()} matched {ok.sum()} |dpos| med/max {np.median(dd) if len(dd) else float('nan'):.3f}/{dd.max() if len(dd) else float('nan'):.3f} cm ; sum dQ {qa:.0f} vs {qb:.0f}")
    # round-to-round displacement within each side
    for lab, R in ((a.label_a, RA), (a.label_b, RB)):
        for s1, s2 in ((1, 2), (2, 3)):
            F1, F2 = R["fit"].get(s1), R["fit"].get(s2)
            if F1 is None or F2 is None: continue
            s = near(F1, J, a.rj); idx = match(F1[s], F2, a.match); ok = idx >= 0
            dd = np.linalg.norm(F2[idx[ok], 2:5] - F1[s][ok, 2:5], axis=1) if ok.any() else np.array([])
            print(f"   {lab} fit{s1}->fit{s2} near-J: matched {ok.sum()}/{s.sum()} |dpos| med/max {np.median(dd) if len(dd) else float('nan'):.3f}/{dd.max() if len(dd) else float('nan'):.3f}")


if __name__ == "__main__":
    main()
