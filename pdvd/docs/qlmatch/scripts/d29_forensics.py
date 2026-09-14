#!/usr/bin/env python3
"""doc qlmatch/29 -- what the matcher did with each truth pair a step lost.  Fork of rl_forensics.py (that script keeps its
hard-coded tm0/rc14 and stays as committed).  (read-only)

    python3 d29_forensics.py --base p99wflip --arm p98voffq > ../d29/forensics_step2.txt

For every objective truth POSITIVE covered on --base and missed on --arm (original truth key, truth mapped through --ref):
  cluster-identity-change  the mapped cluster changed beyond the pre-set bounds between the two arms: npoints ratio outside
                           [0.5, 2] or either x end moved > 10 cm (clustering fed QLMatching a different object);
  no-truth-bundle          no bundle on the mapped cluster at the truth flash in --arm;
  unmatched                a truth bundle exists, the cluster auto-selected nothing;
  mis-pick                 the cluster auto-selected a different flash (winner class as in rl_forensics).
Each row prints the base's selected truth bundle and the arm's truth bundle (ks, chi2/ndf, strength, contained, flags).
Aggregates: classes; for pairs that are NOT identity changes, the truth bundle's flag transitions and median ks / c2n /
strength on both arms.  Then the new phantoms (objective negatives rejected on --base, auto on --arm) with their flags.
"""
import argparse
import collections
import statistics

import d29_common as C

FLAG_CH = {"at_cathode": "C", "at_x_boundary": "B", "two_boundary": "2", "close_to_PMT": "P", "window_truncated": "W",
           "consistent": "c", "xtpc_consistent": "x", "xtpc_pin": "p", "xtpc_scenario1": "1", "xtpc_cathode_rescued": "r",
           "spec_end": "s", "contained": "k"}
NPTS_RATIO, X_MOVE_CM = (0.5, 2.0), 10.0


def flags(b):
    return "".join(ch for k, ch in FLAG_CH.items() if b.get(k))


def brow(t, b):
    return (f"t={t:9.1f} ks={b['ks_dis']:.3f} c2n={b['chi2'] / max(b['ndf'], 1):6.1f} str={b['strength']:.3f} "
            f"pred/meas={sum(b['pred_pe']) / max(b['total_PE'], 1e-9):5.2f} fl=[{flags(b)}]")


def at(lst, time):
    return [(t, b) for t, b in lst if abs(t - time) <= C.TOL]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--ref", default="keep")
    a = ap.parse_args()
    T = C.truth()
    print(f"# doc qlmatch/29 -- forensics {a.base} -> {a.arm} (d29_forensics.py; truth map '{a.ref}'; identity bounds "
          f"npoints ratio {NPTS_RATIO}, x end move {X_MOVE_CM} cm)")
    cls, winner = collections.Counter(), collections.Counter()
    flag_lost, flag_gain = collections.Counter(), collections.Counter()
    metrics = collections.defaultdict(list)
    new_phantoms = []
    for idx in C.IDX:
        evt = C.S.evt_of_idx(idx)
        B, A = C.Arm(a.base, idx, ref=a.ref), C.Arm(a.arm, idx, ref=a.ref)
        for e in T[evt]:
            if e["conf"] not in C.S.OBJECTIVE_TIERS:
                continue
            if not e["positive"]:
                sb, _ = B.negative_status(e)
                sa, ua = A.negative_status(e)
                if (sb, sa) == ("rejected", "phantom"):
                    t, b = at(A.autos[ua], e["time"])[0]
                    new_phantoms.append(f"  evt{evt} truth uid {e['uid']} -> {ua} {'top' if b['apa'] == 4 else 'bottom'}  {brow(t, b)}")
                continue
            sb, ub = B.positive_status(e)
            sa, ua = A.positive_status(e)
            if (sb, sa) != ("covered", "missed"):
                continue
            cb, ca = B.clusters[ub], A.clusters[ua]
            ratio = ca["npoints"] / max(cb["npoints"], 1)
            dx0, dx1 = abs(min(ca["x"]) - min(cb["x"])), abs(max(ca["x"]) - max(cb["x"]))
            vol = "top" if ca["apa"] == 4 else "bottom"
            head = (f"evt{evt} truth uid {e['uid']} t={e['time']:.1f} {vol}: {a.base} uid {ub} n={cb['npoints']} "
                    f"x[{min(cb['x']):.0f},{max(cb['x']):.0f}] -> {a.arm} uid {ua} n={ca['npoints']} x[{min(ca['x']):.0f},{max(ca['x']):.0f}]")
            base_sel = at(B.autos[ub], e["time"])
            base_row = brow(*base_sel[0]) if base_sel else "(none)"
            identity = not (NPTS_RATIO[0] <= ratio <= NPTS_RATIO[1]) or dx0 > X_MOVE_CM or dx1 > X_MOVE_CM
            truth_b = at(A.bundles[ua], e["time"])
            if identity:
                c = "cluster-identity-change"
            elif not truth_b:
                c = "no-truth-bundle"
            elif not A.autos[ua]:
                c = "unmatched"
            else:
                c = "mis-pick"
            cls[c] += 1
            cls[f"{c} ({vol})"] += 1
            print(f"{head}  {c.upper()}")
            print(f"    {a.base} truth (selected): {base_row}")
            if truth_b:
                tb = max(truth_b, key=lambda x: x[1]["strength"])
                print(f"    {a.arm} truth bundle   : {brow(*tb)}")
                if not identity and base_sel:
                    bb = base_sel[0][1]
                    for k in FLAG_CH:
                        if bb.get(k) and not tb[1].get(k):
                            flag_lost[k] += 1
                        if tb[1].get(k) and not bb.get(k):
                            flag_gain[k] += 1
                    for side, x in (("base", bb), ("arm", tb[1])):
                        metrics[f"ks {side}"].append(x["ks_dis"])
                        metrics[f"c2n {side}"].append(x["chi2"] / max(x["ndf"], 1))
                        metrics[f"strength {side}"].append(x["strength"])
            if c == "mis-pick":
                t, sb_ = A.autos[ua][0]
                w = ("winner-pinned" if sb_.get("xtpc_pin") else "winner-sc1-nopin" if sb_.get("xtpc_scenario1")
                     else "winner-str0-rescuepick" if sb_["strength"] == 0.0 else "winner-plain-lasso")
                winner[w] += 1
                print(f"    {a.arm} picked         : {brow(t, sb_)}  ({w})")
    print(f"\nclasses: {dict(cls)}")
    print(f"mis-pick winners: {dict(winner)}")
    print(f"truth-bundle flags lost (not identity changes): {dict(flag_lost)}")
    print(f"truth-bundle flags gained (not identity changes): {dict(flag_gain)}")
    for k in sorted(metrics):
        print(f"  median {k}: {statistics.median(metrics[k]):.3f} (n={len(metrics[k])})")
    print(f"\nnew phantoms (objective negatives rejected on {a.base}, auto on {a.arm}): {len(new_phantoms)}")
    for r in new_phantoms:
        print(r)


if __name__ == "__main__":
    main()
