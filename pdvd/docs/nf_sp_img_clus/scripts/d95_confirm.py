#!/usr/bin/env python3
"""doc pdvd/95 -- verify the confirmation arm against the measurement arm's own twin.

WHY THIS EXISTS, and what it replaces.

The house form of proof A is "PRE + the MEASURED arm's TLA vs POST unset -> 0 lines", i.e. the
flipped file compiles to exactly what the arm that measured the trade ran (doc pdvd/93).  That
form is unavailable here, and pretending otherwise would be the dishonest move:

  * the measurement arm p95vq2db ran michel_q2d_region_cm = 40 cm, which is the DIAGNOSTIC
    radius -- deliberately wide, so the radial profile could be swept offline from one arm;
  * production carries a TIGHT radius (the owner's constraint: the Michel sits near the stop),
    so the two key sets differ in exactly one value.

So proof A here proves the file compiles to PRODUCTION's four keys, and this script closes the
remaining gap -- that those keys were measured -- by a route that is actually stronger than a
config diff:

  1. every cell in the measurement arm carries d_stop_cm, and the offline sum reproduces the
     C++ region scalars EXACTLY (596/596 candidates, 0 differing, d95_region.py --twin).  The
     R=10 reading is therefore not an extrapolation from the R=40 arm -- it is a filter over the
     same cells the component itself summed;
  2. filtering that table to d_stop_cm <= R yields precisely the cell set a production run at
     that R would sum: role 1/3/4 cells are emitted regardless of radius, and role-0 cells
     outside R are excluded by the same predicate the C++ applies;
  3. this script compares that prediction, item by item, against what the confirmation arm
     ACTUALLY wrote after running the flipped file with no TLA at all.

A match on every candidate means production runs the reading this doc measured.  A mismatch
means it does not, and the flip does not stand -- there is no third outcome and no tolerance to
tune: the comparison is exact arithmetic on both sides.
"""
import argparse, glob, os, sys
import numpy as np
import uproot

IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"


def combine(q, n, weights=(0.25, 0.25, 1.0), asym=0.04):
    w = [0.0 if n[p] == 0 else weights[p] for p in range(3)]
    if sum(w) <= 0:
        return 0.0
    o = sorted(range(3), key=lambda p: q[p])
    med, mx = q[o[1]], q[o[2]]
    if med + mx > 0 and abs(med - mx) / (med + mx) > asym:
        keep = [o[0], o[1]]
        wk = [w[p] for p in keep]
        return sum(q[p] * w[p] for p in keep) / sum(wk) if sum(wk) > 0 else 0.0
    return sum(q[p] * w[p] for p in range(3)) / sum(w)


def predict(arm, R):
    """The region sums a run at radius R would produce, filtered from the wide arm's cells."""
    out = {}
    for d in sorted(glob.glob("%s/pdvd/work/*_%s" % (IMG, arm))):
        fn = d + "/tracking-pr.root"
        if not os.path.exists(fn) or os.path.getsize(fn) < 100000:
            continue
        try:
            f = uproot.open(fn)
        except Exception:
            continue
        if "T_stm_michel" not in f or "T_stm_michel_2d" not in f:
            continue
        ev = os.path.basename(d)[: -len(arm) - 1]
        t = f["T_stm_michel"].arrays(library="np")
        c = f["T_stm_michel_2d"].arrays(library="np")
        cid = c["cluster_id"]
        for i in range(len(t["cluster_id"])):
            sel = cid == t["cluster_id"][i]
            if not sel.any():
                continue
            ch = np.asarray(c["charge"])[sel]; pm = np.asarray(c["pred_mu"])[sel]
            pa = np.asarray(c["pred_all"])[sel]; xs = np.asarray(c["xshared"])[sel]
            pl = np.asarray(c["plane"])[sel]; ds = np.asarray(c["d_stop_cm"])[sel]
            con = np.where(xs > 0, np.maximum(pa - pm, 0.0), ch - pm)
            m = (ds >= 0) & (ds <= R)
            q = [float(con[m & (pl == p)].sum()) for p in range(3)]
            n = [int((m & (pl == p)).sum()) for p in range(3)]
            out["%s/%d" % (ev, int(t["cluster_id"][i]))] = (q, n, combine(q, n))
    return out


def actual(arm):
    out = {}
    for d in sorted(glob.glob("%s/pdvd/work/*_%s" % (IMG, arm))):
        fn = d + "/tracking-pr.root"
        if not os.path.exists(fn) or os.path.getsize(fn) < 100000:
            continue
        try:
            f = uproot.open(fn)
        except Exception:
            continue
        if "T_stm_michel" not in f:
            continue
        ev = os.path.basename(d)[: -len(arm) - 1]
        t = f["T_stm_michel"].arrays(library="np")
        if "michel_q2d_region_u" not in t:
            sys.exit("%s has no michel_q2d_region_* branches -- is the flip actually in the file?" % arm)
        for i in range(len(t["cluster_id"])):
            out["%s/%d" % (ev, int(t["cluster_id"][i]))] = (
                [float(t["michel_q2d_region_%s" % s][i]) for s in "uvw"],
                [int(t["michel_q2d_region_n_%s" % s][i]) for s in "uvw"],
                float(t["michel_q2d_region"][i]),
                float(t["michel_ke_q2d_region"][i]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="p95vq2db", help="the wide measurement arm (the twin's source)")
    ap.add_argument("--confirm", default="p95vprodb", help="the confirmation arm on the flipped file")
    ap.add_argument("--radius", type=float, default=10.0, help="production's region radius")
    a = ap.parse_args()

    P = predict(a.arm, a.radius)
    A = actual(a.confirm)
    print("twin prediction from %s filtered to R=%.1f: %d candidates" % (a.arm, a.radius, len(P)))
    print("confirmation arm %s                       : %d candidates" % (a.confirm, len(A)))

    only_p = sorted(set(P) - set(A)); only_a = sorted(set(A) - set(P))
    if only_p: print("  only in the prediction: %s" % " ".join(only_p[:10]))
    if only_a: print("  only in the arm       : %s" % " ".join(only_a[:10]))

    ok = bad = 0
    worst = []
    for k in sorted(set(P) & set(A)):
        (pq, pn, pc) = P[k]
        (aq, an, ac, ake) = A[k]
        same = (all(abs(x - y) <= 1e-6 * max(1.0, abs(y)) for x, y in zip(pq, aq))
                and pn == an
                and abs(pc - ac) <= 1e-6 * max(1.0, abs(ac)))
        if same:
            ok += 1
        else:
            bad += 1
            worst.append((k, pq, aq, pn, an, pc, ac))
    print("\n  candidates where the arm reproduces the twin EXACTLY: %d / %d" % (ok, ok + bad))
    for k, pq, aq, pn, an, pc, ac in worst[:15]:
        print("    %-16s predicted u/v/w %12.0f %12.0f %12.0f n %s comb %12.0f" % (k, pq[0], pq[1], pq[2], pn, pc))
        print("    %-16s arm       u/v/w %12.0f %12.0f %12.0f n %s comb %12.0f" % ("", aq[0], aq[1], aq[2], an, ac))
    print("\n  VERDICT: %s" % (
        "PRODUCTION RUNS THE READING THIS DOC MEASURED -- every candidate matches the twin"
        if bad == 0 and not only_p and not only_a
        else "*** THE CONFIRMATION ARM DOES NOT REPRODUCE THE TWIN -- the flip does not stand ***"))


if __name__ == "__main__":
    main()
