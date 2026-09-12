#!/usr/bin/env python3
"""doc pdvd/96 -- verify the confirmation arm against the measurement arm's own twin.

Same role as d95_confirm.py, and the same reason for existing: the house form of proof A ("PRE +
the MEASURED arm's TLA vs POST unset -> 0 lines") is only partly available, because the
measurement arm runs the wide DIAGNOSTIC radius (40 cm) while production carries the tight one
(10 cm) -- and, if the pre-registered rule selects scope 2, a different scope as well.

So proof A proves the file compiles to production's key set, and this script closes the
remaining gap by a route that is stronger than a config diff:

  1. every cell in the measurement arm carries d_stop_cm AND own_blob, and the cell table is
     emitted for every in-region cell REGARDLESS of the scope filter (the C++ pushes the row
     before the accumulators are gated).  So the table from a single scope-1 arm contains
     everything needed to reconstruct scope 0, 1 and 2 exactly;
  2. filtering that table to d_stop_cm <= R and the scope's own predicate yields precisely the
     cell set a production run at that R and scope would sum;
  3. this script compares that prediction, item by item, against what the confirmation arm
     ACTUALLY wrote after running the flipped file with no TLA at all.

A match on every candidate means production runs the reading this doc measured.  A mismatch means
it does not, and the flip does not stand -- there is no third outcome and no tolerance to tune:
the comparison is exact arithmetic on both sides.

THE CLAMP IS PART OF THE PREDICATE.  At scope > 0 the C++ clamps a negative muon prediction to 0
before subtracting (pred_mu < 0 is unphysical and only inflates).  The offline side must clamp
identically or the twin is comparing two different estimators -- so `predict` applies it exactly
when scope > 0, and not at all when scope == 0.
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


def scope_mask(own, scope):
    """The C++ predicate: scope 1 = own != 0 (main + admitted companions), 2 = own & 1 (main)."""
    if scope <= 0:
        return np.ones(len(own), dtype=bool)
    if scope >= 2:
        return (own & 1) != 0
    return own != 0


def predict(arm, R, scope):
    """The region sums a run at radius R and this scope would produce, from the wide arm's cells."""
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
        if "own_blob" not in c:
            sys.exit("%s has no own_blob column -- is michel_q2d_cells on?" % arm)
        cid = c["cluster_id"]
        for i in range(len(t["cluster_id"])):
            sel = cid == t["cluster_id"][i]
            if not sel.any():
                continue
            ch = np.asarray(c["charge"])[sel]; pm = np.asarray(c["pred_mu"])[sel]
            pa = np.asarray(c["pred_all"])[sel]; xs = np.asarray(c["xshared"])[sel]
            pl = np.asarray(c["plane"])[sel]; ds = np.asarray(c["d_stop_cm"])[sel]
            own = np.asarray(c["own_blob"])[sel]
            # the clamp rides with the scope, exactly as the C++ does
            pm_r = np.maximum(pm, 0.0) if scope > 0 else pm
            con = np.where(xs > 0, np.maximum(pa - pm_r, 0.0), ch - pm_r)
            m = (ds >= 0) & (ds <= R) & scope_mask(own, scope)
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
    ap.add_argument("--arm", default="p96vscope", help="the wide measurement arm (the twin's source)")
    ap.add_argument("--confirm", default="p96vprod", help="the confirmation arm on the flipped file")
    ap.add_argument("--radius", type=float, default=10.0, help="production's region radius")
    ap.add_argument("--scope", type=int, default=1, help="production's region scope (0/1/2)")
    a = ap.parse_args()

    P = predict(a.arm, a.radius, a.scope)
    A = actual(a.confirm)
    print("twin prediction from %s at R=%.1f scope=%d: %d candidates" % (a.arm, a.radius, a.scope, len(P)))
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
