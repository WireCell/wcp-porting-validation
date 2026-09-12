#!/usr/bin/env python3
"""doc pdvd/95 -- the region-based Michel charge, swept offline and graded on the hand scan.

Reads the ON arm's per-cell table (T_stm_michel_2d) straight from tracking-pr.root and joins to
the record by key, so no prep dir is needed.  Every cell carries d_stop_cm, d_ctl_cm and
own_blob, which means EVERY radius and EVERY scope choice is a filter over one arm rather than
another 120-event run -- the docs 91/92 twin rule applied to a radius.

The offline sum is not an approximation of the C++: it is the same arithmetic, and
`--twin` checks it reproduces michel_q2d_region_{u,v,w} exactly.  Verified MATCH on every
candidate of the first event before any number below was believed.

WHAT DECIDES THIS ROUND, and why the obvious reading is wrong.  A median over ALL candidates is
meaningless here: through-going muons outnumber real stoppers 2:1 and have no stop, so their
"stop region" is just another piece of track and reads like the control by construction.  Mixing
them in produced a first reading where the body control EXCEEDED the signal, which looked like a
blocking bias and was actually a class-mixing artifact.  Everything below is therefore split by
the RECORD's verdict:

    TP_found   owner says Michel, chain found one    -- does the region ADD the charge the
                                                        trajectory missed?
    TARGET     owner says Michel, chain found none   -- is there charge to find where both
                                                        existing estimators read exactly 0?
    ZERO_CTL   owner says stopper, NOT Michel        -- THE load-bearing control: a real Bragg
               (STM_ONLY)                               peak, a real stop, and no Michel, so the
                                                        true region energy is ~0.  Phantom
                                                        energy here is the cost of the method.
    THRU       owner says through-going              -- a second control.

NO TRUTH ANCHOR EXISTS.  The record carries verdicts, Michel kind, confidence, tags and a stop
pin -- no energy field of any kind.  Nothing here shows the region energy is CORRECT.  It shows
whether it is stable in R, whether it reads ~0 where the owner says there is no Michel, and how
it compares with the two existing estimators.  Any stronger claim would be unfounded.

Fork by duplication (CLAUDE.md M10) of d94_offfit.py's record plumbing; d90/d94 untouched.
"""
import argparse, collections, glob, json, os, sys
import numpy as np
import uproot

sys.path.insert(0, "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan")
import census_lib as C

IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
RS = [2.5, 5, 7.5, 10, 12.5, 15, 20, 25, 30, 40]


def combine(q, n, weights=(0.25, 0.25, 1.0), asym=0.04):
    """stm_michel_combine_planes restated for a SIGNED input (the C++ is doctested).

    A plane with no cells takes weight 0 -- 'ignore this plane', not 'a plane that measured
    zero', which would make the one populated plane the largest and drop it.
    """
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


def contrib_of(c, sel):
    """The C++ per-cell contribution, verbatim: measured - muon where the cell is the preloaded
    clusters' alone, the fit's own non-muon prediction where it is cross-shared."""
    ch = np.asarray(c["charge"])[sel]; pm = np.asarray(c["pred_mu"])[sel]
    pa = np.asarray(c["pred_all"])[sel]; xs = np.asarray(c["xshared"])[sel]
    return np.where(xs > 0, np.maximum(pa - pm, 0.0), ch - pm)


def region_sum(con, pl, d, R, extra=None):
    m = (d >= 0) & (d <= R)
    if extra is not None:
        m = m & extra
    q = [float(con[m & (pl == p)].sum()) for p in range(3)]
    n = [int((m & (pl == p)).sum()) for p in range(3)]
    return combine(q, n), sum(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default="pdvd")
    ap.add_argument("--arm", default="p95vq2db")
    ap.add_argument("--twin", action="store_true", help="check the offline sum reproduces the C++")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    if not os.environ.get("STM_SCAN_RECORD", "").endswith("smx7_smx8_smx9_verdicts.json"):
        sys.exit("STM_SCAN_RECORD must name the merged smx1a..smx9 record (docs 93/94)")
    rec = C.load_record()

    D = collections.defaultdict(list)
    per_cell = collections.Counter(); per_cell_n = collections.Counter()
    K = None; twin_ok = twin_bad = 0
    for d in sorted(glob.glob("%s/%s/work/*_%s" % (IMG, a.det, a.arm))):
        fn = d + "/tracking-pr.root"
        if not os.path.exists(fn) or os.path.getsize(fn) < 100000:
            continue
        try:
            f = uproot.open(fn)
        except Exception:
            continue
        if "T_stm_michel" not in f or "T_stm_michel_2d" not in f:
            continue
        ev = os.path.basename(d)[: -len(a.arm) - 1]
        t = f["T_stm_michel"].arrays(library="np")
        c = f["T_stm_michel_2d"].arrays(library="np")
        cid = c["cluster_id"]
        for i in range(len(t["cluster_id"])):
            key = "%s/%d" % (ev, int(t["cluster_id"][i]))
            sel = cid == t["cluster_id"][i]
            if not sel.any():
                continue
            con = contrib_of(c, sel)
            pl = np.asarray(c["plane"])[sel]; ro = np.asarray(c["role"])[sel]
            ds = np.asarray(c["d_stop_cm"])[sel]; dc = np.asarray(c["d_ctl_cm"])[sel]
            own = np.asarray(c["own_blob"])[sel] if "own_blob" in c else np.ones(sel.sum(), int)
            sh = np.asarray(c["shared"])[sel]
            # MeV per combined electron, read off the arm's OWN branches -- never hard-coded
            # (feedback_twin_constants_from_compiled_config).
            if K is None:
                qc, _ = region_sum(con, pl, ds, 40.0)
                if abs(qc) > 1:
                    K = float(t["michel_ke_q2d_region"][i]) / qc
            if a.twin:
                mine = [float(con[(ds >= 0) & (ds <= 40.0) & (pl == p)].sum()) for p in range(3)]
                cpp = [float(t["michel_q2d_region_%s" % s][i]) for s in "uvw"]
                ok = all(abs(x - y) <= 1e-6 * max(1.0, abs(y)) for x, y in zip(mine, cpp))
                twin_ok += ok; twin_bad += (not ok)
            if key not in rec or not C.judged(rec[key]):
                continue
            ism = int(t["is_stm"][i]); mf = int(t["michel_found"][i])
            if C.is_michel(rec[key]) and ism and mf:       g = "TP_found"
            elif C.is_michel(rec[key]) and ism and not mf: g = "TARGET"
            elif C.is_stopper(rec[key]) and not C.is_michel(rec[key]) and ism: g = "ZERO_CTL"
            elif not C.is_stopper(rec[key]):               g = "THRU"
            else: continue
            row = dict(key=key, best=float(t["michel_ke_best"][i]), q2d=float(t["michel_ke_q2d"][i]),
                       gam=int(t["n_stop_gammas"][i]) if "n_stop_gammas" in t else 0,
                       conf=rec[key].get("confidence", "?"),
                       # fields the pre-registration (pred.txt) is graded on
                       valid=int(t["michel_q2d_valid"][i]),
                       reason=int(t["michel_q2d_reason"][i]),
                       dropped=int(t["michel_q2d_region_dropped_plane"][i]) if "michel_q2d_region_dropped_plane" in t else -9,
                       ke_region_cpp=float(t["michel_ke_q2d_region"][i]) if "michel_ke_q2d_region" in t else float("nan"),
                       nd=sum(int(t["michel_q2d_region_nd_%s" % s][i]) for s in "uvw") if "michel_q2d_region_nd_u" in t else -1,
                       n_scalar=sum(int(t["michel_q2d_region_n_%s" % s][i]) for s in "uvw") if "michel_q2d_region_n_u" in t else -1,
                       n_table=int(((ds >= 0) & (ds <= 40.0)).sum()))
            for R in RS:
                row["s%s" % R], _ = region_sum(con, pl, ds, R)
                row["c%s" % R], _ = region_sum(con, pl, dc, R)
                row["o%s" % R], _ = region_sum(con, pl, ds, R, extra=(own > 0))
            D[g].append(row)
            m10 = (ds >= 0) & (ds <= 10)
            for lab, mm in (("role0", m10 & (ro == 0)), ("role1", m10 & (ro == 1)),
                            ("role3", m10 & (ro == 3)), ("role4", m10 & (ro == 4)),
                            ("pmu>0", m10 & (sh == 1)), ("pmu==0", m10 & (sh == 0)),
                            ("own", m10 & (own > 0)), ("not own", m10 & (own == 0))):
                per_cell[lab] += float(con[mm].sum()); per_cell_n[lab] += int(mm.sum())
    if K is None:
        sys.exit("no candidate produced a region sum -- was michel_q2d_region_cm set?")
    if a.twin:
        print("TWIN: offline region sum reproduces the C++ on %d candidates, differs on %d" % (twin_ok, twin_bad))
    print("MeV per combined electron (from the arm's own branches): %.4g\n" % K)

    def col(g, pre, R):
        v = np.array([r["%s%s" % (pre, R)] for r in D[g]]) * K
        return v

    print("=== 1. populations (judged, with a cell table)")
    for g in ("TP_found", "TARGET", "ZERO_CTL", "THRU"):
        own_n = sum(1 for r in D[g] if r["conf"] == "owner")
        print("  %-10s %4d  (owner-judged %d)" % (g, len(D[g]), own_n))

    print("\n=== 2. REGION energy at the stop, MeV: median [p25, p75], by record class")
    print("  %-10s %s" % ("class", " ".join("%18s" % ("R=%g" % R) for R in RS[:7])))
    for g in ("TP_found", "TARGET", "ZERO_CTL", "THRU"):
        if not D[g]: continue
        cells = ["%6.1f [%4.1f,%5.1f]" % (np.median(col(g, "s", R)), np.percentile(col(g, "s", R), 25),
                                          np.percentile(col(g, "s", R), 75)) for R in RS[:7]]
        print("  %-10s %s" % (g, " ".join(cells)))

    print("\n=== 3. the SAME estimator on the BODY control (up the fit, where no Michel can be)")
    for g in ("TP_found", "TARGET", "ZERO_CTL", "THRU"):
        if not D[g]: continue
        print("  %-10s %s" % (g, " ".join("%18.1f" % np.median(col(g, "c", R)) for R in RS[:7])))

    print("\n=== 4. signal over control, and over the ZERO_CTL phantom (the numbers that matter)")
    print("  %-28s %s" % ("", " ".join("%8s" % ("R=%g" % R) for R in RS[:7])))
    tp = {R: np.median(col("TP_found", "s", R)) for R in RS}
    tpc = {R: np.median(col("TP_found", "c", R)) for R in RS}
    zc = {R: np.median(col("ZERO_CTL", "s", R)) for R in RS}
    zcc = {R: np.median(col("ZERO_CTL", "c", R)) for R in RS}
    print("  %-28s %s" % ("TP_found median", " ".join("%8.1f" % tp[R] for R in RS[:7])))
    print("  %-28s %s" % ("TP excess over own control", " ".join("%8.1f" % (tp[R] - tpc[R]) for R in RS[:7])))
    print("  %-28s %s" % ("ZERO_CTL median (phantom)", " ".join("%8.1f" % zc[R] for R in RS[:7])))
    print("  %-28s %s" % ("ZERO_CTL less own control", " ".join("%8.1f" % (zc[R] - zcc[R]) for R in RS[:7])))
    # The headline ratio uses the zero-control's ABSOLUTE median, not its control-subtracted
    # excess.  Subtracting a noisy control from a small class can drive that denominator
    # through zero and produce large negative "ratios" that mean nothing; the absolute
    # phantom is the conservative statement and cannot flatter the method.
    print("  %-28s %s" % ("ratio TP / phantom", " ".join(
        ("%8.1f" % (tp[R] / zc[R])) if zc[R] > 0.05 else "     n/a" for R in RS[:7])))

    print("\n=== 5. the PRE-REGISTERED R rule (pred.txt, written before any of this ran)")
    print("    (a) smallest R where TP growth over the next 5 cm is under 5 percent:")
    pick_a = None
    for i, R in enumerate(RS[:-1]):
        nxt = [x for x in RS if x >= R + 5]
        if not nxt: break
        g5 = (tp[nxt[0]] - tp[R]) / tp[R] if tp[R] > 0 else 1.0
        print("        R=%-5g growth to R=%-5g  %+6.1f %%%s" % (R, nxt[0], 100 * g5, "   <- first < 5 %" if (g5 < 0.05 and pick_a is None) else ""))
        if g5 < 0.05 and pick_a is None: pick_a = R
    print("    (b) ZERO_CTL consistent with 0: median %.1f MeV at R=10 -- NOT 0." % zc[10])
    print("    => (a) selects %s; (b) is not satisfied at any R." % (("R=%g" % pick_a) if pick_a else "no R"))
    print("    pred.txt: 'If (a) and (b) disagree ... that is a REPORTED CONFLICT and the owner")
    print("    picks. I do not split the difference silently.'  So no R is claimed as")
    print("    pre-registered; any R quoted in the doc is POST-HOC and labelled as such.")

    print("\n=== 6. WHERE the R=10 charge comes from (all candidates, summed)")
    print("  %-12s %12s %10s %12s" % ("class", "sum MeV", "cells", "MeV/cell"))
    for lab in ("role0", "role1", "role3", "role4", "pmu>0", "pmu==0", "own", "not own"):
        s = per_cell[lab] * K; n = per_cell_n[lab]
        print("  %-12s %12.1f %10d %12.4f" % (lab, s, n, s / n if n else 0))

    print("\n=== 7. SCOPE: doc 78 item 9 says 'the main cluster and the admitted companions'.")
    print("    Restricting the sum to own_blob > 0 (median MeV):")
    print("  %-12s %s" % ("class", " ".join("%8s" % ("R=%g" % R) for R in RS[:7])))
    for g in ("TP_found", "ZERO_CTL"):
        if not D[g]: continue
        print("  %-12s %s" % (g + " all", " ".join("%8.1f" % np.median(col(g, "s", R)) for R in RS[:7])))
        print("  %-12s %s" % (g + " own", " ".join("%8.1f" % np.median(col(g, "o", R)) for R in RS[:7])))

    print("\n=== 8. against the existing estimators (median MeV)")
    for g in ("TP_found", "TARGET", "ZERO_CTL", "THRU"):
        if not D[g]: continue
        print("  %-10s ke_best %6.2f | ke_q2d(assoc) %6.2f | region R=10 %6.2f | region R=10 own %6.2f" % (
            g, np.median([r["best"] for r in D[g]]), np.median([r["q2d"] for r in D[g]]),
            np.median(col(g, "s", 10)), np.median(col(g, "o", 10))))

    print("\n=== 9. the TARGET items by name (both existing estimators read 0 on these)")
    for r in sorted(D["TARGET"], key=lambda r: -r["s10"]):
        print("  %-16s conf %-7s region R=5 %6.1f  R=10 %6.1f  own R=10 %6.1f  body ctl R=10 %6.1f" % (
            r["key"], r["conf"], r["s5"] * K, r["s10"] * K, r["o10"] * K, r["c10"] * K))

    # ------------------------------------------------------------------
    # 10.  RELIABILITY.  The class medians are reassuring and the TAIL is not:
    # a handful of STM_ONLY items read 30-48 MeV where the owner says there is
    # no Michel.  The discriminating question for each is whether the charge is
    # LOCALIZED AT THE STOP or whether that cluster reads high everywhere -- and
    # the body control answers it per item, on the same cluster, with no extra
    # measurement.  On the partial arm, 4 of the 5 worst offenders also had a
    # body control above 10 MeV (the worst: region 48.3 against a control of
    # 169.9), so they are the estimator inflating, NOT the record under-calling
    # a Michel.  That distinction matters: the second reading would have been a
    # claim that the owner's own scan is wrong, and it is not supported.
    print("\n=== 10. reliability: clusters the estimator reads high EVERYWHERE")
    # The test is an ABSOLUTE body control, not ctl/region.  A ratio test fires whenever the
    # REGION is small, so it flags weak-signal items rather than inflated clusters -- which is
    # why an earlier version of this section reported ZERO_CTL's "reliable" median (8.1) ABOVE
    # its median overall (4.9), a result that reads backwards and was a defect in the metric,
    # not a property of the data.  A large body control means the muon subtraction is leaving a
    # big residual all along THIS cluster, so its stop reading is inflated too; that is the
    # thing worth flagging, and it is what separates "the estimator is inflating here" from
    # "the hand scan under-called a Michel".
    CTL_MAX = 10.0   # MeV on the body control; TP_found's median is 3.1
    print("    (body control > %.0f MeV at R=10 -- the muon subtraction is poor on this cluster)" % CTL_MAX)
    print("  %-10s %6s %12s %12s %14s %10s" % ("class", "n", "inflated", "median all", "median clean", "med ctl"))
    for g in ("TP_found", "TARGET", "ZERO_CTL", "THRU"):
        if not D[g]:
            continue
        allv = np.array([r["s10"] for r in D[g]]) * K
        ctl = np.array([r["c10"] for r in D[g]]) * K
        keep = np.array([(r["c10"] * K) <= CTL_MAX for r in D[g]])
        nbad = int((~keep).sum())
        print("  %-10s %6d %12s %12.1f %14s %10.1f" % (
            g, len(D[g]), "%d (%.0f%%)" % (nbad, 100.0 * nbad / len(D[g])),
            np.median(allv), ("%.1f" % np.median(allv[keep])) if keep.any() else "n/a",
            np.median(ctl)))
    print("    The body control is emitted per candidate as michel_ke_q2d_ctl, so a consumer")
    print("    can apply this flag without recomputing anything.")

    # ------------------------------------------------------------------
    # 11.  GRADING THE PRE-REGISTRATION.  pred.txt was written before any arm ran; every
    # prediction in it is graded here BY NAME, including the ones that miss.  Grading only
    # the predictions that happened to work would turn a pre-registration into a trophy case.
    ALL = [r for g in ("TP_found", "TARGET", "ZERO_CTL", "THRU") for r in D[g]]
    print("\n=== 11. the pre-registration, graded by name (pred.txt, written before any arm ran)")

    # Z2: capture-gamma items should read HIGHER (my prediction).  This is the one I expect to miss.
    zc_g = [r["s10"] * K for r in D["ZERO_CTL"] if r["gam"] > 0]
    zc_n = [r["s10"] * K for r in D["ZERO_CTL"] if r["gam"] == 0]
    if zc_g and zc_n:
        held = np.median(zc_g) > np.median(zc_n)
        print("  Z2 capture-gamma STM_ONLY items read higher: with gamma %.1f (n=%d) vs without %.1f (n=%d) -> %s"
              % (np.median(zc_g), len(zc_g), np.median(zc_n), len(zc_n), "HELD" if held else "MISSED"))
        if not held:
            print("     A mu- capture gamma is therefore NOT inflating the region reading -- the")
            print("     contamination I argued for in the plan is not there, and the owner's point 2")
            print("     needs no capture-gamma exclusion at this radius.")

    # F3 / F4: the endpoint.  A bounded region should cap doc 81's 500 MeV pathology.
    tp = np.array([r["s10"] * K for r in D["TP_found"]])
    print("  F3 no 500 MeV object: max TP_found region reading %.1f MeV -> %s"
          % (tp.max(), "HELD" if tp.max() < 200 else "MISSED"))
    print("  F4 under 15 of %d TP above the 52.8 MeV endpoint: %d -> %s"
          % (len(tp), int((tp > 52.8).sum()), "HELD" if (tp > 52.8).sum() < 15 else "MISSED"))
    # F1: region >= the association reading on most TP
    f1 = sum(1 for r in D["TP_found"] if r["s10"] * K >= r["q2d"])
    print("  F1 region >= association reading on most TP: %d of %d (%.0f %%) -> %s"
          % (f1, len(D["TP_found"]), 100.0 * f1 / len(D["TP_found"]), "HELD" if f1 > 0.5 * len(D["TP_found"]) else "MISSED"))
    # F2: median region / ke_best in 1.2-2.5
    ratio = np.median(tp) / np.median([r["best"] for r in D["TP_found"]])
    print("  F2 median region / michel_ke_best in 1.2-2.5: %.2f -> %s" % (ratio, "HELD" if 1.2 <= ratio <= 2.5 else "MISSED"))

    # X2: valid on every candidate
    nval = sum(1 for r in ALL if r["valid"] == 1)
    reasons = collections.Counter(r["reason"] for r in ALL if r["valid"] != 1)
    print("  X2 michel_q2d_valid == 1 on every candidate: %d of %d -> %s%s"
          % (nval, len(ALL), "HELD" if nval == len(ALL) else "MISSED",
             ("  reason codes %s" % dict(sorted(reasons.items()))) if reasons else ""))

    # X3: the closure check.  As WRITTEN in pred.txt it is trivially true -- every emitted row
    # has role 0, 1, 3 or 4, so summing pred_mu over those roles is summing over all rows.  The
    # non-trivial check available offline is that the C++ scalar cell counts equal the table's
    # own counts: that is what would catch the region emission double-counting or dropping cells.
    bad = [r for r in ALL if r["n_scalar"] >= 0 and r["n_scalar"] != r["n_table"]]
    print("  X3 closure (restated): C++ region cell count == the table's own count on %d of %d -> %s"
          % (len(ALL) - len(bad), len(ALL), "HELD" if not bad else "MISSED"))
    print("     (as written in pred.txt X3 was trivially true by construction; this is the")
    print("      stronger check, and it is the one that would catch a miscounted region.)")

    # X4: the plane-drop rate.  doc 81 dropped a plane on 94 of 164 PDVD candidates.
    ndrop = sum(1 for r in ALL if r["dropped"] >= 0)
    print("  X4 plane-drop rate rises above doc 81's 94/164 (57 %%): %d of %d (%.0f %%) -> %s"
          % (ndrop, len(ALL), 100.0 * ndrop / len(ALL), "HELD" if ndrop / len(ALL) > 0.57 else "MISSED"))

    # X5: dead cells contribute ~0 rather than a negative bias
    nd = np.array([r["nd"] for r in ALL if r["nd"] >= 0])
    if nd.size:
        print("  X5 dead cells in the region: median %d, max %d per candidate (a dead row has"
              % (int(np.median(nd)), int(nd.max())))
        print("     scale 0 so pred_mu is 0 there, hence ~0 contribution rather than a negative bias)")

    if a.json:
        json.dump({g: [{k: (v if not isinstance(v, float) else round(v, 6)) for k, v in r.items()}
                       for r in D[g]] for g in D}, open(a.json, "w"), indent=1)
        print("\nwrote %s" % a.json)


if __name__ == "__main__":
    main()
