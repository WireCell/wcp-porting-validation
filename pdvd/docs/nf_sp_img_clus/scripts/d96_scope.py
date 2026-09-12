#!/usr/bin/env python3
"""doc pdvd/96 -- the region's SCOPE, swept offline from one arm and graded on the hand scan.

Reads the scope arm's per-cell table (T_stm_michel_2d) straight from tracking-pr.root and joins
to the record by key.  The C++ pushes a cell row BEFORE the accumulators are gated, so a single
scope-1 arm's table contains every in-region cell with its own_blob bits -- which means scope 0,
1 and 2 are all filters over ONE arm rather than three 120-event runs.  Same twin rule docs 91/92
applied to a radius, now applied to a scope.

WHAT CHANGED SINCE doc 95, and why its sec 4.3 numbers are predictions here rather than baselines:

  * own_blob as doc 95 shipped it carries bit 1 (the main cluster) and bit 2 (an admitted
    UNFITTED companion).  n_dot_clusters_unfit is 0 on all 596 PDVD candidates, so bit 2 never
    fired and the column was main-cluster-only.  A companion that DID produce segments is
    preloaded, its charge IS in the union maps, and it set no bit at all.  So "own_blob > 0" was
    never doc 78 item 9's "the main cluster and the admitted companions", which is what doc 95
    sec 9 item 1 claimed.  At scope > 0 bit 4 marks a preloaded fitted companion.
  * the own test now sweeps every (face, wire) the channel maps to.  Testing fw.front() alone was
    cosmetic for a diagnostic column and a systematic loss on wrapped channels once it filters.
  * a negative muon prediction is clamped to 0 before subtracting.

THE MEDIANS ARE TAKEN ON THE QUANTITY PRODUCTION PUBLISHES.  stm_michel_charge_to_energy_model
returns 0.0 for dQ <= 0, so michel_ke_q2d_region is floored at zero while a signed offline sum is
not -- and a NARROWED scope pushes more candidates to or below zero, so the floor bites harder
here than it did in doc 95.  Grading the rule on a signed quantity would select a scope on a
number no downstream consumer can see.  The signed version is printed alongside, as a diagnostic.

NO TRUTH ANCHOR EXISTS.  The record carries verdicts, Michel kind, confidence, tags and a stop
pin -- no energy field of any kind.  A better ratio here is better SELECTION, never evidence that
either number is CORRECT.

Fork by duplication (CLAUDE.md M10) of d95_region.py; d95 untouched.
"""
import argparse, collections, glob, json, os, sys
import numpy as np
import uproot

sys.path.insert(0, "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan")
import census_lib as C

IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
RS = [5, 10, 15, 20]
RMAX = 40.0
SCOPES = (0, 1, 2)


def combine(q, n, weights=(0.25, 0.25, 1.0), asym=0.04):
    """stm_michel_combine_planes restated for a SIGNED input (the C++ is doctested)."""
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
    """The C++ predicate: 1 = own != 0 (main + admitted companions), 2 = own & 1 (main alone)."""
    if scope <= 0:
        return np.ones(len(own), dtype=bool)
    if scope >= 2:
        return (own & 1) != 0
    return own != 0


def contrib_of(c, sel, scope):
    """The C++ per-cell contribution.  The clamp rides with the scope, exactly as the C++ does:
    at scope > 0 a negative muon prediction is treated as 0 before subtracting."""
    ch = np.asarray(c["charge"])[sel]; pm = np.asarray(c["pred_mu"])[sel]
    pa = np.asarray(c["pred_all"])[sel]; xs = np.asarray(c["xshared"])[sel]
    pm_r = np.maximum(pm, 0.0) if scope > 0 else pm
    return np.where(xs > 0, np.maximum(pa - pm_r, 0.0), ch - pm_r)


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
    ap.add_argument("--arm", default="p96vscope", help="the scope-1 measurement arm at R=40")
    ap.add_argument("--twin", action="store_true", help="check the offline sum reproduces the C++")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    if not os.environ.get("STM_SCAN_RECORD", "").endswith("smx7_smx8_smx9_verdicts.json"):
        sys.exit("STM_SCAN_RECORD must name the merged smx1a..smx9 record (docs 93/94)")
    rec = C.load_record()

    D = collections.defaultdict(list)
    Ks = []
    twin_ok = twin_bad = 0
    own_hist = collections.Counter()
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
        if "own_blob" not in c:
            sys.exit("%s has no own_blob column -- is michel_q2d_cells on?" % a.arm)
        cid = c["cluster_id"]
        for i in range(len(t["cluster_id"])):
            key = "%s/%d" % (ev, int(t["cluster_id"][i]))
            sel = cid == t["cluster_id"][i]
            if not sel.any():
                continue
            pl = np.asarray(c["plane"])[sel]
            ds = np.asarray(c["d_stop_cm"])[sel]; dc = np.asarray(c["d_ctl_cm"])[sel]
            own = np.asarray(c["own_blob"])[sel]
            for v in np.unique(own):
                own_hist[int(v)] += int((own == v).sum())
            con = {s: contrib_of(c, sel, s) for s in SCOPES}

            # MeV per combined electron, read off the arm's OWN branches (never hard-coded,
            # feedback_twin_constants_from_compiled_config).  DEFECT FIXED vs d95_region.py:
            # that script took the FIRST candidate with |q| > 1, so a candidate whose combined
            # charge was NEGATIVE -- whose energy the C++ floors to 0.0 -- would have set
            # K = 0 and silently zeroed every number in the report.  It survived only because
            # the first candidate happened to be positive.  Here K is a median over every
            # candidate with a strictly positive sum, and its spread is printed.
            q40, _ = region_sum(con[1], pl, ds, RMAX, extra=scope_mask(own, 1))
            ke40 = float(t["michel_ke_q2d_region"][i]) if "michel_ke_q2d_region" in t else 0.0
            if q40 > 1 and ke40 > 0:
                Ks.append(ke40 / q40)

            if a.twin:
                mine = [float(con[1][(ds >= 0) & (ds <= RMAX) & scope_mask(own, 1) & (pl == p)].sum())
                        for p in range(3)]
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
                       conf=rec[key].get("confidence", "?"),
                       valid=int(t["michel_q2d_valid"][i]))
            for s in SCOPES:
                m_s = scope_mask(own, s)
                for R in RS:
                    row["s%d_%g" % (s, R)], row["n%d_%g" % (s, R)] = region_sum(con[s], pl, ds, R, extra=m_s)
                row["c%d_10" % s], _ = region_sum(con[s], pl, dc, 10, extra=m_s)
            D[g].append(row)

    if not Ks:
        sys.exit("no candidate produced a positive region sum -- was michel_q2d_region_cm set?")
    K = float(np.median(Ks))
    spread = (max(Ks) - min(Ks)) / K if K else float("inf")
    # K is MeV per combined electron and is a single CONSTANT (doc 95 measured a relative spread
    # of 3e-15 over 120 candidates).  A large spread is not a curiosity to print -- it means the
    # C++ energy was produced by a DIFFERENT cell set than the offline sum, and every MeV below
    # would be silently mis-scaled.  Caught exactly this way by a smoke run against doc 95's arm,
    # which was built at scope 0 while the offline filter here is scope 1: K came out 5.31e-05
    # with spread 4.2e+02 instead of 4.27e-05, inflating every energy by 24 %.  Abort, loudly.
    if spread > 1e-6:
        sys.exit("K is NOT constant: relative spread %.3g over %d candidates (median %.6g).\n"
                 "The C++ michel_ke_q2d_region was produced by a different cell set than the sum\n"
                 "computed here -- almost always because --arm names an arm run at a DIFFERENT\n"
                 "scope (or with no scope knob at all) than the filter this script applies.\n"
                 "Run it against the scope-1 measurement arm." % (spread, len(Ks), K))
    if a.twin:
        print("TWIN: offline scope-1 sum reproduces the C++ on %d candidates, differs on %d" % (twin_ok, twin_bad))
    print("MeV per combined electron: %.6g  (median of %d candidates, full spread %.2g relative)\n" % (K, len(Ks), spread))

    print("=== 0. own_blob in the scope arm -- the positive control for the whole round")
    tot = sum(own_hist.values()) or 1
    for v in sorted(own_hist):
        what = {0: "neither", 1: "main cluster", 2: "unfitted companion", 4: "FITTED companion"}.get(v, "combination")
        print("  own_blob=%d  %9d cells (%5.1f %%)  %s" % (v, own_hist[v], 100.0 * own_hist[v] / tot, what))
    if not any(v & 4 for v in own_hist):
        print("  NOTE bit 4 never fires: no preloaded FITTED companion covers any region cell, so")
        print("  scope 1 and scope 2 are the SAME SET here and 'own' means the main cluster alone.")
        print("  The doc must say that rather than claim doc 78 item 9's scope.")

    def mev(v, floored=True):
        """Production's convention: the energy model returns 0.0 for a non-positive charge."""
        x = np.asarray(v, dtype=float) * K
        return np.maximum(x, 0.0) if floored else x

    print("\n=== 1. populations (judged, with a cell table)")
    for g in ("TP_found", "TARGET", "ZERO_CTL", "THRU"):
        own_n = sum(1 for r in D[g] if r["conf"] == "owner")
        print("  %-10s %4d  (owner-judged %d)" % (g, len(D[g]), own_n))

    print("\n=== 2. region energy at R=10 by scope, MeV -- FLOORED (what production publishes)")
    print("  %-10s %14s %14s %14s" % ("class", "scope 0 (all)", "scope 1 (own)", "scope 2 (main)"))
    med = {}
    for g in ("TP_found", "TARGET", "ZERO_CTL", "THRU"):
        if not D[g]: continue
        cells = []
        for s in SCOPES:
            v = mev([r["s%d_10" % s] for r in D[g]])
            med[(g, s)] = float(np.median(v))
            cells.append("%14.2f" % med[(g, s)])
        print("  %-10s %s" % (g, " ".join(cells)))

    print("\n=== 2b. the same, SIGNED (diagnostic only -- NOT what the rule is graded on)")
    for g in ("TP_found", "ZERO_CTL"):
        if not D[g]: continue
        cells = ["%14.2f" % np.median(mev([r["s%d_10" % s] for r in D[g]], floored=False)) for s in SCOPES]
        print("  %-10s %s" % (g, " ".join(cells)))
    for g in ("TP_found", "TARGET", "ZERO_CTL", "THRU"):
        if not D[g]: continue
        for s in SCOPES:
            neg = sum(1 for r in D[g] if r["s%d_10" % s] <= 0)
            if neg:
                print("      %-10s scope %d: %d of %d candidates at or below zero (floored to 0.00)"
                      % (g, s, neg, len(D[g])))

    print("\n=== 3. the PRE-REGISTERED scope rule (pred.txt, written before any arm ran)")
    print("    PRIMARY  ratio = median(TP_found) / median(ZERO_CTL) at R=10, floored")
    print("    GUARD    median(TP_found) must not fall below 28 MeV")
    print("    TIE-BREAK within 10 % on the ratio, prefer scope 1 (doc 78 item 9's own scope)")
    ratio = {}
    for s in SCOPES:
        tp = med.get(("TP_found", s), 0.0); zc = med.get(("ZERO_CTL", s), 0.0)
        ratio[s] = (tp / zc) if zc > 0.05 else float("nan")
        print("    scope %d: TP %6.2f  phantom %5.2f  ratio %6.2f  guard(>=28) %s"
              % (s, tp, zc, ratio[s], "PASS" if tp >= 28.0 else "FAIL"))
    eligible = [s for s in SCOPES if s > 0 and med.get(("TP_found", s), 0) >= 28.0]
    if not eligible:
        print("    => CONFLICT: the guard fails for every scope > 0.  pred.txt: 'the restriction")
        print("       costs more signal than it buys and I do NOT flip.  Reported, owner picks.'")
        pick = None
    else:
        best = max(eligible, key=lambda s: (ratio[s] if ratio[s] == ratio[s] else -1))
        pick = best
        if 1 in eligible and ratio[1] == ratio[1] and ratio[best] == ratio[best]:
            if ratio[best] - ratio[1] <= 0.10 * ratio[best]:
                pick = 1
        print("    => best on the ratio is scope %d; tie-break applied -> SELECTED SCOPE %d" % (best, pick))
        if ratio.get(0) == ratio.get(0) and ratio[pick] <= ratio[0]:
            print("    NOTE the selected scope does NOT beat scope 0 (%.2f vs %.2f).  That is a"
                  % (ratio[pick], ratio[0]))
            print("    negative result and the doc reports it as one.")

    print("\n=== 4. against the existing estimators (median MeV, floored)")
    for g in ("TP_found", "TARGET", "ZERO_CTL", "THRU"):
        if not D[g]: continue
        print("  %-10s ke_best %6.2f | ke_q2d(assoc) %6.2f | scope0 %6.2f | scope1 %6.2f | scope2 %6.2f" % (
            g, np.median([r["best"] for r in D[g]]), np.median([r["q2d"] for r in D[g]]),
            med[(g, 0)], med[(g, 1)], med[(g, 2)]))

    print("\n=== 5. the pre-registration, graded by name")
    t1 = med.get(("TP_found", 1), 0.0); t2 = med.get(("TP_found", 2), 0.0); t0 = med.get(("TP_found", 0), 0.0)
    d12 = abs(t1 - t2) / t1 if t1 else 0.0
    print("  S1 scopes 1 and 2 within 5 %% on TP_found: %.2f vs %.2f (%.1f %%) -> %s"
          % (t1, t2, 100 * d12, "HELD" if d12 <= 0.05 else "MISSED -- fitted companions carry real charge at the stop"))
    r1 = ratio.get(1, float("nan"))
    print("  S2 the ratio improves on scope 0's and lands in 7.3-8.5: scope0 %.2f -> scope1 %.2f -> %s"
          % (ratio.get(0, float("nan")), r1, "HELD" if (r1 == r1 and 7.3 <= r1 <= 8.5) else "MISSED"))
    fall = (t0 - t1) / t0 if t0 else 0.0
    print("  S3 TP_found falls 5-15 %% from scope 0: %.2f -> %.2f (%.1f %%) -> %s"
          % (t0, t1, 100 * fall, "HELD" if 0.05 <= fall <= 0.15 else "MISSED"))
    print("  S4 (the fw fix recovers cells) and S5 (the clamp lowers sums by < 1 %) are graded in")
    print("     d96_pedestal.py, which has the per-cell populations those predictions are about.")

    print("\n=== 6. per-class cell counts at R=10 (how much the filter removes)")
    print("  %-10s %12s %12s %12s" % ("class", "scope 0", "scope 1", "scope 2"))
    for g in ("TP_found", "TARGET", "ZERO_CTL", "THRU"):
        if not D[g]: continue
        print("  %-10s %12d %12d %12d" % (g, int(np.median([r["n0_10"] for r in D[g]])),
                                          int(np.median([r["n1_10"] for r in D[g]])),
                                          int(np.median([r["n2_10"] for r in D[g]]))))

    print("\n=== 7. the TARGET items by name (both existing estimators read 0 on these)")
    for r in sorted(D["TARGET"], key=lambda r: -r["s1_10"]):
        print("  %-16s conf %-7s scope0 %6.1f  scope1 %6.1f  scope2 %6.1f  body ctl %6.1f" % (
            r["key"], r["conf"], max(r["s0_10"] * K, 0), max(r["s1_10"] * K, 0),
            max(r["s2_10"] * K, 0), max(r["c1_10"] * K, 0)))

    if a.json:
        json.dump({"K": K, "classes": {g: D[g] for g in D}}, open(a.json, "w"), indent=1, default=float)
        print("\nwrote %s" % a.json)


if __name__ == "__main__":
    main()
