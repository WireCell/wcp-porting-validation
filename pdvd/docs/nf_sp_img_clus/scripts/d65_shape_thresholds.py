#!/usr/bin/env python3
"""doc pdvd/65 (T7) -- the two shape tests, re-scored offline.

(a) From the PUBLISHED per-item fields (contrast, contrast_expected, ks_mu,
    ks_flat): a 2-D grid over bragg_contrast_min x ks_margin, is_stm-style
    TP/FP/purity/efficiency/F1 against the scan's stopper verdict, judged
    items with a valid profile.  This reproduces doc 56 sec 4's one-at-a-time
    table as the first row/column and extends it jointly.
(b) The WINDOW knobs (bragg_tail_hi_cm, bragg_plateau_lo/hi_cm,
    compare_range_cm) cannot be swept from published scalars; they are
    recomputed from profile(pay) + the committed reference curve
    (prep-pdvd/dqdx_ref_pdvd.json), after an agreement gate: at the shipped
    windows the recomputation must reproduce the published contrast /
    contrast_expected (max |delta| quoted).  The live cut
    (profile_min_dqdx_frac x mip_dqdx = 0.15 x 55000) is applied as the chain
    does; ks is kslike_compare's shape-only statistic (unit-sum CDF max
    difference), re-implemented in numpy.

Nothing here moves a knob: CLAUDE.md sec 5.7 -- the operating point is the
owner's call.  Usage: python3 d65_shape_thresholds.py <prep_dir>
"""
import sys, json, collections
import numpy as np
HERE = "/home/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan"
sys.path.insert(0, HERE)
import census_lib as C

prep = sys.argv[1]
rec = C.load_record()
P, missing = C.load_payloads(prep, rec)
J = [k for k in rec if k in P and C.judged(rec[k])]
K = {k: C.is_stopper(rec[k]) for k in J}
MIP = 55000.0; LIVE = 0.15 * MIP

def score(sel):
    tp = sum(1 for k in J if sel.get(k) and K[k]); fp = sum(1 for k in J if sel.get(k) and not K[k])
    fn = sum(1 for k in J if not sel.get(k) and K[k])
    pu = tp / max(tp + fp, 1); ef = tp / max(tp + fn, 1); f1 = 2 * pu * ef / max(pu + ef, 1e-9)
    return tp, fp, fn, pu, ef, f1

# ---------------------------------------------------------------- (a) grid
print("=== (a) published fields: shape-test-only verdict, judged items with a valid profile ===")
V = {k: P[k]["verdict"] for k in J if P[k]["verdict"]["bragg_valid"] and (P[k]["verdict"]["contrast_expected"] or 0) > 0}
print("n %d, stoppers %d  (the other reject bits are NOT applied here: this is the two shape tests alone)" % (len(V), sum(K[k] for k in V)))
X = {k: v["contrast"] / v["contrast_expected"] for k, v in V.items()}
Y = {k: v["ks_flat"] - v["ks_mu"] for k, v in V.items()}
cm_grid = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9)
km_grid = (-0.05, -0.02, 0.0, 0.02, 0.05)
print("%-8s" % "ks_margin" + "".join("%18s" % ("cmin %.2f" % c) for c in cm_grid))
for m in km_grid:
    row = []
    for c in cm_grid:
        sel = {k: (X[k] >= c and Y[k] > m) for k in V}
        tp, fp, fn, pu, ef, f1 = score(sel); row.append("%3d/%3d %.3f" % (tp, fp, f1))
    print("%-8.2f" % m + "".join("%18s" % r for r in row))
print("(cells: TP/FP F1; shipped = cmin 0.60, ks_margin 0.00)")
sel = {k: (X[k] >= 0.6 and Y[k] > 0.0) for k in V}; print("shipped both tests alone: TP %d FP %d FN %d purity %.3f eff %.3f F1 %.3f" % score(sel))
full = {k: bool(V[k]["is_stm"]) for k in V}; print("full chain verdict on the same items: TP %d FP %d FN %d purity %.3f eff %.3f F1 %.3f" % score(full))

# ------------------------------------------------------------- (b) windows
print("\n=== (b) window recomputation from the profile + reference curve ===")
ref = json.load(open(C.PREP_DEFAULT + "/dqdx_ref_pdvd.json"))
g = ref["grid"]                                      # {n, start, step} in cm; curves are 401-point lists in e/cm
mu_rr = g["start"] + g["step"] * np.arange(g["n"])
mu_q = np.asarray(ref["muon"], float)
def mu_at(rr): return np.interp(rr, mu_rr, mu_q)

def contrast_at(pay, tail=(0.5, 3.0), plat=(20.0, 40.0)):
    rr, q, xyz = C.profile(pay)
    live = q >= LIVE                       # the chain's live cut (profile_min_dqdx_frac x mip_dqdx)
    rr, q = rr[live], q[live]
    if rr.size == 0: return None
    total = float(rr.max()) if rr.size else 0.0
    plo, phi = plat
    if total < phi: plo, phi = plo * 0.5, phi * 0.5      # short_track halving, as in C++
    t = (rr >= tail[0]) & (rr <= tail[1]); p = (rr >= plo) & (rr <= phi)
    if t.sum() < 3 or p.sum() < 3: return None
    tm, pm = np.median(q[t]), np.median(q[p])
    if pm <= 0: return None
    exp = np.median(mu_at(rr[t])) / max(np.median(mu_at(rr[p])), 1e-9)
    return tm / pm, exp

def ks_at(pay, cmp_range=35.0):
    rr, q, xyz = C.profile(pay)
    live = q >= LIVE; rr, q = rr[live], q[live]
    m = rr <= cmp_range
    if m.sum() < 3: return None
    test = q[m]; ref_mu = mu_at(rr[m]); ref_flat = np.full_like(test, MIP)
    def kslike(a, b):
        a = np.cumsum(a / a.sum()); b = np.cumsum(b / b.sum()); return float(np.abs(a - b).max())
    return kslike(test, ref_mu), kslike(test, ref_flat)

# agreement gate at the shipped windows
dc = []; de = []; dk = []
for k in V:
    c = contrast_at(P[k]); v = V[k]
    if c: dc.append(abs(c[0] - v["contrast"])); de.append(abs(c[1] - v["contrast_expected"]))
    kk = ks_at(P[k])
    if kk: dk.append(abs((kk[1] - kk[0]) - (v["ks_flat"] - v["ks_mu"])))
dc, de, dk = np.array(dc), np.array(de), np.array(dk)
print("agreement at the SHIPPED windows (n %d): |d contrast| median %.4f p90 %.4f max %.4f; |d expected| median %.4f max %.4f; |d (ks_flat-ks_mu)| median %.4f p90 %.4f max %.4f"
      % (len(dc), np.median(dc), np.percentile(dc, 90), dc.max(), np.median(de), de.max(), np.median(dk), np.percentile(dk, 90), dk.max()))
print("(the C++ live profile also drops dx<=0 / dQ<0 rows and uses the un-rounded fit; the payload is rounded to 0.1 e/cm and 0.01 cm -- residuals of that size are the floor)")

def sweep(label, fn):
    print("\n--- %s ---" % label)
    print("%-28s %4s %4s %4s %7s %7s %7s" % ("setting", "TP", "FP", "FN", "purity", "eff", "F1"))
    for name, sel in fn():
        print("%-28s %4d %4d %4d %7.3f %7.3f %7.3f" % ((name,) + score(sel)))

def tail_sweep():
    for hi in (2.0, 3.0, 4.0, 5.0):
        sel = {}
        for k in V:
            c = contrast_at(P[k], tail=(0.5, hi)); kk = ks_at(P[k])
            sel[k] = bool(c and kk and c[0] >= 0.6 * c[1] and (kk[1] - kk[0]) > 0)
        yield "tail 0.5-%.1f cm" % hi, sel
def plat_sweep():
    for lo, hi in ((15.0, 30.0), (20.0, 40.0), (25.0, 50.0), (20.0, 60.0)):
        sel = {}
        for k in V:
            c = contrast_at(P[k], plat=(lo, hi)); kk = ks_at(P[k])
            sel[k] = bool(c and kk and c[0] >= 0.6 * c[1] and (kk[1] - kk[0]) > 0)
        yield "plateau %.0f-%.0f cm" % (lo, hi), sel
def cmp_sweep():
    for cr in (25.0, 35.0, 45.0, 60.0):
        sel = {}
        for k in V:
            c = contrast_at(P[k]); kk = ks_at(P[k], cmp_range=cr)
            sel[k] = bool(c and kk and c[0] >= 0.6 * c[1] and (kk[1] - kk[0]) > 0)
        yield "compare_range %.0f cm" % cr, sel
sweep("bragg_tail_hi_cm (contrast >= 0.6 x expected AND ks_flat > ks_mu, recomputed)", tail_sweep)
sweep("bragg_plateau window", plat_sweep)
sweep("compare_range_cm (KS window)", cmp_sweep)
