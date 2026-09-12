#!/usr/bin/env python3
"""doc pdvd/96 item 2 -- the muon prediction's bias, DIAGNOSED not fixed.

WHY THIS IS A DIAGNOSIS AND NOT A KNOB.  Any real fix lives in masked_response_prediction /
TrackFitting, which PDHD, SBND and uBooNE all share -- CLAUDE.md sec 5 item 3 (a production file
with other live consumers) and M10.  It is not a PDVD knob flip and is not attempted here.

WHAT doc 95 sec 9 ITEM 2 GOT WRONG, and this script's job is to publish the correction.  It said:
"Cells the muon fit models carry a uniform +0.084 MeV/cell residual (role 1) -- the fit slightly
under-predicts everywhere ... its mechanism is unexamined."  Both halves are wrong:

  * NOT UNIFORM.  0.084 MeV/cell is a MEAN over a strongly prediction-dependent quantity.  Binned
    by pred_mu, the MEDIAN residual/prediction runs negative in the middle deciles and rises to
    about +0.10 at the top -- the fit slightly OVER-predicts where it deposits moderately and
    under-predicts where the charge is highest, which is where the Bragg peak lives.
  * NOT UNEXAMINED.  doc 42 measured the signed prediction bias per plane (PDVD U -0.221,
    V -0.217, W -0.101) and doc 44 sec 7 named the leading candidate -- charge-dependent
    whitening, since total_err carries a (charge * rel_uncer) term, so an upward-fluctuating cell
    buys itself a larger sigma and is down-weighted.  doc 42 sec 7.2 also REFUTES the obvious
    clipped-window explanation empirically: dilating the footprint to Chebyshev <= 2, which holds
    100 % of the prediction, makes the bias WORSE.

So the question is not "is there a pedestal" but "how much of it is an additive floor the model
cannot produce, and how much is a fractional deficit in the prediction".  That split is what
decides whether a fix belongs in TrackFitting at all.

Fork by duplication (CLAUDE.md M10); reads only, changes nothing.
"""
import argparse, collections, glob, os, sys
import numpy as np
import uproot

sys.path.insert(0, "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan")
import census_lib as C

IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default="pdvd")
    ap.add_argument("--arm", default="p96vscope")
    a = ap.parse_args()

    if not os.environ.get("STM_SCAN_RECORD", "").endswith("smx7_smx8_smx9_verdicts.json"):
        sys.exit("STM_SCAN_RECORD must name the merged smx1a..smx9 record (docs 93/94)")
    rec = C.load_record()

    cols = ["cluster_id", "role", "own_blob", "charge", "pred_mu", "pred_all",
            "xshared", "flag", "plane", "d_stop_cm"]
    acc = {c: [] for c in cols}
    klass = []           # per-cell record class, for the per-class split
    has_dq = None
    nfile = 0
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
        if has_dq is None and "T_stm_michel_pts" in f:
            has_dq = any(b.lower() in ("dq", "dqdx", "dq_fit") for b in f["T_stm_michel_pts"].keys())
        ev = os.path.basename(d)[: -len(a.arm) - 1]
        t = f["T_stm_michel"].arrays(library="np")
        c = f["T_stm_michel_2d"].arrays(cols, library="np")
        cid = c["cluster_id"]
        kof = {}
        for i in range(len(t["cluster_id"])):
            key = "%s/%d" % (ev, int(t["cluster_id"][i]))
            g = "other"
            if key in rec and C.judged(rec[key]):
                ism = int(t["is_stm"][i]); mf = int(t["michel_found"][i])
                if C.is_michel(rec[key]) and ism and mf: g = "TP_found"
                elif C.is_michel(rec[key]) and ism and not mf: g = "TARGET"
                elif C.is_stopper(rec[key]) and not C.is_michel(rec[key]) and ism: g = "ZERO_CTL"
                elif not C.is_stopper(rec[key]): g = "THRU"
            kof[int(t["cluster_id"][i])] = g
        for cc in cols:
            acc[cc].append(c[cc])
        klass.append(np.array([kof.get(int(x), "other") for x in cid]))
        nfile += 1
    if not nfile:
        sys.exit("no files read for arm %s" % a.arm)
    d = {c: np.concatenate(v) for c, v in acc.items()}
    kl = np.concatenate(klass)
    n = len(d["role"])
    print("files %d, cells %d\n" % (nfile, n))

    res = d["charge"] - d["pred_mu"]

    print("=== 1. per-role residual (charge - pred_mu), charge units")
    print("  %4s %10s %11s %11s %12s %10s" % ("role", "ncells", "mean_res", "med_res", "mean_predmu", "frac_pmu0"))
    for r in sorted(np.unique(d["role"])):
        m = d["role"] == r
        print("  %4d %10d %11.1f %11.1f %12.1f %10.3f" % (
            r, m.sum(), res[m].mean(), np.median(res[m]), d["pred_mu"][m].mean(),
            float((d["pred_mu"][m] == 0).mean())))

    print("\n=== 2. P1: is the role-1 residual a FLAT offset or PROPORTIONAL to the prediction?")
    print("    A flat offset would give a constant mean_res and a median ratio near 0 at every")
    print("    decile.  A fractional deficit would give a roughly CONSTANT median ratio.")
    def profile(mask, label):
        pm = d["pred_mu"][mask]; rr = res[mask]
        if mask.sum() < 200:
            print("  %-10s too few cells (%d)" % (label, mask.sum())); return
        qs = np.quantile(pm, np.linspace(0, 1, 11))
        out = []
        for i in range(10):
            lo, hi = qs[i], qs[i + 1]
            s = (pm >= lo) & ((pm <= hi) if i == 9 else (pm < hi))
            if s.sum() == 0:
                out.append(float("nan")); continue
            p = pm[s]; ok = p > 0
            out.append(float(np.median(rr[s][ok] / p[ok])) if ok.sum() else float("nan"))
        print("  %-10s median(res/pred) by decile: %s" % (label, " ".join("%6.3f" % x for x in out)))
    base = (d["role"] == 1) & (d["xshared"] == 0)
    profile(base, "ALL")
    for g in ("TP_found", "TARGET", "ZERO_CTL", "THRU"):
        profile(base & (kl == g), g)
    print("    P1 holds if the shape (negative in the middle, rising to ~+0.10 at the top) is")
    print("    the same inside every class -- i.e. it is a property of the FIT, not of a class.")

    print("\n=== 3. P2: the top decile against doc 42's measured B_foot (PDVD U -0.221, V -0.217, W -0.101)")
    for p, nm in ((0, "U"), (1, "V"), (2, "W")):
        m = base & (d["plane"] == p)
        if m.sum() < 200:
            print("  plane %s: too few cells" % nm); continue
        pm = d["pred_mu"][m]; rr = res[m]
        hi = pm >= np.quantile(pm, 0.9)
        ok = hi & (pm > 0)
        top = float(np.median(rr[ok] / pm[ok])) if ok.sum() else float("nan")
        b42 = {"U": 0.221, "V": 0.217, "W": 0.101}[nm]
        good = top == top and b42 / 1.5 <= top <= b42 * 1.5
        print("  plane %s: top-decile median res/pred %+0.3f vs doc 42 |B_foot| %.3f -> %s"
              % (nm, top, b42, "consistent (within 1.5x)" if good else "NOT within 1.5x"))

    print("\n=== 4. P3: the per-plane closure test, sum(pred_mu) / sum(fitted dQ over mask_mu)")
    if not has_dq:
        print("  UNRUNNABLE.  T_stm_michel_pts persists cluster_id / role / seg_id / x / y / z only --")
        print("  there is no per-point fitted dQ in the output, and the closure ratio cannot be")
        print("  formed from what is written.  pred.txt declared this risk in advance and committed")
        print("  to reporting it as unrunnable rather than substituting a weaker proxy that happens")
        print("  to be computable.  It moves to the hand-off: emitting the fit's dQ per point is a")
        print("  one-branch change in TrackFitting's writer and would make this test possible.")
    else:
        print("  per-point dQ IS present -- compute the ratio here before claiming anything.")

    print("\n=== 5. P4: how much of the phantom is cells the muon fit models NOT AT ALL?")
    print("    Role 0 carries no muon prediction, so nothing is subtracted and its contribution is")
    print("    the raw measurement.  If the pedestal were mostly a prediction defect, role 1 would")
    print("    sit well ABOVE role 0 per cell.  It does not.")
    zc = (kl == "ZERO_CTL") & (d["d_stop_cm"] >= 0) & (d["d_stop_cm"] <= 10)
    if zc.sum():
        tot = float(res[zc].sum())
        z0 = zc & (d["pred_mu"] == 0)
        frac = float(res[z0].sum()) / tot if tot else float("nan")
        print("  ZERO_CTL within R=10: %d cells, total excess %.4g; from pred_mu == 0 cells %.4g (%.1f %%)"
              % (zc.sum(), tot, res[z0].sum(), 100 * frac))
        print("  P4 (>= 40 %% from unmodelled cells) -> %s" % ("HELD" if frac >= 0.40 else "MISSED"))

    print("\n=== 6. S5 + the clamp: the negative-prediction population")
    neg = d["pred_mu"] < 0
    tot_all = float(res.sum())
    print("  cells with pred_mu < 0: %d of %d (%.4f %%), min %.4g" % (neg.sum(), n, 100.0 * neg.mean(), d["pred_mu"].min()))
    if neg.sum():
        print("  their (charge - pred_mu): %.4g of %.4g total (%.3f %%)" % (res[neg].sum(), tot_all, 100.0 * res[neg].sum() / tot_all))
        r10 = neg & (d["d_stop_cm"] >= 0) & (d["d_stop_cm"] <= 10)
        print("  within R=10 of a stop: %d cells" % r10.sum())
        # what the clamp actually removes: (charge - pred) - (charge - 0) = -pred  (positive here)
        removed = float((-d["pred_mu"][neg]).sum())
        print("  the clamp removes exactly %.4g charge units = %.3f %% of the all-cell total"
              % (removed, 100.0 * removed / tot_all))
        print("  S5 (clamp lowers sums, by under 1 %%) -> %s"
              % ("HELD" if 0 <= removed / tot_all < 0.01 else "MISSED"))

    print("\n=== 7. S4: does sweeping every (face, wire) RECOVER cells?")
    own = d["own_blob"]
    vals = sorted(int(v) for v in np.unique(own))
    print("  arm %s: own_blob values %s; own != 0 on %.1f %% of cells (%d of %d)"
          % (a.arm, vals, 100.0 * (own != 0).mean(), (own != 0).sum(), n))
    print("  doc 95 measured 69.9 % (954333 of 1365558) with the unswept, bit-4-less test.")
    if not any(v & 4 for v in vals):
        print("  S4 NOT GRADED ON THIS ARM.  No own_blob value carries bit 4, which means either")
        print("  this arm predates the fix (scope 0 keeps doc 95's single-(face,wire), bits-1-2")
        print("  test verbatim) or no preloaded fitted companion covers any region cell.  Those")
        print("  two are indistinguishable from the column alone, so quoting a number here would")
        print("  be comparing doc 95's arm against itself.  Re-run on the scope arm to grade S4.")
    else:
        print("  S4 HELD if this is HIGHER than 69.9 % -- a wrapped channel that failed the")
        print("  single-(face,wire) test can now pass, so the fix should ADD own cells.")

    print("\n=== 8. a caution this round learned the hard way")
    print("  The POOLED top-decile ratio mixes planes whose biases differ by 2x (doc 42: U -0.221,")
    print("  V -0.217, W -0.101).  A pooled number that happens to match one plane's B_foot is a")
    print("  coincidence of weighting, not agreement -- sec 3 grades per plane for that reason.")


if __name__ == "__main__":
    main()
