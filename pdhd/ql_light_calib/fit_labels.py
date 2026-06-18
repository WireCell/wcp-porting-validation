#!/usr/bin/env python3
"""Tune the PDHD QLMatching optical model from run-29107 evt-983 hand-scan labels.

Label-driven first pass (cf. SBND sbnd_xin/ql_pe_error.py). The saved labels
(work/ql_labels/labels-evt983.json) already carry, per hand-picked match, the
per-channel measured PE (op_pes), its error (op_pe_err), and the model-predicted
PE (pred_pes, computed with the CURRENT vuv_eff), plus metrics.ks_dis/total_PE.
So we fit directly from the labels -- no calib re-read needed.

PDHD optical-channel blocks (verified from the calib dump opdets x/z):
    ch  0- 79  +x side  (side 1)            -- the CALIBRATED anchor (pred/meas ~ 0.9)
    ch 80-119  -x APA2  (self-trigger half)
    ch120-159  -x APA0  (full-data-stream half, gain-biased low: "APA0")
    ch 80-159  = drift side 0 (-x)

Three knobs are tuned SELF-CONSISTENTLY (user-chosen), in order:
  1. vuv_eff (global light yield): anchored on side 1 so pred/meas -> 1 there.
     vuv_eff_new = vuv_eff_old * g, g = median_match(sum meas / sum pred) over side-1
     channels of side-1 matches. (QtoL stays 1; it is degenerate with vuv_eff.)
  2. measured_pe_scale on APA0 (ch120-159): the -x full-stream readout under-reports
     PE; we scale the MEASUREMENT up to match the RECALIBRATED prediction:
     s = g * median_match(sum pred_old / sum meas) over APA0 of side-0 matches.
     (Anchoring to old pred gives ~1.8; folding in the lower vuv_eff -> ~1.57, the
     self-consistent value the user picked. The brighter 2.3 tail is high-ks
     saturation, excluded by the low-ks cut. APA2 shows the same ~1.7 elevation but
     per the APA0-only instruction is left to the global/saturation terms.)
  3. PE-error fraction: method-of-moments fit of E[(pred-meas)^2] = meas + a*pred^2
     (a = frac^2) on the CORRECTED model (APA0 meas scaled, pred * vuv_eff factor g).

Selection = the user's cuts: sizable (metrics.total_PE > TOT_MIN) AND low-ks
(metrics.ks_dis < KS_MAX). One event, a handful of points per block -- this SEEDS
first-pass values, it does not "calibrate the model" (cross-check run-wide with
fit.py's auto-anchors). The low-ks cut uses the PRE-tuning model -> mild circularity,
acceptable for a first pass.

Usage:  python3 fit_labels.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = os.path.join(HERE, "..", "work", "ql_labels", "labels-evt983.json")

VUV_EFF_OLD = 0.0145     # current cfg/.../pdhd/qlmatching.jsonnet light yield
APA0 = range(120, 160)   # -x full-data-stream half ("APA0")
APA2 = range(80, 120)    # -x self-trigger half
SIDE1 = range(0, 80)     # +x side (the calibrated anchor)
TOT_MIN = 3000.0         # "sizable" flash cut (metrics.total_PE)
KS_MAX = 0.2             # "low-ks" cut (metrics.ks_dis)


def bsum(a, rng):
    return float(sum(a[i] for i in rng))


def main():
    M = json.load(open(LABELS))["matches"]
    sel = [m for m in M
           if (m["metrics"].get("total_PE", 0) > TOT_MIN
               and m["metrics"].get("ks_dis", 9) is not None
               and m["metrics"]["ks_dis"] < KS_MAX)]
    print("matches: %d total, %d selected (total_PE>%g & ks<%g)"
          % (len(M), len(sel), TOT_MIN, KS_MAX))

    # ---- raw per-block pred/meas (transparency) ----
    print("\nraw pred/meas (vs current model), per block, over selected:")
    apa0_ratio = []
    for name, rng in (("ch0-39", range(0, 40)), ("ch40-79", range(40, 80)),
                      ("ch80-119 APA2", APA2), ("ch120-159 APA0", APA0)):
        r = []
        for m in sel:
            o, p = bsum(m["op_pes"], rng), bsum(m["pred_pes"], rng)
            if o > 50 and p > 5:
                r.append(p / o)
        if r:
            print("  %-16s n=%2d  median=%.3f  mean=%.3f" % (name, len(r), np.median(r), np.mean(r)))
            if rng is APA0:
                apa0_ratio = r

    # ---- 1. vuv_eff: anchor on side-1 matches, side-1 channels ----
    g_per_match = []
    for m in sel:
        if m.get("apa") != 1:
            continue
        o, p = bsum(m["op_pes"], SIDE1), bsum(m["pred_pes"], SIDE1)
        if p > 50:
            g_per_match.append(o / p)            # meas/pred ; <1 => pred over-predicts
    f = float(np.median(g_per_match)) if g_per_match else 1.0
    vuv_eff_new = VUV_EFF_OLD * f
    print("\nvuv_eff anchor (side-1 matches): meas/pred median g = %.3f  (n=%d)"
          % (f, len(g_per_match)))
    print("  -> vuv_eff = %.5f * %.3f = %.5f" % (VUV_EFF_OLD, f, vuv_eff_new))

    # ---- 2. APA0 measured scale, self-consistent with the recalibrated pred ----
    APA0_SCALE = round(f * float(np.median(apa0_ratio)), 2) if apa0_ratio else 1.0
    print("\nAPA0 measured_pe_scale = g * median(pred_old/meas|APA0) = %.3f * %.3f = %.2f"
          % (f, float(np.median(apa0_ratio)), APA0_SCALE))

    # ---- 3. PE-error fraction on the CORRECTED model ----
    # corrected meas: scale APA0 (ch120-159) up; corrected pred: * f (vuv_eff bump).
    # The common pe_err_frac is the INTRINSIC per-channel scatter, so it is fit on the
    # CALIBRATED channels (side 1 + APA0); APA2 (ch80-119) carries the same ~1.7
    # over-prediction as APA0 but, per the APA0-only instruction, is NOT measured-scaled
    # -> its mismatch is a known SYSTEMATIC, not random scatter, and is excluded from the
    # frac fit (folding it in would wrongly inflate the error on every channel). The
    # "incl APA2" number is printed for transparency.
    def frac_fit(include_apa2):
        preds, meass = [], []
        for m in sel:
            side = m.get("apa")
            if side == 1:
                chans = SIDE1
            else:
                chans = range(80, 160) if include_apa2 else APA0
            op = np.array(m["op_pes"], float)
            pr = np.array(m["pred_pes"], float) * f
            op[120:160] *= APA0_SCALE
            for ch in chans:
                if op[ch] > 0 or pr[ch] > 0:
                    meass.append(op[ch]); preds.append(pr[ch])
        pred = np.array(preds); meas = np.array(meass)
        fitm = pred > 0
        pf, mf = pred[fitm], meas[fitm]
        Y = (pf - mf) ** 2 - mf
        a = float(np.sum(Y) / np.sum(pf ** 2))
        return float(np.sqrt(max(a, 0.0))), float(np.sum(mf) / np.sum(pf)), pf.size

    frac, norm, n = frac_fit(include_apa2=False)
    frac2, norm2, n2 = frac_fit(include_apa2=True)
    print("\nPE-error fit (corrected model):")
    print("  calibrated chans (side1+APA0): frac=%.2f (%.0f%%)  meas/pred=%.2f  N=%d"
          % (frac, 100 * frac, norm, n))
    print("  incl APA2 (transparency):      frac=%.2f (%.0f%%)  meas/pred=%.2f  N=%d"
          % (frac2, 100 * frac2, norm2, n2))

    # ---- config snippet ----
    print("\n--- cfg/pgrapher/experiment/pdhd/qlmatching.jsonnet ---")
    print("  local vuv_eff = %.5f;" % vuv_eff_new)
    print("  measured_pe_scale: std.makeArray(160, function(i) if i>=120 then %.2f else 1.0)," % APA0_SCALE)
    print("  pe_err_frac: %.2f,  pe_err_floor: 0.3,  pe_err_knee: 1.0," % round(frac, 2))


if __name__ == "__main__":
    main()
