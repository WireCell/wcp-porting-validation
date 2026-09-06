#!/usr/bin/env python3
"""doc 47 sec 10 -- the field-response budget.

Round 4 asks whether the ProtoDUNE-only remainder of the constant transverse
smearing is a FIELD RESPONSE modelling error: the FR files are one-dimensional
path scans (every path at `wirepos = 0`; `Response::wire_region_average` says so
itself: "specific to Garfield 1D line of paths"), while the real anode is 3-D.

The structural point this script quantifies: in the SIMULATION the sim convolves
with FR `F` and SP deconvolves with `wire_region_average(F)` -- the SAME file --
so an error in `F` cancels by construction.  The simulated `c` is therefore a
FLOOR, and `c_data (-) c_sim` in quadrature is the budget available to any
data-only effect (FR mismatch, cross-talk, ...).

Three things are computed, none of which needs a new job:

  A. the budget itself, with the bootstrap errors both sides carry;
  B. the `nsigma` truncation control -- `DepoTransform` hard-codes nsigma = 3, so
     each depo's Gaussian is CUT at +-3 sigma and the simulation cannot put
     diffusion charge at +-2 wires at all.  If the >=2-wire ring moves between
     the nsigma 3 and 5 arms, part of the budget in A is a sim artefact;
  C. the arithmetic that decides WHICH kind of FR error is left: solve for the
     single Gaussian width that would reproduce the measured beyond-+-2 share
     and compare it to the measured core width.  If the two disagree, no
     broadening -- of the response, of diffusion, of anything -- can be the
     cause, and only a LONG-RANGE (few-wire) coupling survives.

Usage:
  d47_fr_budget.py --sim-root /home/xqian/tmp/xtrack --figs <figs dir> [--out <prefix>]
"""
import argparse
import csv
import importlib.util
import math
import os
import sys

import numpy as np

# pitch [mm] per plane -- the DET table of d44_sigma_fit.py (pdvd/sbnd) and of
# its pdhd fork.  Repeated here so this script needs only one import.
PITCH = {"pdhd": (4.6693, 4.6693, 4.7920),
         "pdvd": (7.65, 7.65, 5.10),
         "sbnd": (3.00, 3.00, 3.00)}
PLANES = ("U", "V", "W")


def load_estimator(path):
    spec = importlib.util.spec_from_file_location("d44_sigma_fit", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_tsv(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def fnum(row, key, default=float("nan")):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-root", default="/home/xqian/tmp/xtrack")
    ap.add_argument("--figs", required=True, help="pdvd/docs/nf_sp_img_clus/figs")
    ap.add_argument("--pdhd-figs", default=None, help="pdhd/docs/figs (d02_sigma_fit.tsv)")
    ap.add_argument("--sigma-fit", default=None, help="path to d44_sigma_fit.py")
    ap.add_argument("--apa-bins", default=None,
                    help="_bins.tsv from d44_sigma_fit.py --det pdhd --split apa (sec D)")
    ap.add_argument("--apa-fit", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    est = load_estimator(a.sigma_fit or os.path.join(here, "d44_sigma_fit.py"))

    # ---- data side: the joint share-matched constants (doc 44 / doc pdhd/02)
    # and the base-selection ring shares of sec 9 (47_tail_<det>_cuts.tsv).
    data_fit, data_cut = {}, {}
    for det in ("pdhd", "pdvd", "sbnd"):
        if det == "pdhd":
            p = os.path.join(a.pdhd_figs or os.path.join(here, "../../../../pdhd/docs/figs"),
                             "d02_sigma_fit.tsv")
        else:
            p = os.path.join(a.figs, "44_sigma_%s_fit.tsv" % det)
        for r in read_tsv(p):
            if r["label"] == "all" and r["est"] == "share" and r["plane"].endswith("(joint)"):
                data_fit[(det, r["plane"][0])] = r
        for r in read_tsv(os.path.join(a.figs, "47_tail_%s_cuts.tsv" % det)):
            if r["sel"] == "base":
                data_cut[(det, r["plane"])] = r

    # ---- simulation side: S1 (production chain) and nsig5 (the truncation
    # control), both on the `gauss` tag, from the arms of sec 3.
    sim_shape, sim_fit = {}, {}
    for det in ("pdhd", "pdvd", "sbnd"):
        for arm in ("S1", "S2", "nsig5", "S6b"):
            sp = os.path.join(a.sim_root, det, "ana", "%s_gauss_shape.tsv" % arm)
            fp = os.path.join(a.sim_root, det, "ana", "%s_gauss_fit.tsv" % arm)
            if os.path.exists(sp):
                for r in read_tsv(sp):
                    sim_shape[(det, arm, r["plane"])] = r
            if os.path.exists(fp):
                for r in read_tsv(fp):
                    if r["label"] == "all" and r["est"] == "share" and r["plane"].endswith("(joint)"):
                        sim_fit[(det, arm, r["plane"][0])] = r

    rows = []
    print("A. the budget.  c_data and c_sim are the joint share-matched constants;")
    print("   the sim value is a FLOOR because sim and SP share the field response file,")
    print("   so an FR error cancels there by construction.  d_quad = sqrt(max(cd^2-cs^2,0)).")
    print("   %-5s %-2s | %14s %14s | %8s %8s | %s"
          % ("det", "pl", "c_data [mm]", "c_sim [mm]", "d_quad", "in pitch", "compatible?"))
    for det in ("pdhd", "pdvd", "sbnd"):
        for i, pl in enumerate(PLANES):
            df, sf = data_fit.get((det, pl)), sim_fit.get((det, "S1", pl))
            if df is None or sf is None:
                continue
            cd, ed = fnum(df, "c_eff_mm"), fnum(df, "c_err")
            cs, es = fnum(sf, "c_eff_mm"), fnum(sf, "c_err")
            d2 = cd * cd - cs * cs
            dq = math.sqrt(d2) if d2 > 0 else 0.0
            # error on the quadrature difference by propagation on c^2
            e2 = math.hypot(2 * cd * ed, 2 * cs * es)
            dqe = e2 / (2 * dq) if dq > 0 else float("nan")
            sig = d2 / e2 if e2 > 0 else float("nan")
            print("   %-5s %-2s | %7.3f+-%-5.3f %7.3f+-%-5.3f | %8.3f %8.3f | %s"
                  % (det, pl, cd, ed, cs, es, dq, dq / PITCH[det][i],
                     "consistent (%.1f sigma)" % sig if sig < 2 else "EXCESS %.1f sigma" % sig))
            rows.append(dict(det=det, plane=pl, c_data_mm="%.4g" % cd, c_data_err="%.3g" % ed,
                             c_sim_mm="%.4g" % cs, c_sim_err="%.3g" % es,
                             d_quad_mm="%.4g" % dq, d_quad_err="%.3g" % dqe,
                             d_quad_pitch="%.4g" % (dq / PITCH[det][i]),
                             nsigma_excess="%.3g" % sig))

    print()
    print("B. the nsigma truncation control.  DepoTransform hard-codes nsigma = 3, so a")
    print("   depo's Gaussian is truncated at +-3 sigma; the sim's >=2-wire ring can only")
    print("   come from the FR's own long-range response and from decon ringing, never")
    print("   from diffusion.  The `nsig5` arm is S2 + nsigma 5 (run_d47_xtrack_arms.sh:53),")
    print("   so the control pair is nsig5 vs S2 -- BOTH noiseless.  S1 is shown beside it")
    print("   because the ROI's noise-scaled thresholds are what suppress the ring (sec 9.5),")
    print("   and comparing nsig5 to S1 would read that noise difference as truncation.")
    print("   %-5s %-2s | %9s %9s %9s | %9s %9s %9s | %s"
          % ("det", "pl", "pm2 S1", "pm2 S2", "pm2 n5", "bey S1", "bey S2", "bey n5", "truncation"))
    for det in ("pdhd", "pdvd", "sbnd"):
        for pl in PLANES:
            s1, s3, s5 = (sim_shape.get((det, k, pl)) for k in ("S1", "S2", "nsig5"))
            if s3 is None or s5 is None or s1 is None:
                continue
            p1, p3, p5 = (fnum(x, "meas_pm2") for x in (s1, s3, s5))
            b1, b3, b5 = (fnum(x, "meas_beyond") for x in (s1, s3, s5))
            dpm = abs(p5 - p3) + abs(b5 - b3)
            print("   %-5s %-2s | %9.5f %9.5f %9.5f | %9.5f %9.5f %9.5f | %s"
                  % (det, pl, p1, p3, p5, b1, b3, b5,
                     "inert (d=%.5f)" % dpm if dpm < 4e-3 else "MOVES (d=%.5f)" % dpm))
            rows_b = dict(det=det, plane=pl,
                          sim_pm2_S1="%.5g" % p1, sim_pm2_S2_nsig3="%.5g" % p3, sim_pm2_nsig5="%.5g" % p5,
                          sim_beyond_S1="%.5g" % b1, sim_beyond_S2_nsig3="%.5g" % b3,
                          sim_beyond_nsig5="%.5g" % b5)
            for r in rows:
                if r["det"] == det and r["plane"] == pl:
                    r.update(rows_b)

    print()
    print("C. can ONE Gaussian do both?  sig_core is the width the share match returns;")
    print("   sig_need is the width a single Gaussian would need to put the measured")
    print("   share beyond +-2 wires.  A broadening hypothesis -- wider true response,")
    print("   more diffusion, a wider FR -- requires the two to agree.")
    print("   %-5s %-2s | %8s %8s %8s | %8s %8s | %s"
          % ("det", "pl", "meas>2", "gaus>2", "excess", "sig_core", "sig_need", "ratio"))
    for det in ("pdhd", "pdvd", "sbnd"):
        for i, pl in enumerate(PLANES):
            dc = data_cut.get((det, pl))
            if dc is None:
                continue
            pitch = PITCH[det][i]
            sig_core = fnum(dc, "sig_mm") / pitch
            ext = fnum(dc, "extent")
            meas_b = fnum(dc, "meas_beyond")
            gaus_b = fnum(dc, "gaus_beyond")
            # widen a single Gaussian until its beyond-+-2 share matches the data
            f = lambda g: est.ring_shares(g, ext)[3]
            need, hit = est._bisect(f, meas_b, 0.02, 6.0)
            print("   %-5s %-2s | %8.5f %8.5f %+8.5f | %8.3f %8.3f | x%.2f%s"
                  % (det, pl, meas_b, gaus_b, meas_b - gaus_b, sig_core, need,
                     need / sig_core if sig_core > 0 else float("nan"),
                     "  (ceiling)" if hit else ""))
            for r in rows:
                if r["det"] == det and r["plane"] == pl:
                    r.update(sig_core_pitch="%.4g" % sig_core, sig_need_pitch="%.4g" % need,
                             sig_need_mm="%.4g" % (need * pitch),
                             sig_ratio="%.4g" % (need / sig_core if sig_core > 0 else float("nan")),
                             data_meas_beyond="%.5g" % meas_b, data_gaus_beyond="%.5g" % gaus_b,
                             extent="%.4g" % ext)

    # ---- D. the PDHD APA split at FIXED D_T -------------------------------
    # PDHD deconvolves APA0 against np04hd-garfield-6paths-mcmc-bestfit (an MCMC
    # refit to PDHD data, planes ordered U,W,V -- hence sp.jsonnet:167's
    # plane2layer [0,2,1]) and APA1-3 against the generic dune-garfield-1d565.
    # If the data-only excess is a field-response modelling error, the APA whose
    # response was refit TO THIS DETECTOR should carry less of it.
    # Per-APA joint D_T is not usable (the lever arm collapses: 3.4-11.2 cm2/s
    # against 8.3 for the whole sample), so c is re-solved with D_T FIXED to the
    # all-APA value:  c^2 = sum_w (sigma^2 - 2 D t) / sum_w.
    if a.apa_bins:
        rowsb = read_tsv(a.apa_bins)
        dfit = [r for r in read_tsv(a.apa_fit or a.apa_bins.replace("_bins", "_fit"))
                if r["label"] == "all" and r["est"] == "share" and r["plane"] == "U(joint)"]
        DT = fnum(dfit[0], "DT_eff_cm2s") if dfit else float("nan")
        print()
        print("D. PDHD by APA, D_T fixed at the all-APA joint value %.2f cm2/s." % DT)
        print("   APA0 is deconvolved against the MCMC refit to PDHD data, APA1-3 against")
        print("   the generic DUNE response.  sim is the matched simulation arm (S6b carries")
        print("   the APA0 machinery -- plane2layer, fltresp, the _APA1 Wiener set), and in")
        print("   BOTH sim arms the field response cancels, so `excess` is the data-only part.")
        print("   %-8s %-2s | %7s %8s | %8s %8s | %s"
              % ("sel", "pl", "n_bins", "c_fix[mm]", "c_sim", "excess", "field response SP used"))
        for lab in ["all"] + ["apa:apa%d" % k for k in range(4)]:
            for i, pl in enumerate(PLANES):
                b = [r for r in rowsb if r["label"] == lab and r["est"] == "share" and r["plane"] == pl]
                if not b:
                    continue
                w = np.array([1.0 / max(fnum(r, "sig2_err"), 1e-9) ** 2 for r in b])
                s2 = np.array([fnum(r, "sig2_eff_mm2") for r in b])
                t = np.array([fnum(r, "t_us") * 1e-6 for r in b])
                c2 = float((w * (s2 - 2 * DT * t * 100.0)).sum() / w.sum())
                c = math.sqrt(c2) if c2 > 0 else -math.sqrt(-c2)
                arm = "S6b" if lab == "apa:apa0" else "S1"
                sf = sim_fit.get(("pdhd", arm, pl))
                cs = fnum(sf, "c_eff_mm") if sf else float("nan")
                ex = math.sqrt(max(c * abs(c) - cs * cs, 0.0)) if c > 0 else float("nan")
                print("   %-8s %-2s | %7d %8.3f | %8.3f %8.3f | %s"
                      % (lab, pl, len(b), c, cs, ex,
                         "mcmc-bestfit (refit to PDHD data)" if lab == "apa:apa0"
                         else ("dune-garfield-1d565 (generic)" if lab.startswith("apa") else "")))
                rows.append(dict(det="pdhd", plane=pl, sel=lab, c_fixdt_mm="%.4g" % c,
                                 c_sim_arm=arm, c_sim_fixdt="%.4g" % cs, excess_fixdt="%.4g" % ex,
                                 DT_fixed="%.4g" % DT))

    # ---- E. the field-response mismatch arms (PDVD) -----------------------
    # S1  sim & SP both protodunevd_FR_imbalance3p_260501 (production)  MATCHED
    # FRc sim & SP both protodunevd_FR_3view_speed1d55                  MATCHED
    # FRm sim speed1d55, SP imbalance3p   -- true response WIDER than the model
    # FRr sim imbalance3p, SP speed1d55   -- true response NARROWER than the model
    # The two files share origin (181 mm), period, pitches and plane order and
    # differ by ~1.7x in the +-1-wire response amplitude, so FRm/FRr bracket a
    # REAL disagreement between two accepted PDVD field response models.
    fr_arms = [("S1", "matched, imbalance3p (production)"),
               ("FRc", "matched, speed1d55"),
               ("FRm", "MISMATCH: true wider than model"),
               ("FRr", "MISMATCH: true narrower than model")]
    have = {}
    for arm, _ in fr_arms:
        fp = os.path.join(a.sim_root, "pdvd", "ana", "%s_gauss_fit.tsv" % arm)
        if os.path.exists(fp):
            for r in read_tsv(fp):
                if r["label"] == "all" and r["est"] == "share" and r["plane"].endswith("(joint)"):
                    have[(arm, r["plane"][0])] = fnum(r, "c_eff_mm")
    if len(have) >= 6:
        print()
        print("E. PDVD field-response MISMATCH arms.  In every other simulation arm the sim")
        print("   and SP share the response file, so an FR error cancels; these break that.")
        print("   `added` is the quadrature difference against the arm's own MATCHED control,")
        print("   i.e. the constant a realistic FR modelling error alone would produce.")
        print("   %-4s | %7s %7s %7s | %7s %7s %7s | %s"
              % ("arm", "cU", "cV", "cW", "addU", "addV", "addW", "what it is"))
        for arm, what in fr_arms:
            ref = {"S1": None, "FRc": None, "FRm": "FRc", "FRr": "S1"}[arm]
            cs = [have.get((arm, pl), float("nan")) for pl in PLANES]
            if ref is None:
                print("   %-4s | %7.3f %7.3f %7.3f | %7s %7s %7s | %s"
                      % (arm, cs[0], cs[1], cs[2], "-", "-", "-", what))
            else:
                add = []
                for i, pl in enumerate(PLANES):
                    r = have.get((ref, pl), float("nan"))
                    d = cs[i] ** 2 - r ** 2
                    add.append(math.sqrt(d) if d > 0 else 0.0)
                print("   %-4s | %7.3f %7.3f %7.3f | %7.3f %7.3f %7.3f | %s vs %s"
                      % (arm, cs[0], cs[1], cs[2], add[0], add[1], add[2], what, ref))
                for i, pl in enumerate(PLANES):
                    rows.append(dict(det="pdvd", plane=pl, sel="fr:%s" % arm,
                                     c_fr_mm="%.4g" % cs[i], fr_ref=ref,
                                     c_fr_added_mm="%.4g" % add[i]))

    if a.out:
        keys = sorted({k for r in rows for k in r})
        keys = ["det", "plane"] + [k for k in keys if k not in ("det", "plane")]
        with open(a.out + ".tsv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, delimiter="\t", restval="")
            w.writeheader()
            w.writerows(rows)
        print("\n  -> %s.tsv" % a.out)


if __name__ == "__main__":
    sys.exit(main())
