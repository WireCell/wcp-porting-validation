#!/usr/bin/env python3
"""Transfer the kine_* recombination survival factors from uBooNE to SBND by
the table-integrated recombination ratio (docs/pr/10; knobs from docs/pr/2
sec 2e(iii-a)).

The uBooNE energy reco divides the summed charge by a flat "average
recombination survival": 0.7 (track), 0.5 (shower-flagged), 0.35 (proton),
tuned at 0.273 kV/cm.  Those numbers are empirical -- they also absorb
non-recombination effects (0.7 != the pointwise Box R(MIP) ~ 0.63) -- so the
defensible move is a RATIO transfer that keeps the empirical content: scale
each factor by

    ratio = R_eff^SBND / R_eff^uB

where R_eff is the effective survival of that particle class,

    R_eff(L) = sum_i R(dEdx_i) dE_i / sum_i dE_i     (energy-weighted mean R;
                                                      identically Q_tot*W/E_tot,
                                                      the exact factor that makes
                                                      E_reco = Q W / R_eff correct)

integrated over the class dE/dx profile from the NIST/PDG stopping tables
(energy_loss/pion_travel/stopping.root).  Because only the RATIO enters, the
profile choice is second-order; the spread over track lengths is quoted as the
systematic.

Models:
  * uBooNE side: the official/ArgoNeuT Modified Box, alpha=0.93, beta=0.212,
    rho=1.38 g/cm3, at E = 0.273 kV/cm -- the parameterization the uBooNE
    calibration chain is anchored to.
  * SBND side: the free-power Modified Box fitted to SBND stopping tracks
    (doc 55 sec 7g), R = ln(A+u)/u, u = k*(dEdx/2.1)^p, read from the
    canonical block of ../nusel_display/stm_ref_dqdx.json.  The C
    normalization is deliberately EXCLUDED: it is degenerate with gain /
    electron-lifetime calibration (fit_recombination.py docstring), i.e. it
    belongs with kine_fudge_factor, not with recombination survival.
  * cross-check: official Box at 0.5 kV/cm (pure field scaling, no SBND fit).

Class profiles (rr in cm along a stopping track of length L):
  * track  -> muon table, L in {10, 30, 50, 100}
  * shower -> electron table, L = 15 (the table is physical only to
    rr ~ 15 cm -- beyond is the ele1.dat clamp, docs/energy_loss_overview.md)
  * proton -> proton table, L in {3, 10, 30}

Also computed here (docs/pr/2 sec 2e(i) third item): the single-photon
mean-dedx threshold transfer.  The legacy cut 2.3 MeV/cm is quoted on the
INLINE uBooNE scale (inverse Box with A=1.0, B=0.255 at 0.273); mapping it to
charge and back through the official uBooNE Box gives the threshold on the
physical dE/dx scale -- which is what a model-consistent SBND reconstruction
(sp_dedx_use_recomb_model=true) sees.

Run:
    python3 derive_kine_recom_factors.py
"""

import json
import os
import numpy as np
import uproot

HERE = os.path.dirname(os.path.abspath(__file__))
ELOSS = os.path.normpath(os.path.join(HERE, "../../../..", "energy_loss/pion_travel"))
STOPPING = os.path.join(ELOSS, "stopping.root")
CANON = os.path.join(HERE, "..", "nusel_display", "stm_ref_dqdx.json")

RHO = 1.38          # g/cm3 (MicroBooNE convention, as in convert_field.C)
BOX_AB = (0.93, 0.212)
E_UB, E_SBND = 0.273, 0.5   # kV/cm
W_ION = 23.6e-6     # MeV per pair

UB_FACTORS = {"track": 0.7, "shower": 0.5, "proton": 0.35}
PROFILES = {"track": ("muon", (10.0, 30.0, 50.0, 100.0)),
            "shower": ("electron", (15.0,)),
            "proton": ("proton", (3.0, 10.0, 30.0))}


def R_box(dedx, efield, ab=BOX_AB, rho=RHO):
    a, b = ab
    xi = b / (rho * efield) * np.asarray(dedx, dtype=float)
    return np.log(a + xi) / xi


def load_canonical():
    ck = json.load(open(CANON))["_meta"]["canonical_keys"]
    return float(ck["A"]), float(ck["k"]), float(ck["p"]), float(ck["C"])


def R_freepower(dedx, A, k, p, pivot=2.1):
    u = k * (np.asarray(dedx, dtype=float) / pivot) ** p
    return np.log(A + u) / u


def load_dedx(particle):
    g = uproot.open(STOPPING)[particle]
    return (np.asarray(g.member("fX"), dtype=float),
            np.asarray(g.member("fY"), dtype=float))


def r_eff(x, y, L, rfun):
    """Energy-weighted mean survival over rr in (0, L] on a fine grid."""
    rr = np.arange(0.005, L, 0.01)
    dedx = np.interp(rr, x, y)
    de = dedx * 0.01
    return float(np.sum(rfun(dedx) * de) / np.sum(de))


def main():
    A, k, p, C = load_canonical()
    print(f"SBND free-power canonical: A={A} k={k} p={p} (C={C} EXCLUDED, "
          "degenerate with gain/fudge)")
    print(f"uBooNE reference: official Box alpha={BOX_AB[0]} beta={BOX_AB[1]} "
          f"rho={RHO} at {E_UB} kV/cm")
    print()

    header = (f"{'class':7s} {'L_cm':>6s} {'R_uB':>7s} {'R_fp':>7s} "
              f"{'R_box05':>7s} {'ratio_fp':>8s} {'ratio_box':>9s}")
    print(header)
    results = {}
    for cls, (particle, lengths) in PROFILES.items():
        x, y = load_dedx(particle)
        ratios_fp, ratios_box = [], []
        for L in lengths:
            r_ub = r_eff(x, y, L, lambda d: R_box(d, E_UB))
            r_fp = r_eff(x, y, L, lambda d: R_freepower(d, A, k, p))
            r_b5 = r_eff(x, y, L, lambda d: R_box(d, E_SBND))
            ratios_fp.append(r_fp / r_ub)
            ratios_box.append(r_b5 / r_ub)
            print(f"{cls:7s} {L:6.1f} {r_ub:7.4f} {r_fp:7.4f} {r_b5:7.4f} "
                  f"{r_fp / r_ub:8.4f} {r_b5 / r_ub:9.4f}")
        results[cls] = (float(np.mean(ratios_fp)),
                        float(np.min(ratios_fp)), float(np.max(ratios_fp)),
                        float(np.mean(ratios_box)))

    print()
    print("== proposed SBND kine_* recombination factors "
          "(uBooNE factor x mean free-power ratio) ==")
    for cls, (mean_fp, lo, hi, mean_box) in results.items():
        old = UB_FACTORS[cls]
        new = old * mean_fp
        print(f"  {cls:7s}: {old:.2f} x {mean_fp:.4f} = {new:.4f}   "
              f"(ratio spread {lo:.4f}..{hi:.4f}; official-Box cross-check "
              f"would give {old * mean_box:.4f})")

    # ---- single-photon mean-dedx threshold transfer ----
    print()
    print("== single-photon sp_mean_dedx_cut transfer ==")
    a_in, b_in = 1.0, 0.255            # the INLINE uBooNE parameterization
    bp_in = b_in / (RHO * E_UB)
    cut_inline = 2.3                   # MeV/cm on the inline scale
    dqdx = np.log(a_in + bp_in * cut_inline) / (bp_in * W_ION)   # e/cm
    bp_ub = BOX_AB[1] / (RHO * E_UB)
    cut_true = (np.exp(dqdx * W_ION * bp_ub) - BOX_AB[0]) / bp_ub
    print(f"  2.3 MeV/cm (inline A=1.0/B=0.255 scale) = {dqdx:,.0f} e/cm "
          f"= {cut_true:.3f} MeV/cm on the official-Box physical scale")
    print("  -> with sp_dedx_use_recomb_model=true (model-consistent dE/dx) the")
    print(f"     equivalent threshold is sp_mean_dedx_cut = {cut_true:.2f}")


if __name__ == "__main__":
    main()
