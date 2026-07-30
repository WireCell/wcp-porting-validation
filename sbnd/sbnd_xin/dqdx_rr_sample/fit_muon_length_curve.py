#!/usr/bin/env python3
"""Derive the SBND muon median-dQ/dx-vs-length envelope from the stopping-muon
tables (docs/pr/10; the knob is `muon_dqdx_curve` on TaggerCheckNeutrino).

The uBooNE prototype cuts muon candidates with the empirical envelope

    f_emp(L) = 0.8866 + 0.9533 * (18 cm / L)^0.4234        [x mip_dqdx_median]

at nine tagger sites: the MEDIAN dQ/dx of a stopping muon of length L is
Bragg-inflated at small L (the peak occupies a large fraction of the samples)
and relaxes toward the plateau as L grows.  The refit provenance is visible in
the prototype source itself -- every site carries the commented-out predecessor
`0.85 + 0.95*sqrt(25/L)` (same family, exponent pinned at 0.5).

Method, validated on uBooNE before being trusted on SBND:

  1. From the uBooNE muon table (stopping_ave_dQ_dx.root, 0.273 kV/cm,
     e/cm vs rr) compute the median of dQ/dx(rr), rr in (0, L], for each L,
     normalized by 43e3 (the scale the nine cuts multiply) -> f_tab_uB(L).
  2. g(L) = f_emp(L) / f_tab_uB(L) is the empirical margin the uBooNE tune
     carries over the bare table median (acceptance headroom + real-data
     spread).  If the method is sound g(L) is a mild, slowly-varying factor.
  3. Same median from the SBND table (stopping_ave_dQ_dx_sbnd.root,
     0.5 kV/cm), normalized by 48000 (the production mip_dqdx_median the
     cuts multiply on SBND) -> f_tab_sbnd(L).
  4. Margin-preserving transfer: f_target(L) = f_tab_sbnd(L) * g(L), fitted
     with the SAME functional form c0 + c1*(18/L)^c2 (pivot kept at 18 cm --
     it is degenerate with c1) over L in [4, 120] cm, log-uniform weight.

Caveats stated up front:
  * 48000 is itself a ratio-preserving PLACEHOLDER (docs/pr/2 sec 2e(ii-a)),
    not an SBND median measurement; f_target scales as 1/mip_dqdx_median, so
    a future measurement rescales c0 and c1 by 48000/measured.
  * the tables end at rr = 59.5 cm; beyond that the plateau value is used
    (linterp-style clamp), so L >~ 120 cm medians are plateau-dominated by
    construction.  The fit range stops at 120 cm for exactly that reason.
  * the tables carry convert_field.C's x0.85 normalization; it cancels in
    g(L) only to the extent it is common to both detectors' calibrations --
    which is the same assumption the mip_dqdx_median transfer already makes.

Run:
    python3 fit_muon_length_curve.py
Reads  ../../..//energy_loss/pion_travel/stopping_ave_dQ_dx{,_sbnd}.root
Writes muon_length_curve.tsv, muon_length_curve.png; prints the fitted
[c0, c1, 18, c2] ready for --tla-code 'muon_dqdx_curve=[...]'.
"""

import os
import numpy as np
import uproot

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ELOSS = os.path.normpath(os.path.join(HERE, "../../../..", "energy_loss/pion_travel"))

UB_TABLE = os.path.join(ELOSS, "stopping_ave_dQ_dx.root")
SBND_TABLE = os.path.join(ELOSS, "stopping_ave_dQ_dx_sbnd.root")

UB_MIP_MEDIAN = 43e3     # e/cm, the scale the nine uBooNE cut sites multiply
SBND_MIP_MEDIAN = 48e3   # e/cm, SBND production mip_dqdx_median (placeholder)

EMP = (0.8866, 0.9533, 18.0, 0.4234)   # the prototype envelope
FIT_PIVOT = 18.0                        # cm, kept fixed (degenerate with c1)
L_FIT = np.exp(np.linspace(np.log(4.0), np.log(120.0), 240))   # cm
RR_STEP = 0.1                           # cm, sampling pitch along the track


def load_muon(path):
    g = uproot.open(path)["muon"]
    x = np.asarray(g.member("fX"), dtype=float)
    y = np.asarray(g.member("fY"), dtype=float)
    return x, y


def table_eval(x, y, rr):
    """Linear interp with flat clamp at both ends (toolkit linterp behavior)."""
    return np.interp(rr, x, y)


def median_dqdx(x, y, L):
    rr = np.arange(RR_STEP / 2, L, RR_STEP)
    return float(np.median(table_eval(x, y, rr)))


def f_emp(L):
    c0, c1, piv, p = EMP
    return c0 + c1 * (piv / L) ** p


def fit_envelope(L, target):
    """Least squares of c0 + c1*(18/L)^c2: c2 by two-stage grid scan,
    (c0, c1) by exact linear solve at each c2."""
    def solve(p):
        basis = (FIT_PIVOT / L) ** p
        A = np.stack([np.ones_like(L), basis], axis=1)
        coef, *_ = np.linalg.lstsq(A, target, rcond=None)
        resid = target - A @ coef
        return coef, float(np.sqrt(np.mean(resid ** 2)))

    best = None
    for p in np.arange(0.05, 1.501, 0.005):
        coef, rms = solve(p)
        if best is None or rms < best[2]:
            best = (p, coef, rms)
    p0 = best[0]
    for p in np.arange(p0 - 0.005, p0 + 0.005, 1e-4):
        coef, rms = solve(p)
        if rms < best[2]:
            best = (p, coef, rms)
    p, (c0, c1), rms = best
    return c0, c1, p, rms


def main():
    ub_x, ub_y = load_muon(UB_TABLE)
    sb_x, sb_y = load_muon(SBND_TABLE)

    f_tab_ub = np.array([median_dqdx(ub_x, ub_y, L) for L in L_FIT]) / UB_MIP_MEDIAN
    f_tab_sb = np.array([median_dqdx(sb_x, sb_y, L) for L in L_FIT]) / SBND_MIP_MEDIAN
    femp = f_emp(L_FIT)
    g = femp / f_tab_ub
    target = f_tab_sb * g

    c0, c1, p, rms = fit_envelope(L_FIT, target)
    fit = c0 + c1 * (FIT_PIVOT / L_FIT) ** p

    print("== uBooNE method validation ==")
    print("  empirical / table-median ratio g(L): "
          f"min {g.min():.4f}  max {g.max():.4f}  mean {g.mean():.4f}")
    print("== SBND fit (normalized to mip_dqdx_median = %.0f) ==" % SBND_MIP_MEDIAN)
    print(f"  muon_dqdx_curve = [{c0:.4f}, {c1:.4f}, {FIT_PIVOT:.0f}, {p:.4f}]")
    print(f"  fit rms = {rms:.2e} (dimensionless), max |fit-target| = "
          f"{np.max(np.abs(fit - target)):.2e}")
    for L in (5, 10, 18, 30, 50, 100):
        print(f"  L = {L:5.1f} cm: uB emp {f_emp(L):.3f}  "
              f"SBND target {np.interp(L, L_FIT, target):.3f}  "
              f"SBND fit {c0 + c1 * (FIT_PIVOT / L) ** p:.3f}")

    out = os.path.join(HERE, "muon_length_curve.tsv")
    with open(out, "w") as fp:
        fp.write("# L_cm\tf_tab_uB\tf_emp_uB\tg\tf_tab_sbnd\tf_target_sbnd\tf_fit_sbnd\n")
        fp.write(f"# fit: c0={c0:.6f} c1={c1:.6f} pivot={FIT_PIVOT} power={p:.6f} rms={rms:.3e}\n")
        fp.write(f"# norms: uB {UB_MIP_MEDIAN:.0f} e/cm, SBND {SBND_MIP_MEDIAN:.0f} e/cm (placeholder)\n")
        for i, L in enumerate(L_FIT):
            fp.write(f"{L:.3f}\t{f_tab_ub[i]:.5f}\t{femp[i]:.5f}\t{g[i]:.5f}\t"
                     f"{f_tab_sb[i]:.5f}\t{target[i]:.5f}\t{fit[i]:.5f}\n")
    print(f"wrote {out}")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    ax = axes[0]
    ax.plot(L_FIT, f_tab_ub, label="uB table median / 43e3")
    ax.plot(L_FIT, femp, label="uB empirical envelope")
    ax.plot(L_FIT, g, "--", label="ratio g(L) = emp/table")
    ax.set_xscale("log")
    ax.set_xlabel("track length L [cm]")
    ax.set_ylabel("median dQ/dx  [x mip_dqdx_median]")
    ax.set_title("method validation on uBooNE")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(L_FIT, f_tab_sb, label="SBND table median / 48e3")
    ax.plot(L_FIT, target, label="target = table x g(L)")
    ax.plot(L_FIT, fit, "--",
            label=f"fit [{c0:.4f}, {c1:.4f}, 18, {p:.4f}]")
    ax.plot(L_FIT, femp, ":", color="gray", label="uB envelope (for reference)")
    ax.set_xscale("log")
    ax.set_xlabel("track length L [cm]")
    ax.set_title("SBND transfer + refit")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    png = os.path.join(HERE, "muon_length_curve.png")
    figure.savefig(png, dpi=130)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
