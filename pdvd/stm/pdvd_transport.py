#!/usr/bin/env python3
"""PDVD LAr transport numbers from the data-calibrated drift velocity.

Inverts the production Q/L drift speed (1.48073 mm/us, run_clus_evt.sh
PDVD_DRIFT_SPEED_{BOT,TOP}_MMUS, doc qlmatch/06) through the BNL LAr-properties
mobility parameterisation (https://lar.bnl.gov/properties/, formulas read out of
the page's assets/trans.js + assets/index.js) to get the drift field, then the
longitudinal and transverse diffusion coefficients at that field:

    mu(E,T)   = (a0 + a1 E + a2 E^1.5 + a3 E^2.5) /
                (1 + (a1/a0) E + a4 E^2 + a5 E^3) * (T/89 K)^-3/2     [cm^2/V/s]
    eps_L(E,T)= (b0 + b1 E + b2 E^2) / (1 + (b1/b0) E + b3 E^2) * (T/87 K)  [eV]
    D_L       = mu * eps_L                                          [cm^2/s]
    D_T       = D_L / (1 + (E/mu) dmu/dE)      (site: 0.1 % forward difference)

a = {551.6, 7953.7*0.9, 4440.43, 4.29, 43.63, 0.2053} (Walkowiak/ICARUS rational
fit, a1 scaled 0.9 to match uB 1.101 mm/us @ 273 V/cm 89 K and PDSP 1.560 @
486.7 V/cm 87.7 K).  b = {0.0075, -13.376, -10.9568, 646.523} is the site's 2026
global refit of eps_L that includes the DarkSide / MicroBooNE / ProtoDUNE-SP
longitudinal-diffusion measurements (it is what the calculator uses; the older
Li et al. 2016 set {0.0075, 742.9, 3269.6, 31678.2} gives ~60 % larger D).

The mobility half is imported from energy_loss/docs/deduce_efield.py (same
coefficients, already validated there); LArSoft's Walkowiak/ICARUS drift-velocity
branch is shown beside it as a cross-check on E.

Temperature: PDVD has no configured LAr temperature.  dunecore's
protodunevd_detproperties says 87.68 K; the site default is 87.3 K.  Owner
decision 2026-09-02: trust the velocity, treat T as the soft input, and let
the dQ/dx-vs-residual-range comparison with data confirm the field.

Repro:
    cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
    python3 stm/pdvd_transport.py            # prints the doc-25 sec 7a / sec 8 tables
    python3 stm/pdvd_transport.py --tsv stm/pdvd_transport.tsv
"""
import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EL_DOCS = os.path.normpath(os.path.join(HERE, "..", "..", "..", "energy_loss", "docs"))
sys.path.insert(0, EL_DOCS)
from deduce_efield import mu_bnl, v_bnl, v_larsoft, invert  # noqa: E402

B0, B1, B2, B3 = 0.0075, -13.376, -10.9568, 646.523   # 2026 refit (site)
B_OLD = (0.0075, 742.9, 3269.6, 31678.2)              # Li et al. 2016
T1 = 87.0

V_PROD = 1.48073        # mm/us, production Q/L (both crates)
V_OLD = 1.568           # mm/us, earlier PDVD calibration (params.jsonnet)
T_DUNECORE = 87.68      # K
T_SITE = 87.3           # K
E_PLANNED = 0.495       # kV/cm, dunecore "planned for PDVD"


def eps_l(E, T, b=(B0, B1, B2, B3)):
    b0, b1, b2, b3 = b
    return (b0 + b1 * E + b2 * E * E) / (1 + (b1 / b0) * E + b3 * E * E) * (T / T1)


def diffusion(E, T, b=(B0, B1, B2, B3)):
    """(DL, DT) in cm^2/s at field E [kV/cm], temperature T [K]."""
    mu = mu_bnl(E, T)
    dl = mu * eps_l(E, T, b)
    dmu = (mu_bnl(E * 1.001, T) - mu) / (0.001 * E)
    dt = dl / (1 + E / mu * dmu)
    return dl, dt


def row(v, T):
    E = invert(v_bnl, v, T)
    El = invert(v_larsoft, v, T)
    dl, dt = diffusion(E, T)
    dlo, dto = diffusion(E, T, B_OLD)
    return dict(T=T, v=v, E=E, E_larsoft=El, mu=mu_bnl(E, T), epsL=eps_l(E, T),
                DL=dl, DT=dt, DL_old=dlo, DT_old=dto)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tsv", help="also write the field/diffusion table here")
    ap.add_argument("--drift-cm", type=float, default=338.55,
                    help="full drift distance for the sigma table (cm)")
    args = ap.parse_args()

    print("# anchors (site quotes uB 1.101 @ 0.273 kV/cm 89 K, PDSP 1.560 @ 0.4867 kV/cm 87.7 K)")
    print(f"  v_bnl(0.273, 89.0) = {v_bnl(0.273, 89.0):.4f}   v_bnl(0.4867, 87.7) = {v_bnl(0.4867, 87.7):.4f}")
    print()
    hdr = "T_K\tv_mm_us\tE_kVcm\tE_larsoft\tmu_cm2Vs\tepsL_eV\tDL_cm2s\tDT_cm2s\tDL_oldfit\tDT_oldfit"
    rows = []
    for T in (T_SITE, 87.5, T_DUNECORE, 88.0, 89.0):
        for v in (V_PROD, V_OLD):
            rows.append(row(v, T))
    print("# drift field and diffusion implied by the drift velocity")
    print(hdr)
    for r in rows:
        print(f"{r['T']:.2f}\t{r['v']:.5f}\t{r['E']:.4f}\t{r['E_larsoft']:.4f}\t{r['mu']:.1f}\t"
              f"{r['epsL']:.5f}\t{r['DL']:.3f}\t{r['DT']:.3f}\t{r['DL_old']:.3f}\t{r['DT_old']:.3f}")
    print()
    print("# forward check: the planned field")
    for T in (T_SITE, T_DUNECORE):
        dl, dt = diffusion(E_PLANNED, T)
        print(f"  E={E_PLANNED} kV/cm T={T} K: v_bnl={v_bnl(E_PLANNED, T):.4f} "
              f"v_larsoft={v_larsoft(E_PLANNED, T):.4f} mm/us  DL={dl:.3f} DT={dt:.3f}")
    print()

    # doc 25 sec 8: sigma vs drift at the adopted point
    r = row(V_PROD, T_DUNECORE)
    DL, DT = r["DL"], r["DT"]
    add_sigma_L = 1.0 / (2 * math.pi * 0.12) * V_PROD      # mm  (Gaus_wide 0.12 MHz)
    col_w = (1 / math.sqrt(math.pi)) / 10.0 * 5.10 * 0.2    # mm
    print(f"# adopted: E={r['E']:.4f} kV/cm at T={T_DUNECORE} K, DL={DL:.3f}, DT={DT:.3f} cm^2/s")
    print(f"# add_sigma_L={add_sigma_L:.4f} mm  col_sigma_w_T={col_w:.4f} mm  (doc 25 sec 7b)")
    print("drift_cm\tt_us\tsigL_diff_mm\tsigL_tot_mm\tsigL_ticks\tsigT_diff_mm\tsigT_W_tot_mm\tsigT_W_pitch")
    for d in (0.0, 50.0, 100.0, 200.0, 300.0, args.drift_cm):
        t = max(50.0, d * 10.0 / V_PROD)               # us, min_drift_time floor
        sl = math.sqrt(2 * DL * t * 1e-6) * 10.0      # cm^2/s * us -> cm -> mm
        st = math.sqrt(2 * DT * t * 1e-6) * 10.0
        sl_tot = math.hypot(sl, add_sigma_L)
        st_tot = math.hypot(st, col_w)
        print(f"{d:.2f}\t{t:.1f}\t{sl:.3f}\t{sl_tot:.3f}\t{sl_tot / (0.5 * V_PROD):.2f}\t"
              f"{st:.3f}\t{st_tot:.3f}\t{st_tot / 5.10:.3f}")

    if args.tsv:
        with open(args.tsv, "w") as fh:
            fh.write("# PDVD drift field / diffusion vs assumed LAr temperature; "
                     "generated by pdvd/stm/pdvd_transport.py (BNL lar.bnl.gov/properties parameterisation)\n")
            fh.write(hdr + "\n")
            for r in rows:
                fh.write(f"{r['T']:.2f}\t{r['v']:.5f}\t{r['E']:.4f}\t{r['E_larsoft']:.4f}\t{r['mu']:.1f}\t"
                         f"{r['epsL']:.5f}\t{r['DL']:.3f}\t{r['DT']:.3f}\t{r['DL_old']:.3f}\t{r['DT_old']:.3f}\n")
        print(f"wrote {args.tsv}")


if __name__ == "__main__":
    main()
