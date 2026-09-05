#!/usr/bin/env python3
"""PDHD LAr transport numbers from the data-calibrated drift velocity.

Forked BY DUPLICATION from pdvd/stm/pdvd_transport.py (doc pdvd/25 sec 7a/sec 8,
doc pdvd/29 sec 2); the PDVD file is untouched.  Only the anchors change: PDHD's
drift velocity, temperature and drift length.

Inverts the calibrated PDHD drift speed (1.576 mm/us -- cfg/pgrapher/experiment/
pdhd/params.jsonnet lar.drift_speed and pdhd/clus.jsonnet's local drift_speed,
which MUST agree; see pdhd/docs/clustering-algorithm.md for the four-crosser
cathode-registration calibration) through the BNL LAr-properties mobility
parameterisation (https://lar.bnl.gov/properties/) to get the drift field, then
the longitudinal and transverse diffusion coefficients at that field:

    mu(E,T)   = (a0 + a1 E + a2 E^1.5 + a3 E^2.5) /
                (1 + (a1/a0) E + a4 E^2 + a5 E^3) * (T/89 K)^-3/2     [cm^2/V/s]
    eps_L(E,T)= (b0 + b1 E + b2 E^2) / (1 + (b1/b0) E + b3 E^2) * (T/87 K)  [eV]
    D_L       = mu * eps_L                                          [cm^2/s]
    D_T       = D_L / (1 + (E/mu) dmu/dE)      (site: 0.1 % forward difference)

Temperature: dunecore's protodunehd_detproperties says T = 87.68 K, Efield
[0.4867, ...] kV/cm and Electronlifetime 35 ms (dunecore/dunecore/Utilities/
detectorproperties_dune.fcl:108-117).  The BNL site's own PDSP anchor is
v = 1.560 mm/us at 486.7 V/cm, 87.7 K -- i.e. the SAME nominal ProtoDUNE drift
field, so the 1.576 mm/us calibration sits ~1 % above the nominal velocity and
the inverted field ~2 % above the nominal 486.7 V/cm.

The mobility half is imported from energy_loss/docs/deduce_efield.py, exactly as
the PDVD script does.

Repro:
    cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd
    python3 stm/pdhd_transport.py                    # the doc tables
    python3 stm/pdhd_transport.py --tsv stm/pdhd_transport.tsv
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

V_PROD = 1.576          # mm/us, calibrated PDHD drift speed (params.jsonnet, clus.jsonnet)
V_ALT = (1.565, 1.585, 1.6)   # the other calibration passes / the old default
T_DUNECORE = 87.68      # K, dunecore protodunehd_detproperties
T_SITE = 87.3           # K, lar.bnl.gov default
E_NOMINAL = 0.4867      # kV/cm, dunecore protodunehd_detproperties Efield[0]
DRIFT_CM = 357.985      # cm, PDHD dvm a0f0pA |FV_x| (clus.jsonnet)

# The two SP-derived smearing terms (cfg/pgrapher/experiment/pdhd/sp-filters.jsonnet)
GAUS_WIDE_MHZ = 0.12    # hf('Gaus_wide', sigma)
WIRE_IND = 0.75         # wf('Wire_ind', sigma = 1/sqrt(pi) * WIRE_IND)
WIRE_COL = 10.0         # wf('Wire_col', sigma = 1/sqrt(pi) * WIRE_COL)
PITCH_UV_MM = 4.6693    # protodunehd-wires-larsoft-v1, U/V
PITCH_W_MM = 4.7920     # protodunehd-wires-larsoft-v1, W


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
    ap.add_argument("--drift-cm", type=float, default=DRIFT_CM,
                    help="full drift distance for the sigma table (cm)")
    args = ap.parse_args()

    print("# anchors (site quotes uB 1.101 @ 0.273 kV/cm 89 K, PDSP 1.560 @ 0.4867 kV/cm 87.7 K)")
    print(f"  v_bnl(0.273, 89.0) = {v_bnl(0.273, 89.0):.4f}   v_bnl(0.4867, 87.7) = {v_bnl(0.4867, 87.7):.4f}")
    print()
    hdr = "T_K\tv_mm_us\tE_kVcm\tE_larsoft\tmu_cm2Vs\tepsL_eV\tDL_cm2s\tDT_cm2s\tDL_oldfit\tDT_oldfit"
    rows = []
    for T in (T_SITE, T_DUNECORE, 88.0):
        for v in (V_PROD,) + V_ALT:
            rows.append(row(v, T))
    print("# drift field and diffusion implied by the drift velocity")
    print(hdr)
    for r in rows:
        print(f"{r['T']:.2f}\t{r['v']:.5f}\t{r['E']:.4f}\t{r['E_larsoft']:.4f}\t{r['mu']:.1f}\t"
              f"{r['epsL']:.5f}\t{r['DL']:.3f}\t{r['DT']:.3f}\t{r['DL_old']:.3f}\t{r['DT_old']:.3f}")
    print()
    print("# forward check: the dunecore nominal field")
    for T in (T_SITE, T_DUNECORE):
        dl, dt = diffusion(E_NOMINAL, T)
        print(f"  E={E_NOMINAL} kV/cm T={T} K: v_bnl={v_bnl(E_NOMINAL, T):.4f} "
              f"v_larsoft={v_larsoft(E_NOMINAL, T):.4f} mm/us  DL={dl:.3f} DT={dt:.3f}")
    print()

    # the adopted point + the sigma-vs-drift table (doc pdvd/25 sec 8 shape)
    r = row(V_PROD, T_DUNECORE)
    DL, DT = r["DL"], r["DT"]
    add_sigma_L = 1.0 / (2 * math.pi * GAUS_WIDE_MHZ) * V_PROD           # mm
    sp = 1.0 / math.sqrt(math.pi)
    ind_u = sp / WIRE_IND * PITCH_UV_MM * 0.3                            # mm
    ind_v = sp / WIRE_IND * PITCH_UV_MM * 0.5
    col_w = sp / WIRE_COL * PITCH_W_MM * 0.2
    print(f"# adopted: E={r['E']:.4f} kV/cm at T={T_DUNECORE} K, DL={DL:.4f}, DT={DT:.4f} cm^2/s")
    print(f"# add_sigma_L={add_sigma_L:.4f} mm   (1/(2 pi {GAUS_WIDE_MHZ} MHz) * {V_PROD} mm/us)")
    print(f"# SP closed-form transverse SEEDS: ind_u={ind_u:.4f} ind_v={ind_v:.4f} col_w={col_w:.4f} mm")
    print(f"#   (WITHDRAWN as a derivation on PDVD -- doc pdvd/44; seed only, measure and replace)")
    print(f"# true SP wire-filter spatial sigma: ind={sp/WIRE_IND*PITCH_UV_MM*0.5:.4f} mm "
          f"({1.0/(2*math.pi*sp*WIRE_IND):.4f} pitch), col={sp/WIRE_COL*PITCH_W_MM*0.5:.4f} mm "
          f"({1.0/(2*math.pi*sp*WIRE_COL):.4f} pitch)")
    print("drift_cm\tt_us\tsigL_diff_mm\tsigL_tot_mm\tsigL_ticks\tsigT_diff_mm\tsigT_W_tot_mm\tsigT_W_pitch")
    for d in (0.0, 50.0, 100.0, 200.0, 300.0, args.drift_cm):
        t = max(50.0, d * 10.0 / V_PROD)               # us, min_drift_time floor
        sl = math.sqrt(2 * DL * t * 1e-6) * 10.0       # cm^2/s * us -> cm -> mm
        st = math.sqrt(2 * DT * t * 1e-6) * 10.0
        sl_tot = math.hypot(sl, add_sigma_L)
        st_tot = math.hypot(st, col_w)
        print(f"{d:.2f}\t{t:.1f}\t{sl:.3f}\t{sl_tot:.3f}\t{sl_tot / (0.5 * V_PROD):.2f}\t"
              f"{st:.3f}\t{st_tot:.3f}\t{st_tot / PITCH_W_MM:.3f}")

    if args.tsv:
        with open(args.tsv, "w") as fh:
            fh.write("# PDHD drift field / diffusion vs assumed LAr temperature; "
                     "generated by pdhd/stm/pdhd_transport.py (BNL lar.bnl.gov/properties parameterisation)\n")
            fh.write(hdr + "\n")
            for r in rows:
                fh.write(f"{r['T']:.2f}\t{r['v']:.5f}\t{r['E']:.4f}\t{r['E_larsoft']:.4f}\t{r['mu']:.1f}\t"
                         f"{r['epsL']:.5f}\t{r['DL']:.3f}\t{r['DT']:.3f}\t{r['DL_old']:.3f}\t{r['DT_old']:.3f}\n")
        print(f"wrote {args.tsv}")


if __name__ == "__main__":
    main()
