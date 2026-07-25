#!/usr/bin/env python3
"""Anchor the STM tagger's stopping-muon dQ/dx reference curve to a detector.

The STM tagger (clus/src/TaggerCheckSTM.cxx, eval_stm_core) decides "stopping"
by comparing the fitted dQ/dx profile near a track end against

  ref_muon[i] = MuonDeDx(residual_range_i + offset_length)      # ke/cm
  ref_flat[i] = 50e3                                           # ke/cm, hardcoded

`MuonDeDx` is a LinterpFunction whose 60 samples (start 0.5 cm, step 1 cm) are
the point-by-point contents of the prototype's
`prototype_base/input_data_files/stopping_ave_dQ_dx.root:muon` TGraph.  Those
numbers are CHARGE per cm, so they carry a recombination model -- and the
prototype's is uBooNE's 0.273 kV/cm drift field.  SBND runs at 0.5 kV/cm.

This script quantifies that, with no fitting and no free parameters:

  1. reproduce the table from the muon RANGE table (CSDA dE/dx = dKE/dR)
     pushed through the modified-box model at the uBooNE field -- this is the
     test that establishes WHICH field the table is anchored to;
  2. invert each table point through the uBooNE box model to recover the
     effective dE/dx it encodes;
  3. push that dE/dx back out at SBND's 0.5 kV/cm, under both candidate
     recombination parameter sets, giving the residual-range-dependent
     rescale factor the SBND reference curve would need.

Modified box model (WireCell Gen::BoxRecombination, gen/src/RecombinationModels.cxx):

    xi   = B * (dE/dx) / (E * rho)
    R    = ln(A + xi) / xi
    dQ/dx = R * (dE/dx) / W_i

Note the closed form dQ/dx = ln(A + xi) * (E*rho) / (B * W_i): at fixed field
the charge depends on dE/dx only inside the logarithm, which is why the two
candidate (A,B) sets agree at uBooNE's field and diverge at SBND's.

Repro:
    python3 sbnd_xin/stm_dqdx_reference.py
Reference: sbnd_xin/docs/47_stm-bragg-reference-sbnd-retune.md
"""

import math
import re
import sys
from pathlib import Path

# Recombination parameter sets, (A, B, label).
#   WCP/toolkit: cfg/pgrapher/experiment/sbnd/clus.jsonnet sbnd_box_recomb
#                and qlport/uboone-mabc.jsonnet uBooNE_box_recomb_model
#   LArSoft:     larsim ModBoxA / ModBoxB defaults (ArgoNeuT fit)
WCP_BOX = (1.0, 0.255, "WCP box A=1.0 B=0.255")
LARSOFT_BOX = (0.93, 0.212, "LArSoft ModBox A=0.93 B=0.212")

RHO = 1.38  # g/cm^3, as configured in both jsonnet recombination models
W_I = 23.6e-6  # MeV per ionization electron, as configured
E_UBOONE = 0.273  # kV/cm
E_SBND = 0.5  # kV/cm

DEFAULT_JSONNET = Path(
    "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/particle_dataset.jsonnet"
)


def dqdx(dedx, A, B, E, rho=RHO, wi=W_I):
    """Modified box model: MeV/cm of restricted energy loss -> electrons/cm."""
    xi = B * dedx / (E * rho)
    return math.log(A + xi) / xi * dedx / wi


def dedx_from_dqdx(q, A, B, E, rho=RHO, wi=W_I):
    """Invert the box model for dE/dx (monotonic, so bisect)."""
    lo, hi = 1e-3, 1e3
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if dqdx(mid, A, B, E, rho, wi) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _numbers(text):
    return [float(x) for x in re.findall(r"-?\d+\.?\d*(?:[eE]-?\d+)?", text)]


def _array(block, key):
    start = block.index(key + ":")
    open_b = block.index("[", start)
    return _numbers(block[open_b : block.index("]", open_b) + 1])


def read_tables(path):
    """Return (dedx_values, range_coords_cm, range_values_MeV) for the muon."""
    src = Path(path).read_text()
    dedx_block = src[src.index("muon_dEdx_function") : src.index("electron_dEdx_function")]
    range_block = src[src.index("muon_range_function") : src.index("electron_range_function")]
    return (
        _array(dedx_block, "values"),
        _array(range_block, "coords"),
        _array(range_block, "values"),
    )


def interp(xs, ys, x):
    for k in range(1, len(xs)):
        if xs[k] >= x:
            f = (x - xs[k - 1]) / (xs[k] - xs[k - 1])
            return ys[k - 1] + f * (ys[k] - ys[k - 1])
    return ys[-1]


def main(argv):
    path = Path(argv[1]) if len(argv) > 1 else DEFAULT_JSONNET
    table, rcoords, rvalues = read_tables(path)
    print(f"MuonDeDx: {len(table)} samples, residual range 0.5 .. "
          f"{len(table) - 0.5:.1f} cm (step 1 cm), then clamped flat")
    print(f"MuonRange: {len(rcoords)} samples, {rcoords[0]:.3g} .. {rcoords[-1]:.4g} cm "
          f"-> {rvalues[0]:.3g} .. {rvalues[-1]:.4g} MeV\n")

    ranges = [0.5, 1.5, 2.5, 4.5, 9.5, 14.5, 19.5, 29.5, 39.5, 49.5, 59.5]

    print("(1) is the table anchored to the uBooNE field?  CSDA dE/dx = dKE/dR from the")
    print("    range table, pushed through the box model at E = 0.273 kV/cm:")
    print(f"{'R [cm]':>7} {'table':>9} {'dEdx_csda':>10} {'Q(csda,uB)':>11} {'Q/table':>8}")
    for R in ranges:
        h = 0.05
        csda = (interp(rcoords, rvalues, R + h) - interp(rcoords, rvalues, R - h)) / (2 * h)
        q = dqdx(csda, *WCP_BOX[:2], E_UBOONE)
        t = table[int(R - 0.5)]
        print(f"{R:7.1f} {t:9.0f} {csda:10.3f} {q:11.0f} {q / t:8.3f}")
    print("    Ratio ~1.00 at the Bragg peak and rising to ~1.15 on the plateau is the")
    print("    signature of a RESTRICTED energy loss (escaping delta rays) at the uBooNE")
    print("    field -- not of a different field, which would scale the whole curve.\n")

    print("(2)+(3) effective dE/dx encoded by each table point, and the SBND rescale:")
    header = (f"{'R [cm]':>7} {'table':>9} {'dEdx_eff':>9}"
              f" {'Q_SBND(WCP)':>12} {'x':>6} {'Q_SBND(LArSoft)':>16} {'x':>6}")
    print(header)
    for R in ranges:
        t = table[int(R - 0.5)]
        d = dedx_from_dqdx(t, *WCP_BOX[:2], E_UBOONE)
        q_wcp = dqdx(d, *WCP_BOX[:2], E_SBND)
        q_ls = dqdx(d, *LARSOFT_BOX[:2], E_SBND)
        print(f"{R:7.1f} {t:9.0f} {d:9.3f} {q_wcp:12.0f} {q_wcp / t:6.3f}"
              f" {q_ls:16.0f} {q_ls / t:6.3f}")

    print("\nflat MIP reference (hardcoded 50e3 in TaggerCheckSTM):")
    d = dedx_from_dqdx(50e3, *WCP_BOX[:2], E_UBOONE)
    print(f"  50000 e/cm at E=0.273 <=> dE/dx = {d:.3f} MeV/cm")
    for A, B, label in (WCP_BOX, LARSOFT_BOX):
        q = dqdx(d, A, B, E_SBND)
        print(f"  same dE/dx at E=0.500, {label}: {q:.0f} e/cm  (x{q / 50e3:.3f})")

    print("\nBragg contrast (R=0.5 cm) / (plateau R=29.5 cm):")
    print(f"  table as shipped        : {table[0] / table[29]:.3f}")
    for A, B, label in (WCP_BOX, LARSOFT_BOX):
        num = dqdx(dedx_from_dqdx(table[0], *WCP_BOX[:2], E_UBOONE), A, B, E_SBND)
        den = dqdx(dedx_from_dqdx(table[29], *WCP_BOX[:2], E_UBOONE), A, B, E_SBND)
        print(f"  SBND, {label}: {num / den:.3f}")
    print("  kslike_compare is area-normalized, so the contrast -- not the scale --")
    print("  is what moves ks1.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
