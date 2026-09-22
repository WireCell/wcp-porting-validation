#!/usr/bin/env python3
"""plot_muon_energy.py -- the three muon kinetic-energy estimates of the stopping muons.

    python3 scripts/plot_muon_energy.py [release_dir] [--out figs/muon_energy.png]

Population: T_stm rows with is_stm == 1.
  muon_ke_range  from the reconstructed track length through the CSDA range table  [MeV]
  muon_ke_dqdx   calorimetric: the fitted dQ/dx summed through the recombination model
  muon_ke_mcs    multiple-Coulomb-scattering fit of the trajectory (muon_mcs_* branches hold
                 the fit details; muon_mcs_amb is the ambiguity flag)
Panels: range vs MCS, range vs dQ/dx, the MCS/range ratio vs length, muon_len spectrum.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stm_release as sr
import relplot as rp
import matplotlib.pyplot as plt

BR = ["cluster_id", "is_stm", "muon_len", "muon_ke_range", "muon_ke_dqdx", "muon_ke_mcs", "muon_mcs_amb", "muon_mcs_nsegs", "muon_mcs_bad_path"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("release", nargs="?", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rel = sr.release_dir(a.release)
    R = {k: [] for k in BR}; det = None
    for path in sr.event_files(rel):
        ev = sr.open_event(path); det = det or sr.event_info(ev)["det"].upper()
        if ev["T_stm"].num_entries == 0:
            continue
        c = sr.candidates(ev, BR)
        for i in range(len(c["cluster_id"])):
            if c["is_stm"][i] == 1:
                for k in BR:
                    R[k].append(c[k][i])
    R = {k: np.asarray(v, float) for k, v in R.items()}
    n = len(R["cluster_id"]); good = (R["muon_ke_mcs"] > 0) & (R["muon_mcs_bad_path"] == 0)
    print(f"{det}: {n} stopping muons; MCS available on {good.sum()}; median KE range {np.median(R['muon_ke_range']):.0f} MeV, "
          f"median MCS/range {np.median(R['muon_ke_mcs'][good] / R['muon_ke_range'][good]):.3f}, median dQdx/range {np.median(R['muon_ke_dqdx'] / R['muon_ke_range']):.3f}")
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    lim = max(R["muon_ke_range"].max(), R["muon_ke_mcs"].max()) * 1.05
    ax[0, 0].plot(R["muon_ke_range"][good], R["muon_ke_mcs"][good], "o", color=rp.BLUE, ms=5, alpha=0.8, label="MCS fit ok")
    ax[0, 0].plot(R["muon_ke_range"][~good], R["muon_ke_mcs"][~good], "x", color=rp.RED, ms=6, label="MCS bad path / absent")
    ax[0, 0].plot([0, lim], [0, lim], color=rp.INK2, lw=1, ls="--"); ax[0, 0].set_xlim(0, lim); ax[0, 0].set_ylim(0, lim)
    ax[0, 0].set_xlabel("muon_ke_range [MeV]"); ax[0, 0].set_ylabel("muon_ke_mcs [MeV]"); ax[0, 0].legend(); ax[0, 0].set_title(f"{det}: {n} stopping muons")
    lim2 = max(R["muon_ke_range"].max(), R["muon_ke_dqdx"].max()) * 1.05
    ax[0, 1].plot(R["muon_ke_range"], R["muon_ke_dqdx"], "o", color=rp.ORANGE, ms=5, alpha=0.8)
    ax[0, 1].plot([0, lim2], [0, lim2], color=rp.INK2, lw=1, ls="--"); ax[0, 1].set_xlim(0, lim2); ax[0, 1].set_ylim(0, lim2)
    ax[0, 1].set_xlabel("muon_ke_range [MeV]"); ax[0, 1].set_ylabel("muon_ke_dqdx [MeV]"); ax[0, 1].set_title("calorimetric vs range")
    ax[1, 0].plot(R["muon_len"][good], R["muon_ke_mcs"][good] / R["muon_ke_range"][good], "o", color=rp.BLUE, ms=5, alpha=0.8)
    ax[1, 0].axhline(1, color=rp.INK2, lw=1, ls="--"); ax[1, 0].set_ylim(0, 3)
    ax[1, 0].set_xlabel("muon_len [cm]"); ax[1, 0].set_ylabel("KE MCS / KE range"); ax[1, 0].set_title("MCS resolution vs track length")
    ax[1, 1].hist(R["muon_len"], np.linspace(0, 500, 26), histtype="step", color=rp.AQUA, lw=2)
    ax[1, 1].set_xlabel("muon_len [cm]"); ax[1, 1].set_ylabel("muons"); ax[1, 1].set_title("reconstructed stopping-muon length")
    rp.finish(fig, a.out or f"{rel}/figs/muon_energy.png", "T_stm rows with is_stm == 1")


if __name__ == "__main__":
    main()
