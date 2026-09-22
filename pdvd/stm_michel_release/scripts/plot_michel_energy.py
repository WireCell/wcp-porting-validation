#!/usr/bin/env python3
"""plot_michel_energy.py -- the Michel electron charge energy over the release, and the
2-D-cell sum it is built from.

    python3 scripts/plot_michel_energy.py [release_dir] [--out figs/michel_energy.png]

Population: T_stm rows with is_stm == 1 and michel_found == 1.
  michel_ke_q2d_region   the production estimator: 2-D charge within 10 cm of the stop in the
                         candidate's own cells minus the muon fit's prediction, three planes
                         combined, through the recombination model  [MeV]
  michel_ke_best         the chain's association estimate (fitted dQ/dx path + unfitted charge)
  michel_ke_q2d_ctl      the SAME region sum 35 cm back up the muon body: the per-item floor
Panels: (1) spectra of the two estimators; (2) region vs best per candidate; (3) the per-plane
region charge re-derived from T_michel_2d against the stored michel_q2d_region_{u,v,w}
(the data-structure check: same numbers to 1e-6); (4) the body control.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stm_release as sr
import relplot as rp
import matplotlib.pyplot as plt

BR = ["cluster_id", "is_stm", "michel_found", "michel_ke_q2d_region", "michel_ke_best", "michel_ke_q2d_ctl",
      "michel_q2d_region_u", "michel_q2d_region_v", "michel_q2d_region_w", "michel_ke_charge", "michel_conn_type"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("release", nargs="?", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rel = sr.release_dir(a.release)
    R = {k: [] for k in BR}; twin_cpp = []; twin_off = []
    det = None
    for path in sr.event_files(rel):
        ev = sr.open_event(path)
        det = det or sr.event_info(ev)["det"].upper()
        if ev["T_stm"].num_entries == 0:
            continue
        c = sr.candidates(ev, BR)
        for i in range(len(c["cluster_id"])):
            if not (c["is_stm"][i] == 1 and c["michel_found"][i] == 1):
                continue
            for k in BR:
                R[k].append(c[k][i])
            off, n = sr.michel_region_sum(ev, int(c["cluster_id"][i]))
            twin_off.append(off); twin_cpp.append([c["michel_q2d_region_u"][i], c["michel_q2d_region_v"][i], c["michel_q2d_region_w"][i]])
    R = {k: np.asarray(v, float) for k, v in R.items()}
    twin_off = np.asarray(twin_off); twin_cpp = np.asarray(twin_cpp)
    n = len(R["cluster_id"])
    dev = np.abs(twin_off - twin_cpp) / np.maximum(1.0, np.abs(twin_cpp))
    print(f"{det}: {n} STM+Michel candidates; region twin (offline sum from T_michel_2d vs stored) max relative deviation {dev.max():.2e}")
    reg, best, ctl = R["michel_ke_q2d_region"], R["michel_ke_best"], R["michel_ke_q2d_ctl"]
    print(f"  michel_ke_q2d_region: median {np.median(reg):.1f} MeV, mean {reg.mean():.1f}, > 52.8 MeV: {(reg > 52.8).sum()}")
    print(f"  michel_ke_best      : median {np.median(best):.1f} MeV, mean {best.mean():.1f}")
    print(f"  body control (ctl)  : median {np.median(ctl):.1f} MeV")

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    bins = np.linspace(0, 120, 31)
    ax[0, 0].hist(reg, bins, histtype="step", color=rp.BLUE, lw=2, label="michel_ke_q2d_region (production)")
    ax[0, 0].hist(best, bins, histtype="step", color=rp.ORANGE, lw=2, label="michel_ke_best")
    ax[0, 0].axvline(52.8, color=rp.INK2, lw=1, ls="--"); ax[0, 0].text(53.5, ax[0, 0].get_ylim()[1] * 0.9, "52.8 MeV endpoint", color=rp.INK2, fontsize=8)
    ax[0, 0].set_xlabel("Michel energy [MeV]"); ax[0, 0].set_ylabel("candidates"); ax[0, 0].legend(); ax[0, 0].set_title(f"{det}: {n} STM+Michel candidates")
    ax[0, 1].plot(best, reg, "o", color=rp.BLUE, ms=5, alpha=0.8)
    lim = max(best.max(), reg.max()) * 1.05 if n else 1
    ax[0, 1].plot([0, lim], [0, lim], color=rp.INK2, lw=1, ls="--"); ax[0, 1].set_xlim(0, lim); ax[0, 1].set_ylim(0, lim)
    ax[0, 1].set_xlabel("michel_ke_best [MeV]"); ax[0, 1].set_ylabel("michel_ke_q2d_region [MeV]"); ax[0, 1].set_title("region vs association estimator")
    for p, col, lab in zip(range(3), [rp.BLUE, rp.ORANGE, rp.AQUA], ["U", "V", "W"]):
        ax[1, 0].plot(twin_cpp[:, p] / 1e6, twin_off[:, p] / 1e6, "o", color=col, ms=4, label=f"plane {lab}")
    lim = max(1e-3, twin_cpp.max() / 1e6 * 1.05) if n else 1
    ax[1, 0].plot([0, lim], [0, lim], color=rp.INK2, lw=1, ls="--")
    ax[1, 0].set_xlabel("stored michel_q2d_region_{u,v,w} [10$^6$ e]"); ax[1, 0].set_ylabel("re-derived from T_michel_2d [10$^6$ e]")
    ax[1, 0].set_title(f"data-structure check: max rel. deviation {dev.max():.1e}"); ax[1, 0].legend()
    ax[1, 1].hist(ctl, bins, histtype="step", color=rp.VIOLET, lw=2, label="michel_ke_q2d_ctl (35 cm up the body)")
    ax[1, 1].hist(reg, bins, histtype="step", color=rp.BLUE, lw=1.2, label="michel_ke_q2d_region")
    ax[1, 1].set_xlabel("[MeV]"); ax[1, 1].set_ylabel("candidates"); ax[1, 1].legend(); ax[1, 1].set_title("the body control: the same sum where no Michel is")
    rp.finish(fig, a.out or f"{rel}/figs/michel_energy.png", "T_stm rows with is_stm == 1 and michel_found == 1; region twin = stm_release.michel_region_sum")


if __name__ == "__main__":
    main()
