#!/usr/bin/env python3
"""Plot the PDHD effective transverse smearing: sigma_eff vs drift time per plane,
the joint fit, and the configured model -- the money plot of
pdhd/docs/stm-tagger-chain.md sec 8.

Loosely modelled on pdvd/docs/nf_sp_img_clus/scripts/d44_sigma_plots.py but
written fresh (that script's layout is tied to its own multi-arm comparison).

Repro:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd
  python3 docs/scripts/pdhd_sigma_plots.py --bins docs/figs/pdhd_sigma_bins.tsv \
      --fit docs/figs/pdhd_sigma_fit.tsv --out docs/figs/pdhd_sigma
"""
import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLANES = ["U", "V", "W"]
COL = {"U": "#1f77b4", "V": "#2ca02c", "W": "#d62728"}


def rows(path):
    with open(path) as fh:
        return [r for r in csv.DictReader((l for l in fh if not l.startswith("#")), delimiter="\t")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", required=True)
    ap.add_argument("--fit", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="all")
    a = ap.parse_args()

    B = [r for r in rows(a.bins) if r.get("label", "all") == a.label]
    F = [r for r in rows(a.fit) if r["label"] == a.label]
    joint = {r["plane"]: r for r in F if r["plane"].endswith("(joint)") and r["est"] == "share"}

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    for ax, P in zip(axes, PLANES):
        b = [r for r in B if r["plane"] == P and r["est"] == "share"]
        t = np.array([float(r["t_us"]) for r in b])
        se = np.array([float(r["sig2_eff_mm2"]) for r in b])
        ee = np.array([float(r["sig2_err"]) for r in b])
        sm = np.array([float(r["sig_model_mm"]) for r in b])
        sig = np.sqrt(np.maximum(se, 1e-9))
        err = 0.5 * ee / np.maximum(sig, 1e-6)
        ax.errorbar(t, sig, yerr=err, fmt="o", color=COL[P], label="measured $\\sigma_{eff}$ (share-matched)")
        ax.plot(t, sm, "k--", lw=1.4, label="configured model")
        j = joint.get(P + "(joint)")
        if j:
            DT = float(j["DT_eff_cm2s"]); c = float(j["c_eff_mm"])
            tt = np.linspace(0, max(t) * 1.05, 80)
            # sigma^2 = 2 DT t + c^2 ; DT [cm2/s], t [us] -> mm^2
            ax.plot(tt, np.sqrt(np.maximum(2 * DT * tt * 1e-6, 0) * 100 + c ** 2), "-",
                    color=COL[P], lw=1.6,
                    label="joint fit  $D_T$=%.1f, c=%.2f mm" % (DT, c))
        ax.set_title("PDHD %s" % P)
        ax.set_xlabel("drift time [$\\mu$s]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel("transverse $\\sigma$ [mm]")
    axes[0].set_ylim(0, 5.0)
    fig.tight_layout()
    fig.savefig(a.out + "_fit.png", dpi=130)
    print("wrote", a.out + "_fit.png")

    # ring-share shape panel
    S = rows(a.out + "_shape.tsv")
    fig2, ax = plt.subplots(figsize=(6.4, 4.0))
    x = np.arange(4)
    w = 0.2
    for k, P in enumerate(PLANES):
        r = [s for s in S if s["plane"] == P]
        if not r:
            continue
        r = r[0]
        meas = [float(r["meas_%s" % k2]) for k2 in ("centre", "pm1", "pm2", "beyond")]
        mod = [float(r["gaus_model_%s" % k2]) for k2 in ("centre", "pm1", "pm2", "beyond")]
        ax.bar(x + (k - 1) * w, meas, w * 0.45, color=COL[P], label="%s measured" % P)
        ax.bar(x + (k - 1) * w + w * 0.45, mod, w * 0.45, color=COL[P], alpha=0.4,
               hatch="//", label="%s configured model" % P)
    ax.set_xticks(x); ax.set_xticklabels(["centre", "$\\pm$1", "$\\pm$2", "beyond"])
    ax.set_ylabel("share of profile charge")
    ax.set_title("PDHD stacked centroid-aligned wire profile")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(alpha=0.3, axis="y")
    fig2.tight_layout()
    fig2.savefig(a.out + "_shape.png", dpi=130)
    print("wrote", a.out + "_shape.png")


if __name__ == "__main__":
    main()
