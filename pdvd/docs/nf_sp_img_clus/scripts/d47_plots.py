#!/usr/bin/env python3
"""doc pdvd/47 -- figures: (a) the constant c per plane across the arms of each detector,
against the data constants; (b) the sub-pitch phase dependence of c on the production arm.

Usage: d47_plots.py --summary figs/47_sim_summary.tsv --figs figs --out figs/47
"""
import argparse, csv, glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = {"pdhd": (3.210, 2.717, 1.460), "pdvd": (2.300, 2.304, 1.176), "sbnd": (1.316, 1.321, 0.378)}
FILTER = {"pdhd": (3.405, 3.405, 0.034), "pdvd": (0.213, 0.213, 0.036), "sbnd": (1.405, 1.405, 0.159)}
ARMS = [("S0v2", "splat", "truth\nsplat"), ("S3", "gauss", "no diff.\nno noise"), ("S2", "gauss", "no noise"), ("S5", "gauss", "no wire\nfilter"),
        ("S1n05", "gauss", "noise\nx0.5"), ("S1", "gauss", "production\nsim"), ("S1n2", "gauss", "noise\nx2")]


def rd(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True); ap.add_argument("--figs", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = rd(a.summary)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), sharey=False)
    for ax, det in zip(axs, ("pdhd", "pdvd", "sbnd")):
        x = np.arange(len(ARMS)); w = 0.26
        for ip, (p, col) in enumerate(zip("UVW", ("#1f77b4", "#2ca02c", "#d62728"))):
            vals, errs = [], []
            for arm, tag, _ in ARMS:
                r = [s for s in S if s["det"] == det and s["arm"] == arm and s["tag"] == tag]
                vals.append(float(r[0]["c%s_share" % p]) if r else np.nan); errs.append(float(r[0]["c%serr_share" % p]) if r else 0)
            ax.bar(x + (ip - 1) * w, vals, w, yerr=errs, color=col, label="%s (sim)" % p)
            ax.axhline(DATA[det][ip], color=col, ls="--", lw=1.2, label="%s data" % p if ip == 0 else None)
            ax.axhline(FILTER[det][ip], color=col, ls=":", lw=1.0)
        ax.set_xticks(x); ax.set_xticklabels([t for _, _, t in ARMS], fontsize=8)
        ax.set_title("%s: share-matched c per plane (dashed = data, dotted = wire-filter kernel)" % det.upper(), fontsize=9)
        ax.set_ylabel("c [mm]"); ax.grid(axis="y", alpha=0.3)
        if det == "pdhd":
            ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(a.out + "_arms.png", dpi=130); plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, det in zip(axs, ("pdhd", "pdvd", "sbnd")):
        for lab, ls in (("S1_gauss", "-"), ("S3_gauss", "--")):
            f = os.path.join(a.figs, "47_%s_%s_phase_fit.tsv" % (det, lab))
            if not os.path.exists(f):
                continue
            F = rd(f)
            for p, col in zip("UVW", ("#1f77b4", "#2ca02c", "#d62728")):
                ys = []
                for q in ("q1", "q2", "q3", "q4"):
                    r = [x for x in F if x["label"] == "phase:" + q and x["est"] == "share" and x["plane"] == p + "(joint)"]
                    ys.append(abs(float(r[0]["c_eff_mm"])) if r else np.nan)
                ax.plot([-0.375, -0.125, 0.125, 0.375], ys, marker="o", ls=ls, color=col, label="%s %s" % (p, "production" if lab.startswith("S1") else "no diffusion"))
        ax.set_xlabel("sub-pitch phase of the true position (0 = wire centre, +-0.5 = region boundary)")
        ax.set_ylabel("c [mm]"); ax.set_title("%s: c vs impact phase" % det.upper(), fontsize=9); ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(a.out + "_phase.png", dpi=130); plt.close(fig)
    print("->", a.out + "_arms.png", a.out + "_phase.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
