#!/usr/bin/env python3
"""doc pdvd/47 sec 8 -- the phase-window summary figure.

Reads the combined table written by d47_phase_table.py --append and draws, per detector and
plane, the ratio <sigma_eff>(outer quartiles) / <sigma_eff>(inner quartiles) -- the
"boundary vs centre" contrast the mechanism of sec 4.2 predicted -- for every variant:
the published (phase-averaged) inversion, the corrected inversion on the truth phase, the
corrected inversion on data-like phase estimators (the profile's own centroid; the truth
smeared by 0.26 wire), and the data themselves.  A phase-independent width sits at 1.

Usage: d47_phase_plot.py --table figs/47_phase_table.tsv --out figs/47_phase2.png
"""
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d47_phase_artefact import read_tsv  # noqa: E402

VARIANTS = [("{d}_S1_published_old", "sim S1, published (phase-averaged inversion)", "tab:red", "v"),
            ("{d}_S1_truth", "sim S1, corrected, true phase", "tab:blue", "o"),
            ("{d}_S1_centroid", "sim S1, corrected, centroid phase", "tab:cyan", "s"),
            ("{d}_S1_jitter026", "sim S1, corrected, phase + 0.26 wire", "tab:purple", "D"),
            ("{d}_S3_truth", "sim S3 (no diffusion/noise), true phase", "tab:green", "^"),
            ("data", "data, corrected inversion", "k", "*")]


def ratio(rows, tag, plane):
    r = {x["window"]: x for x in rows if x["tag"] == tag and x["plane"] == plane}
    if not all(w in r for w in ("q1", "q2", "q3", "q4")):
        return np.nan, np.nan
    out = np.average([r["q1"]["sig_mean_mm"], r["q4"]["sig_mean_mm"]])
    ins = np.average([r["q2"]["sig_mean_mm"], r["q3"]["sig_mean_mm"]])
    eo = 0.5 * np.hypot(r["q1"]["sig_err_mm"], r["q4"]["sig_err_mm"])
    ei = 0.5 * np.hypot(r["q2"]["sig_err_mm"], r["q3"]["sig_err_mm"])
    return out / ins, (out / ins) * np.hypot(eo / out, ei / ins)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    T = read_tsv(a.table)
    dets = ["pdhd", "pdvd", "sbnd"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, d in zip(axes, dets):
        for iv, (pat, lab, col, mk) in enumerate(VARIANTS):
            tag = ("data_" + d) if pat == "data" else "sim_" + pat.format(d=d)
            xs, ys, es = [], [], []
            for ip, pl in enumerate("UVW"):
                v, e = ratio(T, tag, pl)
                if np.isfinite(v):
                    xs.append(ip + (iv - 2.5) * 0.11); ys.append(v); es.append(e)
            ax.errorbar(xs, ys, yerr=es, ls="none", marker=mk, ms=6, color=col,
                        label=lab if d == "pdhd" else None, capsize=2)
        ax.axhline(1.0, color="0.5", lw=1, ls="--")
        ax.set_xticks(range(3)); ax.set_xticklabels(list("UVW"))
        ax.set_title(d.upper()); ax.grid(alpha=0.25, axis="y")
    axes[0].set_ylabel(r"$\langle\sigma\rangle$ boundary quartiles / centre quartiles")
    axes[0].set_yscale("log"); axes[0].set_ylim(0.08, 12)
    for ax in axes:
        ax.set_yticks([0.1, 0.2, 0.5, 1, 2, 5, 10]); ax.set_yticklabels(["0.1", "0.2", "0.5", "1", "2", "5", "10"])
    fig.legend(loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.13))
    fig.suptitle("doc 47 sec 8: the sub-pitch-phase contrast is set by the inversion and by the phase estimator, not by the detector", fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(a.out, dpi=140, bbox_inches="tight")
    print("-> %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
