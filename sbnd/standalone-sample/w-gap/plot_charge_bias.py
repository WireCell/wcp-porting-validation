#!/usr/bin/env python3
"""Per-channel relative charge bias vs total truth charge.

For each input "file:alg" pair, integrate the reco charge per channel
(recob::Wire, scaled x50 to electrons) and the truth charge per channel
(sim::SimChannels numElectrons) over all ticks, then plot

    bias = (Q_reco - Q_truth) / Q_truth      vs      Q_truth

one point per (event, channel) with Q_truth > --min-truth.

Reuses the PyROOT data layer of compare_wires_viewer.py, so it must run in
the sbndcode environment (SL7 container + setup-local-opt.sh).

Examples:
    python plot_charge_bias.py                       # the 4 defaults
    python plot_charge_bias.py rb_mean/sp.root:gauss rb_none/sp.root:dnnsp
    python plot_charge_bias.py --plane w --min-truth 5e3 -o bias_w.png
"""

import argparse
import os
import sys

import numpy as np

from compare_wires_viewer import WireFile, NCH_U, NCH_V, NCH_APA

DEFAULT_BASE = "/exp/sbnd/app/users/yuhw/wcp-porting-img/sbnd/standalone-sample/w-gap/rebase-scan"
DEFAULT_PAIRS = [
    "rb_mean/sp.root:gauss",
    "rb_mean/sp.root:dnnsp",
    "rb_none/sp.root:gauss",
    "rb_none/sp.root:dnnsp",
]

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
          "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
MARKERS = ["o", "s", "^", "v", "D", "P", "X", "*"]


def plane_mask(nch, plane):
    """Boolean mask over channel index for plane u/v/w (both APAs)."""
    if plane == "all":
        return np.ones(nch, dtype=bool)
    lo, hi = {"u": (0, NCH_U),
              "v": (NCH_U, NCH_U + NCH_V),
              "w": (NCH_U + NCH_V, NCH_APA)}[plane]
    m = np.zeros(nch, dtype=bool)
    for apa in (0, 1):
        a, b = apa * NCH_APA + lo, apa * NCH_APA + hi
        m[a:min(b, nch)] = True
    return m


def main():
    p = argparse.ArgumentParser(
        description="Per-channel relative charge bias vs truth charge.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("pairs", nargs="*", default=DEFAULT_PAIRS,
                   help="list of file:alg pairs (file relative to --base; "
                        "alg = gauss/wiener/dnnsp/...)")
    p.add_argument("--base", default=DEFAULT_BASE, help="base path for files")
    p.add_argument("--plane", default="all", choices=["all", "u", "v", "w"],
                   help="restrict to one plane (both APAs)")
    p.add_argument("--min-truth", type=float, default=1.0,
                   help="skip channels with Q_truth below this (electrons)")
    p.add_argument("--events", type=int, default=None,
                   help="max events per file (default: all)")
    p.add_argument("--xbins", default="1e4,1e6,100",
                   help="x (truth charge) log binning for the 2D hists: min,max,nbins")
    p.add_argument("--ybins", default="-1,1,100",
                   help="y (bias) linear binning for the 2D hists: min,max,nbins")
    p.add_argument("--xbins1d", default="1e4,1e6,25",
                   help="x log binning for the 1D comparison figures: min,max,nbins")
    p.add_argument("--ineff-thresh", type=float, default=0.90,
                   help="bias < -thresh counts as in-efficiency; |bias| < thresh is 'normal'")
    p.add_argument("-o", "--output", default="charge_bias.png",
                   help="output figure path; one file per pair with the pair "
                        "label inserted before the extension")
    args = p.parse_args()

    xlo, xhi, xn = (float(v) for v in args.xbins.split(","))
    ylo, yhi, yn = (float(v) for v in args.ybins.split(","))
    xedges = np.logspace(np.log10(xlo), np.log10(xhi), int(xn) + 1)
    yedges = np.linspace(ylo, yhi, int(yn) + 1)
    x1lo, x1hi, x1n = (float(v) for v in args.xbins1d.split(","))
    xedges1d = np.logspace(np.log10(x1lo), np.log10(x1hi), int(x1n) + 1)
    xcenters1d = np.sqrt(xedges1d[:-1] * xedges1d[1:])  # geometric centers

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    files = {}  # path -> WireFile (cache; pairs may share files)
    base_out, ext_out = os.path.splitext(args.output)
    ext_out = ext_out or ".png"
    results = []  # (label, q_t, bias) per pair, for the 1D comparison figures

    print(f"{'input':40s} {'npts':>7s} {'median':>8s} {'mean':>8s} {'rms':>8s}")
    for i, pair in enumerate(args.pairs):
        path_rel, _, alg = pair.rpartition(":")
        if not path_rel:
            sys.exit(f"bad pair '{pair}', expected file:alg")
        path = path_rel if os.path.isabs(path_rel) else os.path.join(args.base, path_rel)
        if path not in files:
            files[path] = WireFile(path)
        wf = files[path]
        nev = wf.nevents if args.events is None else min(args.events, wf.nevents)

        truths, biases = [], []
        for entry in range(nev):
            reco = wf.dense(entry, alg)          # electrons (x50 applied)
            truth = wf.dense(entry, "simchannel")
            nch = min(reco.shape[0], truth.shape[0])
            q_r = reco[:nch].sum(axis=1)
            q_t = truth[:nch].sum(axis=1)
            sel = (q_t > args.min_truth) & plane_mask(nch, args.plane)
            truths.append(q_t[sel])
            biases.append((q_r[sel] - q_t[sel]) / q_t[sel])
        q_t = np.concatenate(truths)
        bias = np.concatenate(biases)

        label = f"{path_rel}:{alg}"
        print(f"{label:40s} {bias.size:7d} {np.median(bias):8.4f} "
              f"{bias.mean():8.4f} {bias.std():8.4f}")

        H, _, _ = np.histogram2d(q_t, bias, bins=[xedges, yedges])
        fig, ax = plt.subplots(figsize=(9, 6.5))
        pcm = ax.pcolormesh(xedges, yedges, H.T, cmap="viridis",
                            norm=LogNorm(vmin=1, vmax=max(H.max(), 1)))
        fig.colorbar(pcm, ax=ax, label="channels / bin")
        ax.axhline(0, color="w", lw=0.8, alpha=0.7)
        ax.set_xscale("log")
        ax.set_xlabel("total truth charge on channel  [electrons]")
        ax.set_ylabel("(Q_reco - Q_truth) / Q_truth")
        ax.set_title(f"{label}   (plane: {args.plane}, nev={nev}, "
                     f"med {np.median(bias):+.4f})")
        fig.tight_layout()
        tag = label.replace("/", "_").replace(":", "_").replace(".root", "")
        outname = f"{base_out}_{tag}{ext_out}"
        fig.savefig(outname, dpi=130)
        plt.close(fig)
        print(f"wrote {outname}")
        results.append((label, q_t, bias))

    # ---- 1D comparison: in-efficiency counts vs truth charge --------------
    th = args.ineff_thresh
    LINESTYLES = ["-", "--", ":", "-."]
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for i, (label, q_t, bias) in enumerate(results):
        n_ineff, _ = np.histogram(q_t[bias < -th], bins=xedges1d)
        ntot = int((bias < -th).sum())
        color = COLORS[i % len(COLORS)]
        # gauss inputs get hollow markers so coincident gauss/dnnsp points
        # remain distinguishable without shifting positions
        hollow = label.rpartition(":")[2] == "gauss"
        ax.plot(xcenters1d, n_ineff, MARKERS[i % len(MARKERS)], ms=6, color=color,
                markerfacecolor="none" if hollow else color,
                ls=LINESTYLES[i % len(LINESTYLES)], lw=1.0, alpha=0.9,
                label=f"{label}  (total {ntot})")
    ax.set_xscale("log")
    ax.set_xlabel("total truth charge on channel  [electrons]")
    ax.set_ylabel(f"n channels with bias < -{th:g}")
    ax.set_title(f"in-efficiency channels  (plane: {args.plane})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    outname = f"{base_out}_ineff{ext_out}"
    fig.savefig(outname, dpi=130)
    plt.close(fig)
    print(f"wrote {outname}")

    # ---- 1D comparison: mean bias of normal channels -----------------------
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for i, (label, q_t, bias) in enumerate(results):
        sel = np.abs(bias) < th
        q, b = q_t[sel], bias[sel]
        idx = np.digitize(q, xedges1d) - 1
        means = np.full(len(xcenters1d), np.nan)
        for j in range(len(xcenters1d)):
            bj = b[idx == j]
            if bj.size:
                means[j] = bj.mean()
        ax.plot(xcenters1d, means, "o-", ms=3.5, lw=1.0,
                color=COLORS[i % len(COLORS)], label=label)
    ax.axhline(0, color="k", lw=0.8, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("total truth charge on channel  [electrons]")
    ax.set_ylabel("mean (Q_reco - Q_truth) / Q_truth")
    ax.set_title(f"mean bias of normal channels |bias| < {th:g}  (plane: {args.plane})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    outname = f"{base_out}_meanbias{ext_out}"
    fig.savefig(outname, dpi=130)
    plt.close(fig)
    print(f"wrote {outname}")


if __name__ == "__main__":
    main()
