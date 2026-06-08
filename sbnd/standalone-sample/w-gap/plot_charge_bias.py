#!/usr/bin/env python3
"""Per-channel relative charge bias vs total truth charge.

For each input "file:alg" pair, integrate the reco charge per channel
(recob::Wire, scaled x50 to electrons) and the truth charge per channel
(sim::SimChannels numElectrons) over a tick window, then plot

    bias = (Q_reco - Q_truth) / Q_truth      vs      Q_truth

one point per (event, channel) with Q_truth > --min-truth.

Outputs (into the dir of -o):
  * <out>_<pair>            : 2D bias-vs-truth hist per pair (full ticks, all planes)
  * <out>_ineff, _meanbias  : 1D comparisons (full ticks, all planes)
  * <out>_grid/             : the two 1D metrics for every tick x plane combo
                              (3 tick windows x 4 planes = 12 configs)

Reuses the PyROOT data layer of compare_wires_viewer.py, so it must run in
the sbndcode environment (SL7 container + setup-local-opt.sh).

Examples:
    python plot_charge_bias.py
    python plot_charge_bias.py rb_mean/sp.root:gauss rb_none/sp.root:dnnsp
    python plot_charge_bias.py --tick-split 200 --min-truth 5e3
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
LINESTYLES = ["-", "--", ":", "-."]
PLANE_CFGS = ["u", "v", "w", "all"]


def plane_sel(chan, plane):
    """Boolean mask selecting absolute channel numbers in the given plane."""
    if plane == "all":
        return np.ones(len(chan), dtype=bool)
    pos = np.asarray(chan) % NCH_APA   # within-APA position (both APAs share layout)
    if plane == "u":
        return pos < NCH_U
    if plane == "v":
        return (pos >= NCH_U) & (pos < NCH_U + NCH_V)
    return pos >= NCH_U + NCH_V        # w


def extract_windows(wf, alg, nev, windows, min_truth):
    """One event loop -> {window_name: (chan, q_t, bias)} concatenated arrays.

    windows: list of (name, t0, t1) tick ranges; t1=None means "to end".
    Channels are kept (absolute index) so plane masking can be applied later.
    """
    acc = {name: ([], [], []) for name, _, _ in windows}
    for entry in range(nev):
        reco = wf.dense(entry, alg)            # electrons (x50 applied)
        truth = wf.dense(entry, "simchannel")
        nch = min(reco.shape[0], truth.shape[0])
        nt = min(reco.shape[1], truth.shape[1])
        chan_all = np.arange(nch)
        for name, t0, t1 in windows:
            b = max(0, t0)
            e = nt if t1 is None else min(t1, nt)
            q_r = reco[:nch, b:e].sum(axis=1)
            q_t = truth[:nch, b:e].sum(axis=1)
            sel = q_t > min_truth
            acc[name][0].append(chan_all[sel])
            acc[name][1].append(q_t[sel])
            acc[name][2].append((q_r[sel] - q_t[sel]) / q_t[sel])
    out = {}
    for name in acc:
        c, q, b = acc[name]
        out[name] = (np.concatenate(c), np.concatenate(q), np.concatenate(b))
    return out


def fig_hist2d(label, q_t, bias, xedges, yedges, title, outname, plt, LogNorm):
    """2D bias-vs-truth-charge histogram for one input."""
    H, _, _ = np.histogram2d(q_t, bias, bins=[xedges, yedges])
    fig, ax = plt.subplots(figsize=(9, 6.5))
    pcm = ax.pcolormesh(xedges, yedges, H.T, cmap="viridis",
                        norm=LogNorm(vmin=1, vmax=max(H.max(), 1)))
    fig.colorbar(pcm, ax=ax, label="channels / bin")
    ax.axhline(0, color="w", lw=0.8, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("total truth charge on channel  [electrons]")
    ax.set_ylabel("(Q_reco - Q_truth) / Q_truth")
    med = np.median(bias) if bias.size else float("nan")
    ax.set_title(f"{label}   ({title}, med {med:+.4f})")
    fig.tight_layout()
    fig.savefig(outname, dpi=130)
    plt.close(fig)
    print(f"wrote {outname}")


def fig_ineff(series, xedges1d, xcenters1d, th, title, outname, plt):
    """series: list of (label, chan, q_t, bias).  In-eff counts vs truth charge."""
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for i, (label, chan, q_t, bias) in enumerate(series):
        n_ineff, _ = np.histogram(q_t[bias < -th], bins=xedges1d)
        ntot = int((bias < -th).sum())
        color = COLORS[i % len(COLORS)]
        hollow = label.rpartition(":")[2] == "gauss"
        ax.plot(xcenters1d, n_ineff, MARKERS[i % len(MARKERS)], ms=6, color=color,
                markerfacecolor="none" if hollow else color,
                ls=LINESTYLES[i % len(LINESTYLES)], lw=1.0, alpha=0.9,
                label=f"{label}  (total {ntot})")
    ax.set_xscale("log")
    ax.set_xlabel("total truth charge on channel  [electrons]")
    ax.set_ylabel(f"n channels with bias < -{th:g}")
    ax.set_title(f"in-efficiency channels  ({title})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outname, dpi=130)
    plt.close(fig)
    print(f"wrote {outname}")


def fig_meanbias(series, xedges1d, xcenters1d, th, title, outname, plt):
    """series: list of (label, chan, q_t, bias).  Mean bias of normal channels."""
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for i, (label, chan, q_t, bias) in enumerate(series):
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
    ax.set_title(f"mean bias of normal channels |bias| < {th:g}  ({title})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outname, dpi=130)
    plt.close(fig)
    print(f"wrote {outname}")


def main():
    p = argparse.ArgumentParser(
        description="Per-channel relative charge bias vs truth charge.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("pairs", nargs="*", default=DEFAULT_PAIRS,
                   help="list of file:alg pairs (file relative to --base; "
                        "alg = gauss/wiener/dnnsp/...)")
    p.add_argument("--base", default=DEFAULT_BASE, help="base path for files")
    p.add_argument("--min-truth", type=float, default=1.0,
                   help="skip channels with Q_truth below this (electrons)")
    p.add_argument("--events", type=int, default=None,
                   help="max events per file (default: all)")
    p.add_argument("--tick-split", type=int, default=200,
                   help="boundary between the early/late tick windows for the grid")
    p.add_argument("--xbins", default="1e2,1e6,100",
                   help="x (truth charge) log binning for the 2D hists: min,max,nbins")
    p.add_argument("--ybins", default="-1,1,100",
                   help="y (bias) linear binning for the 2D hists: min,max,nbins")
    p.add_argument("--xbins1d", default="1e2,1e6,35",
                   help="x log binning for the 1D comparison figures: min,max,nbins")
    p.add_argument("--ineff-thresh", type=float, default=0.90,
                   help="bias < -thresh counts as in-efficiency; |bias| < thresh is 'normal'")
    p.add_argument("-o", "--output", default="charge_bias.png",
                   help="output figure path; per-pair tag inserted before the extension")
    args = p.parse_args()

    xlo, xhi, xn = (float(v) for v in args.xbins.split(","))
    ylo, yhi, yn = (float(v) for v in args.ybins.split(","))
    xedges = np.logspace(np.log10(xlo), np.log10(xhi), int(xn) + 1)
    yedges = np.linspace(ylo, yhi, int(yn) + 1)
    x1lo, x1hi, x1n = (float(v) for v in args.xbins1d.split(","))
    xedges1d = np.logspace(np.log10(x1lo), np.log10(x1hi), int(x1n) + 1)
    xcenters1d = np.sqrt(xedges1d[:-1] * xedges1d[1:])  # geometric centers

    split = args.tick_split
    # tick windows for the grid: (name, label, t0, t1)
    tick_cfgs = [
        (f"t0-{split}",   f"ticks [0,{split})",   0,     split),
        (f"t{split}-max", f"ticks [{split},max)", split, None),
        ("t0-max",        "all ticks",            0,     None),
    ]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    files = {}  # path -> WireFile (cache; pairs may share files)
    base_out, ext_out = os.path.splitext(args.output)
    ext_out = ext_out or ".png"
    griddir = base_out + "_grid"
    os.makedirs(griddir, exist_ok=True)

    windows = [(name, t0, t1) for name, _, t0, t1 in tick_cfgs]
    # data[tickname] = list over pairs of (label, chan, q_t, bias)
    data = {name: [] for name, _, _, _ in tick_cfgs}

    print(f"{'input':32s} {'tickwin':12s} {'npts':>7s} {'median':>8s} {'mean':>8s} {'rms':>8s}")
    for i, pair in enumerate(args.pairs):
        path_rel, _, alg = pair.rpartition(":")
        if not path_rel:
            sys.exit(f"bad pair '{pair}', expected file:alg")
        path = path_rel if os.path.isabs(path_rel) else os.path.join(args.base, path_rel)
        if path not in files:
            files[path] = WireFile(path)
        wf = files[path]
        nev = wf.nevents if args.events is None else min(args.events, wf.nevents)

        per = extract_windows(wf, alg, nev, windows, args.min_truth)
        label = f"{path_rel}:{alg}"
        for name in per:
            chan, q_t, bias = per[name]
            data[name].append((label, chan, q_t, bias))
            print(f"{label:32s} {name:12s} {bias.size:7d} {np.median(bias):8.4f} "
                  f"{bias.mean():8.4f} {bias.std():8.4f}")

        # per-pair 2D hist: full ticks, all planes (unchanged top-level output)
        _, q_t, bias = per["t0-max"]
        tag = label.replace("/", "_").replace(":", "_").replace(".root", "")
        fig_hist2d(label, q_t, bias, xedges, yedges,
                   f"all ticks/planes, nev={nev}",
                   f"{base_out}_{tag}{ext_out}", plt, LogNorm)

    th = args.ineff_thresh

    # top-level 1D comparisons (full ticks, all planes) -- as before
    fig_ineff(data["t0-max"], xedges1d, xcenters1d, th, "plane: all, all ticks",
              f"{base_out}_ineff{ext_out}", plt)
    fig_meanbias(data["t0-max"], xedges1d, xcenters1d, th, "plane: all, all ticks",
                 f"{base_out}_meanbias{ext_out}", plt)

    # grid: 3 tick windows x 4 planes, both metrics
    for tname, tlabel, _, _ in tick_cfgs:
        for plane in PLANE_CFGS:
            # apply the plane mask to all three arrays consistently
            series = []
            for (lab, c, q, b) in data[tname]:
                m = plane_sel(c, plane)
                series.append((lab, c[m], q[m], b[m]))
            title = f"plane: {plane}, {tlabel}"
            fig_ineff(series, xedges1d, xcenters1d, th, title,
                      os.path.join(griddir, f"ineff_{tname}_{plane}{ext_out}"), plt)
            fig_meanbias(series, xedges1d, xcenters1d, th, title,
                         os.path.join(griddir, f"meanbias_{tname}_{plane}{ext_out}"), plt)
            # one 2D bias-vs-truth hist per input for this config
            for (lab, c, q, b) in series:
                ptag = lab.replace("/", "_").replace(":", "_").replace(".root", "")
                fig_hist2d(lab, q, b, xedges, yedges, title,
                           os.path.join(griddir, f"hist2d_{tname}_{plane}_{ptag}{ext_out}"),
                           plt, LogNorm)

    nconf = len(tick_cfgs) * len(PLANE_CFGS)
    print(f"grid written to {griddir}/ ({nconf} configs x [ineff, meanbias, "
          f"{len(args.pairs)} x hist2d])")


if __name__ == "__main__":
    main()
