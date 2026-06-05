#!/usr/bin/env python3
"""
Overlay 1D waveforms for a single channel across multiple processing algorithms.

The input ROOT file holds TH2F histograms named:

    h<plane><alg><apa>        e.g.  hw_gauss0

where
    <plane> in {u, v, w}
    <alg>   one of the processing stages (orig, raw, gauss, wiener, ...)
    <apa>   in {0, 1}

For each TH2F the x-axis is the (absolute) channel number and the y-axis is the
time tick; the bin content is the ADC / signal value.  This script picks one
channel column out of every selected histogram and overlays the resulting 1D
waveforms (value vs. time tick) on a single figure.

Because the different algorithms live on wildly different amplitude scales
(raw ADC ~ thousands, ROI masks ~ 0/1, etc.) the waveforms are, by default,
normalized so that every curve shares the same min/max level, making their
shapes directly comparable.

"""

import argparse
import re
import sys

import numpy as np
import uproot

# Default ordering of algorithms (nice, roughly chronological pipeline order).
# Any algorithm found in the file that is not listed here is appended at the end.
ALG_ORDER = [
    "orig",
    "raw",
    "loose_lf",
    "tight_lf",
    "wiener",
    "gauss",
    "extend_roi",
    "shrink_roi",
    "break_roi_1st",
    "break_roi_2nd",
    "cleanup_roi",
]

NAME_RE = re.compile(r"^h([uvw])_(.+?)([01])$")

# Per-algorithm plot style overrides. Anything not listed falls back to the
# turbo colormap with a plain solid line.
STYLE = {
    "raw":      dict(color="#2ca02c", linestyle="-", marker=None, lw=1.3),
    "tight_lf": dict(color="#d62728", linestyle="None", marker=".", ms=4),
}


def discover(file):
    """Return {plane: {apa: {alg: histname}}} for all TH2 in the file."""
    out = {}
    for key in file.keys():
        name = key.split(";")[0]
        m = NAME_RE.match(name)
        if not m:
            continue
        plane, alg, apa = m.group(1), m.group(2), m.group(3)
        out.setdefault(plane, {}).setdefault(apa, {})[alg] = name
    return out


def sort_algs(algs):
    known = [a for a in ALG_ORDER if a in algs]
    extra = sorted(a for a in algs if a not in ALG_ORDER)
    return known + extra


def channel_to_index(hist, channel):
    """Return the x-bin index (0-based, no flow bins) holding `channel`."""
    edges = hist.axis(0).edges()
    if channel < edges[0] or channel > edges[-1]:
        return None
    # np.searchsorted gives the bin to the right of `channel`.
    idx = int(np.searchsorted(edges, channel, side="right") - 1)
    idx = max(0, min(idx, len(edges) - 2))
    return idx


def normalize(wf, mode):
    """Scale a waveform so curves share a common level."""
    wf = wf.astype(float)
    if mode == "none":
        return wf
    if mode == "minmax":  # -> [0, 1]
        lo, hi = np.nanmin(wf), np.nanmax(wf)
        return (wf - lo) / (hi - lo) if hi > lo else wf - lo
    if mode == "maxabs":  # -> [-1, 1], preserves baseline sign
        m = np.nanmax(np.abs(wf))
        return wf / m if m > 0 else wf
    raise ValueError(mode)


def main():
    p = argparse.ArgumentParser(
        description="Overlay 1D waveforms across algorithms for one channel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-f", "--file", default="sbnd-data-check.root",
                   help="input ROOT file")
    p.add_argument("-p", "--plane", default="w", choices=["u", "v", "w"],
                   help="wire plane")
    p.add_argument("-a", "--apa", default="0", choices=["0", "1"],
                   help="APA index")
    p.add_argument("-c", "--channel", type=int, default=5259,
                   help="absolute channel number to visualize")
    p.add_argument("--algs", default="all",
                   help="comma-separated list of algorithms, or 'all'")
    p.add_argument("--norm", default="minmax",
                   choices=["minmax", "maxabs", "none"],
                   help="per-waveform scaling so curves share a level")
    p.add_argument("--tmin", type=float, default=None, help="min time tick to plot")
    p.add_argument("--tmax", type=float, default=None, help="max time tick to plot")
    p.add_argument("-o", "--output", default=None,
                   help="save figure to this path instead of showing it")
    p.add_argument("--list", action="store_true",
                   help="list available planes/apas/algs and exit")
    args = p.parse_args()

    file = uproot.open(args.file)
    catalog = discover(file)

    if args.list:
        for plane in sorted(catalog):
            for apa in sorted(catalog[plane]):
                algs = sort_algs(catalog[plane][apa].keys())
                print(f"plane={plane} apa={apa}: {', '.join(algs)}")
        return

    if args.plane not in catalog or args.apa not in catalog[args.plane]:
        sys.exit(f"No histograms for plane={args.plane} apa={args.apa}")
    available = catalog[args.plane][args.apa]

    if args.algs == "all":
        algs = sort_algs(available.keys())
    else:
        algs = [a.strip() for a in args.algs.split(",") if a.strip()]
        missing = [a for a in algs if a not in available]
        if missing:
            sys.exit(f"Unknown algs {missing}; available: {sort_algs(available.keys())}")

    # Import matplotlib only now so --list/-h stay fast and headless-safe.
    import matplotlib
    if args.output:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 6))
    cmap = plt.get_cmap("turbo")
    n = len(algs)
    plotted = 0

    for i, alg in enumerate(algs):
        hist = file[available[alg]]
        idx = channel_to_index(hist, args.channel)
        if idx is None:
            edges = hist.axis(0).edges()
            print(f"  skip {alg}: channel {args.channel} outside "
                  f"[{edges[0]:.0f}, {edges[-1]:.0f}]")
            continue
        values = hist.values()          # shape (nchan, ntick)
        wf = values[idx, :]
        ticks = hist.axis(1).edges()
        centers = 0.5 * (ticks[:-1] + ticks[1:])

        sel = np.ones_like(centers, dtype=bool)
        if args.tmin is not None:
            sel &= centers >= args.tmin
        if args.tmax is not None:
            sel &= centers <= args.tmax

        y = normalize(wf[sel], args.norm)
        style = STYLE.get(alg, dict(color=cmap(i / max(1, n - 1)), lw=1.1))
        ax.plot(centers[sel], y, label=alg, **style)
        plotted += 1

    if plotted == 0:
        sys.exit("Nothing to plot.")

    ylabel = {
        "minmax": "normalized signal  (min=0, max=1)",
        "maxabs": "normalized signal  (|max|=1)",
        "none": "signal value",
    }[args.norm]
    ax.set_xlabel("time tick")
    ax.set_ylabel(ylabel)
    ax.set_title(f"plane={args.plane}  APA={args.apa}  channel={args.channel}"
                 f"   (norm={args.norm})")
    ax.legend(ncol=2, fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=130)
        print(f"wrote {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
