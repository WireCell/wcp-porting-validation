#!/usr/bin/env python3
"""SBND cross-APA flash coincidence analysis.

Flash information is recorded separately for each APA/TPC (APA0, APA1) in
opflash_apa{0,1}.tar.gz, for both simulation (mc) and real data. Each archive
holds, per event, an array [nflash, 313]: column 0 = flash time in ns,
columns 1..312 = per-channel PE.

This script:
  1. Dumps every flash (mc+data, both APAs, all events) to flash_dump.csv.
  2. Computes the cross-APA time difference dt = t(APA0) - t(APA1), within the
     same event, two ways: all-pairs and nearest-neighbor.
  3. Plots dt histograms (mc vs data) in a wide +-10 us window and a data-driven
     zoom window, plus the per-flash total-PE distributions.

Usage:
    python3 flash_coincidence.py [--zoom US]
"""
import argparse
import io
import os
import re
import tarfile

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(HERE, "input_files")
PICS = os.path.join(HERE, "pics")
DUMP = os.path.join(HERE, "flash_dump.csv")

NS_PER_US = 1000.0
EV_RE = re.compile(r"opflash_tensor_(\d+)_0_array\.npy$")

# mode -> input subdirectory holding opflash_apa{0,1}.tar.gz
MODES = {
    "mc": os.path.join(INPUT, "input-10evt-mc"),
    "data": os.path.join(INPUT, "input-10evt-data"),
}


def load_archive(path):
    """Return {event_id: ndarray[nflash, 313]} from one opflash tar.gz."""
    out = {}
    with tarfile.open(path, "r:gz") as t:
        for m in t.getmembers():
            mo = EV_RE.search(m.name)
            if not mo:
                continue
            ev = int(mo.group(1))
            out[ev] = np.load(io.BytesIO(t.extractfile(m).read()))
    return out


def load_mode(d):
    """Return (apa0, apa1) dicts of {event_id: array} for one mode dir."""
    a0 = load_archive(os.path.join(d, "opflash_apa0.tar.gz"))
    a1 = load_archive(os.path.join(d, "opflash_apa1.tar.gz"))
    return a0, a1


def flash_records(mode, apa, store):
    """Yield (mode, event, apa, idx, time_ns, total_pe, nchan_hit) per flash."""
    for ev in sorted(store):
        arr = store[ev]
        pe = arr[:, 1:]
        total_pe = pe.sum(axis=1)
        nchan_hit = (pe > 0).sum(axis=1)
        for i in range(arr.shape[0]):
            yield (mode, ev, apa, i, float(arr[i, 0]),
                   float(total_pe[i]), int(nchan_hit[i]))


def dump_csv(loaded):
    """Write the single dedicated dump file; return total flash count."""
    n = 0
    with open(DUMP, "w") as f:
        f.write("mode,event,apa,flash_index,time_ns,total_pe,nchan_hit\n")
        for mode in ("mc", "data"):
            a0, a1 = loaded[mode]
            for apa, store in ((0, a0), (1, a1)):
                for rec in flash_records(mode, apa, store):
                    f.write("%s,%d,%d,%d,%.6f,%.6f,%d\n" % rec)
                    n += 1
    return n


def coincidences(a0, a1):
    """All-pairs and nearest-neighbor dt (ns), within matched events only.

    dt = t(APA0 flash) - t(APA1 flash).
    Returns (allpairs, nearest) as 1-D ndarrays in ns.
    """
    allpairs = []
    nearest = []
    common = sorted(set(a0) & set(a1))
    assert set(a0) == set(a1), (
        "event IDs differ between APA0 and APA1: %s vs %s" % (sorted(a0), sorted(a1)))
    for ev in common:
        t0 = a0[ev][:, 0]
        t1 = a1[ev][:, 0]
        if t0.size == 0 or t1.size == 0:
            continue
        # all pairs: t0[i] - t1[j]
        diff = t0[:, None] - t1[None, :]
        allpairs.append(diff.ravel())
        # nearest neighbor, both directions (signed dt to closest other-APA flash)
        # for each APA0 flash -> nearest APA1
        nearest.append(diff[np.arange(t0.size), np.argmin(np.abs(diff), axis=1)])
        # for each APA1 flash -> nearest APA0
        nearest.append(-diff[np.argmin(np.abs(diff), axis=0), np.arange(t1.size)])
    allpairs = np.concatenate(allpairs) if allpairs else np.empty(0)
    nearest = np.concatenate(nearest) if nearest else np.empty(0)
    return allpairs, nearest


def coincident_pe_pairs(a0, a1, win_ns):
    """Total-PE pairs (pe_apa0, pe_apa1) for coincident flashes.

    For each APA0 flash, match it to the nearest APA1 flash in the same event;
    keep the pair only if |dt| <= win_ns. Returns ndarray [npair, 2].
    """
    pairs = []
    for ev in sorted(set(a0) & set(a1)):
        arr0, arr1 = a0[ev], a1[ev]
        t0, t1 = arr0[:, 0], arr1[:, 0]
        if t0.size == 0 or t1.size == 0:
            continue
        pe0 = arr0[:, 1:].sum(axis=1)
        pe1 = arr1[:, 1:].sum(axis=1)
        diff = np.abs(t0[:, None] - t1[None, :])
        j = np.argmin(diff, axis=1)              # nearest APA1 for each APA0
        ok = diff[np.arange(t0.size), j] <= win_ns
        for i in np.where(ok)[0]:
            pairs.append((pe0[i], pe1[j[i]]))
    return np.array(pairs) if pairs else np.empty((0, 2))


def make_pe2d_plot(loaded, win_ns, outfile):
    """Log-log scatter of coincident PE(APA0) vs PE(APA1), data vs mc overlaid."""
    fig, ax = plt.subplots(figsize=(7.5, 7))
    styles = {"mc": dict(marker="x", color="tab:blue", s=45, linewidths=1.3),
              "data": dict(marker="o", color="tab:red", s=36,
                           facecolors="none", linewidths=1.3)}
    allvals = []
    for mode in ("mc", "data"):
        p = coincident_pe_pairs(loaded[mode][0], loaded[mode][1], win_ns)
        allvals.append(p.ravel())
        # Pearson correlation of log10(PE) for the legend
        r = (np.corrcoef(np.log10(p[:, 0]), np.log10(p[:, 1]))[0, 1]
             if p.shape[0] > 1 else float("nan"))
        ax.scatter(p[:, 0], p[:, 1], label="%s (N=%d, r=%.2f)" % (mode, p.shape[0], r),
                   **styles[mode])
    v = np.concatenate(allvals)
    lo, hi = v[v > 0].min() * 0.7, v.max() * 1.4
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6, label="PE$_0$ = PE$_1$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("APA0 flash total PE")
    ax.set_ylabel("APA1 flash total PE")
    ax.set_title(r"Coincident cross-APA flash PE  ($|\Delta t|\leq%d$ ns)" % win_ns)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3, which="both")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(outfile, dpi=130)
    plt.close(fig)
    print("wrote", outfile)


def hist_panel(ax, mc_us, data_us, win_us, nbins, title):
    """Overlay MC and data dt histograms on one axis."""
    edges = np.linspace(-win_us, win_us, nbins + 1)
    for vals, color, label in ((mc_us, "tab:blue", "mc"),
                               (data_us, "tab:red", "data")):
        inside = np.sum(np.abs(vals) <= win_us)
        frac = inside / vals.size if vals.size else 0.0
        ax.hist(vals, bins=edges, histtype="step", color=color, linewidth=1.6,
                label="%s (N=%d, %.1f%% in window)" % (label, vals.size, 100 * frac))
    ax.set_xlabel(r"$\Delta t = t_{APA0} - t_{APA1}$  [$\mu$s]")
    ax.set_ylabel("flash pairs / bin")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(-win_us, win_us)


def make_coinc_plot(mc, data, win_us, nbins, title, outfile):
    fig, ax = plt.subplots(figsize=(8, 5))
    hist_panel(ax, mc / NS_PER_US, data / NS_PER_US, win_us, nbins, title)
    fig.tight_layout()
    fig.savefig(outfile, dpi=130)
    plt.close(fig)
    print("wrote", outfile)


def make_pe_plot(loaded, outfile):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, apa in zip(axes, (0, 1)):
        allpe = []
        for mode in ("mc", "data"):
            store = loaded[mode][apa]
            allpe.extend(s[:, 1:].sum(axis=1) for s in store.values())
        edges = np.logspace(np.log10(max(1.0, min(p.min() for p in allpe if p.size))),
                            np.log10(max(p.max() for p in allpe if p.size)), 40)
        for mode, color in (("mc", "tab:blue"), ("data", "tab:red")):
            store = loaded[mode][apa]
            pe = np.concatenate([s[:, 1:].sum(axis=1) for s in store.values()])
            ax.hist(pe, bins=edges, histtype="step", color=color, linewidth=1.6,
                    label="%s (N=%d)" % (mode, pe.size))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("total PE per flash")
        ax.set_title("APA%d" % apa)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("flashes / bin")
    fig.suptitle("Per-flash total-PE distribution (mc vs data)")
    fig.tight_layout()
    fig.savefig(outfile, dpi=130)
    plt.close(fig)
    print("wrote", outfile)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoom", type=float, default=None,
                    help="zoom half-window in us (default: 1.0)")
    ap.add_argument("--win-ns", type=float, default=80.0,
                    help="coincidence half-window in ns for the PE 2D plot (default: 80)")
    args = ap.parse_args()

    os.makedirs(PICS, exist_ok=True)
    loaded = {mode: load_mode(d) for mode, d in MODES.items()}

    n = dump_csv(loaded)
    print("flash_dump.csv: %d flashes" % n)

    coinc = {}
    for mode in ("mc", "data"):
        a0, a1 = loaded[mode]
        coinc[mode] = coincidences(a0, a1)  # (allpairs, nearest) in ns
    mc_all, mc_near = coinc["mc"]
    da_all, da_near = coinc["data"]

    # ---- wide +-10 us ----
    make_coinc_plot(mc_all, da_all, 10.0, 80,
                    r"All-pairs cross-APA $\Delta t$  (wide $\pm10\,\mu$s)",
                    os.path.join(PICS, "flash_coinc_allpairs_wide.png"))
    make_coinc_plot(mc_near, da_near, 10.0, 80,
                    r"Nearest-neighbor cross-APA $\Delta t$  (wide $\pm10\,\mu$s)",
                    os.path.join(PICS, "flash_coinc_nearest_wide.png"))

    # ---- PE distribution ----
    make_pe_plot(loaded, os.path.join(PICS, "flash_pe_dist.png"))

    # ---- coincident cross-APA PE 2D scatter (|dt| <= win_ns) ----
    make_pe2d_plot(loaded, args.win_ns, os.path.join(PICS, "flash_pe2d_coinc.png"))

    # ---- zoom window ----
    # The wide plots show the coincidence peak is confined to the central bin,
    # i.e. far narrower than the +-10 us window. The nearest-neighbor tail (flashes
    # with no real partner) makes high percentiles useless for sizing the peak, so
    # zoom to a fixed +-1 us by default with fine 20 ns bins to resolve the core.
    both_near = np.concatenate([mc_near, da_near])
    core = both_near[np.abs(both_near) <= NS_PER_US]
    print("nearest-neighbor |dt|<=1us core: N=%d, 68%%=%.1f ns, 95%%=%.1f ns"
          % (core.size, np.percentile(np.abs(core), 68), np.percentile(np.abs(core), 95)))
    zoom = args.zoom if args.zoom is not None else 1.0
    zbins = int(round(2 * zoom * NS_PER_US / 20.0))  # ~20 ns bins
    print("zoom half-window = %.3f us (%d bins)" % (zoom, zbins))

    make_coinc_plot(mc_all, da_all, zoom, zbins,
                    r"All-pairs cross-APA $\Delta t$  (zoom $\pm%.2g\,\mu$s)" % zoom,
                    os.path.join(PICS, "flash_coinc_allpairs_zoom.png"))
    make_coinc_plot(mc_near, da_near, zoom, zbins,
                    r"Nearest-neighbor cross-APA $\Delta t$  (zoom $\pm%.2g\,\mu$s)" % zoom,
                    os.path.join(PICS, "flash_coinc_nearest_zoom.png"))

    # ---- fine zoom: +-0.2 us, 5 ns bins, to resolve the ~16 ns core ----
    fine = 0.2
    fbins = int(round(2 * fine * NS_PER_US / 5.0))  # ~5 ns bins
    make_coinc_plot(mc_all, da_all, fine, fbins,
                    r"All-pairs cross-APA $\Delta t$  (fine $\pm%.2g\,\mu$s)" % fine,
                    os.path.join(PICS, "flash_coinc_allpairs_fine.png"))
    make_coinc_plot(mc_near, da_near, fine, fbins,
                    r"Nearest-neighbor cross-APA $\Delta t$  (fine $\pm%.2g\,\mu$s)" % fine,
                    os.path.join(PICS, "flash_coinc_nearest_fine.png"))

    # ---- summary stats for the doc ----
    def stats(v):
        return (v.size, np.median(v) / NS_PER_US,
                np.percentile(np.abs(v), 68) / NS_PER_US,
                np.sum(np.abs(v) <= zoom * NS_PER_US))
    print("\nsummary (zoom=%.3f us):" % zoom)
    for name, v in (("mc nearest", mc_near), ("data nearest", da_near),
                    ("mc allpairs", mc_all), ("data allpairs", da_all)):
        n_, med, p68, ninz = stats(v)
        print("  %-13s N=%5d  median=%.3f us  |dt|68%%=%.3f us  in-zoom=%d (%.1f%%)"
              % (name, n_, med, p68, ninz, 100 * ninz / n_))


if __name__ == "__main__":
    main()
