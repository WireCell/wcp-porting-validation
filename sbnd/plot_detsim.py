#!/usr/bin/env python3
"""Visualize artROOT detsim files: SimChannels, recob::Wire, raw::RawDigit.

Default: opens a window with two 2D heatmaps (SimChannels + Wire).

With --draw-rawdigits: adds a third 2D panel showing pedestal-subtracted
  raw::RawDigit waveforms; in --interactive mode the 1D panel overlays
  all three (simchannel, wire, rawdigit).

With --interactive: adds a 1D waveform panel below; click on any 2D
  plot to display the waveform for that channel.

With --out-prefix PREFIX: saves PDFs and .npy arrays, no GUI.

Example:
  python plot_detsim.py --input ../dune/dune10kt-vd/detsim.root \\
      --channel-min 0 --channel-max 2000 --tick-min 0 --tick-max 4095

  python plot_detsim.py --input ../dune/dune10kt-vd/detsim.root \\
      --channel-min 0 --channel-max 2000 --interactive --draw-rawdigits
"""

import argparse
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

DEFAULT_INPUT = "standalone-sample/2025f-mc.root"
# Defaults below target SBND art ROOT files (producer=simtpc2d, process=DetSim,
# with the `.obj` suffix that TTreeReaderArray needs to read the vector
# payload of an art product).  Override via --simch-branch / --wire-branch /
# --rawdigit-branch when running on other detectors (e.g. DUNE uses
# producer=tpcrawdecoder, process=detsim).
DEFAULT_SIMCH_TAG = "simpleSC"
_SIMCH_BRANCH_TEMPLATE = "sim::SimChannels_simtpc2d_{tag}_DetSim.obj"
DEFAULT_WIRE_TAG = "dnnsp"
_WIRE_BRANCH_TEMPLATE = "recob::Wires_simtpc2d_{tag}_DetSim.obj"
DEFAULT_RAWDIGIT_TAG = "daq"
_RAWDIGIT_BRANCH_TEMPLATE = "raw::RawDigits_simtpc2d_{tag}_DetSim.obj"

# Warn if the dense array would exceed this many elements
_ARRAY_WARN_LIMIT = 500_000_000


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", default=DEFAULT_INPUT, help="artROOT file path")
    p.add_argument("--entry", type=int, default=0, help="Events tree entry (default 0)")
    p.add_argument(
        "--simch-tag",
        default=DEFAULT_SIMCH_TAG,
        metavar="TAG",
        help=f"SimChannels product instance tag (default: {DEFAULT_SIMCH_TAG})",
    )
    p.add_argument(
        "--simch-branch",
        default=None,
        help="Override SimChannels branch name directly (overrides --simch-tag)",
    )
    p.add_argument(
        "--wire-tag",
        default=DEFAULT_WIRE_TAG,
        metavar="TAG",
        help="recob::Wire product instance tag: gauss|wiener|dnnsp (default: gauss)",
    )
    p.add_argument(
        "--wire-branch",
        default=None,
        help="Override wire branch name directly (overrides --wire-tag)",
    )
    p.add_argument(
        "--draw-rawdigits",
        action="store_true",
        help="Also read and display raw::RawDigit waveforms (adds a 3rd 2D panel; "
             "the 1D panel shows simchannel, wire, and rawdigit overlaid). Default off.",
    )
    p.add_argument(
        "--rawdigit-tag",
        default=DEFAULT_RAWDIGIT_TAG,
        metavar="TAG",
        help=f"raw::RawDigit product instance tag (default: {DEFAULT_RAWDIGIT_TAG})",
    )
    p.add_argument(
        "--rawdigit-branch",
        default=None,
        help="Override RawDigit branch name directly (overrides --rawdigit-tag)",
    )
    p.add_argument("--channel-min", type=int, default=None, metavar="N")
    p.add_argument("--channel-max", type=int, default=None, metavar="N")
    p.add_argument("--tick-min", type=int, default=None, metavar="N")
    p.add_argument("--tick-max", type=int, default=None, metavar="N")
    p.add_argument(
        "--simch-tdc-offset",
        type=int,
        default=2990,
        metavar="N",
        help="Subtract N from each SimChannel TDC so deposits line up with the "
             "Wire/RawDigit tick axis. Default 2990 matches SBND 2025f-mc "
             "(SimChannel TDCs start at 2990 there). Use 0 to disable shifting.",
    )
    p.add_argument(
        "--vmax-percentile",
        type=float,
        default=99.0,
        help="Percentile of non-zero values used for auto vmax (default 99.0)",
    )
    p.add_argument("--cmap", default="YlOrRd", help="Matplotlib colormap (default YlOrRd)")
    p.add_argument(
        "--interactive",
        action="store_true",
        help="Enable click-to-show-1D waveform mode (adds a third panel below the 2D plots)",
    )
    p.add_argument(
        "--out-prefix",
        default=None,
        metavar="PREFIX",
        help="Save PREFIX_simch.pdf / PREFIX_wire.pdf / .npy instead of opening a GUI",
    )
    return p.parse_args()


def import_root():
    try:
        import ROOT  # noqa: PLC0415

        ROOT.gErrorIgnoreLevel = ROOT.kWarning
        return ROOT
    except ImportError:
        sys.exit("PyROOT not available — source your LArSoft/DUNE setup first.")


def open_tree(ROOT, filename):
    f = ROOT.TFile.Open(filename)
    if not f or f.IsZombie():
        sys.exit(f"Cannot open file: {filename}")
    tree = f.Get("Events")
    if not tree:
        sys.exit(f"No 'Events' TTree found in {filename}")
    return f, tree


def _vec_to_numpy(vec):
    """Convert a ROOT std::vector<float> to a numpy array efficiently."""
    try:
        return np.frombuffer(vec.data(), dtype=np.float32, count=vec.size()).copy()
    except Exception:
        return np.array([vec[j] for j in range(vec.size())], dtype=np.float32)


def read_simchannels(ROOT, tree, branch, entry, ch_min, ch_max, tick_min, tick_max,
                     tdc_offset=0):
    """Return (deposits, totals).

    deposits: sparse dict {channel: {tdc_shifted: total_electrons}} restricted to
      the range, with tdc_shifted = (raw_tdc - tdc_offset).  Use tdc_offset to
      align SimChannel TDC with the Wire/RawDigit tick axis.
    totals: dict with charge_all, energy_all, charge_in_range, energy_in_range.
    """
    reader = ROOT.TTreeReader(tree)
    simchs = ROOT.TTreeReaderArray("sim::SimChannel")(reader, branch)
    # SetEntry returns kEntryValid=0 on success; treat any non-zero as failure
    if reader.SetEntry(entry) != 0:
        sys.exit(f"Entry {entry} not found; SimChannels branch: {branch!r}")

    deposits = defaultdict(lambda: defaultdict(float))
    charge_all = 0.0
    energy_all = 0.0
    charge_in_range = 0.0
    energy_in_range = 0.0
    n = simchs.GetSize()
    print(f"  SimChannels: {n} objects in branch (tdc_offset={tdc_offset})",
          file=sys.stderr)

    for i in range(n):
        sc = simchs.At(i)
        ch = int(sc.Channel())
        ch_in = (ch_min is None or ch >= ch_min) and (ch_max is None or ch <= ch_max)
        for pair in sc.TDCIDEMap():
            tdc = int(pair.first) - tdc_offset
            charge = 0.0
            energy = 0.0
            for ide in pair.second:
                charge += ide.numElectrons
                energy += ide.energy
            charge_all += charge
            energy_all += energy
            tdc_in = (tick_min is None or tdc >= tick_min) and (tick_max is None or tdc <= tick_max)
            if ch_in and tdc_in:
                charge_in_range += charge
                energy_in_range += energy
                deposits[ch][tdc] += charge

    print(f"  SimChannels: {len(deposits)} channels with deposits after filter", file=sys.stderr)
    totals = dict(
        charge_all=charge_all,
        energy_all=energy_all,
        charge_in_range=charge_in_range,
        energy_in_range=energy_in_range,
    )
    return deposits, totals


def read_wires(ROOT, tree, branch, entry, ch_min, ch_max, tick_min, tick_max):
    """Return (wire_data, totals).

    wire_data: dict {channel: np.ndarray} with full signal per channel, channel-filtered.
    totals: dict with adc_all (sum over every wire / every tick) and adc_in_range
      (channel-filtered and tick-window-restricted).
    """
    reader = ROOT.TTreeReader(tree)
    wires = ROOT.TTreeReaderArray("recob::Wire")(reader, branch)
    if reader.SetEntry(entry) != 0:
        sys.exit(f"Entry {entry} not found; Wire branch: {branch!r}")

    wire_data = {}
    adc_all = 0.0
    adc_in_range = 0.0
    n = wires.GetSize()
    print(f"  recob::Wire: {n} objects in branch", file=sys.stderr)

    for i in range(n):
        w = wires.At(i)
        ch = int(w.Channel())
        sig = _vec_to_numpy(w.Signal())
        adc_all += float(sig.sum())
        if ch_min is not None and ch < ch_min:
            continue
        if ch_max is not None and ch > ch_max:
            continue
        wire_data[ch] = sig
        lo = tick_min if tick_min is not None else 0
        hi = (tick_max + 1) if tick_max is not None else len(sig)
        lo = max(0, lo)
        hi = min(len(sig), hi)
        if lo < hi:
            adc_in_range += float(sig[lo:hi].sum())

    print(f"  recob::Wire: {len(wire_data)} channels after filter", file=sys.stderr)
    totals = dict(adc_all=adc_all, adc_in_range=adc_in_range)
    return wire_data, totals


def _short_vec_to_numpy(vec):
    try:
        return np.frombuffer(vec.data(), dtype=np.int16, count=vec.size()).astype(
            np.float32, copy=True
        )
    except Exception:
        return np.array([vec[j] for j in range(vec.size())], dtype=np.float32)


def read_rawdigits(ROOT, tree, branch, entry, ch_min, ch_max, tick_min, tick_max):
    """Return (rd_data, totals).

    rd_data: dict {channel: np.ndarray} of pedestal-subtracted, uncompressed
      ADC waveforms, channel-filtered.
    totals: dict with adc_all and adc_in_range (pedestal-subtracted, summed).
    Compression is uncompressed via raw::Uncompress when not kNone (==0).
    """
    reader = ROOT.TTreeReader(tree)
    digits = ROOT.TTreeReaderArray("raw::RawDigit")(reader, branch)
    if reader.SetEntry(entry) != 0:
        sys.exit(f"Entry {entry} not found; RawDigit branch: {branch!r}")

    rd_data = {}
    adc_all = 0.0
    adc_in_range = 0.0
    n = digits.GetSize()
    print(f"  raw::RawDigit: {n} objects in branch", file=sys.stderr)

    uncomp_buf = ROOT.std.vector("short")()
    warned_compress = False

    for i in range(n):
        rd = digits.At(i)
        ch = int(rd.Channel())
        nsamp = int(rd.Samples())
        comp = int(rd.Compression())
        if comp == 0:
            sig = _short_vec_to_numpy(rd.ADCs())
        else:
            uncomp_buf.clear()
            uncomp_buf.resize(nsamp)
            try:
                ROOT.raw.Uncompress(rd.ADCs(), uncomp_buf, rd.Compression())
                sig = _short_vec_to_numpy(uncomp_buf)
            except Exception:
                if not warned_compress:
                    print(
                        f"  WARNING: cannot call raw::Uncompress (compression={comp}); "
                        "falling back to raw ADCs (may be compressed)",
                        file=sys.stderr,
                    )
                    warned_compress = True
                sig = _short_vec_to_numpy(rd.ADCs())
        ped = float(rd.GetPedestal())
        sig = sig - ped
        adc_all += float(sig.sum())
        if ch_min is not None and ch < ch_min:
            continue
        if ch_max is not None and ch > ch_max:
            continue
        rd_data[ch] = sig
        lo = tick_min if tick_min is not None else 0
        hi = (tick_max + 1) if tick_max is not None else len(sig)
        lo = max(0, lo)
        hi = min(len(sig), hi)
        if lo < hi:
            adc_in_range += float(sig[lo:hi].sum())

    print(f"  raw::RawDigit: {len(rd_data)} channels after filter", file=sys.stderr)
    totals = dict(adc_all=adc_all, adc_in_range=adc_in_range)
    return rd_data, totals


def choose_range(label, req_min, req_max, values):
    lo = req_min if req_min is not None else int(min(values))
    hi = req_max if req_max is not None else int(max(values))
    print(f"  {label} range: [{lo}, {hi}]", file=sys.stderr)
    return lo, hi


def build_simch_array(deposits, ch_range, tick_range):
    ch_lo, ch_hi = ch_range
    tick_lo, tick_hi = tick_range
    nch = ch_hi - ch_lo + 1
    ntick = tick_hi - tick_lo + 1
    _warn_size(nch, ntick)
    arr = np.zeros((nch, ntick), dtype=np.float64)
    for ch, tdc_map in deposits.items():
        ci = ch - ch_lo
        if ci < 0 or ci >= nch:
            continue
        for tdc, charge in tdc_map.items():
            ti = tdc - tick_lo
            if 0 <= ti < ntick:
                arr[ci, ti] += charge
    return arr


def build_wire_array(wire_data, ch_range, tick_range):
    ch_lo, ch_hi = ch_range
    tick_lo, tick_hi = tick_range
    nch = ch_hi - ch_lo + 1
    ntick = tick_hi - tick_lo + 1
    arr = np.zeros((nch, ntick), dtype=np.float32)
    for ch, sig in wire_data.items():
        ci = ch - ch_lo
        if ci < 0 or ci >= nch:
            continue
        src_lo = tick_lo
        src_hi = min(tick_lo + ntick, len(sig))
        if src_lo >= len(sig):
            continue
        length = src_hi - src_lo
        arr[ci, :length] = sig[src_lo:src_hi]
    return arr


def _warn_size(nch, ntick):
    total = nch * ntick
    if total > _ARRAY_WARN_LIMIT:
        mb = total * 8 / 1e6
        print(
            f"  WARNING: array is {nch}×{ntick} = {total:,} elements (~{mb:.0f} MB). "
            "Consider narrowing --channel-min/max or --tick-min/max.",
            file=sys.stderr,
        )


def _auto_vmax(arr, percentile):
    nonzero = arr[arr != 0]
    if len(nonzero) == 0:
        return 1.0
    return float(np.percentile(nonzero, percentile))


def _imshow_kwargs(arr, cmap, vmax_percentile):
    return dict(
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=0.0,
        vmax=_auto_vmax(arr, vmax_percentile),
        interpolation="nearest",
    )


def _draw_2d_panel(ax, arr, extent, title, cbar_label, cmap, vmax_percentile, fig):
    im = ax.imshow(arr.T, extent=extent, **_imshow_kwargs(arr, cmap, vmax_percentile))
    fig.colorbar(im, ax=ax, label=cbar_label, pad=0.02)
    ax.set_title(title)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Tick")
    return im


def show_2d_only(simch_arr, wire_arr, rd_arr, ch_range, tick_range, args):
    """Display 2 (or 3) stacked 2D heatmaps, no click handler."""
    ch_lo, ch_hi = ch_range
    tick_lo, tick_hi = tick_range
    extent = [ch_lo - 0.5, ch_hi + 0.5, tick_lo - 0.5, tick_hi + 0.5]

    nrows = 3 if rd_arr is not None else 2
    fig, axes = plt.subplots(nrows, 1, figsize=(14, 4 * nrows))
    if nrows == 2:
        ax_sc, ax_wr = axes
    else:
        ax_sc, ax_wr, ax_rd = axes

    _draw_2d_panel(ax_sc, simch_arr, extent,
                   f"sim::SimChannels ({args.simch_tag})", "electrons",
                   args.cmap, args.vmax_percentile, fig)
    _draw_2d_panel(ax_wr, wire_arr, extent,
                   f"recob::Wire ({args.wire_tag})",
                   f"ADC — recob::Wire ({args.wire_tag})",
                   args.cmap, args.vmax_percentile, fig)
    if rd_arr is not None:
        _draw_2d_panel(ax_rd, rd_arr, extent,
                       f"raw::RawDigit ({args.rawdigit_tag}, ped-subtracted)",
                       f"ADC — raw::RawDigit ({args.rawdigit_tag})",
                       args.cmap, args.vmax_percentile, fig)

    suptitle = (
        f"{args.input}  entry={args.entry}  simch={args.simch_tag}  wire={args.wire_tag}"
    )
    if rd_arr is not None:
        suptitle += f"  rawdigit={args.rawdigit_tag}"
    fig.suptitle(suptitle, fontsize=10)
    plt.tight_layout()
    plt.show()


def show_interactive(simch_arr, wire_arr, rd_arr, ch_range, tick_range, args):
    ch_lo, ch_hi = ch_range
    tick_lo, tick_hi = tick_range
    extent = [ch_lo - 0.5, ch_hi + 0.5, tick_lo - 0.5, tick_hi + 0.5]
    tick_axis = np.arange(tick_lo, tick_hi + 1)

    has_rd = rd_arr is not None
    if has_rd:
        fig, axes = plt.subplots(
            4, 1, figsize=(14, 13), gridspec_kw={"height_ratios": [2, 2, 2, 1.4]}
        )
        ax_sc, ax_wr, ax_rd, ax_1d = axes
    else:
        fig, axes = plt.subplots(
            3, 1, figsize=(14, 11), gridspec_kw={"height_ratios": [2, 2, 1.2]}
        )
        ax_sc, ax_wr, ax_1d = axes
        ax_rd = None

    _draw_2d_panel(ax_sc, simch_arr, extent,
                   f"sim::SimChannels ({args.simch_tag})", "electrons",
                   args.cmap, args.vmax_percentile, fig)
    _draw_2d_panel(ax_wr, wire_arr, extent,
                   f"recob::Wire ({args.wire_tag})",
                   f"ADC — recob::Wire ({args.wire_tag})",
                   args.cmap, args.vmax_percentile, fig)
    if has_rd:
        _draw_2d_panel(ax_rd, rd_arr, extent,
                       f"raw::RawDigit ({args.rawdigit_tag}, ped-subtracted)",
                       f"ADC — raw::RawDigit ({args.rawdigit_tag})",
                       args.cmap, args.vmax_percentile, fig)

    ax_1d.set_title("1D waveform — click on any 2D plot to select channel")
    ax_1d.set_xlabel("Tick")

    vlines = [ax_sc.axvline(x=ch_lo, color="cyan", lw=0.8, ls="--", visible=False),
              ax_wr.axvline(x=ch_lo, color="cyan", lw=0.8, ls="--", visible=False)]
    if has_rd:
        vlines.append(ax_rd.axvline(x=ch_lo, color="cyan", lw=0.8, ls="--", visible=False))

    click_axes = (ax_sc, ax_wr) + ((ax_rd,) if has_rd else ())
    twin_state = {"ax2": None}

    def update_1d(ch):
        idx = ch - ch_lo
        if idx < 0 or idx >= simch_arr.shape[0]:
            return

        ax_1d.cla()
        if twin_state["ax2"] is not None:
            try:
                twin_state["ax2"].remove()
            except Exception:
                pass
            twin_state["ax2"] = None

        simch_integral = float(simch_arr[idx].sum())
        wire_integral = float(wire_arr[idx].sum())

        # Left axis: ADC waveforms (wire + optional rawdigit)
        ax_1d.plot(tick_axis, wire_arr[idx], color="steelblue", lw=0.9,
                   label=f"Wire ({args.wire_tag}) ch={ch}  ∫={wire_integral:.3g} ADC·tick")
        if has_rd:
            rd_integral = float(rd_arr[idx].sum())
            ax_1d.plot(tick_axis, rd_arr[idx], color="forestgreen", lw=0.7, alpha=0.8,
                       label=f"RawDigit ({args.rawdigit_tag}) ch={ch}  "
                             f"∫={rd_integral:.3g} ADC·tick")
        ax_1d.set_ylabel("ADC", color="steelblue")
        ax_1d.tick_params(axis="y", labelcolor="steelblue")
        ax_1d.axhline(0, color="black", lw=0.3, ls=":")

        # Right (twin) axis: SimChannels electrons
        ax2 = ax_1d.twinx()
        twin_state["ax2"] = ax2
        ax2.plot(tick_axis, simch_arr[idx], color="darkorange", lw=0.9,
                 label=f"SimCh ch={ch}  ∫={simch_integral:.3g} e⁻")
        ax2.set_ylabel("electrons", color="darkorange")
        ax2.tick_params(axis="y", labelcolor="darkorange")

        title = (
            f"Channel {ch}   "
            f"∫SimCh = {simch_integral:.4g} e⁻   "
            f"∫Wire({args.wire_tag}) = {wire_integral:.4g} ADC·tick"
        )
        if has_rd:
            title += f"   ∫RawDigit({args.rawdigit_tag}) = {float(rd_arr[idx].sum()):.4g} ADC·tick"
        ax_1d.set_title(title)
        ax_1d.set_xlabel("Tick")

        lines1, labels1 = ax_1d.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax_1d.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

        for vl in vlines:
            vl.set_xdata([ch, ch])
            vl.set_visible(True)

        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes not in click_axes:
            return
        if event.xdata is None:
            return
        ch = int(round(event.xdata))
        ch = max(ch_lo, min(ch_hi, ch))
        update_1d(ch)

    fig.canvas.mpl_connect("button_press_event", on_click)
    suptitle = (
        f"{args.input}  entry={args.entry}  simch={args.simch_tag}  wire={args.wire_tag}"
    )
    if has_rd:
        suptitle += f"  rawdigit={args.rawdigit_tag}"
    fig.suptitle(suptitle, fontsize=10)
    plt.tight_layout()
    plt.show()


def save_static(simch_arr, wire_arr, rd_arr, ch_range, tick_range, args):
    ch_lo, ch_hi = ch_range
    tick_lo, tick_hi = tick_range
    extent = [ch_lo - 0.5, ch_hi + 0.5, tick_lo - 0.5, tick_hi + 0.5]
    prefix = args.out_prefix

    panels = [
        (simch_arr, f"simch_{args.simch_tag}", "electrons"),
        (wire_arr, f"wire_{args.wire_tag}", f"ADC ({args.wire_tag})"),
    ]
    if rd_arr is not None:
        panels.append((rd_arr, f"rawdigit_{args.rawdigit_tag}",
                       f"ADC ({args.rawdigit_tag}, ped-sub)"))
    for arr, label, unit in panels:
        fig, ax = plt.subplots(figsize=(14, 5))
        im = ax.imshow(
            arr.T, extent=extent, **_imshow_kwargs(arr, args.cmap, args.vmax_percentile)
        )
        fig.colorbar(im, ax=ax, label=unit)
        ax.set_title(label)
        ax.set_xlabel("Channel")
        ax.set_ylabel("Tick")
        fig.suptitle(f"{args.input}  entry={args.entry}", fontsize=9)
        plt.tight_layout()
        pdf_path = f"{prefix}_{label}.pdf"
        npy_path = f"{prefix}_{label}.npy"
        fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        np.save(npy_path, arr)
        print(f"Saved {pdf_path}  {npy_path}", file=sys.stderr)


def main():
    args = parse_args()
    simch_branch = args.simch_branch or _SIMCH_BRANCH_TEMPLATE.format(tag=args.simch_tag)
    wire_branch = args.wire_branch or _WIRE_BRANCH_TEMPLATE.format(tag=args.wire_tag)
    rawdigit_branch = (
        args.rawdigit_branch
        or _RAWDIGIT_BRANCH_TEMPLATE.format(tag=args.rawdigit_tag)
    )

    ROOT = import_root()
    f, tree = open_tree(ROOT, args.input)

    print("Reading SimChannels...", file=sys.stderr)
    deposits, sc_totals = read_simchannels(
        ROOT, tree, simch_branch, args.entry,
        args.channel_min, args.channel_max, args.tick_min, args.tick_max,
        tdc_offset=args.simch_tdc_offset,
    )

    print("Reading recob::Wire...", file=sys.stderr)
    wire_data, wire_totals = read_wires(
        ROOT, tree, wire_branch, args.entry,
        args.channel_min, args.channel_max, args.tick_min, args.tick_max,
    )

    rd_data = None
    rd_totals = None
    if args.draw_rawdigits:
        print("Reading raw::RawDigits...", file=sys.stderr)
        rd_data, rd_totals = read_rawdigits(
            ROOT, tree, rawdigit_branch, args.entry,
            args.channel_min, args.channel_max, args.tick_min, args.tick_max,
        )

    ch_filt = (args.channel_min, args.channel_max)
    tk_filt = (args.tick_min, args.tick_max)
    print("=" * 60)
    print("Totals:")
    print(f"  1. SimChannels charge (all):       {sc_totals['charge_all']:.6g} e-")
    print(f"  2. SimChannels energy (all):       {sc_totals['energy_all']:.6g} MeV")
    print(f"  3. recob::Wire ADC (all):          {wire_totals['adc_all']:.6g} ADC*tick")
    print(f"  4. SimChannels charge in ch={ch_filt}, tick={tk_filt}: "
          f"{sc_totals['charge_in_range']:.6g} e-")
    print(f"  5. SimChannels energy in ch={ch_filt}, tick={tk_filt}: "
          f"{sc_totals['energy_in_range']:.6g} MeV")
    print(f"  6. recob::Wire ADC in ch={ch_filt}, tick={tk_filt}:    "
          f"{wire_totals['adc_in_range']:.6g} ADC*tick")
    if rd_totals is not None:
        print(f"  7. raw::RawDigit ADC (all, ped-sub):       "
              f"{rd_totals['adc_all']:.6g} ADC*tick")
        print(f"  8. raw::RawDigit ADC in ch={ch_filt}, tick={tk_filt}: "
              f"{rd_totals['adc_in_range']:.6g} ADC*tick")
    print("=" * 60)

    all_chs = list(deposits.keys()) + list(wire_data.keys())
    if rd_data is not None:
        all_chs += list(rd_data.keys())
    if not all_chs:
        sys.exit("No channels found in the requested range.")

    all_tdcs = [tdc for d in deposits.values() for tdc in d]
    wire_ticks = [len(sig) - 1 for sig in wire_data.values() if len(sig) > 0]
    if rd_data is not None:
        wire_ticks += [len(sig) - 1 for sig in rd_data.values() if len(sig) > 0]

    ch_range = choose_range("channel", args.channel_min, args.channel_max, all_chs)

    # Determine tick range: prefer user limits; fall back to data extents
    tick_values = (all_tdcs or [0]) + (
        [args.tick_min or 0] + ([max(wire_ticks)] if wire_ticks else [])
    )
    tick_range = choose_range("tick", args.tick_min, args.tick_max, tick_values)

    print("Building dense arrays...", file=sys.stderr)
    simch_arr = build_simch_array(deposits, ch_range, tick_range)
    wire_arr = build_wire_array(wire_data, ch_range, tick_range)
    rd_arr = build_wire_array(rd_data, ch_range, tick_range) if rd_data is not None else None

    if args.out_prefix:
        save_static(simch_arr, wire_arr, rd_arr, ch_range, tick_range, args)
    elif args.interactive:
        show_interactive(simch_arr, wire_arr, rd_arr, ch_range, tick_range, args)
    else:
        show_2d_only(simch_arr, wire_arr, rd_arr, ch_range, tick_range, args)

    f.Close()


if __name__ == "__main__":
    main()
