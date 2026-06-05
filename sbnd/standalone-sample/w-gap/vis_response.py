#!/usr/bin/env python3
"""
View the OmnibusSigProc W-plane debug dump written by decon_2D_init():

    osp_predecon_w_anode<N>.npz
        data_predecon : (fft_nwires, fft_nticks)  pre-decon data (post-NF, FFT-padded)
        response      : (nrel_wires, fft_nticks)  time-domain overall response FRxER;
                        rows are RELATIVE wires of the wire-region-averaged kernel
                        (the fine impact positions were averaged per wire in
                        init_overall_response), central wire = row `wire_shift`
        meta          : (1, 5) = [pad_nwires, nwires, nticks, wire_shift, period]

Examples:
    python vis_response.py -f osp_predecon_w_anode0.npz                      # response, central wire, 1D
    python vis_response.py --mode 2d                                         # response, 2D
    python vis_response.py --wires -2,-1,0,1,2 --tmax 400                    # several relative wires
    python vis_response.py --what data -c 5259 --mode 1d                     # pre-decon waveform of channel 5259
    python vis_response.py --what data --mode 2d                             # pre-decon matrix
"""

import argparse
import sys

import numpy as np

# first W-plane channel (= nwire_u + nwire_v), for --channel on the data matrix
W_CHANNEL_OFFSET = 3968


def main():
    p = argparse.ArgumentParser(
        description="View W-plane overall response / pre-decon data dump.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-f", "--file", default="osp_predecon_w_anode0.npz",
                   help="input npz dump")
    p.add_argument("--what", default="response", choices=["response", "data"],
                   help="which array to view")
    p.add_argument("--mode", default="1d", choices=["1d", "2d"],
                   help="single-curve overlay or 2D image")
    p.add_argument("--wires", default="0",
                   help="comma-separated RELATIVE wires for response 1d (0=central)")
    p.add_argument("-c", "--channel", type=int, default=None,
                   help="absolute channel for data 1d (default: central wire row)")
    p.add_argument("--tmin", type=float, default=None, help="min time tick")
    p.add_argument("--tmax", type=float, default=None, help="max time tick")
    p.add_argument("-o", "--output", default=None,
                   help="save figure instead of showing")
    args = p.parse_args()

    npz = np.load(args.file)
    pad_nwires, nwires, nticks, wire_shift, period = npz["meta"][0]
    pad_nwires, nwires, nticks, wire_shift = map(int, (pad_nwires, nwires, nticks, wire_shift))

    import matplotlib
    if args.output:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    if args.what == "response":
        arr = npz["response"]           # (nrel, fft_nticks)
        nrel = arr.shape[0]
        ticks = np.arange(arr.shape[1])

        if args.mode == "2d":
            im = ax.imshow(arr, aspect="auto", origin="lower", cmap="RdBu_r",
                           vmin=-np.abs(arr).max(), vmax=np.abs(arr).max(),
                           extent=[0, arr.shape[1], -wire_shift - 0.5, nrel - wire_shift - 0.5])
            fig.colorbar(im, ax=ax, label="response  [ADC per electron-ish]")
            ax.set_ylabel("relative wire (0 = central)")
            ax.set_title(f"{args.file}: overall response (FR x ER), {nrel} relative wires")
        else:
            sel = np.ones_like(ticks, dtype=bool)
            if args.tmin is not None:
                sel &= ticks >= args.tmin
            if args.tmax is not None:
                sel &= ticks <= args.tmax
            for w in [int(s) for s in args.wires.split(",") if s.strip()]:
                row = wire_shift + w
                if not 0 <= row < nrel:
                    print(f"  skip relative wire {w}: row {row} outside [0, {nrel})")
                    continue
                ax.plot(ticks[sel], arr[row, sel], lw=1.2,
                        label=f"rel wire {w:+d}" + ("  (central)" if w == 0 else ""))
            ax.set_ylabel("response amplitude")
            ax.set_title(f"{args.file}: overall response (FR x ER) vs time tick "
                         f"(tick = {period/1000:.0f} ns)")
            ax.legend(fontsize=8)
    else:
        arr = npz["data_predecon"]      # (fft_nwires, fft_nticks)
        ticks = np.arange(arr.shape[1])

        if args.mode == "2d":
            v = np.percentile(np.abs(arr), 99.5)
            im = ax.imshow(arr, aspect="auto", origin="lower", cmap="RdBu_r",
                           vmin=-v, vmax=v)
            fig.colorbar(im, ax=ax, label="value")
            ax.set_ylabel(f"row (wire + pad {pad_nwires}); W chans start at row {pad_nwires}")
            ax.set_title(f"{args.file}: pre-decon data (post-NF), "
                         f"{nwires} wires + padding")
        else:
            if args.channel is not None:
                row = args.channel - W_CHANNEL_OFFSET + pad_nwires
                lbl = f"channel {args.channel} (row {row})"
            else:
                row = pad_nwires + nwires // 2
                lbl = f"central row {row} (channel {row - pad_nwires + W_CHANNEL_OFFSET})"
            if not 0 <= row < arr.shape[0]:
                sys.exit(f"row {row} outside [0, {arr.shape[0]})")
            sel = np.ones_like(ticks, dtype=bool)
            if args.tmin is not None:
                sel &= ticks >= args.tmin
            if args.tmax is not None:
                sel &= ticks <= args.tmax
            ax.plot(ticks[sel], arr[row, sel], lw=1.0, color="#2ca02c", label=lbl)
            ax.set_ylabel("value")
            ax.set_title(f"{args.file}: pre-decon waveform")
            ax.legend(fontsize=8)

    ax.set_xlabel("time tick")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if args.output:
        fig.savefig(args.output, dpi=130)
        print(f"wrote {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
