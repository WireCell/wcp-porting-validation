#!/usr/bin/env python3
"""Physics-inset generators for the PDHD NF and SP algorithm diagrams.

All insets are real ProtoDUNE-HD data, run 027409 evt 0, APA 1 (a reference
APA; APA0 V-plane is anomalous).  One V-plane induction channel (offline ch
3757) is threaded through the whole chain so the two diagrams tell one story:

  NF diagram
    nf_noise_rms.png  : per-channel noise RMS, pre-NF vs post-NF (V plane) --
                        the headline effect of coherent-noise subtraction.
    nf_waveform.png   : ch 3757 raw ADC input (pre-NF) vs NF output --
                        baseline flattened, coherent wiggle removed.
  SP diagram
    sp_filters.png    : the analytic SP frequency filters actually applied
                        (Wiener_tight_{U,V,W} + Gaus_wide).
    sp_decon_kernel_2d.png : 2D time-domain field response (V plane) = the
                        deconvolution kernel SP inverts.
    sp_waveform.png   : ch 3757 NF-cleaned ADC (bipolar induction) vs the
                        deconvolved charge (gauss) -- what SP recovers.

Sources (same event for every inset):
  input_data_14_old_coh_grouping/run027409/evt_0/
      protodunehd-orig-frames-anode1.tar.bz2       (pre-NF raw ADC, 512 ns)
      protodunehd-sp-frames-raw-anode1.tar.bz2     (NF output, 500 ns)
      protodunehd-sp-frames-anode1.tar.bz2         (SP gauss/wiener, 500 ns)
  nf_plot/noise_rms/noise_rms_data{,_prenf}.npz    (pre/post-NF RMS)
Field response: dune-garfield-1d565.json.bz2 (APA1-3), via WIRECELL_PATH.

Output: pdhd/pics/nfsp_src/*.png
"""
import io
import os
import tarfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
PDHD = os.path.dirname(HERE)
SRC = os.path.join(HERE, "nfsp_src")
EVT = os.path.join(PDHD, "input_data_14_old_coh_grouping", "run027409", "evt_0")
RMS = os.path.join(PDHD, "nf_plot", "noise_rms")

# reference V-plane induction channel threaded through both diagrams
CH = 3757
ANODE = 1
V_LOCAL = (800, 1600)          # V-plane local row band in an APA frame
ORIG_TICK_US = 0.512           # pre-NF DAQ period (resampled to 0.500 by NF driver)
NF_TICK_US = 0.500

C_NF = "#c85a11"               # NF (orange family)
C_SP = "#1f4e9b"               # SP (blue family)
C_OUT = "#2e7d4f"              # deconvolved output (green)
C_RAW = "#8a8a8a"              # pre-NF / raw reference (grey)


def load_frame(path):
    with tarfile.open(path, "r:bz2") as tf:
        fr = ch = None
        for m in tf.getmembers():
            if m.name.startswith("frame_"):
                fr = np.load(io.BytesIO(tf.extractfile(m).read()))
            elif m.name.startswith("channels_"):
                ch = np.load(io.BytesIO(tf.extractfile(m).read()))
    return fr, ch


def row_of(chans, ch_id):
    return int(np.where(chans == ch_id)[0][0])


# --------------------------------------------------------------------------
def make_noise_rms():
    post = np.load(os.path.join(RMS, "noise_rms_data.npz"))
    pre = np.load(os.path.join(RMS, "noise_rms_data_prenf.npz"))
    key = "anode%d_V" % ANODE
    ch_post, r_post = post[key + "_chan"], post[key + "_rms"]
    r_pre = pre[key + "_rms"]
    x = np.arange(len(r_post))
    ok = (r_post > 0) & (r_pre > 0)      # drop masked/dead channels (rms=0)

    fig, ax = plt.subplots(figsize=(5.0, 2.7))
    ax.plot(x[ok], r_pre[ok], lw=0.7, color=C_RAW, label="pre-NF  (median %.1f ADC)"
            % np.median(r_pre[ok]))
    ax.plot(x[ok], r_post[ok], lw=0.7, color=C_NF, label="post-NF (median %.1f ADC)"
            % np.median(r_post[ok]))
    ax.set_xlabel("V-plane channel index", fontsize=9)
    ax.set_ylabel("noise RMS [ADC]", fontsize=9)
    ax.set_title("per-channel noise RMS  —  APA1 V plane  (run 027409 data)",
                 fontsize=8.5)
    ax.set_ylim(0, np.percentile(r_pre[ok], 99) * 1.15)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
    ax.margins(x=0)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    out = os.path.join(SRC, "nf_noise_rms.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out, "pre med %.2f post med %.2f"
          % (np.median(r_pre[ok]), np.median(r_post[ok])))


def make_nf_coherent_2d():
    """Channel x time ADC heatmap of a V-plane FEMB block, pre-NF vs post-NF.

    Coherent (common-mode) noise appears as vertical stripes shared across the
    whole block at a given tick; PDHDCoherentNoiseSub subtracts the per-group
    median so the stripes flatten.  A signal-free time window is auto-selected
    so the stripes are not confused with real charge.
    """
    orig, co = load_frame(os.path.join(EVT, "protodunehd-orig-frames-anode1.tar.bz2"))
    raw, cr = load_frame(os.path.join(EVT, "protodunehd-sp-frames-raw-anode1.tar.bz2"))
    lo, hi = 800, 880                            # 80 V-plane channels (2 FEMBs)
    ob = orig[lo:hi].astype(float)
    ob -= np.median(ob, axis=1, keepdims=True)   # per-channel pedestal subtract
    rb = raw[lo:hi].astype(float)
    # pick a quiet 300-tick window (smallest block-summed |ADC|) in the NF output
    nwin, span = 300, rb.shape[1]
    step = 100
    best, bt = None, 0
    for s in range(0, span - nwin, step):
        e = np.abs(rb[:, s:s + nwin]).sum()
        if best is None or e < best:
            best, bt = e, s
    # map the same physical time window into the (512 ns) orig grid
    ot0 = int(bt * NF_TICK_US / ORIG_TICK_US)
    ov = ob[:, ot0:ot0 + nwin]
    rv = rb[:, bt:bt + nwin]
    vmax = np.percentile(np.abs(np.concatenate([ov, rv], axis=1)), 97.0)

    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.7), sharey=True)
    for ax, blk, ttl in ((axes[0], ov, "pre-NF"), (axes[1], rv, "post-NF")):
        im = ax.imshow(blk, origin="lower", aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, interpolation="nearest",
                       extent=(bt * NF_TICK_US, (bt + nwin) * NF_TICK_US, lo, hi))
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel("time [µs]", fontsize=8.5)
        ax.tick_params(labelsize=7.5)
    axes[0].set_ylabel("V-plane channel", fontsize=8.5)
    cb = fig.colorbar(im, ax=axes, fraction=0.045, pad=0.03)
    cb.set_label("ADC", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.suptitle("coherent-noise subtraction  —  APA1 V block (quiet window)",
                 fontsize=8.5)
    out = os.path.join(SRC, "nf_coherent_2d.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out, "window tick %d vmax %.1f" % (bt, vmax))


def make_sp_waveform():
    raw, cr = load_frame(os.path.join(EVT, "protodunehd-sp-frames-raw-anode1.tar.bz2"))
    gau, cg = load_frame(os.path.join(EVT, "protodunehd-sp-frames-anode1.tar.bz2"))
    wr, wg = raw[row_of(cr, CH)], gau[row_of(cg, CH)]
    t = np.arange(len(wr)) * NF_TICK_US
    p = int(wg.argmax())
    t0, t1 = t[p] - 120 * NF_TICK_US, t[p] + 160 * NF_TICK_US

    fig, ax = plt.subplots(figsize=(5.0, 2.7))
    ax.plot(t, wr, lw=0.8, color=C_SP, label="NF-cleaned ADC (bipolar induction)")
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_xlim(t0, t1)
    mr = np.abs(wr[(t >= t0) & (t <= t1)]).max()
    ax.set_ylim(-1.5 * mr, 1.5 * mr)
    ax.set_xlabel("time [µs]", fontsize=9)
    ax.set_ylabel("ADC", color=C_SP, fontsize=9)
    ax.tick_params(axis="y", labelcolor=C_SP, labelsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax2 = ax.twinx()
    ax2.plot(t, wg, lw=1.1, color=C_OUT, label="deconvolved charge (gauss)")
    mg = wg[(t >= t0) & (t <= t1)].max()
    ax2.set_ylim(-1.5 * mg, 1.5 * mg)
    ax2.set_ylabel("deconvolved charge", color=C_OUT, fontsize=9)
    ax2.tick_params(axis="y", labelcolor=C_OUT, labelsize=8)
    ax.set_title("ch %d (V) — bipolar ADC → unipolar charge" % CH, fontsize=8.5)
    l1, la1 = ax.get_legend_handles_labels()
    l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, fontsize=7, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    out = os.path.join(SRC, "sp_waveform.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


def hf_filter(f, sigma, power):
    """WCT HfFilter: exp(-0.5 (f/sigma)^power), H(0)=0 (flag=true)."""
    h = np.exp(-0.5 * (f / sigma) ** power)
    h[0] = 0.0
    return h


def make_sp_filters():
    # PDHD APA1-3 filters (sp-filters.jsonnet / pdhd/docs/sp.md)
    wiener = {"U": (0.221933, 6.55413), "V": (0.222723, 8.75998),
              "W": (0.225567, 3.47846)}
    gaus_wide = (0.12, 2)
    f = np.linspace(0, 0.5, 1200)     # MHz (SP grid max_freq = 1 MHz -> Nyquist 1)
    fig, ax = plt.subplots(figsize=(5.0, 2.7))
    for pl, col in zip("UVW", [C_SP, "#3d78c9", C_OUT]):
        s, p = wiener[pl]
        ax.plot(f, hf_filter(f, s, p), lw=1.3, color=col,
                label="Wiener_tight_%s" % pl)
    ax.plot(f, hf_filter(f, *gaus_wide), lw=1.4, ls="--", color="#b0413e",
            label="Gaus_wide (charge)")
    ax.set_xlabel("frequency [MHz]", fontsize=9)
    ax.set_ylabel("filter gain", fontsize=9)
    ax.set_title("SP deconvolution filters  —  APA1–3", fontsize=8.5)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = os.path.join(SRC, "sp_filters.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


def make_decon_kernel_2d():
    from wirecell.sigproc.response import persist
    from wirecell.sigproc.response.arrays import pr2array
    from wirecell.util.fileio import wirecell_path
    fn = "dune-garfield-1d565.json.bz2"
    path = None
    for d in wirecell_path():
        if os.path.exists(os.path.join(d, fn)):
            path = os.path.join(d, fn)
            break
    if path is None:
        raise FileNotFoundError(fn)
    fr = persist.load(path)
    pr = [p for p in fr.planes if p.planeid == 1][0]     # V plane
    r210, _ = pr2array(pr)
    nt = r210.shape[1]
    per_wire = r210.reshape(-1, 10, nt).mean(axis=1)
    nw = per_wire.shape[0]
    period_us = fr.period / 1000.0
    t = np.arange(nt) * period_us
    env = np.abs(per_wire).sum(axis=0)
    active = np.where(env > 0.02 * env.max())[0]
    t0, t1 = max(0, active[0] - 8), min(nt, active[-1] + 12)
    wire_off = np.arange(nw) - nw // 2
    keep = np.abs(wire_off) <= 6
    disp = per_wire[keep, t0:t1]
    wire_off = wire_off[keep]
    t = t - t[t0]
    vmax = np.percentile(np.abs(disp), 88.0)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    extent = (t[t0], t[t1 - 1], wire_off[0] - 0.5, wire_off[-1] + 0.5)

    fig, ax = plt.subplots(figsize=(5.0, 2.7))
    im = ax.imshow(disp, origin="lower", aspect="auto", extent=extent,
                   cmap="RdBu_r", norm=norm, interpolation="nearest")
    ax.set_xlabel("time [µs]", fontsize=9)
    ax.set_ylabel("wire offset", fontsize=9)
    ax.set_title("field response — V plane  (deconvolution kernel)", fontsize=8.5)
    ax.tick_params(labelsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("induced current [a.u.]", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    out = os.path.join(SRC, "sp_decon_kernel_2d.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


def main():
    os.makedirs(SRC, exist_ok=True)
    make_noise_rms()
    make_nf_coherent_2d()
    make_sp_waveform()
    make_sp_filters()
    make_decon_kernel_2d()


if __name__ == "__main__":
    main()
