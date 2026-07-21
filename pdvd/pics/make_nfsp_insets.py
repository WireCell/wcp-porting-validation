#!/usr/bin/env python3
"""Physics-inset generators for the PDVD NF and SP algorithm diagrams.

Counterpart of pdhd/pics/make_nfsp_insets.py (PD-HD original).  All frame
insets are real ProtoDUNE-VD data, run 039252 evt 298567 (the hand-scan
reference event, same event as the clustering+Q/L diagram), anode 0 (bottom
CRP, BDE side — the resampled 512→500 ns side).  V-plane illustrates.

  NF diagram
    nf_noise_rms.png    : per-channel noise RMS, pre-NF vs post-NF (V plane) --
                          the headline effect of coherent-noise subtraction.
    nf_coherent_2d.png  : V-plane conduit block, pre-NF vs post-NF, signal-free
                          window -- the coherent stripes flatten.
  SP diagram
    sp_filters.png      : the analytic SP frequency filters actually applied
                          (Wiener_tight_{U,V,W} + Gaus_wide; bottom = top).
    sp_decon_kernel_2d.png : 2D time-domain field response (V plane) = the
                          deconvolution kernel SP inverts.
    sp_waveform.png     : NF-cleaned ADC (bipolar induction) vs the deconvolved
                          charge (gauss) for one V channel.
    dnn_roi.png         : gauss with traditional ROI vs gauss with DNN-ROI for
                          the same V-plane region (run 039253 A/B processings)
                          -- what the neural-network ROI decision changes.

Sources:
  input_data/run039252/evt_0/protodune-{orig,sp-frames-raw,sp-frames}-anode0.tar.bz2
  input_data/run039253_{nodnn,dnn}_nol1sp/evt_0/protodune-sp-frames-anode0.tar.bz2
Field response: protodunevd_FR_imbalance3p_260501.json.bz2 via WIRECELL_PATH.

Output: pdvd/pics/nfsp_src/*.png
"""
import io
import os
import tarfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# Insets are baked rasters placed onto a 3840-px master canvas, so their
# internal text must be large enough to survive downscaling (fonts set here;
# the master font bumps in make_*_diagram.py do NOT touch raster text).
plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 15,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12.5,
})
DPI = 240

HERE = os.path.dirname(os.path.abspath(__file__))
PDVD = os.path.dirname(HERE)
SRC = os.path.join(HERE, "nfsp_src")
EVT = os.path.join(PDVD, "input_data", "run039252", "evt_0")
DNN_A = os.path.join(PDVD, "input_data", "run039253_nodnn_nol1sp", "evt_0")
DNN_B = os.path.join(PDVD, "input_data", "run039253_dnn_nol1sp", "evt_0")

ANODE = 0                      # bottom CRP (BDE) -- the resampled side
V_LOCAL = (476, 952)           # V-plane local row band in an anode frame
ORIG_TICK_US = 0.512           # pre-NF DAQ period, bottom (resampled by NF driver)
NF_TICK_US = 0.500

C_NF = "#c85a11"               # NF (orange family)
C_SP = "#1f4e9b"               # SP (blue family)
C_OUT = "#2e7d4f"              # deconvolved output (green)
C_RAW = "#8a8a8a"              # pre-NF / raw reference (grey)
C_DNN = "#0f8a8a"              # DNN-ROI (teal)


def load_frame(path, prefix="frame_"):
    with tarfile.open(path, "r:bz2") as tf:
        fr = ch = None
        for m in tf.getmembers():
            if m.name.startswith(prefix) and fr is None:
                fr = np.load(io.BytesIO(tf.extractfile(m).read()))
            elif m.name.startswith("channels_") and ch is None:
                ch = np.load(io.BytesIO(tf.extractfile(m).read()))
    return fr, ch


def robust_rms(a):
    """Per-channel 1.4826*MAD -- immune to sparse real signal."""
    med = np.median(a, axis=1, keepdims=True)
    return 1.4826 * np.median(np.abs(a - med), axis=1)


# --------------------------------------------------------------------------
def make_noise_rms():
    orig, _ = load_frame(os.path.join(
        EVT, "protodune-orig-frames-anode%d.tar.bz2" % ANODE))
    raw, _ = load_frame(os.path.join(
        EVT, "protodune-sp-frames-raw-anode%d.tar.bz2" % ANODE))
    lo, hi = V_LOCAL
    r_pre = robust_rms(orig[lo:hi].astype(float))
    r_post = robust_rms(raw[lo:hi].astype(float))
    x = np.arange(hi - lo)
    ok = (r_post > 0) & (r_pre > 0)      # drop masked/dead channels

    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.plot(x[ok], r_pre[ok], lw=1.0, color=C_RAW,
            label="pre-NF  (median %.1f ADC)" % np.median(r_pre[ok]))
    ax.plot(x[ok], r_post[ok], lw=1.0, color=C_NF,
            label="post-NF (median %.1f ADC)" % np.median(r_post[ok]))
    ax.set_xlabel("V-plane channel index")
    ax.set_ylabel("noise RMS [ADC]")
    ax.set_title("per-channel noise RMS  ·  CRP0 V  (run 039252)")
    ax.set_ylim(0, np.percentile(r_pre[ok], 99) * 1.15)
    # traces sit at 3-9 ADC; the band below ~3 ADC on the right is empty
    ax.legend(loc="lower right", framealpha=0.9)
    ax.margins(x=0)
    fig.tight_layout()
    out = os.path.join(SRC, "nf_noise_rms.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out, "pre med %.2f post med %.2f nmasked %d"
          % (np.median(r_pre[ok]), np.median(r_post[ok]), int((~ok).sum())))


def make_nf_coherent_2d():
    """Channel x time ADC heatmap of a V-plane block, pre-NF vs post-NF.

    Coherent (common-mode) noise appears as vertical stripes shared across the
    block at a given tick; PDVDCoherentNoiseSub subtracts the per-conduit-group
    median so the stripes flatten.  A signal-free time window is auto-selected.
    """
    orig, _ = load_frame(os.path.join(
        EVT, "protodune-orig-frames-anode%d.tar.bz2" % ANODE))
    raw, _ = load_frame(os.path.join(
        EVT, "protodune-sp-frames-raw-anode%d.tar.bz2" % ANODE))
    # start above the hot/dead rows at the V-band edge (ch 476-500)
    lo, hi = V_LOCAL[0] + 24, V_LOCAL[0] + 104   # 80 V-plane channels
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
    ovb = ob[:, ot0:ot0 + nwin]
    rvb = rb[:, bt:bt + nwin]
    vmax = np.percentile(np.abs(np.concatenate([ovb, rvb], axis=1)), 97.0)

    from matplotlib.ticker import MaxNLocator
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 3.0), sharey=True)
    for ax, blk, ttl in ((axes[0], ovb, "pre-NF"), (axes[1], rvb, "post-NF")):
        im = ax.imshow(blk, origin="lower", aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, interpolation="nearest",
                       extent=(bt * NF_TICK_US, (bt + nwin) * NF_TICK_US,
                               lo, hi))
        ax.set_title(ttl)
        ax.set_xlabel("time [µs]")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune='upper'))
    axes[0].set_ylabel("V-plane channel")
    cb = fig.colorbar(im, ax=axes, fraction=0.045, pad=0.03)
    cb.set_label("ADC")
    fig.suptitle("CRP0 V block  ·  signal-free window", y=1.02, fontsize=14)
    out = os.path.join(SRC, "nf_coherent_2d.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out, "window tick %d vmax %.1f" % (bt, vmax))


def pick_v_channel(raw, gau):
    """Deterministic pick: strongest clean bipolar V channel (both NF lobes
    strong relative to noise, decon signal present, live channel)."""
    lo, hi = V_LOCAL
    seg = raw[lo:hi].astype(float)
    gseg = gau[lo:hi].astype(float)
    rms = robust_rms(seg)
    pos = seg.max(axis=1)
    neg = -seg.min(axis=1)
    ptk = np.abs(seg).argmax(axis=1)
    bip = np.minimum(pos, neg) / (rms + 1e-6)
    ok = ((ptk > 1000) & (ptk < seg.shape[1] - 1000) & (rms > 1.0)
          & (gseg.max(axis=1) > 0))
    return lo + int(np.argmax(np.where(ok, bip, -1)))


def make_sp_waveform():
    raw, _ = load_frame(os.path.join(
        EVT, "protodune-sp-frames-raw-anode%d.tar.bz2" % ANODE))
    gau, _ = load_frame(os.path.join(
        EVT, "protodune-sp-frames-anode%d.tar.bz2" % ANODE), "frame_gauss")
    row = pick_v_channel(raw, gau)
    wr, wg = raw[row].astype(float), gau[row].astype(float)
    t = np.arange(len(wr)) * NF_TICK_US
    p = int(np.abs(wr).argmax())
    t0, t1 = t[p] - 120 * NF_TICK_US, t[p] + 160 * NF_TICK_US

    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.plot(t, wr, lw=1.1, color=C_SP, label="NF ADC (bipolar)")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xlim(t0, t1)
    mr = np.abs(wr[(t >= t0) & (t <= t1)]).max()
    ax.set_ylim(-1.5 * mr, 1.5 * mr)
    ax.set_xlabel("time [µs]")
    ax.set_ylabel("ADC", color=C_SP)
    ax.tick_params(axis="y", labelcolor=C_SP)
    ax2 = ax.twinx()
    ax2.plot(t, wg, lw=1.4, color=C_OUT, label="decon (gauss)")
    mg = max(wg[(t >= t0) & (t <= t1)].max(), 1e-9)
    ax2.set_ylim(-1.5 * mg, 1.5 * mg)
    ax2.set_ylabel("decon charge", color=C_OUT)
    ax2.tick_params(axis="y", labelcolor=C_OUT)
    ax.set_title("ch %d (V) — bipolar ADC → unipolar charge" % row)
    l1, la1 = ax.get_legend_handles_labels()
    l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, loc="lower right", framealpha=0.9,
              fontsize=10.5)
    fig.tight_layout()
    out = os.path.join(SRC, "sp_waveform.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out, "row", row)


def hf_filter(f, sigma, power):
    """WCT HfFilter: exp(-0.5 (f/sigma)^power), H(0)=0 (flag=true)."""
    h = np.exp(-0.5 * (f / sigma) ** power)
    h[0] = 0.0
    return h


def make_sp_filters():
    # PDVD filters (cfg/pgrapher/experiment/protodunevd/sp-filters.jsonnet;
    # bottom (_b) and top (_t) values are identical -- the split is structural)
    wiener = {"U": (0.148788, 3.76194), "V": (0.1596568, 4.36125),
              "W": (0.13623, 3.35324)}
    gaus_wide = (0.12, 2)
    f = np.linspace(0, 0.5, 1200)     # MHz
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    for pl, col in zip("UVW", [C_SP, "#3d78c9", C_OUT]):
        s, p = wiener[pl]
        ax.plot(f, hf_filter(f, s, p), lw=1.6, color=col,
                label="Wiener_tight_%s" % pl)
    ax.plot(f, hf_filter(f, *gaus_wide), lw=1.7, ls="--", color="#b0413e",
            label="Gaus_wide (charge)")
    ax.set_xlabel("frequency [MHz]")
    ax.set_ylabel("filter gain")
    ax.set_title("SP deconvolution filters  —  bottom = top")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = os.path.join(SRC, "sp_filters.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def make_decon_kernel_2d():
    from wirecell.sigproc.response import persist
    from wirecell.sigproc.response.arrays import pr2array
    from wirecell.util.fileio import wirecell_path
    fn = "protodunevd_FR_imbalance3p_260501.json.bz2"
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

    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    im = ax.imshow(disp, origin="lower", aspect="auto", extent=extent,
                   cmap="RdBu_r", norm=norm, interpolation="nearest")
    ax.set_xlabel("time [µs]")
    ax.set_ylabel("wire offset")
    ax.set_title("field response — V plane")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("induced current [a.u.]")
    fig.tight_layout()
    out = os.path.join(SRC, "sp_decon_kernel_2d.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def make_dnn_compare():
    """gauss with traditional ROI vs gauss with DNN-ROI, same V-plane region.

    Uses the run-039253 A/B processings (evt_0): identical NF+decon, only the
    final ROI decision differs -- what DNNROIFinding changes on real data.
    """
    ga, _ = load_frame(os.path.join(
        DNN_A, "protodune-sp-frames-anode%d.tar.bz2" % ANODE), "frame_gauss")
    gb, _ = load_frame(os.path.join(
        DNN_B, "protodune-sp-frames-anode%d.tar.bz2" % ANODE), "frame_gauss")
    lo, hi = V_LOCAL
    a = ga[lo:hi].astype(float)
    b = gb[lo:hi].astype(float)
    # auto-select the busiest 400-tick window of the DNN output (most charge)
    nwin, step = 400, 100
    best, bt = -1, 0
    for s in range(0, b.shape[1] - nwin, step):
        e = b[:, s:s + nwin].sum()
        if e > best:
            best, bt = e, s
    av = a[:, bt:bt + nwin]
    bv = b[:, bt:bt + nwin]
    vmax = np.percentile(np.concatenate([av, bv], axis=1), 99.5)

    fig, axes = plt.subplots(1, 2, figsize=(5.6, 3.0), sharey=True)
    for ax, blk, ttl in ((axes[0], av, "traditional ROI"),
                         (axes[1], bv, "DNN-ROI")):
        im = ax.imshow(blk, origin="lower", aspect="auto", cmap="viridis",
                       vmin=0, vmax=vmax, interpolation="nearest",
                       extent=(bt * NF_TICK_US, (bt + nwin) * NF_TICK_US,
                               lo, hi))
        ax.set_title(ttl)
        ax.set_xlabel("time [µs]")
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune='upper'))
    axes[0].set_ylabel("V-plane channel")
    cb = fig.colorbar(im, ax=axes, fraction=0.045, pad=0.03)
    cb.set_label("decon charge")
    fig.suptitle("CRP0 V  ·  run 039253  ·  same decon, ROI decision differs",
                 y=1.02, fontsize=13.5)
    out = os.path.join(SRC, "dnn_roi.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out, "window tick %d vmax %.1f" % (bt, vmax))


def main():
    os.makedirs(SRC, exist_ok=True)
    make_noise_rms()
    make_nf_coherent_2d()
    make_sp_waveform()
    make_sp_filters()
    make_decon_kernel_2d()
    make_dnn_compare()


if __name__ == "__main__":
    main()
