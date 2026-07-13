#!/usr/bin/env python3
"""Raw vs deconvolved W-plane (collection) waveform alignment: does the
deconvolved (gauss) peak sit on the raw post-NF *rising edge* (charge arrival)
at ctoffset=+4us or at -5.5us?  Colleague's per-hit clock reference; see
pdvd-anode-time-consistency.md section 8.13.

Physics: on a collection wire the raw (post-NF) pulse rises as charge arrives
and PEAKS ~1-2us later (cold-electronics shaping).  The deconvolution removes
FR+electronics and puts a narrow gaussian at the charge-arrival time, i.e. on
the raw LEADING edge -- not the raw peak.

ctoffset only shifts the deconvolved output, never the raw.  -5.5us -> +4us is
an exact rigid +19-tick (+9.5us) roll of the gauss array (verified below against
a real +4us reprocess (_ct4) on the checkpoint event 298567).  So the +4us gauss
is obtained by rolling the stored -5.5us (_ctoff) gauss +19 ticks.

For every isolated high-SNR W hit we report
    d = (gauss_peak_tick - raw_leading_edge_tick) * 0.5us
for both ctoffsets.  Expect d ~ 0 at +4us and d ~ -9.5us at -5.5us.

Repro:
    cd pdvd/docs/qlmatch
    python3 check_raw_decon_alignment.py   # writes raw_decon_align_*.png + prints table
"""
import os, io, tarfile, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/img_plot")
import geom

WORK = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/work"
STORE = geom.load_store("/nfs/data/1/xqian/toolkit-dev/wire-cell-data/protodunevd-wires-larsoft-v5.json.bz2")
TICK_US = 0.5
ROLL = 19          # +9.5us / 0.5us  (the -5.5 -> +4 us shift)

# (run6, idx, anode, side-label).  Spread across runs, BDE (0-3) and TDE (4-7).
CASES = [
    ("039252", 0, 0, "BDE"),   # checkpoint: real _ct4 available
    ("039252", 0, 4, "TDE"),   # checkpoint
    ("039253", 0, 1, "BDE"),
    ("039253", 0, 5, "TDE"),
    ("039349", 0, 2, "BDE"),
    ("039349", 0, 6, "TDE"),
]
N_PER_CASE = 2     # isolated W channels to plot per case


def wchans(anode):
    """WCT channel IDs of the W (collection) plane for this anode."""
    ids = []
    for f in (0, 1):
        ids += [int(c) for c in geom.PlaneGeom(STORE, anode, f, 2).chans]
    return set(ids)


def load_frames(pdir, anode):
    """Return dict with gauss (nch,nt), raw (nch,nt), chans (nch,), evt."""
    out = {}
    tf_path = os.path.join(pdir, "protodune-sp-dnnroi-frames-anode%d.tar.bz2" % anode)
    with tarfile.open(tf_path) as tf:
        for m in tf.getmembers():
            if not m.name.endswith(".npy"):
                continue
            arr = None
            for key, pref in (("gauss", "frame_gauss%d_" % anode),
                              ("raw", "frame_raw%d_" % anode),
                              ("chg", "channels_gauss%d_" % anode),
                              ("chr", "channels_raw%d_" % anode)):
                if m.name.startswith(pref):
                    arr = np.load(io.BytesIO(tf.extractfile(m).read()))
                    out[key] = arr
                    out["evt"] = m.name[len(pref):-4]
    return out


def leading_edge(raw, t_peak, base, val):
    """Sub-tick where raw crosses base+0.5*(val-base) on the rising side
    (last upward crossing before t_peak)."""
    thr = base + 0.5 * (val - base)
    t = t_peak
    while t > 1 and raw[t] > thr:
        t -= 1
    # raw[t] <= thr < raw[t+1]; interpolate
    lo, hi = raw[t], raw[t + 1]
    if hi == lo:
        return float(t)
    return t + (thr - lo) / (hi - lo)


def pick_isolated(fr, wset, n):
    """Pick up to n isolated, high-SNR W channels spread over drift depth."""
    gauss, chans = fr["gauss"], fr["chg"]
    nt = gauss.shape[1]
    cand = []
    for r in range(gauss.shape[0]):
        if int(chans[r]) not in wset:
            continue
        g = gauss[r]
        gmax = g.max()
        if gmax < 4000:
            continue
        tpk = int(g.argmax())
        if tpk < 120 or tpk > nt - 120:
            continue
        # temporal isolation: mask +-30 of the main peak, second peak must be small
        gg = g.copy()
        gg[max(0, tpk - 30):tpk + 30] = 0
        if gg.max() > 0.35 * gmax:
            continue
        cand.append((gmax, r, tpk))
    cand.sort(reverse=True)
    # spread over drift depth (peak tick): greedy pick keeping ticks >800 apart
    chosen, ticks = [], []
    for gmax, r, tpk in cand:
        if all(abs(tpk - t) > 800 for t in ticks):
            chosen.append((r, tpk))
            ticks.append(tpk)
        if len(chosen) >= n:
            break
    if len(chosen) < n:  # relax the spread if too few
        for gmax, r, tpk in cand:
            if (r, tpk) not in chosen:
                chosen.append((r, tpk))
            if len(chosen) >= n:
                break
    return chosen[:n]


rows = []
for run6, idx, anode, side in CASES:
    pdir = os.path.join(WORK, "%s_%d_ctoff" % (run6, idx))
    if not os.path.isdir(pdir):
        print("[skip] missing", pdir); continue
    fr = load_frames(pdir, anode)
    evt = fr["evt"]
    wset = wchans(anode)
    ct4dir = os.path.join(WORK, "%s_%d_ct4" % (run6, idx))
    fr4 = load_frames(ct4dir, anode) if os.path.isdir(ct4dir) else None

    for r, tpk in pick_isolated(fr, wset, N_PER_CASE):
        ch = int(fr["chg"][r])
        g = fr["gauss"][r].astype(float)
        g4 = np.roll(g, ROLL)                       # +4us gauss (rigid roll)
        # raw row for the same channel id
        rr = np.where(fr["chr"] == ch)[0]
        if not len(rr):
            continue
        raw = fr["raw"][rr[0]].astype(float)
        w0, w1 = tpk - 55, tpk + 75
        base = np.median(raw[w0:w1])
        rpk = w0 + int(raw[w0:w1].argmax())
        rval = raw[rpk]
        if rval - base < 40:                        # raw SNR too low
            continue
        t_edge = leading_edge(raw, rpk, base, rval)
        tg_m55 = w0 + int(g[w0:w1].argmax())
        tg_p4 = w0 + int(g4[w0:w1].argmax())
        d_m55 = (tg_m55 - t_edge) * TICK_US
        d_p4 = (tg_p4 - t_edge) * TICK_US

        # real +4us cross-check on the checkpoint event
        real_note = ""
        g_real = None
        if fr4 is not None:
            rr4 = np.where(fr4["chg"] == ch)[0]
            if len(rr4):
                g_real = fr4["gauss"][rr4[0]].astype(float)
                tg_real = w0 + int(g_real[w0:w1].argmax())
                maxdiff = np.abs(g_real[w0:w1] - g4[w0:w1]).max()
                real_note = " realpk=%d roll_vs_real_maxdiff=%.0f" % (tg_real, maxdiff)

        rows.append((run6, evt, anode, side, ch, tpk, d_m55, d_p4, real_note.strip()))

        # ---- plot ----
        fig, axL = plt.subplots(figsize=(9, 4.2))
        tt = np.arange(w0, w1)
        axL.plot(tt, raw[w0:w1], color="0.35", lw=1.4, label="raw (post-NF ADC)")
        axL.axhline(base, color="0.7", lw=0.8, ls=":")
        axL.axvline(t_edge, color="green", lw=1.8, ls="--",
                    label="raw rising edge (50%%)  t=%.1f" % t_edge)
        axL.axvline(rpk, color="0.55", lw=1.0, ls=":", label="raw peak  t=%d" % rpk)
        axL.set_xlabel("tick (0.5 us)")
        axL.set_ylabel("raw ADC (post-NF)", color="0.35")
        axR = axL.twinx()
        axR.plot(tt, g[w0:w1], color="tab:red", lw=1.3,
                 label="gauss decon, ctoffset=-5.5us")
        axR.plot(tt, g4[w0:w1], color="tab:blue", lw=1.8,
                 label="gauss decon, ctoffset=+4us (roll +19)")
        if g_real is not None:
            axR.plot(tt, g_real[w0:w1], color="tab:cyan", lw=0, marker=".",
                     ms=3, label="gauss decon, ctoffset=+4us (real _ct4)")
        axR.axvline(tg_m55, color="tab:red", lw=1.2, ls="-.")
        axR.axvline(tg_p4, color="tab:blue", lw=1.2, ls="-.")
        axR.set_ylabel("gauss decon charge", color="tab:blue")
        axL.set_title(
            "run %s evt %s  anode%d (%s)  W-chan %d\n"
            "gauss_peak - raw_edge:   +4us = %+.1f us    -5.5us = %+.1f us"
            % (run6, evt, anode, side, ch, d_p4, d_m55), fontsize=10)
        lL, laL = axL.get_legend_handles_labels()
        lR, laR = axR.get_legend_handles_labels()
        axR.legend(lL + lR, laL + laR, fontsize=7, loc="upper right")
        fig.tight_layout()
        out = "raw_decon_align_%s_a%d_ch%d.png" % (evt, anode, ch)
        fig.savefig(out, dpi=120)
        plt.close(fig)

print("\n%-8s %-8s %5s %4s %6s %6s | %10s %10s | %s" %
      ("run", "evt", "anode", "side", "Wchan", "tick", "d(+4us)", "d(-5.5us)", "ct4 xcheck"))
print("-" * 96)
for run6, evt, anode, side, ch, tpk, d_m55, d_p4, note in rows:
    print("%-8s %-8s %5d %4s %6d %6d | %+8.2f us %+8.2f us | %s" %
          (run6, evt, anode, side, ch, tpk, d_p4, d_m55, note))
if rows:
    dp4 = np.array([r[7] for r in rows]); dm55 = np.array([r[6] for r in rows])
    print("-" * 96)
    print("MEDIAN  d(+4us) = %+.2f us   d(-5.5us) = %+.2f us   (n=%d)   [expect ~0 vs ~-9.5]"
          % (np.median(dp4), np.median(dm55), len(rows)))
