#!/usr/bin/env python3
"""PDVD light-reco timing diagnostic: raw waveform / decon waveform / flash time.

Everything is plotted on ONE absolute axis: microseconds relative to the
chain tick origin t0 of PDVDOpWaveformSource

    t0 = min over ALL records of  llround(timestamp_us * 62.5)
                                  - (nsamp <= 1024 ? trig_sample : 0)   [ticks]

(16 ns ticks; the min deliberately ignores the opch selection so the three
population branches share one origin -- root/src/PDVDOpWaveformSource.cxx:105).

  * raw / decon come from the dense frames dumped by wct-light-frames.jsonnet
    (tickinfo = [time, tick_ns, tbin0]; column i is absolute tick tbin0+i).
  * flash time is opflash column 0, in NANOSECONDS relative to the same t0
    (OpFlashFinder works in WCT ns).

Outputs (docs/pics/):
  pdvd_light_timing_overview.png   full 7.5 ms window, raw | decon | flash PE
  pdvd_light_timing_zoom.png       one bright flash, cathode | membrane | PMT
  pdvd_light_timing_residual.png   flash time - decon peak time, 414 flashes

Usage: light_timing_check.py [workdir] [run] [event]
"""
import io
import os
import sys
import tarfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PDVD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PICS = os.path.join(PDVD, "docs", "pics")
TICK_NS = 16.0
BRANCHES = ("cathode", "membrane", "pmt")
COL = {"cathode": "#1f77b4", "membrane": "#d62728", "pmt": "#2ca02c"}


def load_frames(workdir, branch, event):
    """-> dict tag -> (frame[nch, nsamp], channels[nch], tbin0)"""
    out = {}
    with tarfile.open(os.path.join(workdir, f"light-frames-{branch}.tar.bz2")) as tf:
        arrs = {m.name: np.load(io.BytesIO(tf.extractfile(m).read()))
                for m in tf.getmembers() if m.name.endswith(".npy")}
    for tag in ("raw", "decon"):
        out[tag] = (arrs[f"frame_{tag}_{event}.npy"],
                    arrs[f"channels_{tag}_{event}.npy"],
                    int(arrs[f"tickinfo_{tag}_{event}.npy"][2]))
    return out


def load_flash(workdir, event):
    with tarfile.open(os.path.join(workdir, "opflash_pdvd-wct.tar.gz")) as tf:
        arrs = {m.name: np.load(io.BytesIO(tf.extractfile(m).read()))
                for m in tf.getmembers() if m.name.endswith(".npy")}
    fl = arrs[f"opflash_tensor_{event}_0_array.npy"]       # [t_ns, PE per opdet...]
    oh = arrs[f"opflash_tensor_{event}_2_array.npy"]       # ophits
    return fl[:, 0], fl[:, 1:].sum(axis=1), fl[:, 1:], oh


def trace_us(frame, tbin0, row):
    v = frame[row].astype(np.float64)
    t = (tbin0 + np.arange(v.size)) * TICK_NS / 1e3
    return t, v


def pedsub(v, n=20):
    return v - np.median(v[:n])


def main():
    workdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        PDVD, "work", "039252_light298567_keep")
    run = int(sys.argv[2]) if len(sys.argv) > 2 else 39252
    event = int(sys.argv[3]) if len(sys.argv) > 3 else 298567
    os.makedirs(PICS, exist_ok=True)

    fr = {b: load_frames(workdir, b, event) for b in BRANCHES}
    t_ns, pe_tot, pe_det, oh = load_flash(workdir, event)
    print(f"run {run} event {event}: {len(t_ns)} flashes, "
          f"t = {t_ns.min()/1e3:.1f} .. {t_ns.max()/1e3:.1f} us")

    # ---- summed cathode decon: the reference light-arrival waveform -------
    cdec, cch, ctb0 = fr["cathode"]["decon"]
    craw, _, ctb0r = fr["cathode"]["raw"]
    Sdec = cdec.sum(axis=0)
    t_dec = (ctb0 + np.arange(Sdec.size)) * TICK_NS / 1e3

    # =====================================================================
    # Figure 1 -- full-window overview
    # =====================================================================
    ch_show = 1010                                   # cathode XA, OpDet 4
    row_r = int(np.nonzero(cch == ch_show)[0][0])
    tr, vr = trace_us(craw, ctb0r, row_r)
    td, vd = trace_us(cdec, ctb0, row_r)
    vr = pedsub(vr)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(4, 1, hspace=0.35, height_ratios=[1, 1, 1, 1.15])
    a0 = fig.add_subplot(gs[0])
    ax = [a0, fig.add_subplot(gs[1], sharex=a0), fig.add_subplot(gs[2], sharex=a0)]
    az = fig.add_subplot(gs[3])
    ax[0].plot(tr, vr, lw=0.4, color=COL["cathode"])
    ax[0].set_ylabel("raw ADC - pedestal")
    ax[0].set_title(f"PDVD run {run} event {event} -- cathode XA opch {ch_show} "
                    f"(full-stream 468800 samples), all-PD flashes overlaid",
                    fontsize=13)
    ax[1].plot(td, vd, lw=0.4, color=COL["cathode"])
    ax[1].set_ylabel("decon (PE / tick)")
    ax[2].vlines(t_ns / 1e3, 1e-1, pe_tot, lw=0.6, color="0.35")
    ax[2].set_yscale("log")
    ax[2].set_ylabel("flash total PE")
    ax[2].set_xlabel(r"time relative to chain $t_0$  [$\mu$s]")
    for a in ax[:2]:
        for t in t_ns[pe_tot > 200] / 1e3:
            a.axvline(t, color="0.75", lw=0.5, zorder=0)
    ax[2].set_xlim(t_dec[0], t_dec[-1])
    for a in ax:
        a.grid(alpha=0.25)
    ax[0].text(0.005, 0.93, "grey lines: flashes with total PE > 200",
               transform=ax[0].transAxes, fontsize=9, color="0.35")

    # bottom row: 150 us zoom of the summed-cathode decon, every flash marked
    z0 = float(t_ns[np.argmax(pe_tot)] / 1e3) - 75.0
    mz = (t_dec > z0) & (t_dec < z0 + 150.0)
    az.plot(t_dec[mz], Sdec[mz], lw=0.6, color=COL["cathode"])
    mf = (t_ns / 1e3 > z0) & (t_ns / 1e3 < z0 + 150.0)
    az.vlines(t_ns[mf] / 1e3, *az.get_ylim(), color="0.6", lw=0.7, zorder=0)
    az.set_xlim(z0, z0 + 150.0)
    az.set_xlabel(r"time relative to chain $t_0$  [$\mu$s]   "
                  r"(150 $\mu$s zoom, 16 cathode channels summed)")
    az.set_ylabel("summed cathode decon")
    az.grid(alpha=0.25)
    az.text(0.005, 0.9, f"grey lines: all {int(mf.sum())} flashes in this window",
            transform=az.transAxes, fontsize=9, color="0.35")
    p1 = os.path.join(PICS, "pdvd_light_timing_overview.png")
    fig.savefig(p1, dpi=110)
    plt.close(fig)
    print("wrote", p1)

    # =====================================================================
    # Figure 2 -- zoom on one bright flash, all three populations
    # =====================================================================
    # pick the brightest flash that also fired >=1 membrane and >=1 PMT OpDet
    MEM = [0, 1, 2, 3, 12, 13, 18, 19]
    PMT = list(range(14, 18)) + list(range(20, 40))
    ok = ((pe_det[:, MEM].max(axis=1) > 1) & (pe_det[:, PMT].max(axis=1) > 1)
          & (pe_tot > 300) & (pe_tot < 3000))       # bright but below the ADC rail
    fi = int(np.argmax(np.where(ok, pe_tot, -1)))
    tf_us = t_ns[fi] / 1e3
    print(f"zoom flash #{fi}: t = {tf_us:.3f} us, total PE = {pe_tot[fi]:.0f}")

    WIN = (-3.0, 6.0)
    fig, ax = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    for j, b in enumerate(BRANCHES):
        raw, chans, tb0r = fr[b]["raw"]
        dec, _, tb0d = fr[b]["decon"]
        # one channel per column: the largest DECON signal in the window;
        # the raw panel above shows the SAME channel.  Samples outside the
        # record are exactly 0 in the raw frame (dense zero-fill) -- masked
        # so the record boundary is not drawn as a signal step.
        best = None
        for r, c in enumerate(chans):
            t, v = trace_us(dec, tb0d, r)
            _, vraw = trace_us(raw, tb0r, r)
            n = min(v.size, vraw.size)               # decon is zero-padded longer
            t, v, vraw = t[:n], v[:n], vraw[:n]
            live = (t > tf_us + WIN[0]) & (t < tf_us + WIN[1]) & (vraw != 0.0)
            if live.sum() < 20:
                continue
            vv = v[live] - np.median(v[live][:10])
            if best is None or vv.max() > best[0]:
                best = (vv.max(), int(c), t[live], vv,
                        pedsub(vraw[live], 10))
        if best is None:
            for i in range(2):
                ax[i, j].text(0.5, 0.5, "no record in window", ha="center",
                              transform=ax[i, j].transAxes)
            continue
        _, bch, bt, bdec, braw = best
        ax[0, j].plot(bt, braw, lw=0.9, color=COL[b])
        ax[1, j].plot(bt, bdec, lw=0.9, color=COL[b])
        for i in range(2):
            ax[i, j].axvline(tf_us, color="k", ls="--", lw=1.2)
            ax[i, j].grid(alpha=0.25)
            ax[i, j].set_title(f"{b} opch {bch} -- "
                               f"{'raw' if i == 0 else 'decon'}", fontsize=11)
        ax[1, j].set_xlabel(r"time rel. $t_0$  [$\mu$s]")
        pk = bt[int(np.argmax(bdec))] - tf_us
        ax[1, j].text(0.02, 0.9, f"decon peak {pk*1e3:+.0f} ns", fontsize=9,
                      transform=ax[1, j].transAxes, color=COL[b])
    ax[0, 0].set_ylabel("raw ADC - pedestal")
    ax[1, 0].set_ylabel("decon (PE / tick)")
    ax[0, 0].set_xlim(tf_us + WIN[0], tf_us + WIN[1])
    fig.suptitle(f"PDVD run {run} event {event} -- flash #{fi} at "
                 f"{tf_us:.3f} " + r"$\mu$s" + f" ({pe_tot[fi]:.0f} PE, dashed); "
                 "raw (top) and decon (bottom) of the brightest channel per population",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p2 = os.path.join(PICS, "pdvd_light_timing_zoom.png")
    fig.savefig(p2, dpi=110)
    plt.close(fig)
    print("wrote", p2)

    # =====================================================================
    # Figure 3 -- residual: flash time vs summed-cathode decon peak
    # =====================================================================
    res, tsel, pesel = [], [], []
    half = int(round(2000.0 / TICK_NS))              # +-2 us search window
    for t, p in zip(t_ns, pe_tot):
        c = int(round(t / TICK_NS)) - ctb0
        lo, hi = max(0, c - half), min(Sdec.size, c + half)
        if hi - lo < 50:
            continue
        k = lo + int(np.argmax(Sdec[lo:hi]))
        res.append(((k + ctb0) * TICK_NS - t) / 1e3)
        tsel.append(t / 1e3)
        pesel.append(p)
    res = np.array(res); tsel = np.array(tsel); pesel = np.array(pesel)
    bright = pesel > 300
    med = np.median(res[bright])
    print(f"residual (PE>300, n={bright.sum()}): median {med*1e3:.0f} ns, "
          f"IQR {np.percentile(res[bright],25)*1e3:.0f}..{np.percentile(res[bright],75)*1e3:.0f} ns")

    fig, ax = plt.subplots(1, 2, figsize=(16, 5.5))
    ax[0].hist(res[bright] * 1e3, bins=np.arange(-1000, 1010, 40),
               color=COL["cathode"], alpha=0.85)
    ax[0].axvline(0, color="k", lw=1)
    ax[0].axvline(med * 1e3, color="r", ls="--", lw=1.2,
                  label=f"median {med*1e3:.0f} ns")
    ax[0].set_xlabel("decon peak time - flash time  [ns]")
    ax[0].set_ylabel(f"flashes (total PE > 300, n={bright.sum()})")
    ax[0].legend()
    ax[0].grid(alpha=0.25)
    sc = ax[1].scatter(tsel[bright], res[bright] * 1e3, s=14,
                       c=np.log10(pesel[bright]), cmap="viridis")
    ax[1].axhline(0, color="k", lw=1)
    ax[1].axhline(med * 1e3, color="r", ls="--", lw=1.2)
    ax[1].set_xlabel(r"flash time rel. $t_0$  [$\mu$s]")
    ax[1].set_ylabel("decon peak - flash  [ns]")
    ax[1].set_ylim(-1000, 1000)
    ax[1].grid(alpha=0.25)
    fig.colorbar(sc, ax=ax[1], label=r"$\log_{10}$ flash PE")
    fig.suptitle(f"PDVD run {run} event {event} -- light-domain flash timing "
                 "closure (no drift with time => no clock/units error)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p3 = os.path.join(PICS, "pdvd_light_timing_residual.png")
    fig.savefig(p3, dpi=110)
    plt.close(fig)
    print("wrote", p3)

    # slope test: constant offset vs growing-with-time
    a, b_ = np.polyfit(tsel[bright], res[bright] * 1e3, 1)
    print(f"linear fit: slope {a:.4f} ns/us, intercept {b_:.1f} ns")

    # =====================================================================
    # Figure 4 -- per-population arrival time: full-stream vs self-trigger
    # =====================================================================
    # For every bright flash, the 20%-of-max rise time of the loudest RAW
    # channel of each population, relative to the flash time.  Raw (not
    # decon) so the comparison is free of any template/kernel alignment.
    ris = {}
    for b in BRANCHES:
        raw, chans, tb0 = fr[b]["raw"]
        vals = []
        for t, p in zip(t_ns, pe_tot):
            if p < 300:
                continue
            c = int(round(t / TICK_NS))
            lo, hi = c - int(3000 / TICK_NS) - tb0, c + int(6000 / TICK_NS) - tb0
            lo, hi = max(0, lo), min(raw.shape[1], hi)
            bv, bt = -1.0, None
            for r in range(raw.shape[0]):
                s = raw[r, lo:hi].astype(np.float64)
                live = s != 0.0
                if live.sum() < 50:
                    continue
                v = s - np.median(s[live][:10])
                v[~live] = 0.0
                if v.max() > bv:
                    bv = v.max()
                    k = int(np.argmax(v))
                    thr = 0.2 * v.max()
                    j = k
                    while j > 0 and v[j] > thr:
                        j -= 1
                    bt = ((lo + tb0 + j) * TICK_NS - t) / 1e3
            if bv > 50:
                vals.append((bt, p))
        ris[b] = np.array(vals)
        print(f"{b:9s} 20%-rise - flash: median {np.median(ris[b][:,0])*1e3:+.0f} ns "
              f"(n={len(ris[b])}, IQR {np.percentile(ris[b][:,0],25)*1e3:+.0f}"
              f"..{np.percentile(ris[b][:,0],75)*1e3:+.0f})")

    fig, ax = plt.subplots(1, 2, figsize=(16, 5.5))
    bins = np.arange(-1.0, 2.01, 0.05)
    for b in BRANCHES:
        m = np.median(ris[b][:, 0])
        ax[0].hist(ris[b][:, 0], bins=bins, histtype="step", lw=1.8,
                   color=COL[b], label=f"{b}  median {m*1e3:+.0f} ns "
                                       f"(n={len(ris[b])})")
        ax[1].scatter(ris[b][:, 1], ris[b][:, 0] * 1e3, s=12, color=COL[b],
                      alpha=0.7, label=b)
    ax[0].axvline(0, color="k", lw=1)
    ax[0].set_xlabel(r"raw 20%-of-max rise time - flash time  [$\mu$s]")
    ax[0].set_ylabel("flashes (total PE > 300)")
    ax[0].legend(fontsize=9)
    ax[0].grid(alpha=0.25)
    ax[1].set_xscale("log")
    ax[1].axhline(0, color="k", lw=1)
    ax[1].set_xlabel("flash total PE")
    ax[1].set_ylabel("rise - flash  [ns]")
    ax[1].set_ylim(-500, 1500)
    ax[1].legend(fontsize=9)
    ax[1].grid(alpha=0.25)
    fig.suptitle("PDVD run %d event %d -- light arrival per readout mode: "
                 "cathode FULL-STREAM vs membrane/PMT SELF-TRIGGER.  The "
                 "+0.7 us offset is flat in PE => not late light, not "
                 "threshold walk." % (run, event), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p4 = os.path.join(PICS, "pdvd_light_timing_population.png")
    fig.savefig(p4, dpi=110)
    plt.close(fig)
    print("wrote", p4)

    # ---- why doc 05 (`05_pdvd-flash-dt.md`) saw +0.000 us instead --------
    # It measures NEAREST-NEIGHBOUR dt to the closest cathode hit >= 3 PE.
    # Cathode hits are dense along the tail of every pulse, so a PMT hit
    # 0.8 us after the leading edge matches a different cathode hit a few
    # tens of ns away and the estimator returns ~0.  Reproduce it here.
    ch, tpk, hpe = oh[:, 0].astype(int), oh[:, 1], oh[:, 5]
    tc = np.sort(tpk[((ch // 1000) == 1) & (hpe >= 3.0)])
    gaps = np.diff(tc)
    tp = tpk[((ch // 1000) == 3) & (hpe >= 0.7)]
    j = np.clip(np.searchsorted(tc, tp), 1, tc.size - 1)
    nn = np.where(np.abs(tp - tc[j - 1]) < np.abs(tp - tc[j]),
                  tp - tc[j - 1], tp - tc[j])
    print(f"doc-05 estimator check: {tc.size} cathode hits >=3PE, "
          f"median gap {np.median(gaps):.0f} ns, "
          f"{100*np.mean(gaps < 800):.0f}% of gaps < 800 ns")
    print(f"  nearest-neighbour pmt-cathode: median {np.median(nn):+.0f} ns "
          f"(68% half-width {0.5*(np.percentile(nn,84)-np.percentile(nn,16)):.0f} ns)"
          f"  <-- doc 05 reports +0 ns; leading-edge estimator says "
          f"{np.median(ris['pmt'][:,0])*1e3:+.0f} ns")

    # =====================================================================
    # Figure 5 -- the same offset, straight from the raw ROOT, many events
    # =====================================================================
    if os.environ.get("PDVD_LT_MULTI", "1") != "0":
        multi_event_scan()


def multi_event_scan(runs=("039252", "039253", "039349"), nevt=5):
    """Self-trigger-vs-full-stream offset without any WCT product.

    For each event: replicate the chain t0, sum the 16 cathode full-stream
    channels, take the 20%-of-max rise of every stream pulse, and for each
    self-trigger record take its own 20%-of-max rise at the tick where
    PDVDOpWaveformSource places it.  The difference is the offset between
    the two DAQ timestamp paths -- it involves no template, no filter and
    no flash finding.
    """
    import uproot
    import glob

    def rise20(v):
        k = int(np.argmax(v))
        thr = 0.2 * v.max()
        j = k
        while j > 0 and v[j] > thr:
            j -= 1
        return j

    TRIG = 64
    res = {b: {} for b in ("membrane", "pmt")}
    for run in runs:
        fs = sorted(glob.glob(os.path.join(
            PDVD, "input_data_light", f"np02vd_raw_run{run}_*_rawwf.root")))
        if not fs:
            continue
        d = uproot.open(fs[0])["rawdump/raw_waveform"].arrays(
            ["event", "opchannel", "nsamp", "timestamp", "adc"], library="np")
        for nm in res:
            res[nm][run] = []
        for ev in sorted(set(d["event"]))[:nevt]:
            sel = np.nonzero(d["event"] == ev)[0]
            starts = {i: int(round(d["timestamp"][i] * 62.5))
                      - (TRIG if d["nsamp"][i] <= 1024 else 0) for i in sel}
            t0 = min(starts.values())
            cidx = [i for i in sel if d["opchannel"][i] // 1000 == 1]
            if not cidx:
                continue
            S = np.zeros(int(d["nsamp"][cidx[0]]))
            for i in cidx:
                a = np.asarray(d["adc"][i], dtype=np.float64)
                S[:a.size] += a - np.median(a[:200])
            off = starts[cidx[0]] - t0
            thr = max(200.0, 8 * np.std(S[:5000]))
            pulses, last = [], -1 << 30
            for k in np.nonzero(S > thr)[0]:
                if k - last < 200:
                    continue
                j = k
                while j > 0 and S[j] > 0.2 * S[k:k + 60].max():
                    j -= 1
                pulses.append(j + off)
                last = k
            pulses = np.array(pulses)
            if pulses.size < 10:
                continue
            for lo, hi, nm in ((2000, 2999, "membrane"), (3000, 3999, "pmt")):
                for i in sel:
                    if not (lo <= d["opchannel"][i] <= hi):
                        continue
                    a = np.asarray(d["adc"][i], dtype=np.float64)
                    if a.size != 1024:
                        continue
                    v = a - np.median(a[:20])
                    if v.max() < 150:
                        continue
                    ar = starts[i] - t0 + rise20(v)
                    dt = (ar - pulses[np.argmin(np.abs(pulses - ar))]) * TICK_NS
                    if abs(dt) < 3000:
                        res[nm][run].append(dt)

    fig, ax = plt.subplots(1, 2, figsize=(16, 5.5))
    bins = np.arange(-1500, 1520, 32)
    for nm in ("membrane", "pmt"):
        allv = np.concatenate([np.array(v) for v in res[nm].values() if len(v)])
        ax[0].hist(allv, bins=bins, histtype="step", lw=1.8, color=COL[nm],
                   label=f"{nm}  median {np.median(allv):+.0f} ns (n={allv.size})")
    ax[0].axvline(0, color="k", lw=1)
    ax[0].set_xlabel("self-trigger 20%-rise - nearest cathode full-stream "
                     r"20%-rise  [ns]")
    ax[0].set_ylabel("records")
    ax[0].legend(fontsize=9)
    ax[0].grid(alpha=0.25)
    xs, lab = [], []
    for k, (nm, run) in enumerate([(n, r) for n in ("membrane", "pmt")
                                   for r in runs]):
        v = np.array(res[nm].get(run, []))
        if not v.size:
            continue
        ax[1].errorbar(k, np.median(v),
                       yerr=[[np.median(v) - np.percentile(v, 25)],
                             [np.percentile(v, 75) - np.median(v)]],
                       fmt="o", color=COL[nm], capsize=4)
        xs.append(k)
        lab.append(f"{nm[:4]}\n{run}")
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels(lab, fontsize=9)
    ax[1].axhline(0, color="k", lw=1)
    ax[1].set_ylabel("median offset  [ns]  (bars = IQR)")
    ax[1].set_ylim(-100, 1000)
    ax[1].grid(alpha=0.25)
    fig.suptitle(f"PDVD self-trigger vs full-stream timestamp offset -- "
                 f"{len(runs)} runs x {nevt} events, raw ROOT only "
                 "(no template, no filter, no flash finder)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p5 = os.path.join(PICS, "pdvd_light_timing_multievent.png")
    fig.savefig(p5, dpi=110)
    plt.close(fig)
    for nm in res:
        for run, v in res[nm].items():
            v = np.array(v)
            if v.size:
                print(f"  {nm:9s} run {run}: n={v.size:5d} median "
                      f"{np.median(v):+.0f} ns  IQR {np.percentile(v,25):+.0f}"
                      f"..{np.percentile(v,75):+.0f}")
    print("wrote", p5)


if __name__ == "__main__":
    sys.exit(main())
