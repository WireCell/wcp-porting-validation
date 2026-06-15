#!/usr/bin/env python3
"""Diagnose WHY the -x full-stream (opch 120-159) reconstructs far more flashes
than the self-trigger APAs over the same 5.5 ms window, with waveform-level
evidence.  Companion to fullstream_compare.py (which only counts coincidences).

The full stream is read out continuously; the self-trigger APAs only in short
1024-tick windows around triggers (here ~4% live-time -> the full stream has
~25x the exposure).  So MORE flashes is partly expected.  The question this
script answers is whether the EXCESS flashes are real (untriggered cosmic light)
or artefacts of a threshold OpHit finder on a long, heavy-tailed decon stream.

Three figures (-> pdhd/pics/, git-ignored):
  1. *_diagnosis.png      (a) per-PE-bin flash-count ratio full/self vs PE, with
                          the live-time exposure bound; (b) per-flash decon
                          peak-vs-area scatter (dominant channel), colored by
                          self-trigger coincidence -- shaped pulses fall on a
                          line, narrow spikes fall off it.
  2. *_waveform_coinc.png   a bright full-stream flash COINCIDENT with a bright
                          self-trigger -x cosmic: both modes' decon waveforms,
                          zoomed to a few us -- the same shaped pulse.
  3. *_waveform_nocoinc.png non-coincident mid-PE (100-800) full-stream flashes:
                          decon waveform shape vs a clean SPE template; the
                          self-trigger is blind (no readout) there.

Reads the existing per-event reco (run_light_fullstream_evt.sh):
  work/<RUN>_fs<evt>/   opflash_pdhd-fullstream-wct.tar.gz + light-frames-fullstream-wct.tar.bz2
  work/<RUN>_snip<evt>/ opflash_pdhd-wct.tar.gz            + light-frames-wct.tar.bz2

Usage: fullstream_diagnose.py <run> <evt>
"""
import os
import sys
import tarfile
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PDHD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(PDHD, "work")
PICS = os.path.join(PDHD, "pics")

SCALE = 100.0          # OpHitFinder m_scale (decon -> scaled units before thresholds)
HIT_THR = 11.0 / SCALE  # full-stream hit_threshold on the pulse peak, decon units (0.11)
START_THR = 3.0 / SCALE  # sliding-window pulse-start adc_threshold, decon units (0.03)
COINC_US = 1.0          # coincidence window


def load_tensors(tgz):
    d = {}
    with tarfile.open(tgz) as t:
        for m in t.getmembers():
            if m.name.endswith("_array.npy"):
                d[int(m.name.split("_")[-2])] = np.load(io.BytesIO(t.extractfile(m).read()))
    return d


def load_frames(bz2, evt):
    out = {}
    with tarfile.open(bz2) as t:
        for tag in ("raw", "decon"):
            out[tag] = np.load(io.BytesIO(t.extractfile("frame_%s_%d.npy" % (tag, evt)).read()))
        out["ch"] = np.load(io.BytesIO(t.extractfile("channels_decon_%d.npy" % evt).read()))
        out["ti"] = np.load(io.BytesIO(t.extractfile("tickinfo_decon_%d.npy" % evt).read()))
    return out  # frame[nch, nt]; ch[nch]; ti=[t0_ns, tick_ns, tbin]


def t_to_sample(t_ns, ti):
    return int(round((t_ns - ti[0]) / ti[1] - ti[2]))


def sample_axis_us(n, ti):
    return (ti[0] + ti[1] * (ti[2] + np.arange(n))) / 1000.0


def nearest_dt(ref, other):
    o = np.sort(other)
    if len(o) == 0:
        return np.full(len(ref), 1e9)
    idx = np.searchsorted(o, ref)
    dt = np.full(len(ref), 1e9)
    for i, x in enumerate(ref):
        for j in (idx[i] - 1, idx[i]):
            if 0 <= j < len(o):
                dt[i] = min(dt[i], abs(x - o[j]))
    return dt


def load(run, evt):
    rp = "%06d" % run
    fsd = "%s/%s_fs%d" % (WORK, rp, evt)
    snd = "%s/%s_snip%d" % (WORK, rp, evt)
    fs = load_tensors("%s/opflash_pdhd-fullstream-wct.tar.gz" % fsd)
    sn = load_tensors("%s/opflash_pdhd-wct.tar.gz" % snd)
    d = dict(
        fs_t=fs[0][:, 0] / 1000.0, fs_pe=fs[1][:, 1], fs_opf=fs[0][:, 1:],
        sn_t=sn[0][:, 0] / 1000.0, sn_pe=sn[1][:, 1], sn_opf=sn[0][:, 1:],
        fsf=load_frames("%s/light-frames-fullstream-wct.tar.bz2" % fsd, evt),
        snf=load_frames("%s/light-frames-wct.tar.bz2" % snd, evt),
    )
    return d


def bright_xup(d, pe_cut=200):
    """Indices of bright self-trigger flashes dominated by -x upper (80-119)."""
    xup = d["sn_opf"][:, 80:120].sum(1)
    tot = d["sn_opf"].sum(1)
    return np.where((d["sn_pe"] > pe_cut) & (xup > 0.5 * tot))[0]


def dom_fs_channel(d, i):
    """Dominant full-stream opch (120-159) for full-stream flash i."""
    return 120 + int(np.argmax(d["fs_opf"][i, 120:160]))


def decon_row(frame, ch_arr, opch):
    return frame[int(np.where(ch_arr == opch)[0][0])]


def peak_area(w, isamp, half=40):
    """Local decon peak and integrated area (over start-threshold span) near isamp."""
    lo, hi = max(0, isamp - half), min(len(w), isamp + half)
    seg = w[lo:hi]
    pk = float(seg.max())
    # area: contiguous above-start-threshold run around the local max
    k = int(np.argmax(seg))
    a, b = k, k
    while a > 0 and seg[a - 1] > START_THR:
        a -= 1
    while b < len(seg) - 1 and seg[b + 1] > START_THR:
        b += 1
    return pk, float(seg[a:b + 1].sum()), (b - a + 1)


def classify_channels(d):
    """Per full-stream channel (120-159): global median, robust RMS, and a class
    -- 'clean', 'offset' (DC baseline shift), or 'ringing' (unstable/bipolar)."""
    fsf = d["fsf"]
    med, rms, cls = {}, {}, {}
    for opch in range(120, 160):
        w = decon_row(fsf["decon"], fsf["ch"], opch)
        m = float(np.median(w))
        r = float(np.median(np.abs(w - m)) * 1.4826)
        med[opch], rms[opch] = m, r
        cls[opch] = "ringing" if r >= 0.1 else ("offset" if abs(m) >= 0.05 else "clean")
    return med, rms, cls


# ----------------------------------------------------------------------------- fig 1
def fig_diagnosis(run, evt, d):
    fsf = d["fsf"]
    # (a) per-PE-bin ratio
    edges = np.array([0, 50, 100, 200, 400, 800, 1600, 3200, 1e9])
    hfs, _ = np.histogram(d["fs_pe"], edges)
    hsn, _ = np.histogram(d["sn_pe"], edges)
    ratio = hfs / np.maximum(hsn, 1e-9)
    # live-time exposure (full stream is continuous; self-trigger ~ nonzero fraction)
    nz = (d["snf"]["raw"] != 0).sum(1)
    live = np.median(nz) / d["snf"]["raw"].shape[1]
    expo = 1.0 / max(live, 1e-9)
    centers = np.sqrt(edges[:-1] * np.where(edges[1:] > 1e8, edges[:-1] * 4, edges[1:]))
    labels = ["%g-%g" % (edges[i], edges[i + 1]) if edges[i + 1] < 1e8 else ">%g" % edges[i]
              for i in range(len(edges) - 1)]

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(ratio))
    ax[0].bar(x, ratio, color="#1f77b4")
    ax[0].axhline(expo, ls="--", c="red", label="live-time bound (%.0fx)" % expo)
    ax[0].axhline(1.0, ls=":", c="gray")
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    ax[0].set(xlabel="flash total PE", ylabel="flash-count ratio  full-stream / self-trigger",
              title="A. Excess peaks at mid-PE, not high-PE\n(bright cosmics ~matched; ratio << 25x exposure)")
    for xi, r in zip(x, ratio):
        ax[0].annotate("%.1f" % r, (xi, r), ha="center", va="bottom", fontsize=7)
    ax[0].legend(fontsize=8)

    # (b) source-channel of the over-producing flashes: non-coincident mid-PE
    # flashes attributed to the channel carrying their pulse.  A couple of
    # pathological channels (ringing / DC-offset) dominate -- that is the answer.
    med, rms, cls = classify_channels(d)
    ref = d["sn_t"][bright_xup(d)]
    dt = nearest_dt(d["fs_t"], ref)
    midnc = np.where((d["fs_pe"] > 100) & (d["fs_pe"] < 800) & (dt > 5.0))[0]
    by_ch = {opch: 0 for opch in range(120, 160)}
    for i in midnc:  # attribute by reconstructed PE (the channel that built the flash)
        by_ch[dom_fs_channel(d, i)] += 1
    chs = np.arange(120, 160)
    counts = np.array([by_ch[c] for c in chs])
    colors = ["#d62728" if cls[c] == "ringing" else ("#ff7f0e" if cls[c] == "offset" else "#1f77b4")
              for c in chs]
    ax[1].bar(chs, counts, color=colors)
    ax[1].set_ylim(0, max(counts.max() * 1.32, 1))
    for c, n in zip(chs, counts):
        if cls[c] != "clean":
            ax[1].annotate("ch%d\n%s\nrms %.2f" % (c, cls[c], rms[c]),
                           (c, n + counts.max() * 0.02), ha="center", va="bottom", fontsize=7)
    nclean = sum(by_ch[c] for c in chs if cls[c] == "clean")
    ax[1].set(xlabel="full-stream opch", ylabel="non-coincident mid-PE flashes (100-800 PE)",
              title="B. The excess comes from 2 bad channels\n(%d of %d on ringing/offset chans; %d on clean chans)"
                    % (len(midnc) - nclean, len(midnc), nclean))

    fig.suptitle("PDHD run %d evt %d: -x full-stream over-production diagnosis" % (run, evt), fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(PICS, exist_ok=True)
    out = "%s/pd_fullstream_%d_evt%d_diagnosis.png" % (PICS, run, evt)
    fig.savefig(out, dpi=95); print("wrote", out)
    return dict(live=live, expo=expo, ratio=dict(zip(labels, ratio)),
                cls=cls, rms=rms, by_ch=by_ch, nmid=len(midnc), nclean=nclean)


# ----------------------------------------------------------------------------- fig 2
def fig_coinc(run, evt, d, win_us=10.0):
    fsf, snf = d["fsf"], d["snf"]
    ref_idx = bright_xup(d)
    ref_t = d["sn_t"][ref_idx]
    # bright coincident full-stream flash
    dt = nearest_dt(d["fs_t"], ref_t)
    cand = np.where((d["fs_pe"] > 800) & (dt < COINC_US))[0]
    i = cand[np.argmax(d["fs_pe"][cand])]
    t0 = d["fs_t"][i]
    # two dominant full-stream channels
    fs_ord = 120 + np.argsort(d["fs_opf"][i, 120:160])[::-1]
    fs_chs = [int(c) for c in fs_ord[:2]]
    # matching self-trigger -x cosmic + its two brightest 80-119 channels
    j = ref_idx[np.argmin(np.abs(ref_t - t0))]
    sn_ord = 80 + np.argsort(d["sn_opf"][j, 80:120])[::-1]
    sn_chs = [int(c) for c in sn_ord[:2]]

    fig, ax = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    ax_us = sample_axis_us(fsf["decon"].shape[1], fsf["ti"])
    sn_us = sample_axis_us(snf["decon"].shape[1], snf["ti"])
    m_fs = np.abs(ax_us - t0) < win_us
    m_sn = np.abs(sn_us - t0) < win_us
    for k, ch in enumerate(fs_chs):
        w = decon_row(fsf["decon"], fsf["ch"], ch)
        ax[0, k].plot(ax_us[m_fs], w[m_fs], color="#d62728", lw=0.8)
        ax[0, k].set_title("full-stream opch %d (PE in flash %.0f)" % (ch, d["fs_opf"][i, ch]), fontsize=9)
    for k, ch in enumerate(sn_chs):
        w = decon_row(snf["decon"], snf["ch"], ch)
        ax[1, k].plot(sn_us[m_sn], w[m_sn], color="#1f77b4", lw=0.8)
        ax[1, k].set_title("self-trigger opch %d (PE in flash %.0f)" % (ch, d["sn_opf"][j, ch]), fontsize=9)
    for a in ax.ravel():
        a.axvline(t0, ls="--", c="green", lw=0.8)
        a.axhline(HIT_THR, ls=":", c="k", lw=0.6)
        a.set_ylabel("decon")
    for a in ax[1]:
        a.set_xlabel("trigger-relative time (us)")
    fig.suptitle("run %d evt %d: COINCIDENT bright cosmic -- full-stream flash PE %.0f at t=%.1f us\n"
                 "self-trigger -x flash PE %.0f at t=%.1f us (dt=%.2f us): the same shaped pulse"
                 % (run, evt, d["fs_pe"][i], t0, d["sn_pe"][j], d["sn_t"][j], abs(t0 - d["sn_t"][j])),
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = "%s/pd_fullstream_%d_evt%d_waveform_coinc.png" % (PICS, run, evt)
    fig.savefig(out, dpi=95); print("wrote", out)
    return dict(fs_flash=i, fs_pe=float(d["fs_pe"][i]), sn_pe=float(d["sn_pe"][j]),
                fs_chs=fs_chs, sn_chs=sn_chs, t0=float(t0))


def spe_template(d, win=40):
    """Clean single-channel SPE-ish shape from the brightest coincident pulse."""
    fsf = d["fsf"]
    ref_t = d["sn_t"][bright_xup(d)]
    dt = nearest_dt(d["fs_t"], ref_t)
    cand = np.where((d["fs_pe"] > 800) & (dt < COINC_US))[0]
    i = cand[np.argmax(d["fs_pe"][cand])]
    opch = dom_fs_channel(d, i)
    w = decon_row(fsf["decon"], fsf["ch"], opch)
    isamp = t_to_sample(d["fs_t"][i] * 1000.0, fsf["ti"])
    k = isamp - win + int(np.argmax(w[isamp - win:isamp + win]))
    tmpl = w[k - win:k + win].copy()
    return tmpl / tmpl.max()  # normalized to unit peak, centered


# ----------------------------------------------------------------------------- fig 3
def fig_nocoinc(run, evt, d, win_us=3.0):
    """One representative non-coincident mid-PE flash per channel class: the
    ringing channel that produces most of them, the DC-offset channel, and a
    clean channel (a genuine untriggered pulse)."""
    fsf = d["fsf"]
    med, rms, cls = classify_channels(d)
    ref_t = d["sn_t"][bright_xup(d)]
    dt = nearest_dt(d["fs_t"], ref_t)
    cand = np.where((d["fs_pe"] > 100) & (d["fs_pe"] < 800) & (dt > 5.0))[0]
    chs = {int(i): dom_fs_channel(d, int(i)) for i in cand}  # PE-dominant channel

    def pick(want, how="median"):  # a representative flash on a class-`want` channel
        sel = [int(i) for i in cand if cls[chs[int(i)]] == want]
        if not sel:
            return None
        peaks = []
        for i in sel:
            w = decon_row(fsf["decon"], fsf["ch"], chs[i])
            peaks.append(peak_area(w, t_to_sample(d["fs_t"][i] * 1000.0, fsf["ti"]))[0])
        # median peak is representative of a class; for the clean class pick the
        # clearest (highest-peak) pulse so the "real pulse" example is unambiguous.
        k = int(np.argmax(peaks)) if how == "max" else int(np.argsort(peaks)[len(peaks) // 2])
        return sel[k], chs[sel[k]]

    nbc = sum(1 for i in cand if cls[chs[int(i)]] != "clean")
    cols = [("ringing", "ringing channel\n(%d/%d of these flashes)" % (
                sum(1 for i in cand if cls[chs[int(i)]] == "ringing"), len(cand))),
            ("offset", "DC-offset channel\n(baseline above threshold)"),
            ("clean", "clean channel\n(genuine untriggered pulse)")]
    tmpl = spe_template(d)
    tx = (np.arange(len(tmpl)) - len(tmpl) // 2) * fsf["ti"][1] / 1000.0
    ax_us = sample_axis_us(fsf["decon"].shape[1], fsf["ti"])

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for c, (cl, sub) in enumerate(cols):
        p = pick(cl, how="max" if cl == "clean" else "median")
        if p is None:
            ax[c].set_title(sub + "\n(none)", fontsize=9); continue
        i, opch = p
        w = decon_row(fsf["decon"], fsf["ch"], opch)
        t0 = d["fs_t"][i]
        isamp = t_to_sample(t0 * 1000.0, fsf["ti"])
        m = np.abs(ax_us - t0) < win_us
        pk, ar, wid = peak_area(w, isamp)
        ax[c].plot(ax_us[m], w[m], color="#7f3fbf", lw=0.9, label="full-stream opch %d" % opch)
        ax[c].plot(t0 + tx, tmpl * max(pk, 0), "g--", lw=1.0, label="clean SPE shape x peak")
        ax[c].axhline(0, ls="-", c="0.7", lw=0.5)
        ax[c].axhline(med[opch], ls="-.", c="orange", lw=0.6, label="channel baseline %.2f" % med[opch])
        ax[c].axvline(t0, ls="--", c="green", lw=0.6)
        ax[c].axhline(HIT_THR, ls=":", c="k", lw=0.6, label="hit threshold 0.11")
        ax[c].set(xlabel="trigger-relative time (us)", ylabel="decon",
                  title="%s\nflash PE %.0f | peak %.2f area %.2f w %d smp | rms %.2f"
                        % (sub, d["fs_pe"][i], pk, ar, wid, rms[opch]))
        ax[c].legend(fontsize=6.5)
    fig.suptitle("run %d evt %d: NON-coincident mid-PE (100-800 PE) flashes by channel class -- "
                 "%d of %d come from the 2 bad channels (self-trigger is blind here)"
                 % (run, evt, nbc, len(cand)), fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    out = "%s/pd_fullstream_%d_evt%d_waveform_nocoinc.png" % (PICS, run, evt)
    fig.savefig(out, dpi=95); print("wrote", out)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); raise SystemExit(1)
    run, evt = int(sys.argv[1]), int(sys.argv[2])
    d = load(run, evt)
    print("full-stream flashes %d   self-trigger flashes %d" % (len(d["fs_t"]), len(d["sn_t"])))
    s = fig_diagnosis(run, evt, d)
    print("self-trigger live-time fraction %.3f -> exposure bound %.0fx" % (s["live"], s["expo"]))
    print("per-PE-bin ratio (full/self):")
    for k, v in s["ratio"].items():
        print("  %-10s %.1f" % (k, v))
    bad = {opch: s["by_ch"][opch] for opch in s["by_ch"] if s["cls"][opch] != "clean"}
    print("non-coincident mid-PE flashes: %d total, %d on clean channels" % (s["nmid"], s["nclean"]))
    print("  pathological channels: %s"
          % ", ".join("ch%d(%s rms%.2f -> %d flashes)" % (c, s["cls"][c], s["rms"][c], n)
                      for c, n in sorted(bad.items())))
    c = fig_coinc(run, evt, d)
    print("coinc example: fs flash %d PE %.0f, sn PE %.0f, fs_chs %s sn_chs %s"
          % (c["fs_flash"], c["fs_pe"], c["sn_pe"], c["fs_chs"], c["sn_chs"]))
    fig_nocoinc(run, evt, d)
