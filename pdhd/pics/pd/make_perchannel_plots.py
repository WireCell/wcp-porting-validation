#!/usr/bin/env python3
"""Per-channel PDHD light deconvolution study (answers the two questions in
pdhd/docs/pdhd-light-raw-data.md follow-up):

 1. raw vs deconvolved waveform for every channel -> is the decon reasonable?
 2. old 2024 FBK/HPK average SPE templates vs the production run28368 v1
    per-channel templates (+ run27950 noise) -> did the v1 switch make any
    channel worse?

Both deconvolutions are computed here with a faithful NumPy replication of
flash/src/OpDecon.cxx; the v1 result is checked against the in-file LArSoft
reference per channel (must agree to ~1e-3) so the old-template result from
the same code path is trustworthy.

Reference event: run 27305 evt 150 (best single-event channel coverage, 0-82).

Outputs: pdhd/pics/pd/decon_ch<NNN>.png (one per channel) + a summary
figure decon_summary.png and a printed per-channel metrics table.
"""
import io
import json
import os
import tarfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# This script lives in pdhd/pics/pd/ (pdhd -> wcp-porting-img repo); the
# converted work dir is found relative to it.  Override with env vars if
# they live elsewhere.
HERE = os.path.dirname(os.path.abspath(__file__))
PDHD = os.path.abspath(os.path.join(HERE, "..", ".."))           # .../pdhd
WORK = os.environ.get("PDHD_WORK", os.path.join(PDHD, "work", "027305_150"))
PICS = HERE
CFG = os.environ.get("PDHD_CFG",
                     "/nfs/data/1/xqian/toolkit-dev/toolkit/cfg/pgrapher/experiment/pdhd")
OLD_SPE = os.path.join(HERE, "pdhd-spe-templates-2024.json")
EVT = "150"
TICK_NS = 16.0
N = 1024

# OpDecon protodunehd_deconvolution parameters.
PRE_TRIGGER = 50
PEDESTAL_BUFFER = 30
LINE_NOISE_RMS = 4.5
POLARITY = -1
POSTFILTER_CUTOFF = 1.5  # MHz


def load_frames(path):
    out = {}
    with tarfile.open(path) as tf:
        for m in tf.getmembers():
            if m.name.endswith(".npy"):
                out[m.name] = np.load(io.BytesIO(tf.extractfile(m).read()))
    return out


def load_templates(spe_json):
    j = json.load(open(spe_json))
    tmpl = [np.array(t["values"], float) for t in j["templates"]]
    for i in range(len(tmpl)):
        w = np.zeros(N)
        w[:min(N, len(tmpl[i]))] = tmpl[i][:N]
        tmpl[i] = w
    c2t = {c: t for c, t in zip(j["channels"], j["template_index"])}
    H = [np.fft.fft(w) for w in tmpl]
    amp = [max(1.0, w.max()) for w in tmpl]
    return c2t, H, amp


def load_noise(noise_json):
    j = json.load(open(noise_json))
    pw = []
    for t in j["templates"]:
        v = np.array(t["values"], float)
        v = np.resize(np.concatenate([v, np.zeros(N // 2 + 1)]), N // 2 + 1) \
            if len(v) < N // 2 + 1 else v[:N // 2 + 1]
        pw.append(v)
    c2n = {c: t for c, t in zip(j["channels"], j["template_index"])}
    return c2n, pw


def build_postfilter():
    sample_freq = 62.5
    df = sample_freq / N
    cutoff_bins = POSTFILTER_CUTOFF / df
    sigma = N * np.sqrt(np.log(2.0)) / (2 * np.pi * cutoff_bins)
    mu = N // 2
    xf = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((np.arange(N) - mu) / sigma) ** 2)
    F = np.fft.fft(xf)
    k = np.arange(N)
    return F * np.exp(-1j * 2 * np.pi * k * mu / N)


POST = build_postfilter()


def deconvolve(adc, H, amp, noise=None):
    """Faithful replication of Flash::OpDecon::deconvolve."""
    nped = PRE_TRIGGER - PEDESTAL_BUFFER
    pedestal = adc[:nped].mean()
    xv = np.zeros(N)
    nin = min(N, len(adc))
    xv[:nin] = POLARITY * (adc[:nin] - pedestal)

    spe_max = xv.max() / amp
    S2 = spe_max * spe_max
    if noise is None:
        N2 = np.full(N, LINE_NOISE_RMS ** 2 * N)
    else:
        k = np.arange(N)
        idx = np.where(k <= N // 2, k, N - k)
        N2 = noise[idx]

    xV = np.fft.fft(xv)
    H2 = np.abs(H) ** 2
    xG = np.conj(H) * S2 / (H2 * S2 + N2)
    xvdec = np.fft.ifft(xG * xV).real

    # AutoScale: filter the SPE response, center (mult by (-1)^k), walk the
    # positive region around the peak, integral -> norm (clamped).
    sign = np.where(np.arange(N) % 2 == 0, 1.0, -1.0)
    x = np.fft.ifft(xG * H * sign).real
    imax = int(np.argmax(x))
    il = imax
    while il > 0 and x[il] > 0:
        il -= 1
    ir = imax
    while ir < N and x[ir] > 0:
        ir += 1
    norm = x[il:min(ir + 1, N)].sum()
    if norm > 1.0 or norm <= 0.0:
        norm = 1.0
    scale = 1.0 / norm

    dec_ped = xvdec[:nped].mean()  # post-BL, before postfilter
    Y = np.fft.fft(xvdec) * POST
    xvdec = np.fft.ifft(Y).real
    return (xvdec - dec_ped) * scale


def snippets(raw_row):
    nz = np.nonzero(raw_row)[0]
    if not len(nz):
        return []
    out = []
    start = nz[0]
    for j in nz[1:]:
        if j >= start + N:
            out.append(start)
            start = j
    out.append(start)
    return out


def main():
    os.makedirs(PICS, exist_ok=True)
    wct = load_frames(f"{WORK}/light-frames-wct.tar.bz2")   # raw + v1 decon (C++)
    ref = load_frames(f"{WORK}/light-frames.tar.bz2")       # LArSoft deconv
    raw_all = wct[f"frame_raw_{EVT}.npy"]
    raw_ch = wct[f"channels_raw_{EVT}.npy"]
    lar_all = ref[f"frame_deconv_{EVT}.npy"]
    lar_ch = ref[f"channels_deconv_{EVT}.npy"]

    c2t_old, H_old, amp_old = load_templates(OLD_SPE)
    c2t_v1, H_v1, amp_v1 = load_templates(f"{CFG}/pdhd-spe-templates.json")
    c2n, noise_pw = load_noise(f"{CFG}/pdhd-noise-templates.json")

    t_us = np.arange(N) * TICK_NS / 1000.0
    rows = []
    chans = sorted(int(c) for c in raw_ch)
    for ch in chans:
        ri = int(np.where(raw_ch == ch)[0][0])
        starts = snippets(raw_all[ri])
        if not starts:
            continue
        # representative snippet = largest deconvolved peak
        best = max(starts, key=lambda s: np.abs(lar_all[int(np.where(lar_ch == ch)[0][0]), s:s + N]).max()
                   if ch in lar_ch else 0)
        sl = slice(best, best + N)
        adc = raw_all[ri, sl].astype(float)
        ped = adc[:20].mean()
        xv = POLARITY * (adc - ped)

        has_v1 = ch in c2t_v1 and c2t_v1[ch] < len(H_v1)
        d_old = deconvolve(adc, H_old[c2t_old[ch]], amp_old[c2t_old[ch]], None) \
            if ch in c2t_old else None
        d_v1 = None
        if has_v1:
            ni = noise_pw[c2n[ch]] if ch in c2n else None
            d_v1 = deconvolve(adc, H_v1[c2t_v1[ch]], amp_v1[c2t_v1[ch]], ni)

        # LArSoft reference for the same snippet (validates d_v1)
        d_lar = lar_all[int(np.where(lar_ch == ch)[0][0]), sl].astype(float) if ch in lar_ch else None
        val = np.nan
        if d_v1 is not None and d_lar is not None and np.abs(d_lar).max() > 1:
            val = np.abs(d_v1 - d_lar).max()

        # metrics: peak and late-tail (5-15 us) mean, integrated area (PE)
        peak_old = d_old.max() if d_old is not None else np.nan
        peak_v1 = d_v1.max() if d_v1 is not None else np.nan
        tail_old = d_old[320:960].mean() if d_old is not None else np.nan
        tail_v1 = d_v1[320:960].mean() if d_v1 is not None else np.nan
        # area over the pulse region [peak-2us, peak+8us] ~ what OpHit integrates
        area_old = area_v1 = np.nan
        if d_v1 is not None:
            ip = int(np.argmax(d_v1))
            lo, hi = max(0, ip - 125), min(N, ip + 500)
            area_v1 = d_v1[lo:hi].sum()
            if d_old is not None:
                area_old = d_old[lo:hi].sum()
        rows.append(dict(ch=ch, has_v1=has_v1, val=val, peak_old=peak_old, peak_v1=peak_v1,
                         tail_old=tail_old, tail_v1=tail_v1, area_old=area_old, area_v1=area_v1))

        # ---- per-channel figure ----
        fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        ax[0].plot(t_us, xv, color="navy", lw=0.7)
        ax[0].axhline(0, color="k", lw=0.5)
        ax[0].set_ylabel("raw ADC\n(baseline-sub, ×polarity)")
        fbk = "FBK" if (ch in c2t_old and c2t_old[ch] == 0) else "HPK"
        ax[0].set_title(f"OpChannel {ch} ({fbk})  —  run 27305 evt 150, "
                        f"snippet @ {best * TICK_NS / 1000:.0f} µs")
        ax[0].grid(alpha=0.3)

        if d_old is not None:
            ax[1].plot(t_us, d_old, color="C0", lw=1.0,
                       label=f"old 2024 {fbk} avg template (flat N²)")
        if d_v1 is not None:
            ax[1].plot(t_us, d_v1, color="C3", lw=1.0,
                       label="v1 per-channel template (+run27950 N²) = LArSoft")
        if not has_v1:
            ax[1].text(0.5, 0.5, "no v1 template (IgnoreChannels)\nLArSoft drops this channel",
                       transform=ax[1].transAxes, ha="center", va="center",
                       color="firebrick", fontsize=11)
        ax[1].axhline(0, color="k", lw=0.5)
        ax[1].set_ylabel("deconvolved\n(≈ PE / tick)")
        ax[1].set_xlabel("time within snippet [µs]")
        ax[1].legend(loc="upper right", fontsize=8)
        ax[1].grid(alpha=0.3)
        if not np.isnan(val):
            ax[1].text(0.02, 0.95, f"v1 vs LArSoft max|Δ| = {val:.2e}",
                       transform=ax[1].transAxes, va="top", fontsize=7, color="gray")
        fig.tight_layout()
        fig.savefig(f"{PICS}/decon_ch{ch:03d}.png", dpi=100)
        plt.close(fig)

    # ---- validation + metrics table ----
    vals = [r["val"] for r in rows if not np.isnan(r["val"])]
    print(f"v1-replication vs LArSoft: {len(vals)} channels checked, "
          f"max|Δ| over all = {max(vals):.2e} (should be ~1e-3 or less)")
    print()
    print(f"{'ch':>3} {'tmpl':>4} {'peak_old':>9} {'peak_v1':>9} {'tail_old':>9} "
          f"{'tail_v1':>9} {'area_old':>9} {'area_v1':>9} {'dArea%':>7}")
    for r in rows:
        if not r["has_v1"]:
            print(f"{r['ch']:>3} {'--':>4}  (no v1 template - LArSoft IgnoreChannels)")
            continue
        fbk = "FBK" if r["tail_old"] is not None else "?"
        dpe = 100 * (r["area_v1"] - r["area_old"]) / r["area_old"] if r["area_old"] else np.nan
        print(f"{r['ch']:>3} {'':>4} {r['peak_old']:>9.2f} {r['peak_v1']:>9.2f} "
              f"{r['tail_old']:>9.3f} {r['tail_v1']:>9.3f} {r['area_old']:>9.1f} "
              f"{r['area_v1']:>9.1f} {dpe:>7.1f}")

    # ---- summary figure: peak-normalized late tail per channel ----
    # Drop dead / near-zero channels (peak < 1 PE): nothing to deconvolve.
    rr = [r for r in rows if r["has_v1"] and r["peak_v1"] >= 1.0]
    chs = [r["ch"] for r in rr]
    fig, ax = plt.subplots(2, 1, figsize=(13, 7.5))
    x = np.arange(len(chs))

    # peak amplitudes: the fast component is well reconstructed both ways.
    ax[0].bar(x - 0.2, [r["peak_old"] for r in rr], 0.4, label="old 2024 avg", color="C0")
    ax[0].bar(x + 0.2, [r["peak_v1"] for r in rr], 0.4, label="v1 (=LArSoft)", color="C3")
    ax[0].set_ylabel("decon peak\n(≈ PE/tick)")
    ax[0].set_title("Per-channel deconvolution — fast-peak amplitude (well reconstructed either way)")
    ax[0].set_xticks(x); ax[0].set_xticklabels(chs, fontsize=6, rotation=90)
    ax[0].legend(); ax[0].grid(alpha=0.3, axis="y")

    # late tail as a fraction of the peak: the over-subtraction, comparable
    # across channels (negative = the slow tail is pushed below zero).
    rel_old = [100 * r["tail_old"] / r["peak_old"] for r in rr]
    rel_v1 = [100 * r["tail_v1"] / r["peak_v1"] for r in rr]
    ax[1].bar(x - 0.2, rel_old, 0.4, label="old 2024 avg", color="C0")
    ax[1].bar(x + 0.2, rel_v1, 0.4, label="v1 (=LArSoft)", color="C3")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_ylabel("late-tail [5–15 µs] mean\nas % of decon peak")
    ax[1].set_xlabel("OpChannel")
    ax[1].set_title("Late-tail level relative to the peak — v1/LArSoft over-subtracts the slow component "
                    "on every channel (driven by the v1 SPE template shape; see ablation in README)")
    ax[1].set_xticks(x); ax[1].set_xticklabels(chs, fontsize=6, rotation=90)
    ax[1].legend(); ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(f"{PICS}/decon_summary.png", dpi=120)
    plt.close(fig)
    print(f"\nwrote {len(rows)} per-channel figures + decon_summary.png to {PICS}")


if __name__ == "__main__":
    main()
