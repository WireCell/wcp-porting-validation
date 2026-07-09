#!/usr/bin/env python3
"""Software-filter determination for the PDVD light deconvolution.

The chain uses the PDHD-style fixed Wiener filter G = conj(H)·R/(|H|²R + 1)
(R = fixed_snr, flat noise) followed by a Gauss post-filter (cutoff in MHz).
PDHD production uses R=0.005, cutoff=1.5 MHz.  PDVD pulse shapes differ per
population (XA slow ~1 us, PMT 2-3 ticks), so re-derive both knobs.

Scan per DAPHNE channel over (fixed_snr, cutoff), aggregate per population:
  - decon noise floor: real quiet waveform stretches (snippet pre-trigger
    regions; pulse-masked cathode stream chunks) passed through the filter,
    in PE/tick units (AutoScale applied) -> sets the OpHit threshold;
  - 1-PE peak height: held-out average 1-PE pulse (tmpl_b) deconvolved with
    the channel template (tmpl_a);
  - significance = 1-PE peak / noise floor  (the quality lever);
  - matched-kernel FWHM (two-pulse separation) and negative sidelobe %.

Also measures the quiet-noise power spectrum per population (flat-noise
Wiener sanity check).

Outputs: docs/pds/filter_scan.png, filter_noise_psd.png; table on stdout.
Usage: filter_scan.py [run]   (default 39252; needs spe_build.py harvest)
"""
import os
import sys

import numpy as np
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pdvd_light as pl
import spe_build as sb
import spe_compare as sc

SNRS = [0.001, 0.002, 0.005, 0.01, 0.02]
CUTOFFS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
NNOISE = 4096      # length of the noise record used for the floor
POPS = ("cathode", "membrane_top", "membrane_bot", "pmt")


def collect_noise(f, channels, nseg=NNOISE):
    """Real quiet-noise records per channel (baseline-subtracted).
    Snippets: concatenated pre-trigger heads.  Full stream: longest
    pulse-free stretches (mask = HPF trace above 4x the quiet sigma)."""
    heads = {ch: [] for ch in channels}
    for evt in f.events:
        for r in f.records(evt):
            ch = r["opch"]
            if ch not in heads:
                continue
            adc = r["adc"]
            if r["nsamp"] <= 1024:
                head = adc[:56]
                heads[ch].append(head - head.mean())
            elif sum(len(s) for s in heads[ch]) < nseg:
                med, _ = pl.pedestal_robust(adc)
                w = adc - med
                hp = sb.hpf(w)
                dsig = np.median(np.abs(np.diff(adc))) * 1.4826 / np.sqrt(2)
                mask = np.abs(hp) > 4 * dsig
                # grow the mask +-100 ticks around every excursion
                idx = np.nonzero(mask)[0]
                if len(idx):
                    starts = idx[np.diff(idx, prepend=-1000) > 1]
                    for i in starts:
                        mask[max(0, i - 100):i + 100] = True
                # stitch quiet stretches (>=512 contiguous) up to nseg,
                # HPF'd (the C++ chain's OpRoi removes the wander before
                # hits; the post-HPF floor is the relevant one)
                quiet = ~mask
                edges = np.diff(quiet.astype(int), prepend=0, append=0)
                r0 = np.nonzero(edges == 1)[0]
                r1 = np.nonzero(edges == -1)[0]
                for a, b in zip(r0, r1):
                    if b - a >= 256:
                        heads[ch].append(hp[a:b])
    out = {}
    for ch, lst in heads.items():
        if not lst:
            continue
        if len(lst[0]) < nseg:      # snippet heads -> concatenate
            cat = np.concatenate(lst)
            if len(cat) < nseg:
                continue
            out[ch] = cat[:nseg]
        else:
            out[ch] = lst[0]
    return out


def filter_response(tmpl, rms, snr, cutoff, n):
    """Return (G*post filter spectrum, autoscale norm) for a record length n."""
    H = np.fft.fft(np.pad(tmpl.astype(float), (0, max(0, n - len(tmpl))))[:n])
    N2 = rms ** 2 * n
    S2 = snr * N2
    xG = np.conj(H) * S2 / (np.abs(H) ** 2 * S2 + N2)
    sign = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    xk = np.fft.ifft(xG * H * sign).real
    im = int(np.argmax(xk)); il = im; ir = im
    while il > 0 and xk[il] > 0:
        il -= 1
    while ir < n and xk[ir] > 0:
        ir += 1
    norm = xk[il:ir + 1].sum()
    if norm > 1.0 or norm <= 0.0:
        norm = 1.0
    return xG * pl.build_postfilter(n, cutoff), norm


def scan_channel(h, noise):
    """-> dict[(snr,cutoff)] = (noise_floor, pe_peak, fwhm, sidelobe_pct)"""
    ta = sc.as_kernel(h["tmpl_a"])
    probe = h["tmpl_b"]
    rms = max(float(h["rms"]), 1.0)
    n = NNOISE
    wave = np.zeros(n)
    wave[100:100 + len(probe)] = probe
    W = np.fft.fft(wave)
    Nz = np.fft.fft(noise[:n])
    out = {}
    for snr in SNRS:
        for cut in CUTOFFS:
            GF, norm = filter_response(ta, rms, snr, cut, n)
            dec = np.fft.ifft(GF * W).real / norm
            nz = np.fft.ifft(GF * Nz).real / norm
            lo = 100 + sb.PRE - 40
            pk = float(dec[lo:100 + sb.PRE + 15].max())
            floor = float(nz.std())
            kern = pl.matched_kernel(ta, fixed_snr=snr, line_noise_rms=rms,
                                     cutoff_mhz=cut, pre=60, post=300)
            k0 = int(np.argmax(kern))
            above = np.nonzero(kern > 0.5)[0]
            fwhm = float(above[-1] - above[0]) if len(above) else np.nan
            side = 100.0 * float(kern[k0:].min())
            out[(snr, cut)] = (floor, pk, fwhm, side)
    return out


def main():
    run = int(sys.argv[1]) if len(sys.argv) > 1 else 39252
    harvest = sc.load_harvest(run)
    harvest = {ch: h for ch, h in harvest.items() if "tmpl_a" in h}
    src = [p for p in pl.data_files() if f"run{run:06d}" in p][0]
    f = pl.LightFile(src)

    # population membership from the data
    chpop = {}
    for evt in f.events[:1]:
        for r in f.records(evt):
            if r["opch"] in harvest:
                chpop[r["opch"]] = r["pop"]

    noise = collect_noise(f, list(harvest))
    print(f"noise records for {len(noise)} channels")

    # noise PSD per population (quiet records)
    plot_psd(noise, chpop, run)

    scans = {}
    for ch, h in harvest.items():
        if ch not in noise:
            continue
        scans[ch] = scan_channel(h, noise[ch])
    print(f"scanned {len(scans)} channels x {len(SNRS)}x{len(CUTOFFS)}")

    # aggregate per population: median significance etc.
    agg = {}
    for pop in POPS:
        chs = [c for c in scans if chpop.get(c) == pop]
        if not chs:
            continue
        agg[pop] = {}
        for key in scans[chs[0]]:
            floor = np.median([scans[c][key][0] for c in chs])
            pk = np.median([scans[c][key][1] for c in chs])
            fwhm = np.median([scans[c][key][2] for c in chs])
            side = np.median([scans[c][key][3] for c in chs])
            agg[pop][key] = (floor, pk, pk / floor, fwhm, side)

    # table
    for pop, d in agg.items():
        print(f"\n== {pop} (median over {sum(1 for c in scans if chpop.get(c)==pop)} ch)")
        print("   snr    cutoff  noise_floor  1PE_peak  signif  FWHM  sidelobe%")
        for (snr, cut), (fl, pk, sg, fw, sd) in sorted(d.items()):
            mark = " <-- PDHD default" if (snr, cut) == (0.005, 1.5) else ""
            print(f"  {snr:6.3f}  {cut:4.1f}   {fl:9.4f}  {pk:8.4f}  "
                  f"{sg:6.1f}  {fw:4.0f}  {sd:+7.1f}{mark}")

    plot_scan(agg, run)


def plot_psd(noise, chpop, run):
    fig, ax = plt.subplots(figsize=(9, 5))
    for pop, color in zip(POPS, ("C0", "C1", "C2", "C3")):
        specs = []
        for ch, nz in noise.items():
            if chpop.get(ch) != pop:
                continue
            ps = np.abs(np.fft.rfft(nz[:NNOISE])) ** 2 / NNOISE
            specs.append(ps)
        if not specs:
            continue
        fr = np.fft.rfftfreq(NNOISE, d=pl.TICK_NS * 1e-3)  # MHz
        m = np.median(specs, axis=0)
        ax.semilogy(fr[1:], m[1:], color=color, lw=1,
                    label=f"{pop} ({len(specs)} ch)")
    ax.set_xlabel("frequency (MHz)"); ax.set_ylabel("noise power (ADC^2)")
    ax.set_title(f"PDVD run {run}: quiet-noise power spectra "
                 f"(flat-noise Wiener sanity check)")
    ax.grid(alpha=0.3); ax.legend()
    out = os.path.join(pl.PICS, "filter_noise_psd.png")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    print("wrote", out)


def plot_scan(agg, run):
    fig, axes = plt.subplots(3, len(agg), figsize=(4.5 * len(agg), 11),
                             squeeze=False)
    for j, (pop, d) in enumerate(agg.items()):
        for i, (title, idx) in enumerate((("1-PE significance", 2),
                                          ("kernel FWHM (ticks)", 3),
                                          ("kernel sidelobe (%)", 4))):
            ax = axes[i][j]
            for snr in SNRS:
                ys = [d[(snr, c)][idx] for c in CUTOFFS]
                ax.plot(CUTOFFS, ys, "o-", ms=3, lw=1,
                        label=f"snr={snr}")
            ax.axvline(1.5, color="k", lw=0.5, ls="--")
            ax.set_xlabel("post-filter cutoff (MHz)")
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(pop)
            if j == 0:
                ax.set_ylabel(title)
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.suptitle(f"PDVD run {run}: fixed-Wiener filter scan "
                 f"(dashed = PDHD 1.5 MHz)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(pl.PICS, "filter_scan.png")
    fig.savefig(out, dpi=110); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
