#!/usr/bin/env python3
"""
Post-NF noise spectrum comparison: PDVD-top, PDVD-bottom, PDHD.

Reads _raw TH2 from one anode/APA per detector (run 039324 anode 0/4,
run 027409 APA 1), masks signal-like samples (Microboone::SignalFilter
algorithm), FFTs each channel, averages |FFT| across all channels in the
plane, and overlays the three detectors.

Outputs (same directory as this script):
  noise_spectrum_compare_U.png
  noise_spectrum_compare_V.png
  noise_spectrum_compare_W.png
"""

import os
import numpy as np
import uproot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPTDIR  = os.path.dirname(os.path.abspath(__file__))
PDVD_WORK  = os.path.join(SCRIPTDIR, '../work/039324_0')
PDHD_WORK  = os.path.join(SCRIPTDIR, '../../pdhd/work/027409_0')

TICK_US    = 0.5       # 500 ns ADC tick
SIG_FACTOR = 4.0       # Microboone::SignalFilter sigFactor
PAD_BINS   = 8         # Microboone::SignalFilter padBins
SAT_ADC    = 4096.0    # saturation cutoff

DETECTORS = [
    {
        'label': 'PDHD APA1',
        'color': 'C0',
        'root':  os.path.join(PDHD_WORK,  'magnify-run027409-evt0-apa1.root'),
        'ident': 1,
    },
    {
        'label': 'PDVD bottom (anode 0)',
        'color': 'C1',
        'root':  os.path.join(PDVD_WORK,  'magnify-run039324-evt0-anode0.root'),
        'ident': 0,
    },
    {
        'label': 'PDVD top (anode 4)',
        'color': 'C2',
        'root':  os.path.join(PDVD_WORK,  'magnify-run039324-evt0-anode4.root'),
        'ident': 4,
    },
]


def signal_mask(wave):
    """Return boolean mask (True = signal-like, exclude). Mirrors Microboone::SignalFilter."""
    good = wave < SAT_ADC
    if not np.any(good):
        return np.zeros(len(wave), dtype=bool)
    p16, p50, p84 = np.percentile(wave[good], [16, 50, 84])
    rms = np.sqrt(((p84 - p50) ** 2 + (p50 - p16) ** 2) / 2.0)
    if rms == 0:
        return np.zeros(len(wave), dtype=bool)
    flag = (np.abs(wave - p50) > SIG_FACTOR * rms).astype(int)
    kernel = np.ones(2 * PAD_BINS + 1, dtype=int)
    return np.convolve(flag, kernel, mode='same').astype(bool)


def plane_mean_spectrum(root_file, hist_name):
    """Return (freq_MHz, mean_|FFT|, n_channels)."""
    f = uproot.open(root_file)
    raw_all = f[hist_name].values()        # (nch, ntick)
    nch, ntick = raw_all.shape
    freq = np.fft.rfftfreq(ntick, d=TICK_US)    # MHz
    accum = np.zeros(len(freq))
    for i in range(nch):
        w     = raw_all[i].astype(float)
        med   = float(np.median(w))
        w0    = w - med
        mask  = signal_mask(w)
        # Anchor baseline to noise-only samples before zeroing signal, so
        # zeroing signal peaks doesn't introduce a spurious DC offset.
        if (~mask).any():
            w0 -= float(w0[~mask].mean())
        w0[mask] = 0.0
        accum += np.abs(np.fft.rfft(w0))
    return freq, accum / nch, nch


def make_compare_plot(plane_label, results, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    fig.suptitle(
        f'Plane {plane_label}  —  post-NF (_raw) noise spectrum comparison  '
        f'(mean |FFT| over plane channels, signal masked)',
        fontsize=11,
    )
    for r in results:
        lbl = f"{r['label']}   (N_ch = {r['nch']})"
        area = np.trapz(r['spec'], r['freq'])
        spec_norm = r['spec'] / area if area > 0 else r['spec']
        axes[0].plot(r['freq'], r['spec'],      color=r['color'], lw=1.3, label=lbl)
        axes[1].plot(r['freq'], spec_norm,      color=r['color'], lw=1.3, label=lbl)
    for ax, ylabel, title in [
        (axes[0], 'mean |FFT| of _raw  (ADC)',        'Absolute'),
        (axes[1], 'mean |FFT| normalized by area  (1/MHz)', 'Normalized by area'),
    ]:
        ax.set_xlabel('frequency (MHz)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xlim(0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='upper right')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f'  wrote {outpath}')


def run():
    plane_specs = [
        (0, 'U', 'hu_raw'),
        (1, 'V', 'hv_raw'),
        (2, 'W', 'hw_raw'),
    ]
    for _, plane_label, prefix in plane_specs:
        print(f'\n=== {plane_label} plane ===')
        results = []
        for d in DETECTORS:
            hname = f'{prefix}{d["ident"]}'
            freq, spec, nch = plane_mean_spectrum(d['root'], hname)
            results.append({**d, 'freq': freq, 'spec': spec, 'nch': nch})
            print(f'  {d["label"]:30s}  {hname:10s}  N_ch={nch}  '
                  f'peak={spec.max():.1f} ADC @ {freq[np.argmax(spec)]:.3f} MHz')
        outpath = os.path.join(SCRIPTDIR, f'noise_spectrum_compare_{plane_label}.png')
        make_compare_plot(plane_label, results, outpath)


if __name__ == '__main__':
    run()
