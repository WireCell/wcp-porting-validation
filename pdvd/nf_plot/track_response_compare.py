#!/usr/bin/env python3
"""
Cross-detector track response comparison: PDVD-top, PDVD-bottom, PDHD, uBooNE.

Computes the analytic FR ⊗ ER perpendicular-line MIP-track response for U and
V planes for each detector and overlays them in two PNGs.

Outputs (same directory as this script):
  track_response_compare_U.png
  track_response_compare_V.png

Each PNG:
  Top panel    — ADC waveform vs time, all four detectors, trough-aligned.
  Bottom panel — |FFT| of the same waveforms.

Amplitudes: absolute ADC per N_MIP (no per-curve rescaling).
Time alignment: each curve is shifted so its negative trough sits at T_REF_US.
"""

import os, json, bz2
import numpy as np
from scipy.signal import resample as sp_resample
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

from wirecell.sigproc.response import persist
from wirecell.sigproc import response as wc_resp
from wirecell import units
from wirecell.util.fileio import wirecell_path

WORKDIR = os.path.dirname(os.path.abspath(__file__))
ADC_TICK_NS = 500.0
T_REF_US    = 20.0   # trough sits here in the comparison plot


# ---------------------------------------------------------------------------
# Helpers (identical to the per-detector scripts)
# ---------------------------------------------------------------------------

def n_mip(pitch_mm):
    return (1.8e6 * (pitch_mm / 10.0) * 0.7) / 23.6


def line_source_response(plane):
    pitch = plane.pitch
    N     = len(plane.paths[0].current)
    by_r  = defaultdict(list)
    for path in plane.paths:
        r  = int(round(path.pitchpos / pitch))
        xi = path.pitchpos - r * pitch
        by_r[r].append((xi, np.asarray(path.current, dtype=float)))
    integral = np.zeros(N)
    for items in by_r.values():
        sym = {xi: I for xi, I in items}
        for xi in list(sym):
            if abs(xi) > 1e-9 and (-xi) not in sym:
                sym[-xi] = sym[xi]
        xis = sorted(sym)
        n   = len(xis)
        w   = np.empty(n)
        w[0]  = (xis[1]  - xis[0])  / 2.0
        w[-1] = (xis[-1] - xis[-2]) / 2.0
        for i in range(1, n - 1):
            w[i] = (xis[i + 1] - xis[i - 1]) / 2.0
        for xi, wi in zip(xis, w):
            integral += wi * sym[xi]
    return integral / pitch


def load_jsonelec(filename):
    for d in wirecell_path():
        cand = os.path.join(d, filename)
        if os.path.exists(cand):
            with bz2.open(cand) as fh:
                data = json.load(fh)
            return np.array(data['times'], dtype=float), np.array(data['amplitudes'], dtype=float)
    raise FileNotFoundError(filename)


# ---------------------------------------------------------------------------
# Per-detector specifications
# ---------------------------------------------------------------------------

DETECTORS = [
    {
        'label':         'PDHD APA1-3',
        'short':         'PDHD',
        'color':         'C0',
        'fr_file':       'dune-garfield-1d565.json.bz2',
        'er_kind':       'cold',
        'er_gain':       14.0 * units.mV / units.fC,
        'er_shaping':    2.2  * units.us,
        'postgain':      1.0,
        'adc_per_mv':    11.70,
        'pad_window_ns': None,
        'n_mip_const':   None,
    },
    {
        'label':         'PDVD bottom',
        'short':         'PDVD-bot',
        'color':         'C1',
        'fr_file':       'protodunevd_FR_imbalance3p_260501.json.bz2',
        'er_kind':       'cold',
        'er_gain':       7.8 * units.mV / units.fC,
        'er_shaping':    2.2 * units.us,
        'postgain':      1.0,
        'adc_per_mv':    11.70,
        'pad_window_ns': 160_000.0,
        'n_mip_const':   None,
    },
    {
        'label':         'PDVD top',
        'short':         'PDVD-top',
        'color':         'C2',
        'fr_file':       'protodunevd_FR_imbalance3p_260501.json.bz2',
        'er_kind':       'jsonelec',
        'er_file':       'dunevd-coldbox-elecresp-top-psnorm_400.json.bz2',
        'postgain':      1.36,
        'adc_per_mv':    8.192,
        'pad_window_ns': 160_000.0,
        'n_mip_const':   None,
    },
    {
        'label':         'uBooNE',
        'short':         'uBooNE',
        'color':         'C3',
        'fr_file':       'ub-10-half.json.bz2',
        'er_kind':       'cold',
        'er_gain':       14.0 * units.mV / units.fC,
        'er_shaping':    2.2  * units.us,
        'postgain':      1.2,
        'adc_per_mv':    2.048,
        'pad_window_ns': None,
        'n_mip_const':   (1.8e6 * 0.3 * 0.7) / 23.6,
    },
]


# ---------------------------------------------------------------------------
# Load + compute
# ---------------------------------------------------------------------------

def load_detector(spec):
    """Load FR and build ER array for one detector. Returns (fr, period_ns, N_fr, er)."""
    print(f"\n--- {spec['label']} ---")
    print(f"  FR: {spec['fr_file']}")
    fr = persist.load(spec['fr_file'], paths=wirecell_path())
    period_ns    = fr.period
    N_fr_native  = len(fr.planes[0].paths[0].current)

    pad = spec['pad_window_ns']
    if pad and pad > N_fr_native * period_ns:
        N_fr = int(round(pad / period_ns))
        print(f"  FR padded: {N_fr_native} → {N_fr} samples ({N_fr * period_ns / 1000:.1f} µs)")
    else:
        N_fr = N_fr_native
        print(f"  FR native: {N_fr} samples ({N_fr * period_ns / 1000:.1f} µs)")

    if spec['er_kind'] == 'cold':
        times = np.arange(N_fr, dtype=float) * period_ns
        er    = np.asarray(wc_resp.electronics(times,
                                               peak_gain=spec['er_gain'],
                                               shaping=spec['er_shaping'],
                                               elec_type='cold'), dtype=float)
        print(f"  ER: cold  gain={spec['er_gain']/(units.mV/units.fC):.1f} mV/fC  "
              f"shaping={spec['er_shaping']/units.us:.1f} µs")
    else:
        er_t, er_a   = load_jsonelec(spec['er_file'])
        er_period    = er_t[1] - er_t[0]
        er_window    = er_t[-1] + er_period
        er           = np.zeros(N_fr)
        if abs(er_period - period_ns) > 1e-6:
            n_resamp = int(round(er_window / period_ns))
            er_resamp = sp_resample(er_a, n_resamp)
            m = min(len(er_resamp), N_fr)
            er[:m] = er_resamp[:m]
        else:
            m = min(len(er_a), N_fr)
            er[:m] = er_a[:m]
        peak_mv_per_fc = er_a[np.argmax(np.abs(er_a))] / (units.mV / units.fC)
        print(f"  ER: JsonElecResponse({spec['er_file']})  "
              f"peak≈{peak_mv_per_fc:.1f} mV/fC")

    print(f"  postgain={spec['postgain']}  ADC/mV={spec['adc_per_mv']}")
    return fr, period_ns, N_fr, er


def compute_plane_wave(fr, period_ns, N_fr, er, plane_id, spec):
    """Compute wave_adc for one plane of a loaded detector."""
    plane = next(pl for pl in fr.planes if pl.planeid == plane_id)
    nmip  = spec['n_mip_const'] if spec['n_mip_const'] else n_mip(plane.pitch)

    fr_line = line_source_response(plane)
    if N_fr > len(fr_line):
        fr_pad          = np.zeros(N_fr)
        fr_pad[:len(fr_line)] = fr_line
        fr_line         = fr_pad

    wave_mv  = -(wc_resp.convolve(fr_line, er) * period_ns * nmip / units.mV * spec['postgain'])
    N_adc    = int(round(N_fr * period_ns / ADC_TICK_NS))
    wave_adc = sp_resample(wave_mv, N_adc) * spec['adc_per_mv']
    tick_us  = ADC_TICK_NS / 1000.0

    pk_pos = wave_adc[np.argmax(wave_adc)]
    pk_neg = wave_adc[np.argmin(wave_adc)]
    print(f"  plane {plane_id}: pitch={plane.pitch:.2f} mm  N_MIP={nmip:.0f}  "
          f"peak={pk_pos:+.1f}  trough={pk_neg:+.1f} ADC")

    return wave_adc, tick_us, nmip, plane.pitch


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_compare_plot(plane_label, results, outpath):
    """
    results: list of dicts with keys label, color, wave_adc, tick_us, nmip, pitch_mm.
    Layout: 2×2.  Left column = absolute ADC.  Right column = trough-normalized.
    """
    tick_us = results[0]['tick_us']  # all 500 ns

    fig, axes = plt.subplots(2, 2, figsize=(22, 9))
    fig.suptitle(f'Plane {plane_label}  —  FR ⊗ ER track response comparison  '
                 f'(MIP perpendicular-line, trough-aligned at t = {T_REF_US:.0f} µs)',
                 fontsize=11)

    for r in results:
        wav   = r['wave_adc']
        N     = len(wav)
        i_neg = int(np.argmin(wav))
        t     = (np.arange(N) - i_neg) * tick_us + T_REF_US
        pk_pos = wav[np.argmax(wav)]
        pk_neg = wav[np.argmin(wav)]

        wav_norm = wav / abs(pk_neg)

        lbl_abs  = (f"{r['label']}  "
                    f"[pitch={r['pitch_mm']:.2f} mm  N_MIP={r['nmip']:.0f}  "
                    f"pk={pk_pos:+.1f}  tr={pk_neg:+.1f} ADC]")
        lbl_norm = r['label']

        # top-left: absolute time domain
        axes[0, 0].plot(t, wav,      color=r['color'], lw=1.5, label=lbl_abs)
        # top-right: normalized time domain
        axes[0, 1].plot(t, wav_norm, color=r['color'], lw=1.5, label=lbl_norm)

        freqs = np.fft.rfftfreq(N, d=tick_us)
        # bottom-left: absolute FFT
        axes[1, 0].plot(freqs, np.abs(np.fft.rfft(wav)),      color=r['color'], lw=1.5, label=r['label'])
        # bottom-right: normalized FFT
        axes[1, 1].plot(freqs, np.abs(np.fft.rfft(wav_norm)), color=r['color'], lw=1.5, label=r['label'])

    for col, ylabel, title in [
        (0, 'ADC  (absolute per N_MIP)', 'Absolute'),
        (1, 'ADC  (normalized: trough = −1)', 'Normalized to trough'),
    ]:
        ax = axes[0, col]
        ax.axhline(0, color='gray', lw=0.5)
        ax.axvline(T_REF_US, color='gray', lw=0.5, ls=':')
        ax.set_xlabel('time relative to trough (µs)')
        ax.set_ylabel(ylabel)
        ax.set_title(f'Time domain — {title}')
        ax.set_xlim(T_REF_US - 30, T_REF_US + 80)
        ax.legend(fontsize=7, loc='upper right')

        ax = axes[1, col]
        ax.set_xlabel('frequency (MHz)')
        ax.set_ylabel('|FFT|' + (' (absolute)' if col == 0 else ' (normalized)'))
        ax.set_title(f'Frequency spectrum — {title}')
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f'\n  wrote {outpath}')


# ---------------------------------------------------------------------------

def run():
    loaded = []
    for spec in DETECTORS:
        fr, period_ns, N_fr, er = load_detector(spec)
        loaded.append((spec, fr, period_ns, N_fr, er))

    plane_map = {0: 'U', 1: 'V'}
    for plane_id, plane_label in plane_map.items():
        print(f"\n=== {plane_label} plane ===")
        results = []
        for spec, fr, period_ns, N_fr, er in loaded:
            wave_adc, tick_us, nmip, pitch_mm = compute_plane_wave(
                fr, period_ns, N_fr, er, plane_id, spec)
            results.append({
                'label':    spec['label'],
                'color':    spec['color'],
                'wave_adc': wave_adc,
                'tick_us':  tick_us,
                'nmip':     nmip,
                'pitch_mm': pitch_mm,
            })
        outpath = os.path.join(WORKDIR, f'track_response_compare_{plane_label}.png')
        make_compare_plot(plane_label, results, outpath)


if __name__ == '__main__':
    run()
