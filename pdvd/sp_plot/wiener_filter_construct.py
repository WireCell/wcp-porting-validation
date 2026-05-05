#!/usr/bin/env python3
"""
Construct the data-driven Wiener filter W(f) = |S(f)|² / (|S(f)|² + |N(f)|²)
for PDHD APA1, PDVD bottom, and PDVD top — U and V induction planes.

Signal |S(f)|²
  Analytic FR ⊗ ER perpendicular-line MIP-track response (one wire pitch),
  i.e. the "standard candle" from track_response_compare.py.  Absolute ADC
  scale is preserved (each detector's own gain, shaping, postgain, ADC/mV),
  so a detector with smaller S/N gets a tighter filter.

Noise |N(f)|²
  Mean *power* per bin — <|FFT(channel)|²> averaged over all channels in the
  plane — from the post-NF _raw TH2 in a magnify ROOT file.  Signal samples
  are masked (Microboone SignalFilter) before the FFT.
  NOTE: the parallel noise_spectrum_compare.py accumulates mean *amplitude*
  <|X|> instead; squaring that would introduce a Jensen-inequality bias.  This
  script accumulates <|X|²> directly.

Window normalization (the "be careful with power vs amplitude" part)
  The noise ROOT files contain N_long = 6000 ticks = 3000 µs.  The Wiener
  filter is constructed on a T_WIN = 100 µs / N_SHORT = 200-tick window.
  For stationary noise with one-sided PSD P(f) [ADC²/MHz]:

    <|X_long(f)|²>  = P(f) · N_long / (2·dt)
    <|X_short(f)|²> = P(f) · N_short / (2·dt)
    ⇒  P_short = P_long · (N_short / N_long)          (linear in window length)

  Amplitude (|X|) would scale as √(N_short/N_long); power scales as
  N_short/N_long.  Both grids share the same Nyquist (1 MHz), so
  P_long is linearly interpolated onto the coarser short grid before scaling.

Signal truncation note
  PDVD's analytic response is zero-padded to 160 µs (320 ticks).  Extracting
  200 ticks centered on the trough clips ~30 µs from the long induction tails.
  This affects |S(f)|² at low frequencies and is intrinsic to the 100 µs
  window choice (not a bug).

Outputs (same directory as this script):
  wiener_filter_U.png
  wiener_filter_V.png

Each PNG has:
  Top panel    — W(f) vs f (0..1 MHz), all three detectors + fitted parametric +
                 production Wiener_tight per detector (dash-dot) + shared Wiener_wide (dashed).
  Bottom panel — same overlays in the time domain (iFFT).
"""

import os, sys
import numpy as np
import uproot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

SCRIPTDIR = os.path.dirname(os.path.abspath(__file__))
NF_PLOT   = os.path.join(SCRIPTDIR, '..', 'nf_plot')
sys.path.insert(0, NF_PLOT)

from track_response_compare import (
    DETECTORS as TR_DETECTORS, load_detector, compute_plane_wave,
)

T_WIN_US = 150.0
TICK_US  = 0.5
N_SHORT  = int(T_WIN_US / TICK_US)   # 300

SIG_FACTOR = 4.0
PAD_BINS   = 8
SAT_ADC    = 4096.0

PDVD_WORK = os.path.join(SCRIPTDIR, '..', 'work', '039324_0')
PDHD_WORK = os.path.join(SCRIPTDIR, '..', '..', 'pdhd', 'work', '027409_0')

# Extend the track-response detector specs with ROOT files and ident for noise.
# Indices 0-2 of TR_DETECTORS are PDHD APA1-3, PDVD bottom, PDVD top.
NOISE_ROOT = {
    'PDHD APA1-3': (os.path.join(PDHD_WORK, 'magnify-run027409-evt0-apa1.root'), 1),
    'PDVD bottom': (os.path.join(PDVD_WORK,  'magnify-run039324-evt0-anode0.root'), 0),
    'PDVD top':    (os.path.join(PDVD_WORK,  'magnify-run039324-evt0-anode4.root'), 4),
}

# Only the three detectors we have noise for (no uBooNE).
DETECTORS = [d for d in TR_DETECTORS if d['label'] in NOISE_ROOT]

# PDHD Wiener_wide parameters per plane (from pdhd/sp-filters.jsonnet lines 70-81).
# Note: PDVD and PDHD share the same wide-Wiener σ/power — the overlay
# is labelled "Wiener_wide (PDHD = PDVD)" to reflect this.
WIENER_WIDE = {
    0: {'sigma': 0.186765, 'power': 5.05429},   # U
    1: {'sigma': 0.1936,   'power': 5.77422},   # V
}

# Production Wiener_tight parameters per detector per plane.
# PDHD: APA1-specific values (matching the noise data source, apa1.root).
#   pdhd/sp-filters.jsonnet lines 58-65 (Wiener_tight_*_APA1)
# PDVD: _b and _t share identical parameters.
#   protodunevd/sp-filters.jsonnet lines 97-100 (Wiener_tight_*_{b,t})
WIENER_TIGHT = {
    'PDHD APA1-3': {
        0: {'sigma': 0.203451,  'power': 5.78093},   # Wiener_tight_U_APA1
        1: {'sigma': 0.160191,  'power': 3.54835},   # Wiener_tight_V_APA1
    },
    'PDVD bottom': {
        0: {'sigma': 0.148788,  'power': 3.76194},   # Wiener_tight_U_b
        1: {'sigma': 0.1596568, 'power': 4.36125},   # Wiener_tight_V_b
    },
    'PDVD top': {
        0: {'sigma': 0.148788,  'power': 3.76194},   # Wiener_tight_U_t (= _b)
        1: {'sigma': 0.1596568, 'power': 4.36125},   # Wiener_tight_V_t (= _b)
    },
}


def wiener_wide_analytic(f_mhz, plane_id):
    """Analytic PDHD/PDVD Wiener_wide filter on the given frequency grid."""
    p = WIENER_WIDE[plane_id]
    H = np.exp(-0.5 * (f_mhz / p['sigma']) ** p['power'])
    H[0] = 0.0   # flag=True: zero DC bin
    return H


# ---------------------------------------------------------------------------
# Fit the data-driven W(f) to the WireCell parametric form:
#   H(f) = exp(-0.5 * (f/sigma)^power)
# H(0) = 1 by construction; the fit targets the falling edge.
# ---------------------------------------------------------------------------

def _wiener_form(f, sigma, power):
    return np.exp(-0.5 * (f / sigma) ** power)


def _log_wiener_form(f, sigma, power):
    """log(H) = -0.5*(f/sigma)^power.  Used for log-space fitting."""
    return -0.5 * (f / sigma) ** power


def fit_wiener_params(f_mhz, W, f_lo=0.1, f_hi=0.5):
    """
    Fit exp(-0.5*(f/sigma)^power) to W over [f_lo, f_hi] MHz in log space.

    Fitting log(W) = -0.5*(f/sigma)^power (rather than W directly) weights
    residuals by 1/W, giving the high-frequency tail stronger relative
    influence.  The fitted curve is therefore slightly more aggressive
    (lower) at the tail than a linear-space fit would produce.

    Returns (sigma_MHz, power) or (None, None) on failure.
    """
    # Exclude the noise-floor tail (W < 0.05).  Near-zero-W bins get very
    # large weight in log space (log(0.02) = -3.9) and may include secondary
    # spectral humps from signal leakage, both of which badly bias σ and
    # power.  W = 0.05 is well inside the physical transition for all three
    # detectors while cutting cleanly before any secondary humps.
    mask  = (f_mhz >= f_lo) & (f_mhz <= f_hi) & (W > 0.05)
    f_fit = f_mhz[mask]
    W_fit = W[mask]
    if len(f_fit) < 3:
        return None, None
    try:
        popt, _ = curve_fit(_log_wiener_form, f_fit, np.log(W_fit),
                            p0=[0.2, 4.0],
                            bounds=([0.02, 0.5], [2.0, 30.0]),
                            maxfev=10000)
        return float(popt[0]), float(popt[1])
    except RuntimeError:
        return None, None


# ---------------------------------------------------------------------------
# Signal-masking helper (verbatim from noise_spectrum_compare.py)
# ---------------------------------------------------------------------------

def signal_mask(wave):
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


# ---------------------------------------------------------------------------
# Noise: mean *power* per bin  (<|X|²>, not <|X|>)
# ---------------------------------------------------------------------------

def plane_mean_power_spectrum(root_file, hist_name):
    """Return (freq_MHz, mean_power_ADC², N_long, n_channels)."""
    f = uproot.open(root_file)
    raw_all       = f[hist_name].values()
    nch, ntick    = raw_all.shape
    freq          = np.fft.rfftfreq(ntick, d=TICK_US)   # MHz
    accum         = np.zeros(len(freq))
    for i in range(nch):
        w    = raw_all[i].astype(float)
        med  = float(np.median(w))
        w0   = w - med
        mask = signal_mask(w)
        if (~mask).any():
            w0 -= float(w0[~mask].mean())
        w0[mask] = 0.0
        accum += np.abs(np.fft.rfft(w0)) ** 2          # power, not amplitude
    return freq, accum / nch, ntick, nch


# ---------------------------------------------------------------------------
# Build signal window (200 ticks centred on the trough)
# ---------------------------------------------------------------------------

def signal_window(wave_adc):
    i_neg  = int(np.argmin(wave_adc))
    half   = N_SHORT // 2
    win    = np.zeros(N_SHORT)
    src_lo = max(0, i_neg - half)
    src_hi = min(len(wave_adc), i_neg + half)
    dst_lo = half - (i_neg - src_lo)
    win[dst_lo:dst_lo + (src_hi - src_lo)] = wave_adc[src_lo:src_hi]
    return win


# ---------------------------------------------------------------------------
# Wiener filter for one (detector, plane)
# ---------------------------------------------------------------------------

def wiener_one(spec, plane_id):
    """
    Returns dict with keys:
      f_short  — frequency grid (MHz), length N_SHORT//2 + 1
      W        — Wiener filter, same length
      t        — time axis (µs), length N_SHORT, centred at 0
      w_t      — iFFT of W (time-domain filter kernel)
      S_pow    — |S(f)|² on f_short grid
      N_pow    — |N(f)|² on f_short grid (rescaled to N_SHORT window)
    """
    root_file, ident = NOISE_ROOT[spec['label']]

    # --- signal ---
    fr, period_ns, N_fr, er = load_detector(spec)
    wave_adc, _, _, _ = compute_plane_wave(fr, period_ns, N_fr, er, plane_id, spec)
    sig_win = signal_window(wave_adc)
    f_short = np.fft.rfftfreq(N_SHORT, TICK_US)          # MHz, 101 bins
    S_pow   = np.abs(np.fft.rfft(sig_win)) ** 2

    # --- noise (rescaled to N_SHORT window) ---
    plane_prefix = {0: 'hu_raw', 1: 'hv_raw', 2: 'hw_raw'}[plane_id]
    hist_name    = f'{plane_prefix}{ident}'
    f_long, P_long, N_long, nch = plane_mean_power_spectrum(root_file, hist_name)
    print(f'  [{spec["label"]}] plane {plane_id}  noise: {nch} ch, '
          f'N_long={N_long}, scale={N_SHORT}/{N_long}={N_SHORT/N_long:.4f}')

    # Interpolate long-window power onto short-window freq grid, then scale.
    P_short = np.interp(f_short, f_long, P_long) * (N_SHORT / N_long)

    # --- Wiener ---
    W   = S_pow / (S_pow + P_short)   # real, in [0, 1]
    w_t = np.fft.fftshift(np.fft.irfft(W, n=N_SHORT))
    t   = (np.arange(N_SHORT) - N_SHORT // 2) * TICK_US

    # Fit the falling edge to the WireCell parametric form.
    sigma_fit, power_fit = fit_wiener_params(f_short, W)

    return {'f_short': f_short, 'W': W, 't': t, 'w_t': w_t,
            'S_pow': S_pow, 'N_pow': P_short,
            'sigma_fit': sigma_fit, 'power_fit': power_fit,
            'label': spec['label'], 'color': spec['color']}


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_plot(plane_label, plane_id, results, outpath):
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    fig.suptitle(
        f'Plane {plane_label}  —  data-driven Wiener filter  '
        f'W(f) = |S|² / (|S|² + |N|²)\n'
        f'Signal: analytic FR⊗ER line-track  |  Noise: post-NF _raw, '
        f'scaled to {T_WIN_US:.0f} µs window',
        fontsize=10,
    )

    for r in results:
        axes[0].plot(r['f_short'], r['W'],   color=r['color'], lw=1.5, label=r['label'])
        axes[1].plot(r['t'],       r['w_t'], color=r['color'], lw=1.5, label=r['label'])
        # Fitted parametric curve (dotted, same colour).
        if r['sigma_fit'] is not None:
            H_fit   = _wiener_form(r['f_short'], r['sigma_fit'], r['power_fit'])
            h_fit_t = np.fft.fftshift(np.fft.irfft(H_fit, n=N_SHORT))
            lbl_fit = (f"  fit σ={r['sigma_fit']:.4f} MHz  pow={r['power_fit']:.3f}")
            axes[0].plot(r['f_short'], H_fit,   color=r['color'], lw=1.2, ls=':',
                         label=r['label'] + lbl_fit)
            axes[1].plot(r['t'],       h_fit_t, color=r['color'], lw=1.2, ls=':')

    f_ref = results[0]['f_short']
    t_ref = results[0]['t']

    # Overlay production Wiener_tight per detector (dash-dot, same colour).
    # PDVD top and bottom share identical tight parameters; their lines overlap.
    seen_tight = set()
    for r in results:
        tp = WIENER_TIGHT.get(r['label'], {}).get(plane_id)
        if tp is None:
            continue
        key = (tp['sigma'], tp['power'])
        H_tight   = _wiener_form(f_ref, tp['sigma'], tp['power'])
        h_tight_t = np.fft.fftshift(np.fft.irfft(H_tight, n=N_SHORT))
        suffix    = '(top = bottom)' if key in seen_tight else ''
        lbl_tight = (f"{r['label']} Wiener_tight {suffix} "
                     f"σ={tp['sigma']:.4f} MHz  pow={tp['power']:.3f}")
        axes[0].plot(f_ref,   H_tight,   color=r['color'], lw=1.5, ls='-.',
                     label=lbl_tight)
        axes[1].plot(t_ref,   h_tight_t, color=r['color'], lw=1.5, ls='-.')
        seen_tight.add(key)

    # Overlay analytic Wiener_wide (identical for PDHD and PDVD — single line).
    H_wide   = wiener_wide_analytic(f_ref, plane_id)
    h_wide_t = np.fft.fftshift(np.fft.irfft(H_wide, n=N_SHORT))
    p        = WIENER_WIDE[plane_id]
    wide_lbl = (f'Wiener_wide (PDHD = PDVD)  '
                f'σ={p["sigma"]:.4f} MHz  pow={p["power"]:.3f}')
    axes[0].plot(f_ref,   H_wide,   color='k', lw=1.5, ls='--', label=wide_lbl)
    axes[1].plot(t_ref,   h_wide_t, color='k', lw=1.5, ls='--', label='Wiener_wide (PDHD = PDVD)')

    axes[0].set_xlabel('frequency (MHz)')
    axes[0].set_ylabel('W(f)  [0..1]')
    axes[0].set_title('Frequency domain')
    axes[0].set_xlim(0, 1.0)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].axhline(0, color='gray', lw=0.5)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)

    axes[1].set_xlabel('time (µs)')
    axes[1].set_ylabel('w(t)  [arb.]')
    axes[1].set_title(f'Time domain  (iFFT of W, {T_WIN_US:.0f} µs window)')
    axes[1].axhline(0, color='gray', lw=0.5)
    axes[1].axvline(0, color='gray', lw=0.5, ls=':')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f'  wrote {outpath}')


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test():
    print('_self_test: checking basic properties ...')
    fr, period_ns, N_fr, er = load_detector(DETECTORS[0])
    wave_adc, _, _, _ = compute_plane_wave(fr, period_ns, N_fr, er, 0, DETECTORS[0])
    sig_win = signal_window(wave_adc)
    f_short = np.fft.rfftfreq(N_SHORT, TICK_US)
    S_pow   = np.abs(np.fft.rfft(sig_win)) ** 2
    # Use a synthetic noise power (flat) to avoid file I/O in the test.
    P_flat  = np.ones_like(S_pow) * float(S_pow.max()) * 0.5
    W       = S_pow / (S_pow + P_flat)
    assert np.all(W >= 0) and np.all(W <= 1), 'W not in [0,1]'
    w_t     = np.fft.fftshift(np.fft.irfft(W, n=N_SHORT))
    # Parseval round-trip: sum(|W|²) == N_SHORT * sum(|w_t|²) / N_SHORT (unnorm)
    assert abs(np.sum(np.abs(W) ** 2) - np.sum(np.abs(np.fft.rfft(w_t)) ** 2)) < 1e-6, \
        'iFFT round-trip failed'
    print('_self_test: OK')


# ---------------------------------------------------------------------------

def run():
    plane_map = {0: 'U', 1: 'V'}
    for plane_id, plane_label in plane_map.items():
        print(f'\n=== Plane {plane_label} ===')
        results = []
        for spec in DETECTORS:
            r = wiener_one(spec, plane_id)
            peak = r['w_t'].max()
            s, p = r['sigma_fit'], r['power_fit']
            fit_str = f'σ={s:.4f} MHz  pow={p:.3f}' if s is not None else 'fit failed'
            print(f'  {spec["label"]:20s}  W_dc={r["W"][0]:.3f}  '
                  f'W_pk={r["W"].max():.3f}  w_t_peak={peak:.4f}  fit: {fit_str}')
            results.append(r)
        outpath = os.path.join(SCRIPTDIR, f'wiener_filter_{plane_label}.png')
        make_plot(plane_label, plane_id, results, outpath)


if __name__ == '__main__':
    _self_test()
    run()
