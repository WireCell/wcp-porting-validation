#!/usr/bin/env python3
"""
compare_lf_filters.py — low-frequency-cutoff (LfFilter) comparison for PDVD and PDHD.

Produces two PNGs in the same directory as this script:

  compare_lf_filters_freq.png   |H(f)| = 1 − exp(−(f/τ)²) in frequency domain
  compare_lf_filters_time.png   Complement G(t) = iFFT[exp(−(f/τ)²)] in time domain
                                  — the Gaussian baseline shape each filter subtracts

Context
-------
The LfFilter is a high-pass component of the 2-D ROI-finding deconvolution.
Three variants are wired into OmnibusSigProc (via the Jsonnet SP config):

  ROI_loose_lf   → decon_2D_looseROI()    (gentlest LF removal)
  ROI_tight_lf   → decon_2D_tightROI()
  ROI_tighter_lf → decon_2D_tighterROI()  (most aggressive LF removal)

The filter formula (util/src/Response.cxx:444):
  L(f) = 1 − exp(−(f/τ)²)     high-pass; L(0) = 0, L(∞) → 1

The complement G(f) = 1 − L(f) = exp(−(f/τ)²) is a Gaussian low-pass.
Its iFFT G(t) is also a Gaussian in time, showing the time-scale of the
baseline that each filter removes from the deconvolved waveform.

Filter parameters (sp-filters.jsonnet):
  protodunevd lines 86–91 — _b and _t are byte-identical; PDVD top = bottom.
  pdhd        lines 41–43

Usage:
  python compare_lf_filters.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

N_TIME       = 6000
TICK_US      = 0.5
MAX_FREQ_MHZ = 1.0

# τ values in MHz
# PDVD: top and bottom are byte-identical
PDVD_LF = {
    'loose':   0.002,    # protodunevd/sp-filters.jsonnet:90-91
    'tight':   0.014,    # :86-87
    'tighter': 0.060,    # :88-89
}
PDHD_LF = {
    'loose':   0.002,    # pdhd/sp-filters.jsonnet:41
    'tight':   0.016,    # :42
    'tighter': 0.080,    # :43
}

# ── filter functions ──────────────────────────────────────────────────────────

def _lf_full(tau):
    """LfFilter in FFT bin order: L[i] = 1 − exp(−(|f_i|/τ)²)."""
    i = np.arange(N_TIME, dtype=float)
    f = i * (2.0 * MAX_FREQ_MHZ / N_TIME)
    f = np.where(f > MAX_FREQ_MHZ, f - 2.0 * MAX_FREQ_MHZ, f)
    f = np.abs(f)
    return 1.0 - np.exp(-(f / tau) ** 2)


def lf_pos_freq(tau):
    """(freq_mhz, H) for the positive-frequency half."""
    L = _lf_full(tau)
    freq = np.arange(N_TIME // 2 + 1) * (2.0 * MAX_FREQ_MHZ / N_TIME)
    return freq, L[:N_TIME // 2 + 1]


def complement_kernel(tau, window_us):
    """
    iFFT of the complement G(f) = exp(−(f/τ)²), centred at t = 0.
    Returns (t_us, g) within |t| ≤ window_us.  Normalised to unit peak.
    """
    G = 1.0 - _lf_full(tau)     # = exp(-(f/tau)²) in FFT order
    g = np.fft.fftshift(np.real(np.fft.ifft(G)))
    t = (np.arange(N_TIME) - N_TIME // 2) * TICK_US
    mask = np.abs(t) <= window_us
    peak = g[mask].max()
    return t[mask], g[mask] / peak


def fwhm_complement_us(tau):
    """
    FWHM (µs) of G(t) = iFFT[exp(−(f/τ)²)].
    Continuous-FT approximation: G(f) ↔ g(t) = τ√π·exp(−π²τ²t²),
    giving σ_t = 1/(√2·π·τ) and FWHM = 2√(2 ln 2)·σ_t = 2√(ln 2)·√2/(π·τ).
    (τ in MHz → result in µs.)
    """
    return 2.0 * np.sqrt(np.log(2)) * np.sqrt(2) / (np.pi * tau)


# ── palette ───────────────────────────────────────────────────────────────────

C = {'loose': '#9467bd', 'tight': '#1f77b4', 'tighter': '#d62728'}
LS_PDVD = '-'
LS_PDHD = '--'
LW = 2.0
VARIANTS = ['loose', 'tight', 'tighter']


# ── plot 1: frequency domain ──────────────────────────────────────────────────

def make_freq_plot():
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(
        'Low-frequency-cutoff filters (LfFilter): L(f) = 1 − exp(−(f/τ)²)\n'
        'Solid = PDVD top/bottom (identical).  Dashed = PDHD.\n'
        'PDVD loose = PDHD loose (τ = 0.002 MHz); only one curve shown.',
        fontsize=10,
    )

    for variant in VARIANTS:
        tau_p = PDVD_LF[variant]
        tau_h = PDHD_LF[variant]
        c = C[variant]

        freq, H_p = lf_pos_freq(tau_p)
        ax.plot(freq, H_p, color=c, ls=LS_PDVD, lw=LW,
                label=f'PDVD {variant}   τ = {tau_p:.3f} MHz')

        if tau_h != tau_p:
            freq, H_h = lf_pos_freq(tau_h)
            ax.plot(freq, H_h, color=c, ls=LS_PDHD, lw=LW,
                    label=f'PDHD {variant}   τ = {tau_h:.3f} MHz')
        else:
            # PDVD == PDHD: amend label instead of drawing a duplicate
            ax.lines[-1].set_label(
                f'PDVD = PDHD {variant}   τ = {tau_p:.3f} MHz'
            )

    ax.axhline(0.5, color='grey', lw=0.8, ls=':', label='H = 0.5')
    # thin vertical reference at Gaus_wide half-power (~0.10 MHz) for context
    ax.axvline(0.10, color='lightgrey', lw=1.0, ls='--', zorder=0,
               label='≈ Gaus_wide half-power (0.10 MHz) for context')

    ax.set_xlim(0, 0.15)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('|H(f)|')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(SCRIPT_DIR, 'compare_lf_filters_freq.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Wrote {out}')


# ── plot 2: time-domain complement ───────────────────────────────────────────

def make_time_plot():
    fig, (ax_wide, ax_zoom) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        'LfFilter complement G(t) = iFFT[exp(−(f/τ)²)] — the Gaussian baseline each filter subtracts\n'
        'Normalised to unit peak.  Left: ±200 µs (shows tight).  '
        'Right: ±30 µs zoom (shows tighter).\n'
        'Loose (FWHM ≈ 530 µs) is nearly flat in both panels — it only removes a DC-like, '
        'slowly drifting baseline.',
        fontsize=10,
    )

    panels = [(ax_wide, 200.0), (ax_zoom, 30.0)]

    for variant in VARIANTS:
        c = C[variant]
        for det, lf_dict, ls in [('PDVD', PDVD_LF, LS_PDVD),
                                  ('PDHD', PDHD_LF, LS_PDHD)]:
            tau = lf_dict[variant]
            # skip duplicate PDVD-loose == PDHD-loose
            if det == 'PDHD' and tau == PDVD_LF[variant]:
                continue

            fw = fwhm_complement_us(tau)
            if fw > 500:
                fw_str = f'{fw:.0f} µs  (>> window)'
            else:
                fw_str = f'{fw:.1f} µs'

            # build label for the wide panel only (shared legend)
            if det == 'PDHD':
                label = f'PDHD {variant}   τ={tau:.3f} MHz   FWHM≈{fw_str}'
            elif PDVD_LF[variant] == PDHD_LF[variant]:
                label = f'PDVD=PDHD {variant}   τ={tau:.3f} MHz   FWHM≈{fw_str}'
            else:
                label = f'PDVD {variant}   τ={tau:.3f} MHz   FWHM≈{fw_str}'

            for ax, win in panels:
                t, g = complement_kernel(tau, win)
                ax.plot(t, g, color=c, ls=ls, lw=LW,
                        label=label if ax is ax_wide else '_nolegend_')

    for ax, win in panels:
        step = 50.0 if win >= 100 else 10.0
        for xtick in np.arange(-win, win + 0.01, step):
            ax.axvline(xtick, color='lightgrey', lw=0.4, ls='--', zorder=0)
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xlim(-win, win)
        ax.set_ylim(-0.05, 1.08)
        ax.set_xlabel('Time (µs)')

    ax_wide.set_title('±200 µs window')
    ax_wide.set_ylabel('G(t) / peak')
    ax_wide.legend(fontsize=8.5, loc='upper right')

    ax_zoom.set_title('±30 µs zoom')
    ax_zoom.set_ylabel('')

    fig.tight_layout()
    out = os.path.join(SCRIPT_DIR, 'compare_lf_filters_time.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Wrote {out}')


# ── self-test ────────────────────────────────────────────────────────────────

def _self_test():
    # L(0) must be 0 (no DC pass-through)
    L = _lf_full(0.014)
    assert L[0] == 0.0, f'L(0) = {L[0]}'
    # L → 1 at Nyquist for reasonable τ
    assert L[N_TIME // 2] > 0.99, f'L(Nyquist) = {L[N_TIME // 2]:.4f}'
    # complement at DC = 1 (pure DC term in G)
    G = 1.0 - L
    assert abs(G[0] - 1.0) < 1e-10, f'G(0) = {G[0]}'
    # FWHM of tighter complement should be ~15-20 µs for PDHD
    fw = fwhm_complement_us(PDHD_LF['tighter'])
    assert 8 < fw < 25, f'PDHD tighter FWHM = {fw:.1f} µs'
    print('_self_test: OK')


def main():
    _self_test()
    make_freq_plot()
    make_time_plot()


if __name__ == '__main__':
    main()
