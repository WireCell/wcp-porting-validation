#!/usr/bin/env python3
"""
compare_lf_filters.py — low-frequency-cutoff (LfFilter) comparison for PDVD and PDHD.

Produces three PNGs in the same directory as this script:

  compare_lf_filters_freq.png     |H(f)| = 1 − exp(−(f/τ)²) in frequency domain
  compare_lf_filters_impulse.png  Actual impulse response l(t) = iFFT[L(f)]
                                    — near-delta spike at t=0 plus negative Gaussian wings
  compare_lf_filters_demo.png     Filter applied to a synthetic waveform
                                    — shows what each variant removes from data

Context
-------
The LfFilter is a high-pass component of the 2-D ROI-finding deconvolution.
Three variants are wired into OmnibusSigProc (via the Jsonnet SP config):

  ROI_loose_lf   → decon_2D_looseROI()    (gentlest LF removal)
  ROI_tight_lf   → decon_2D_tightROI()
  ROI_tighter_lf → decon_2D_tighterROI()  (most aggressive LF removal)

The filter formula (util/src/Response.cxx:444):
  L(f) = 1 − exp(−(f/τ)²)     high-pass; L(0) = 0, L(∞) → 1

Its actual impulse response in the time domain is:
  l(t) = iFFT[L(f)] = δ(t) − g(t)
where g(t) = iFFT[exp(−(f/τ)²)] is a Gaussian of FWHM ≈ 2√(2 ln 2)/(π·τ).
Convolving a waveform with l(t) means: pass it as-is (the δ spike), then
subtract a Gaussian-smeared copy (the −g term) — i.e. high-pass filter.

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


def lf_impulse(tau, window_us):
    """iFFT of L(f) centred at t = 0. Returns (t_us, l) within |t| ≤ window_us."""
    L = _lf_full(tau)
    l = np.fft.fftshift(np.real(np.fft.ifft(L)))
    t = (np.arange(N_TIME) - N_TIME // 2) * TICK_US
    mask = np.abs(t) <= window_us
    return t[mask], l[mask]


def fwhm_wing_us(tau):
    """
    FWHM (µs) of the wing Gaussian g(t) = iFFT[exp(−(f/τ)²)].
    Continuous-FT approximation: FWHM = 2√(2 ln 2)/(π·τ).  (τ in MHz → µs.)
    """
    return 2.0 * np.sqrt(2.0 * np.log(2.0)) / (np.pi * tau)


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


# ── plot 2: actual impulse response ──────────────────────────────────────────

def make_impulse_plot():
    """
    Show the true filter impulse response l(t) = iFFT[L(f)] = δ(t) − g(t).

    Left  (±5 µs):   the spike at t = 0; height just below 1.
    Right (±100 µs): negative wings l(t) for t ≠ 0 (y-axis clipped;
                     spike is off scale and not shown).
    """
    fig, (ax_spike, ax_wings) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        'LfFilter actual impulse response: l(t) = iFFT[L(f)] = δ(t) − g(t)\n'
        'Convolving a waveform with l(t) passes it as-is (the δ spike) while\n'
        'subtracting a Gaussian-smeared copy (the −g wings) — a high-pass filter.\n'
        'Left: spike at t = 0.  Right: negative wings (y clipped; spike omitted).',
        fontsize=10,
    )

    for variant in VARIANTS:
        c = C[variant]
        for det, lf_dict, ls in [('PDVD', PDVD_LF, LS_PDVD),
                                   ('PDHD', PDHD_LF, LS_PDHD)]:
            tau = lf_dict[variant]
            if det == 'PDHD' and tau == PDVD_LF[variant]:
                continue

            # spike height = l(t=0) = iFFT[L][N//2] after fftshift
            l_full = np.fft.fftshift(np.real(np.fft.ifft(_lf_full(tau))))
            spike_h = l_full[N_TIME // 2]
            fw = fwhm_wing_us(tau)
            fw_str = f'{fw:.0f} µs' if fw > 200 else f'{fw:.1f} µs'

            if det == 'PDHD':
                det_lbl = 'PDHD'
            elif PDVD_LF[variant] == PDHD_LF[variant]:
                det_lbl = 'PDVD=PDHD'
            else:
                det_lbl = 'PDVD'

            label = (f'{det_lbl} {variant}  τ={tau:.3f} MHz  '
                     f'spike≈{spike_h:.4f}  wing FWHM≈{fw_str}')

            t5, l5 = lf_impulse(tau, 5.0)
            ax_spike.plot(t5, l5, color=c, ls=ls, lw=LW, label=label)

            t100, l100 = lf_impulse(tau, 100.0)
            ax_wings.plot(t100, l100, color=c, ls=ls, lw=LW, label='_nolegend_')

    for ax in (ax_spike, ax_wings):
        ax.axhline(0, color='k', lw=0.5)
        ax.axvline(0, color='grey', lw=0.5, ls=':')
        ax.set_xlabel('Time (µs)')
        ax.set_ylabel('l(t)')
        ax.grid(True, alpha=0.3)

    ax_spike.set_xlim(-5, 5)
    ax_spike.set_title('±5 µs: spike at t = 0')
    ax_spike.legend(fontsize=8.0, loc='upper right')

    ax_wings.set_xlim(-100, 100)
    ax_wings.set_ylim(-0.07, 0.01)
    ax_wings.set_title('±100 µs: negative wings (y clipped, spike not visible here)')

    fig.tight_layout()
    out = os.path.join(SCRIPT_DIR, 'compare_lf_filters_impulse.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Wrote {out}')


# ── plot 3: demo on synthetic waveform ───────────────────────────────────────

def make_demo_plot():
    """
    Apply each PDVD LfFilter variant to a synthetic waveform.

    Waveform = narrow signal + two sinusoidal baseline components at
    frequencies that straddle the filter cutoffs:
      - Slow baseline:   f = 0.002 MHz (period = 500 µs) → removed by tight/tighter,
                         partially attenuated by loose (L ≈ 0.63 at τ = 0.002 MHz)
      - Medium baseline: f = 0.012 MHz (period ≈ 83 µs)  → passed by loose,
                         ~50 % attenuated by tight, blocked by tighter
    """
    t_us = np.arange(N_TIME, dtype=float) * TICK_US
    center = (N_TIME // 2) * TICK_US  # 1500 µs

    # Narrow signal: Gaussian σ = 3 µs (frequency content well above all cutoffs)
    sig     = 50.0 * np.exp(-0.5 * ((t_us - center) / 3.0) ** 2)
    # Baselines: sinusoids centred at t=0 in a periodic frame
    bl_slow = 30.0 * np.sin(2.0 * np.pi * 0.002 * (t_us - center))
    bl_med  = 20.0 * np.sin(2.0 * np.pi * 0.012 * (t_us - center))

    raw     = sig + bl_slow + bl_med
    raw_fft = np.fft.fft(raw)

    t_rel = t_us - center

    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex='col',
                              gridspec_kw={'width_ratios': [2, 1]})
    fig.suptitle(
        'LfFilter demo: PDVD variants applied to a synthetic waveform\n'
        'Waveform = narrow signal (σ=3 µs, 50 ADC) + slow baseline (f=0.002 MHz, 30 ADC, period=500 µs)'
        ' + medium baseline (f=0.012 MHz, 20 ADC, period≈83 µs)\n'
        'Left: ±500 µs full view.  Right: ±30 µs zoom.  '
        'Grey = raw (reference); colour = filtered.  Dashed = removed component (raw − filtered).',
        fontsize=10,
    )

    WIN_WIDE = 500.0
    WIN_ZOOM = 30.0

    # ── row 0: raw waveform ───────────────────────────────────────────────────
    for col, win in enumerate([WIN_WIDE, WIN_ZOOM]):
        msk = np.abs(t_rel) <= win
        ax = axes[0, col]
        ax.plot(t_rel[msk], raw[msk],     'k-',        lw=1.5, label='raw')
        ax.plot(t_rel[msk], bl_slow[msk], color='C7',  lw=1.0, ls='--',
                label='slow baseline (0.002 MHz)')
        ax.plot(t_rel[msk], bl_med[msk],  color='C7',  lw=1.0, ls=':',
                label='medium baseline (0.012 MHz)')
        ax.axhline(0, color='grey', lw=0.5)
        ax.set_xlim(-win, win)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel('ADC')
            ax.legend(fontsize=8.5, loc='upper right')

    axes[0, 0].set_title('Full ±500 µs window')
    axes[0, 1].set_title('±30 µs zoom around signal')

    # ── rows 1–3: each PDVD variant ──────────────────────────────────────────
    for row, variant in enumerate(VARIANTS, start=1):
        tau      = PDVD_LF[variant]
        L        = _lf_full(tau)
        filtered = np.real(np.fft.ifft(raw_fft * L))
        removed  = raw - filtered

        # transmission at the two baseline frequencies
        L_slow = 1.0 - np.exp(-(0.002 / tau) ** 2)
        L_med  = 1.0 - np.exp(-(0.012 / tau) ** 2)
        row_title = (f'PDVD {variant}  τ={tau:.3f} MHz  '
                     f'[L(0.002 MHz)={L_slow:.2f}  L(0.012 MHz)={L_med:.2f}]')

        for col, win in enumerate([WIN_WIDE, WIN_ZOOM]):
            msk = np.abs(t_rel) <= win
            ax  = axes[row, col]
            ax.plot(t_rel[msk], raw[msk],      color='lightgrey', lw=0.8, label='raw (ref)')
            ax.plot(t_rel[msk], filtered[msk], color=C[variant],  lw=1.5, label='filtered')
            ax.plot(t_rel[msk], removed[msk],  color=C[variant],  lw=1.0, ls='--',
                    label='removed (raw − filtered)')
            ax.axhline(0, color='grey', lw=0.5)
            ax.set_xlim(-win, win)
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel('ADC')
                ax.set_title(row_title, fontsize=9)
                ax.legend(fontsize=8.5, loc='upper right')

    for ax in axes[-1, :]:
        ax.set_xlabel('Time relative to center (µs)')

    fig.tight_layout()
    out = os.path.join(SCRIPT_DIR, 'compare_lf_filters_demo.png')
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
    # G(0) = 1 - L(0) = 1
    G = 1.0 - L
    assert abs(G[0] - 1.0) < 1e-10, f'G(0) = {G[0]}'
    # FWHM of tighter wing should be ~8-25 µs for PDHD
    fw = fwhm_wing_us(PDHD_LF['tighter'])
    assert 8 < fw < 25, f'PDHD tighter wing FWHM = {fw:.1f} µs'
    # impulse response at t=0 should be just below 1
    _, l5 = lf_impulse(PDVD_LF['tighter'], 5.0)
    spike = l5.max()
    assert 0.90 < spike < 1.0, f'tighter spike height = {spike:.4f}'
    print('_self_test: OK')


def main():
    _self_test()
    make_freq_plot()
    make_impulse_plot()
    make_demo_plot()


if __name__ == '__main__':
    main()
