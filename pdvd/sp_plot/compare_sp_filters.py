#!/usr/bin/env python3
"""
compare_sp_filters.py — high-frequency-cutoff filter comparison for PDVD and PDHD.

Produces five PNGs in the same directory as this script:

  1. compare_wiener_wide_freq.png      Wiener wide, frequency domain
  2. compare_wiener_wide_time.png      Wiener wide, time domain (500 ns/tick)
  3. compare_wiener_wide_vs_tight.png  Wide vs tight Wiener: freq + time
  4. compare_gauss.png                 Gaus_wide filter: freq + time
  5. compare_wire_filter.png           Wire filters (induction/collection):
                                       wire-frequency + wire-index spatial domains

Usage:
  python compare_sp_filters.py          # produce all five PNGs
  python compare_sp_filters.py --only 3 # produce only plot 3

Filter parameters sourced from:
  toolkit/cfg/pgrapher/experiment/protodunevd/sp-filters.jsonnet (lines 93-114)
  toolkit/cfg/pgrapher/experiment/pdhd/sp-filters.jsonnet         (lines 45-84)

Filter math verified against:
  toolkit/util/src/Response.cxx:435-444  (hf_filter formula)
  toolkit/sigproc/src/HfFilter.cxx:38-52 (frequency-bin grid)
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── sampling parameters ────────────────────────────────────────────────────────

N_TIME       = 6000    # WCT frame size (ticks)
TICK_US      = 0.5     # tick period µs → Nyquist = 1 MHz
MAX_FREQ_MHZ = 1.0     # Nyquist for time-domain filters

N_WIRE       = 1024    # wire-filter FFT size (display resolution)
MAX_FREQ_WIRE = 1.0    # dimensionless Nyquist for wire filters

# ── filter parameters (sp-filters.jsonnet) ────────────────────────────────────

SQRTPI = np.sqrt(np.pi)

# protodunevd/sp-filters.jsonnet lines 97-109
# _b and _t are byte-identical today; we use one entry per plane.
PDVD_WIENER_TIGHT = {
    'U': dict(sigma=0.148788,  power=3.76194),   # lines 97-98
    'V': dict(sigma=0.1596568, power=4.36125),   # lines 99-100
    'W': dict(sigma=0.13623,   power=3.35324),   # lines 101-102
}
PDVD_WIENER_WIDE = {
    'U': dict(sigma=0.186765,  power=5.05429),   # lines 104-105
    'V': dict(sigma=0.1936,    power=5.77422),   # lines 106-107
    'W': dict(sigma=0.175722,  power=4.37928),   # lines 108-109
}
PDVD_GAUS_WIDE = dict(sigma=0.12, power=2.0)    # lines 94-95

# pdhd/sp-filters.jsonnet lines 48-81
PDHD_WIENER_TIGHT_DEFAULT = {
    'U': dict(sigma=0.221933,  power=6.55413),   # lines 48-50
    'V': dict(sigma=0.222723,  power=8.75998),   # lines 51-53
    'W': dict(sigma=0.225567,  power=3.47846),   # lines 54-56
}
PDHD_WIENER_TIGHT_APA1 = {
    'U': dict(sigma=0.203451,  power=5.78093),   # lines 58-60
    'V': dict(sigma=0.160191,  power=3.54835),   # lines 61-63
    'W': dict(sigma=0.125448,  power=5.27080),   # lines 64-67
}
PDHD_WIENER_WIDE = {                              # lines 70-81 (identical to PDVD wide)
    'U': dict(sigma=0.186765,  power=5.05429),
    'V': dict(sigma=0.1936,    power=5.77422),
    'W': dict(sigma=0.175722,  power=4.37928),
}
PDHD_GAUS_WIDE = dict(sigma=0.12, power=2.0)    # line 46

# Wire filters (σ in dimensionless wire-frequency units; max_freq=1)
# protodunevd/sp-filters.jsonnet lines 111-114
PDVD_WIRE_IND = 5.0  / SQRTPI    # ≈ 2.821
PDVD_WIRE_COL = 10.0 / SQRTPI    # ≈ 5.642
# pdhd/sp-filters.jsonnet lines 83-84
PDHD_WIRE_IND = 0.75 / SQRTPI    # ≈ 0.423
PDHD_WIRE_COL = 10.0 / SQRTPI    # ≈ 5.642

# ── core filter functions ──────────────────────────────────────────────────────
# All match toolkit/util/src/Response.cxx::hf_filter + HfFilter.cxx:38-52.


def _hf_array(N, max_freq, sigma, power, flag):
    """
    Build an HF-filter array in FFT bin order.

    H[i] = exp(-0.5 * |f_i / sigma|^power)
    where f_i = i * 2*max_freq/N, wrapped above max_freq.
    If flag=True, H[0] is forced to 0 (no DC pass-through).
    """
    i = np.arange(N, dtype=float)
    f = i * (2.0 * max_freq / N)
    f = np.where(f > max_freq, f - 2.0 * max_freq, f)
    f = np.abs(f)
    H = np.exp(-0.5 * (f / sigma) ** power)
    if flag:
        H[0] = 0.0
    return H


def hf_pos_freq(N, max_freq, sigma, power, flag):
    """Return (freq_axis, H) for the positive-frequency half of the HF filter."""
    H_full = _hf_array(N, max_freq, sigma, power, flag)
    freq = np.arange(N // 2 + 1) * (2.0 * max_freq / N)
    return freq, H_full[:N // 2 + 1]


def iFFT_kernel(H_full, bin_size):
    """
    Real-valued impulse response of the filter, centred at zero.

    H_full must be in FFT order.  Returns (axis, h) where axis is in units
    of bin_size (µs for time filters, wire-index for wire filters).
    """
    N = len(H_full)
    h = np.fft.fftshift(np.real(np.fft.ifft(H_full)))
    axis = (np.arange(N) - N // 2) * bin_size
    return axis, h


def fwhm_of_kernel(h, bin_size):
    """Estimate FWHM (in bin_size units) of a peaked kernel numerically."""
    peak_val = h.max()
    if peak_val <= 0:
        return float('nan')
    half = peak_val / 2.0
    center = int(np.argmax(h))
    left_cross  = np.where(h[:center] < half)[0]
    right_cross = np.where(h[center:] < half)[0]
    if left_cross.size == 0 or right_cross.size == 0:
        return float('nan')
    return (center + right_cross[0] - left_cross[-1]) * bin_size


def spatial_sigma_wires(sigma_code, power=2.0):
    """
    Approximate spatial σ (in wires) of the wire filter for power=2.
    Derivation: for H(f) = exp(-0.5*(f/σ_code)^2), the iFFT Gaussian has
    σ_spatial = 1 / (π * σ_code).  This is the standard DFT pair relation.
    NOTE: larger σ_code → smaller σ_spatial → LESS smearing.
    """
    if power != 2.0:
        return float('nan')
    return 1.0 / (np.pi * sigma_code)


# ── colour palette ─────────────────────────────────────────────────────────────

C_PDVD  = '#1f77b4'   # blue
C_PDHD  = '#d62728'   # red
C_PDHD1 = '#ff7f0e'   # orange  (PDHD APA1)
C_WIDE  = C_PDVD      # alias for wide curves
C_TIGHT = '#9467bd'   # purple  (PDVD tight)

PLANES = ['U', 'V', 'W']
PLANE_LABELS = ['U (induction)', 'V (induction)', 'W (collection)']


# ── plot 1: Wiener wide — frequency domain ────────────────────────────────────

def make_plot1():
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    fig.suptitle(
        'Wiener wide filters — frequency domain\n'
        'PDVD top/bottom (_b and _t) are byte-identical.\n'
        'PDHD wide has no APA1 override; it is also identical to PDVD wide.',
        fontsize=10,
    )

    for ax, plane, plabel in zip(axes, PLANES, PLANE_LABELS):
        fp = PDVD_WIENER_WIDE[plane]
        freq, H_pdvd = hf_pos_freq(N_TIME, MAX_FREQ_MHZ, fp['sigma'], fp['power'], True)
        fp2 = PDHD_WIENER_WIDE[plane]
        freq2, H_pdhd = hf_pos_freq(N_TIME, MAX_FREQ_MHZ, fp2['sigma'], fp2['power'], True)

        ax.plot(freq, H_pdvd, color=C_PDVD, lw=2.0,
                label=f'PDVD top/bottom   σ={fp["sigma"]:.4f} MHz   power={fp["power"]:.3f}')
        ax.plot(freq2, H_pdhd, color=C_PDHD, lw=1.5, ls='--',
                label=f'PDHD (all APAs)   σ={fp2["sigma"]:.4f} MHz   power={fp2["power"]:.3f}')

        ax.axhline(0.5, color='grey', lw=0.8, ls=':', label='H = 0.5')
        ax.set_xlim(0, 0.5)
        ax.set_ylim(-0.02, 1.05)
        ax.set_ylabel(f'|H(f)|  — plane {plabel}')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Frequency (MHz)')
    fig.tight_layout()
    _savefig(fig, 'compare_wiener_wide_freq.png')


# ── plot 2: Wiener wide — time domain ────────────────────────────────────────

def make_plot2():
    WINDOW_US = 5.0
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    fig.suptitle(
        'Wiener wide filters — time domain (500 ns/tick, normalised to unit peak)\n'
        'Grey dashed verticals mark 500 ns tick boundaries.',
        fontsize=10,
    )

    for ax, plane, plabel in zip(axes, PLANES, PLANE_LABELS):
        fp = PDVD_WIENER_WIDE[plane]
        H_full = _hf_array(N_TIME, MAX_FREQ_MHZ, fp['sigma'], fp['power'], True)
        t, h = iFFT_kernel(H_full, TICK_US)
        fw = fwhm_of_kernel(h, TICK_US)
        mask = np.abs(t) <= WINDOW_US
        peak = h[mask].max()

        fp2 = PDHD_WIENER_WIDE[plane]
        H_full2 = _hf_array(N_TIME, MAX_FREQ_MHZ, fp2['sigma'], fp2['power'], True)
        t2, h2 = iFFT_kernel(H_full2, TICK_US)
        fw2 = fwhm_of_kernel(h2, TICK_US)
        mask2 = np.abs(t2) <= WINDOW_US
        peak2 = h2[mask2].max()

        ax.plot(t[mask], h[mask] / peak, color=C_PDVD, lw=2.0,
                label=f'PDVD top/bottom   FWHM ≈ {fw:.2f} µs')
        # PDHD wide is byte-identical to PDVD wide → curves exactly coincide
        ax.plot(t2[mask2], h2[mask2] / peak2, color=C_PDHD, lw=1.5, ls='--',
                label=f'PDHD (all APAs)   FWHM ≈ {fw2:.2f} µs  ← coincides: same σ/power')

        for xtick in np.arange(-WINDOW_US, WINDOW_US + 0.01, 0.5):
            ax.axvline(xtick, color='lightgrey', lw=0.5, ls='--', zorder=0)
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xlim(-WINDOW_US, WINDOW_US)
        ax.set_ylim(-0.12, 1.1)
        ax.set_ylabel(f'h(t) / peak  — plane {plabel}')
        ax.legend(fontsize=8)

    axes[-1].set_xlabel('Time (µs)')
    fig.tight_layout()
    _savefig(fig, 'compare_wiener_wide_time.png')


# ── plot 3: Wide vs tight Wiener — freq + time ────────────────────────────────

def make_plot3():
    WINDOW_US = 5.0
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        'Wiener wide vs tight + Gaus_wide — frequency domain (top row) and time domain (bottom row)\n'
        'Bottom row normalised to unit peak; grey verticals = 500 ns ticks.',
        fontsize=11,
    )

    C_GAUS = '#2ca02c'   # green for Gaussian

    curves_defs = [
        ('PDVD wide (= PDHD wide)', PDVD_WIENER_WIDE,              C_PDVD,  '-',   2.0),
        ('PDVD tight',              PDVD_WIENER_TIGHT,              C_TIGHT, '--',  2.0),
        ('PDHD tight (APAs 2/3/4)', PDHD_WIENER_TIGHT_DEFAULT,     C_PDHD,  '-.',  1.8),
        ('PDHD tight (APA1)',       PDHD_WIENER_TIGHT_APA1,        C_PDHD1, ':',   1.8),
    ]

    # Gaussian is plane-independent — compute once outside the plane loop
    gp = PDVD_GAUS_WIDE  # same for PDHD and PDVD top/bottom
    freq_g, H_g = hf_pos_freq(N_TIME, MAX_FREQ_MHZ, gp['sigma'], gp['power'], True)
    H_full_g = _hf_array(N_TIME, MAX_FREQ_MHZ, gp['sigma'], gp['power'], True)
    t_g, h_g = iFFT_kernel(H_full_g, TICK_US)
    fw_g = fwhm_of_kernel(h_g, TICK_US)
    mask_g = np.abs(t_g) <= WINDOW_US
    peak_g = h_g[mask_g].max()

    for col, (plane, plabel) in enumerate(zip(PLANES, PLANE_LABELS)):
        ax_f = axes[0, col]
        ax_t = axes[1, col]

        for label, table, color, ls, lw in curves_defs:
            fp = table[plane]
            # frequency
            freq, H = hf_pos_freq(N_TIME, MAX_FREQ_MHZ, fp['sigma'], fp['power'], True)
            ax_f.plot(freq, H, color=color, ls=ls, lw=lw,
                      label=f'{label}\n  σ={fp["sigma"]:.4f}  p={fp["power"]:.3f}')
            # time
            H_full = _hf_array(N_TIME, MAX_FREQ_MHZ, fp['sigma'], fp['power'], True)
            t, h = iFFT_kernel(H_full, TICK_US)
            fw = fwhm_of_kernel(h, TICK_US)
            mask = np.abs(t) <= WINDOW_US
            peak = h[mask].max()
            ax_t.plot(t[mask], h[mask] / peak, color=color, ls=ls, lw=lw,
                      label=f'{label}  FWHM≈{fw:.2f} µs')

        # Gaussian — same curve on every plane panel
        gaus_label_f = f'Gaus_wide (all detectors)\n  σ={gp["sigma"]:.2f} MHz  p={gp["power"]:.0f}'
        gaus_label_t = f'Gaus_wide (all detectors)  FWHM≈{fw_g:.2f} µs'
        ax_f.plot(freq_g, H_g, color=C_GAUS, ls=(0, (3, 1, 1, 1)), lw=1.8,
                  label=gaus_label_f)
        ax_t.plot(t_g[mask_g], h_g[mask_g] / peak_g, color=C_GAUS,
                  ls=(0, (3, 1, 1, 1)), lw=1.8, label=gaus_label_t)

        # frequency panel
        ax_f.axhline(0.5, color='grey', lw=0.8, ls=':', label='H = 0.5')
        ax_f.set_xlim(0, 0.5)
        ax_f.set_ylim(-0.02, 1.05)
        ax_f.set_title(f'Plane {plabel}', fontsize=10)
        ax_f.set_xlabel('Frequency (MHz)')
        ax_f.set_ylabel('|H(f)|' if col == 0 else '')
        ax_f.legend(fontsize=6.5, loc='upper right')
        ax_f.grid(True, alpha=0.3)

        # time panel
        for xtick in np.arange(-WINDOW_US, WINDOW_US + 0.01, 0.5):
            ax_t.axvline(xtick, color='lightgrey', lw=0.5, ls='--', zorder=0)
        ax_t.axhline(0, color='k', lw=0.5)
        ax_t.set_xlim(-WINDOW_US, WINDOW_US)
        ax_t.set_ylim(-0.15, 1.1)
        ax_t.set_xlabel('Time (µs)')
        ax_t.set_ylabel('h(t) / peak' if col == 0 else '')
        ax_t.legend(fontsize=6.5, loc='upper right')

    fig.tight_layout()
    _savefig(fig, 'compare_wiener_wide_vs_tight.png')


# ── plot 4: Gaussian filter — freq + time ─────────────────────────────────────

def make_plot4():
    WINDOW_US = 5.0
    fig, (ax_f, ax_t) = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(
        'Gaus_wide filter (σ = 0.12 MHz, power = 2)\n'
        'Identical for PDVD top/bottom and PDHD.\n'
        'Gaus_tight has σ = 0 (evaluates to all-zero; not shown).',
        fontsize=10,
    )

    fp = PDVD_GAUS_WIDE  # same as PDHD_GAUS_WIDE
    freq, H = hf_pos_freq(N_TIME, MAX_FREQ_MHZ, fp['sigma'], fp['power'], True)
    H_full  = _hf_array(N_TIME, MAX_FREQ_MHZ, fp['sigma'], fp['power'], True)
    t, h    = iFFT_kernel(H_full, TICK_US)
    fw      = fwhm_of_kernel(h, TICK_US)
    mask    = np.abs(t) <= WINDOW_US
    peak    = h[mask].max()

    # frequency
    ax_f.plot(freq, H, color=C_PDVD, lw=2.0,
              label=f'PDVD top/bottom = PDHD   σ=0.12 MHz   power=2')
    ax_f.axhline(0.5, color='grey', lw=0.8, ls=':', label='H = 0.5')
    ax_f.set_xlim(0, 0.5)
    ax_f.set_ylim(-0.02, 1.05)
    ax_f.set_xlabel('Frequency (MHz)')
    ax_f.set_ylabel('|H(f)|')
    ax_f.set_title('Frequency domain')
    ax_f.legend(fontsize=9)
    ax_f.grid(True, alpha=0.3)

    # time
    ax_t.plot(t[mask], h[mask] / peak, color=C_PDVD, lw=2.0,
              label=f'FWHM ≈ {fw:.2f} µs')
    for xtick in np.arange(-WINDOW_US, WINDOW_US + 0.01, 0.5):
        ax_t.axvline(xtick, color='lightgrey', lw=0.5, ls='--', zorder=0)
    ax_t.axhline(0, color='k', lw=0.5)
    ax_t.set_xlim(-WINDOW_US, WINDOW_US)
    ax_t.set_ylim(-0.12, 1.1)
    ax_t.set_xlabel('Time (µs)')
    ax_t.set_ylabel('h(t) / peak')
    ax_t.set_title('Time domain (500 ns/tick)')
    ax_t.legend(fontsize=9)

    fig.tight_layout()
    _savefig(fig, 'compare_gauss.png')


# ── plot 5: Wire filters — induction and collection ────────────────────────────

def make_plot5():
    WINDOW_WIRES = 15
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        'Wire filters — induction (top row) and collection (bottom row)\n'
        'Left: wire-frequency domain (cycles/wire)    '
        'Right: spatial domain (wire index, 1 bin = 1 wire)',
        fontsize=11,
    )

    configs = [
        ('Induction (U/V)', 0, PDVD_WIRE_IND, PDHD_WIRE_IND),
        ('Collection (W)',  1, PDVD_WIRE_COL, PDHD_WIRE_COL),
    ]

    for plane_label, row, sig_pdvd, sig_pdhd in configs:
        ax_f  = axes[row, 0]
        ax_sp = axes[row, 1]

        sp_pdvd = spatial_sigma_wires(sig_pdvd)
        sp_pdhd = spatial_sigma_wires(sig_pdhd)

        # frequency
        freq_p, H_p = hf_pos_freq(N_WIRE, MAX_FREQ_WIRE, sig_pdvd, 2.0, False)
        freq_h, H_h = hf_pos_freq(N_WIRE, MAX_FREQ_WIRE, sig_pdhd, 2.0, False)
        ax_f.plot(freq_p, H_p, color=C_PDVD, lw=2.0,
                  label=f'PDVD top/bottom   σ_code={sig_pdvd:.3f}   → σ_spatial≈{sp_pdvd:.3f} wire')
        ax_f.plot(freq_h, H_h, color=C_PDHD, lw=2.0, ls='--',
                  label=f'PDHD              σ_code={sig_pdhd:.3f}   → σ_spatial≈{sp_pdhd:.3f} wire')
        ax_f.axhline(0.5, color='grey', lw=0.8, ls=':')
        ax_f.set_xlim(0, 0.5)
        ax_f.set_ylim(-0.02, 1.05)
        ax_f.set_title(f'{plane_label} — wire-frequency domain')
        ax_f.set_xlabel('Wire frequency (cycles / wire)')
        ax_f.set_ylabel('|H(f)|')
        ax_f.legend(fontsize=8)
        ax_f.grid(True, alpha=0.3)

        # spatial
        H_full_p = _hf_array(N_WIRE, MAX_FREQ_WIRE, sig_pdvd, 2.0, False)
        H_full_h = _hf_array(N_WIRE, MAX_FREQ_WIRE, sig_pdhd, 2.0, False)
        w_p, h_p = iFFT_kernel(H_full_p, 1)   # bin_size = 1 wire
        w_h, h_h = iFFT_kernel(H_full_h, 1)

        mask = np.abs(w_p) <= WINDOW_WIRES
        ax_sp.plot(w_p[mask], h_p[mask], color=C_PDVD, lw=2.0,
                   label=f'PDVD   σ_spatial≈{sp_pdvd:.3f} wire')
        ax_sp.plot(w_h[mask], h_h[mask], color=C_PDHD, lw=2.0, ls='--',
                   label=f'PDHD   σ_spatial≈{sp_pdhd:.3f} wire')
        for xi in range(-WINDOW_WIRES, WINDOW_WIRES + 1):
            ax_sp.axvline(xi, color='lightgrey', lw=0.4, ls='--', zorder=0)
        ax_sp.axhline(0, color='k', lw=0.5)
        ax_sp.set_xlim(-WINDOW_WIRES, WINDOW_WIRES)
        ax_sp.set_title(f'{plane_label} — spatial domain')
        ax_sp.set_xlabel('Wire index (relative to hit wire)')
        ax_sp.set_ylabel('h(wire)')
        ax_sp.legend(fontsize=8)

        # annotation on induction spatial panel only
        if row == 0:
            ratio = sp_pdhd / sp_pdvd
            ax_sp.annotate(
                f'σ_code in jsonnet is a FREQUENCY parameter.\n'
                f'Larger σ_code → near-flat in freq → delta in spatial → LESS smearing.\n'
                f'Smaller σ_code → fast roll-off → wide Gaussian → MORE smearing.\n\n'
                f'PDHD induction σ_spatial ≈ {ratio:.1f}× wider than PDVD\n'
                f'⇒ PDHD smears more across strips, not PDVD.',
                xy=(0.97, 0.97), xycoords='axes fraction',
                ha='right', va='top', fontsize=7.5,
                bbox=dict(boxstyle='round,pad=0.5', fc='lightyellow', ec='orange', lw=1.2),
            )

    fig.tight_layout()
    _savefig(fig, 'compare_wire_filter.png')


# ── helpers ───────────────────────────────────────────────────────────────────

def _savefig(fig, name):
    out = os.path.join(SCRIPT_DIR, name)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Wrote {out}')


# ── self-test (runs automatically before plots) ────────────────────────────────

def _self_test():
    # HF filter with flag=True must zero the DC bin.
    H = _hf_array(200, 1.0, 0.12, 2.0, True)
    assert H[0] == 0.0, f'flag=True but H[0]={H[0]}'

    # Wire filter (flag=False) must preserve DC (H[0] = 1).
    H2 = _hf_array(200, 1.0, 0.42, 2.0, False)
    assert abs(H2[0] - 1.0) < 1e-10, f'flag=False but H[0]={H2[0]}'

    # Positive-frequency slice: length = N//2 + 1, starts at 0, ends at max_freq.
    freq, H3 = hf_pos_freq(100, 2.0, 0.5, 2.0, True)
    assert len(freq) == 51 and abs(freq[-1] - 2.0) < 1e-10, 'pos-freq slice wrong'

    # FWHM of Wiener wide W should be a few ticks (< 10 µs, > 0.3 µs).
    fp = PDVD_WIENER_WIDE['W']
    fw = fwhm_of_kernel(
        iFFT_kernel(_hf_array(N_TIME, MAX_FREQ_MHZ, fp['sigma'], fp['power'], True), TICK_US)[1],
        TICK_US,
    )
    assert 0.3 < fw < 10.0, f'Wiener wide W FWHM={fw:.3f} µs out of range'

    # spatial_sigma_wires formula: σ_spatial = 1/(π σ_code)
    sig = spatial_sigma_wires(PDHD_WIRE_IND)
    expected = 1.0 / (np.pi * PDHD_WIRE_IND)
    assert abs(sig - expected) < 1e-12, 'spatial_sigma_wires formula wrong'

    print('_self_test: OK')


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--only', type=int, choices=[1, 2, 3, 4, 5],
                        metavar='N', help='produce only plot N (1–5)')
    args = parser.parse_args()

    _self_test()

    makers = [
        (1, make_plot1),
        (2, make_plot2),
        (3, make_plot3),
        (4, make_plot4),
        (5, make_plot5),
    ]
    for n, fn in makers:
        if args.only is None or args.only == n:
            fn()


if __name__ == '__main__':
    main()
