#!/usr/bin/env python3
"""
Plot strip (wire) length vs the peak of the noise amplitude spectrum, for the
induction planes U and V, both bottom and top drift regions.

  bottom : pdvd-bottom-noise-spectra-7d8mVfC-v1.json
  top    : pdvd-top-noise-spectra-v3-nodip.json   (dip-removed top)

For each spectrum entry the peak = max(amps); plotted vs wirelen.  One panel
per plane (U, V), with bottom and top overlaid.

Usage:  ./noise_spectrum_peak_vs_len.py
"""
import os
import bz2
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPTDIR = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(SCRIPTDIR, 'noise_spectrum_region')
WCD = '/nfs/data/1/xning/wirecell-working/wire-cell-data/'

REFS = {
    'bottom': (WCD + 'pdvd-bottom-noise-spectra-7d8mVfC-v1.json.bz2', 'C0'),
    'top':    (WCD + 'pdvd-top-noise-spectra-v3-nodip.json.bz2',      'C3'),
}
PLANES = {'U': 0, 'V': 1}     # induction planes only


# The lowest few frequency bins carry a DC / coherent-low-f spike that varies
# erratically between entries and is NOT the shaper peak.  Taking the raw max
# therefore sometimes latches onto that spike instead of the true ~0.12 MHz
# peak, producing a spurious non-monotonic point at the longest strip.  We
# smooth the spectrum and search for the peak only above a low-frequency
# cutoff so the DC spike cannot win.
# The top spectra have a broad, gentle bump near ~0.05 MHz (bottom peaks
# near ~0.12 MHz).  The very lowest bins (bin 0-1, ~DC) are jagged and can
# spike above that bump, so without enough smoothing the argmax wrongly lands
# at ~0 MHz.  A wider moving average lets the real bump dominate; the small
# low-frequency cutoff drops the residual DC bins.
FLO_MHZ = 0.01          # ignore bins below this when locating the peak (drops DC bin0-1)
SMOOTH = 15             # moving-average window (bins); wide enough for the ~0.05 MHz bump


def smooth(a, k):
    if k <= 1:
        return a
    kern = np.ones(k) / k
    return np.convolve(a, kern, mode='same')


def peaks_by_plane(path):
    """plane -> sorted list of (wirelen, peak_amp).

    Peak = max of the smoothed spectrum above FLO_MHZ."""
    spectra = json.load(bz2.open(path))
    out = {}
    for e in spectra:
        amps = np.asarray(e['amps'], float)
        freqs = np.asarray(e['freqs'], float) * 1e3      # -> MHz
        sm = smooth(amps, SMOOTH)
        mask = freqs >= FLO_MHZ
        peak = float(sm[mask].max())
        out.setdefault(e['plane'], []).append((float(e['wirelen']), peak))
    for p in out:
        out[p].sort()
    return out


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    data = {reg: peaks_by_plane(path) for reg, (path, _) in REFS.items()}

    # one panel per drift region; U and V overlaid within each panel
    regions = list(REFS)                       # ['bottom', 'top']
    fig, axes = plt.subplots(1, len(regions), figsize=(12, 5), sharey=False)
    fig.suptitle('PDVD induction-plane noise spectrum peak vs strip length '
                 '(smoothed %d-bin, f>%.0f kHz)' % (SMOOTH, FLO_MHZ * 1e3),
                 fontsize=12)

    pcolor = {'U': 'C0', 'V': 'C3'}
    for reg, ax in zip(regions, axes):
        for plane, ip in PLANES.items():
            pts = data[reg].get(ip, [])
            if not pts:
                continue
            lens = [w for w, _ in pts]
            pk = [p for _, p in pts]
            ax.plot(lens, pk, 'o-', color=pcolor[plane], lw=1.3, ms=5,
                    label='%s plane' % plane)
        ax.set_title('%s drift' % reg, fontsize=11)
        ax.set_xlabel('strip length [mm]')
        ax.set_ylabel('peak |FFT| amplitude [internal V]')
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, title='plane')

    plt.tight_layout()
    out = os.path.join(OUTDIR, 'noise_spectrum_peak_vs_len.png')
    plt.savefig(out, dpi=130)
    plt.close()
    print('  wrote %s' % out)


if __name__ == '__main__':
    main()
