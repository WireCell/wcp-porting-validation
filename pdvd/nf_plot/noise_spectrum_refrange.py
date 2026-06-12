#!/usr/bin/env python3
"""
Plot the reference PDVD noise spectra at the shortest and longest wire length,
U/V/W overlaid, one figure for the bottom drift and one for the top.

  bottom : pdvd-bottom-noise-spectra-7d8mVfC-v1.json
  top    : pdvd-top-noise-spectra-v3.json

For each plane the spectrum with the smallest and largest 'wirelen' in the file
is drawn (W has a single length, so only one curve).  Amplitudes are plotted
as stored (internal voltage units); frequency converted to MHz.

Usage:  ./noise_spectrum_refrange.py
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
    'bottom': ('pdvd-bottom-noise-spectra-7d8mVfC-v1.json.bz2',
               'pdvd-bottom 7d8mVfC v1'),
    'top':    ('pdvd-top-noise-spectra-v3.json.bz2',
               'pdvd-top v3'),
}
PLANES = ['U', 'V', 'W']


def load_by_plane(path):
    spectra = json.load(bz2.open(path))
    byp = {}
    for e in spectra:
        byp.setdefault(e['plane'], []).append(
            dict(wirelen=float(e['wirelen']),
                 freqs=np.asarray(e['freqs'], float),
                 amps=np.asarray(e['amps'], float)))
    return byp


def plot_region(region):
    fname, label = REFS[region]
    byp = load_by_plane(WCD + fname)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    fig.suptitle('PDVD %s reference noise spectrum -- shortest & longest '
                 'wire length  (%s)' % (region, label), fontsize=12)

    for ip, (plane, ax) in enumerate(zip(PLANES, axes)):
        spectra = sorted(byp[ip], key=lambda e: e['wirelen'])
        short = spectra[0]
        long = spectra[-1]
        ax.plot(short['freqs'] * 1e3, short['amps'], lw=1.1, color='C0',
                label='shortest  %.0f mm' % short['wirelen'])
        if long['wirelen'] != short['wirelen']:
            ax.plot(long['freqs'] * 1e3, long['amps'], lw=1.1, color='C3',
                    label='longest  %.0f mm' % long['wirelen'])
        ax.set_title('%s plane' % plane, fontsize=10)
        ax.set_xlabel('frequency [MHz]')
        ax.set_ylabel('mean |FFT| amplitude [internal V]')
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, title='wire length')

    plt.tight_layout()
    out = os.path.join(OUTDIR, 'noise_spectrum_refrange_%s.png' % region)
    plt.savefig(out, dpi=130)
    plt.close()
    print('  wrote %s' % out)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    for region in ('bottom', 'top'):
        plot_region(region)
    print('done.')


if __name__ == '__main__':
    main()
