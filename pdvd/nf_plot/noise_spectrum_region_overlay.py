#!/usr/bin/env python3
"""
Overlay the anode0 region-derived noise spectrum (noise_spectrum_region.py)
on the previous reference spectra from wire-cell-data, using only the
matching wire length per plane.

Reference: pdvd-bottom-noise-spectra-7d8mVfC-v1.json
  U: 1720.4 mm   V: 1720.4 mm   W: 1679.0 mm   (match my region's lengths)

The reference JSON spectra are prepared at nsamples=6400; my region spectrum
uses a 2500-tick window.  To compare on equal footing the reference amps are
rescaled by sqrt(my_nwin / json_nsamples) -- exactly EmpiricalNoiseModel's
resample() amplitude scaling -- and interpolated onto my frequency grid.

Usage:  ./noise_spectrum_region_overlay.py
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
REGION_NPZ = os.path.join(OUTDIR, 'noise_spectrum_region_anode0.npz')
WCD = '/nfs/data/1/xning/wirecell-working/wire-cell-data/'
REF_JSON = WCD + 'pdvd-bottom-noise-spectra-7d8mVfC-v1.json.bz2'

PLANES = ['U', 'V', 'W']
PLANE_IDX = {'U': 0, 'V': 1, 'W': 2}

# freqs in the JSON are in WCT internal units (1/ns); my region freqs are too.
# plot axis is MHz: f[1/ns] * 1e3 = MHz.


def load_region():
    d = np.load(REGION_NPZ)
    out = {}
    for p in PLANES:
        if (p + '_amp') not in d:
            continue
        meta = d[p + '_meta']    # [n, wirelen, chid_lo, chid_hi, ntick, period]
        out[p] = dict(freq=d[p + '_freq'], amp=d[p + '_amp'],
                      n=int(meta[0]), wirelen=float(meta[1]),
                      nwin=int(meta[4]), period=float(meta[5]))
    return out


def load_ref():
    """plane -> list of dicts {wirelen, nsamples, period, freqs, amps}."""
    spectra = json.load(bz2.open(REF_JSON))
    byp = {}
    for e in spectra:
        byp.setdefault(e['plane'], []).append(
            dict(wirelen=float(e['wirelen']), nsamples=int(e['nsamples']),
                 period=float(e['period']),
                 freqs=np.asarray(e['freqs'], float),
                 amps=np.asarray(e['amps'], float)))
    return byp


def pick_matching(ref_list, target_len):
    """Pick the reference spectrum whose wirelen is closest to target_len."""
    return min(ref_list, key=lambda e: abs(e['wirelen'] - target_len))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--xmax', type=float, default=None,
                    help='upper frequency limit in MHz (e.g. 0.2 to zoom in); '
                         'default shows the full band')
    args = ap.parse_args()
    reg = load_region()
    ref = load_ref()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    zoom = '  (zoom 0-%.2g MHz)' % args.xmax if args.xmax else ''
    fig.suptitle('PDVD anode0 region noise spectrum vs reference '
                 '(pdvd-bottom 7d8mVfC v1, matched wire length)%s' % zoom,
                 fontsize=12)

    for p, ax in zip(PLANES, axes):
        if p not in reg:
            ax.set_title('%s plane (no region data)' % p)
            continue
        r = reg[p]
        # my region curve
        ax.plot(r['freq'] * 1e3, r['amp'], lw=1.0, color='C0',
                label='region data\n%.0f mm (n=%d, %d-tick)'
                % (r['wirelen'], r['n'], r['nwin']))

        # matched reference, rescaled + interpolated
        rl = pick_matching(ref[PLANE_IDX[p]], r['wirelen'])
        scale = np.sqrt(r['nwin'] / rl['nsamples'])           # EmpiricalNoiseModel.resample
        ref_amp = rl['amps'] * scale
        ref_on_grid = np.interp(r['freq'], rl['freqs'], ref_amp)
        ax.plot(r['freq'] * 1e3, ref_on_grid, lw=1.2, color='C3',
                label='reference\n%.0f mm (x sqrt(%d/%d))'
                % (rl['wirelen'], r['nwin'], rl['nsamples']))

        ax.set_title('%s plane' % p, fontsize=10)
        ax.set_xlabel('frequency [MHz]')
        ax.set_ylabel('mean |FFT| amplitude [internal V]')
        ax.set_xlim(0, args.xmax)
        if args.xmax:                       # rescale y to the visible window
            mhz = r['freq'] * 1e3
            sel = mhz <= args.xmax
            ymax = max(r['amp'][sel].max(), ref_on_grid[sel].max())
            ax.set_ylim(0, ymax * 1.05)
        else:
            ax.set_ylim(0, None)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    tag = '_zoom%g' % args.xmax if args.xmax else ''
    out = os.path.join(OUTDIR, 'noise_spectrum_region_overlay_anode0%s.png' % tag)
    plt.savefig(out, dpi=130)
    plt.close()
    print('  wrote %s' % out)


if __name__ == '__main__':
    main()
