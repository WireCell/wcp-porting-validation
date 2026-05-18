#!/usr/bin/env python3
"""
Build replacement PDVD EmpiricalNoiseModel spectra files from the data
extraction (noise_spectrum/noise_spectrum_data.npz, written by noise_spectrum.py).

Writes drop-in spectra files:
  wire-cell-data/pdvd-bottom-noise-spectra-v2.json.bz2   (anodes 0-3)
  wire-cell-data/pdvd-top-noise-spectra-v3.json.bz2      (anodes 4-7)

Normalization -- see noise_spectrum_comparison.md:
  * amps    = measured mean |FFT| in WireCell internal voltage units, as-is
              (noise_spectrum.py already applied span*1e-6/16384 per ADC count).
  * const   = 0.  The measured spectrum already contains the white-noise floor;
              the model forms mu = sqrt(amps^2 + const^2), so const=0 puts the
              whole spectrum in `amps`.
  * period  = 500 ns (the simulation tick; the model runs at 500 ns).
  * nsamples = N chosen so the model resample scale sqrt(10000/N) equals the
              renormalization from the data grid (N_data,T_data) to the sim
              grid (10000,500 ns):  N = round(N_data * 500 / T_data).
              bottom data 6400 @ 500 ns -> 6400 ;  top data 6400 @ 512 ns -> 6250.
  * gain / shaping are copied verbatim from the previous files -- inert
    metadata (EmpiricalNoiseModel applies no gain/shaping correction when
    `chanstat` is empty, which is the PDVD configuration).

Each induction (U,V) plane gets one entry per strip-length bin; the collection
plane (W) is single-length so it gets one entry.
"""
import os
import bz2
import json

import numpy as np

SCRIPTDIR = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(SCRIPTDIR, 'noise_spectrum', 'noise_spectrum_data.npz')
WCD = '/nfs/data/1/xqian/toolkit-dev/wire-cell-data/'

PLANES = ['U', 'V', 'W']
NGRID = 512                              # uniform freq points per entry
FNYQ = 1.0e-3                            # 1/ns; model Nyquist at 500 ns tick

REGION_OUT = {
    'bottom': dict(infile='pdvd-bottom-noise-spectra-v1.json.bz2',
                   outfile='pdvd-bottom-noise-spectra-v2.json.bz2',
                   nsamples=6400),
    'top': dict(infile='pdvd-top-noise-spectra-v2.json.bz2',
                outfile='pdvd-top-noise-spectra-v3.json.bz2',
                nsamples=6250),
}


def load_data():
    """Return {(region,iplane,ibin): dict(freq, amp, meanlen, n)}."""
    npz = np.load(NPZ)
    out = {}
    for key in npz.files:
        if not key.endswith('_amp'):
            continue
        pre = key[:-4]
        region, pl, ib = pre.split('_')
        n, meanlen, ntick, period = npz[pre + '_meta']
        out[(region, PLANES.index(pl), int(ib))] = dict(
            freq=npz[pre + '_freq'], amp=npz[pre + '_amp'],
            meanlen=float(meanlen), n=int(n))
    return out


def metadata(infile):
    """gain/shaping per plane from the previous spectra file (inert metadata)."""
    with bz2.open(WCD + infile) as f:
        entries = json.load(f)
    meta = {}
    for e in entries:
        meta.setdefault(e['plane'], (e['gain'], e['shaping']))
    return meta


def leave_one_out(data, region):
    """Sanity check: drop a middle induction bin, linearly interpolate it from
    its neighbours (as the model would), compare to the measurement."""
    print('  leave-one-out cross-check (%s):' % region)
    for ip in (0, 1):
        ibs = sorted(b for (r, p, b) in data if r == region and p == ip)
        for ib in ibs[1:-1]:
            lo, hi = data[(region, ip, ib - 1)], data[(region, ip, ib + 1)]
            mid = data[(region, ip, ib)]
            mu = ((mid['meanlen'] - lo['meanlen']) /
                  (hi['meanlen'] - lo['meanlen']))
            pred = (1 - mu) * lo['amp'] + mu * hi['amp']
            m = (mid['freq'] > 5e-5) & (mid['freq'] < 9.5e-4)
            rel = np.abs(pred[m] - mid['amp'][m]) / mid['amp'][m]
            print('    %s bin%d  len=%4.0fmm  interp-vs-measured: '
                  'mean %.1f%%  max %.1f%%'
                  % (PLANES[ip], ib, mid['meanlen'],
                     100 * rel.mean(), 100 * rel.max()))


def build_region(region, data):
    cfg = REGION_OUT[region]
    meta = metadata(cfg['infile'])
    grid = np.linspace(0.0, FNYQ, NGRID)            # 1/ns
    entries = []
    for ip in range(3):
        ibs = sorted(b for (r, p, b) in data if r == region and p == ip)
        gain, shaping = meta[ip]
        for ib in ibs:
            d = data[(region, ip, ib)]
            amp = np.interp(grid, d['freq'], d['amp'])  # flat-extrapolated
            entries.append(dict(
                plane=ip,
                wirelen=round(d['meanlen'], 3),         # internal units (mm)
                gain=gain, shaping=shaping,
                const=0.0,
                nsamples=cfg['nsamples'], period=500.0,
                freqs=[float(x) for x in grid],
                amps=[float(x) for x in amp]))
    out = WCD + cfg['outfile']
    with bz2.open(out, 'wt') as f:
        json.dump(entries, f)
    print('  wrote %s  (%d entries: %s)'
          % (out, len(entries),
             ' '.join('%s=%d' % (PLANES[p], sum(1 for e in entries
                                                 if e['plane'] == p))
                      for p in range(3))))


def main():
    data = load_data()
    for region in ('bottom', 'top'):
        print('=== %s ===' % region)
        leave_one_out(data, region)
        build_region(region, data)
    print('done.')


if __name__ == '__main__':
    main()
