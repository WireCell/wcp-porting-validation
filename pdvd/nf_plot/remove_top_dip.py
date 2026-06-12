#!/usr/bin/env python3
"""
Remove the downward spectral dip at ~0.693-0.695 MHz (amps bins 354-355) from
every entry of pdvd-top-noise-spectra-v3.json and write a cleaned copy.

The dip is identical across all 15 entries (all planes, all wire lengths): two
amps bins drop to ~20% of their neighbors.  They are repaired by linear
interpolation between the nearest clean bins on each side, leaving freqs and
all other fields untouched.

Input : <wire-cell-data>/pdvd-top-noise-spectra-v3.json.bz2
Output: <wire-cell-data>/pdvd-top-noise-spectra-v3-nodip.json.bz2
"""
import os
import bz2
import json
import argparse

import numpy as np

WCD = '/nfs/data/1/xning/wirecell-working/wire-cell-data/'
SRC = WCD + 'pdvd-top-noise-spectra-v3.json.bz2'
DST = WCD + 'pdvd-top-noise-spectra-v3-nodip.json.bz2'

DIP_BINS = [354, 355]      # the two dipped amps indices (~0.693, 0.695 MHz)


def repair(amps, bins):
    """Linear-interpolate the given bins from the nearest clean bins outside."""
    a = np.asarray(amps, float).copy()
    lo = min(bins) - 1            # last clean bin before the dip
    hi = max(bins) + 1            # first clean bin after the dip
    for i in bins:
        mu = (i - lo) / (hi - lo)
        a[i] = (1 - mu) * a[lo] + mu * a[hi]
    return a


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--src', default=SRC)
    ap.add_argument('--dst', default=DST)
    args = ap.parse_args()

    spectra = json.load(bz2.open(args.src))
    nfix = 0
    for e in spectra:
        before = [e['amps'][i] for i in DIP_BINS]
        fixed = repair(e['amps'], DIP_BINS)
        e['amps'] = [float(x) for x in fixed]
        after = [e['amps'][i] for i in DIP_BINS]
        nfix += 1
        if nfix <= 1:
            print('  bins %s: before=%s -> after=%s'
                  % (DIP_BINS, ['%.4g' % b for b in before],
                     ['%.4g' % a for a in after]))
    print('  repaired %d entries' % nfix)

    with bz2.open(args.dst, 'wt') as f:
        json.dump(spectra, f)
    print('  wrote %s' % args.dst)


if __name__ == '__main__':
    main()
