#!/usr/bin/env python3
"""
Per-channel electronics-noise frequency spectrum for ProtoDUNE Vertical Drift.

For every channel the real (track) signal is identified; signal-free
channel-events are Fourier transformed and |FFT| is averaged over channels and
events, grouped by drift region (bottom anodes 0-3 / top anodes 4-7), wire
plane (U/V/W) and strip-length bin.

The averaged |FFT| is the *mean amplitude spectrum* in WireCell internal
voltage units -- exactly the `amps` field consumed by EmpiricalNoiseModel.
The ADC->internal-voltage conversion (and the units::volt = 1e-6 factor) is
verified by the Phase-1 closure test; see noise_spectrum_comparison.md.

Usage:
  ./noise_spectrum.py --source data   # run039324, 11 events, post-NF frames
  ./noise_spectrum.py --source sim    # noise-only simulation (existing spectra)

Outputs (noise_spectrum/ next to this script):
  noise_spectrum_<src>.npz            per region/plane/bin: freqs, amp, counts
  noise_spectrum_<src>_<region>.png   per-region U/V/W spectra, bins overlaid
"""
import os
import io
import json
import bz2
import math
import argparse
import tarfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPTDIR = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(SCRIPTDIR, 'noise_spectrum')
WCD = '/nfs/data/1/xqian/toolkit-dev/wire-cell-data/'
WIRES_FILE = WCD + 'protodunevd-wires-larsoft-v3.json.bz2'

PLANES = ['U', 'V', 'W']
REGIONS = {'bottom': [0, 1, 2, 3], 'top': [4, 5, 6, 7]}

# --- ADC count -> WireCell internal voltage units -------------------------
# Digitizer fullscale span in *real* volts; units::volt = 1e-6 internal.  The
# EmpiricalNoiseModel `amps` are in internal units, so
#   amp_internal = adc * span_realvolt * 1e-6 / 16384.
# The 1e-6 is confirmed by the closure test (extracted sim spectrum reproduces
# the spectra-JSON that fed the sim to 0.1%).
UNIT_VOLT = 1.0e-6
NADC = 16384
SPAN = {'bottom': 1.4, 'top': 2.0}      # ADC fullscale [0.2,1.6]V / [0,2]V


def lsb_internal(region):
    return SPAN[region] * UNIT_VOLT / NADC


# strip-length bins (mm, geometry summed-segment length).  Induction (U,V)
# strips span ~10-1720 mm with a large pile-up at the full length 1720.5 mm;
# the collection plane (W) is single-length (~1679 mm) -> one bin.
IND_EDGES = np.array([0., 300., 600., 900., 1200., 1500., 1700., 1741.])

# signal rejection: a channel-event is dropped if any sample exceeds
# SIG_SIGMA times its robust noise rms (6 sigma is never reached by ~6400
# pure-Gaussian samples, so only real signal / hot codes flag).
SIG_SIGMA = 6.0

SOURCES = {
    'data': {
        'label': 'data run039324 (post-NF, 11 evt)',
        'events': list(range(11)),
        'path': os.path.join(SCRIPTDIR, '..', 'input_data', 'run039324',
                             'evt_%d', 'protodune-sp-frames-raw-anode%d.tar.bz2'),
    },
    'sim': {
        'label': 'noise-only simulation (original v1/v2 spectra)',
        'events': [None],
        'path': os.path.join(SCRIPTDIR, '..', '..', 'pdvd_sim', 'work',
                             'noise', 'all', 'pdvd-noise-sim-anode%d.tar.bz2'),
    },
    'simretuned': {
        'label': 'noise-only simulation (data-retuned v2/v3 spectra)',
        'events': [None],
        'path': os.path.join(SCRIPTDIR, '..', '..', 'pdvd_sim', 'work',
                             'noise', 'retuned', 'pdvd-noise-sim-anode%d.tar.bz2'),
    },
}


def geom_lengths():
    """Per-channel summed wire-segment length, WireCell internal units (mm).
    This is the length EmpiricalNoiseModel uses to pick a spectrum
    (m_anode->wires(chid) -> sum of ray_length), so binning data channels by
    it makes the data/sim comparison like-for-like."""
    with bz2.open(WIRES_FILE) as f:
        store = json.load(f)['Store']
    P = np.array([[p['Point']['x'], p['Point']['y'], p['Point']['z']]
                  for p in store['points']])
    clen = {}
    for w in store['wires']:
        ww = w['Wire']
        d = P[ww['tail']] - P[ww['head']]
        clen[ww['channel']] = clen.get(ww['channel'], 0.0) + float(np.sqrt(d @ d))
    return clen


def load_frame(archive):
    """Return (frame [nch,ntick] float64, channels [nch], tick period ns)."""
    with tarfile.open(archive, 'r:bz2') as tf:
        arrs = {m.name: np.load(io.BytesIO(tf.extractfile(m).read()))
                for m in tf.getmembers() if m.name.endswith('.npy')}
    frame = next(v for k, v in arrs.items()
                 if os.path.basename(k).startswith('frame'))
    chans = next(v for k, v in arrs.items()
                 if os.path.basename(k).startswith('channels'))
    tick = next((v for k, v in arrs.items()
                 if os.path.basename(k).startswith('tickinfo')), None)
    period = float(tick[1]) if tick is not None else 500.0
    return np.asarray(frame, dtype=np.float64), np.asarray(chans), period


def split_planes(channels):
    """Index ranges of the U/V/W blocks, split at gaps in channel numbering."""
    gap = np.where(np.diff(channels) > 1)[0]
    starts = [0] + list(gap + 1)
    ends = list(gap + 1) + [len(channels)]
    return list(zip(starts, ends))


def usable(wave):
    """True if a channel-event is non-dead and signal-free.  Robust noise rms
    from the 16-84 percentile spread; reject if any sample is beyond
    SIG_SIGMA of it (real tracks; NF-zeroed channels give rms 0)."""
    p16, p50, p84 = np.percentile(wave, [16, 50, 84])
    rms = 0.5 * (p84 - p16)
    if rms <= 1.0e-6:
        return False
    return not np.any(np.abs(wave - p50) > SIG_SIGMA * rms)


def bin_index(iplane, length):
    """Strip-length bin index: induction by IND_EDGES, collection single bin."""
    if iplane == 2:
        return 0
    return int(min(np.searchsorted(IND_EDGES, length, side='right') - 1,
                   len(IND_EDGES) - 2))


def nbins(iplane):
    return 1 if iplane == 2 else len(IND_EDGES) - 1


def extract(src):
    """Mean |FFT| amplitude (internal V) per region/plane/strip-length bin."""
    spec = SOURCES[src]
    glen = geom_lengths()
    print('=== noise spectrum extraction: %s ===' % spec['label'])
    result = {}
    for region, anodes in REGIONS.items():
        lsb = lsb_internal(region)
        acc, cnt, lensum = {}, {}, {}
        nseen = nused = 0
        ntick = period = freqs = None
        for ev in spec['events']:
            for an in anodes:
                arch = (spec['path'] % an if ev is None
                        else spec['path'] % (ev, an))
                frame, chans, per = load_frame(arch)
                if ntick is None:
                    ntick, period = frame.shape[1], per
                    freqs = np.fft.rfftfreq(ntick, d=period)   # 1/ns
                elif frame.shape[1] != ntick or per != period:
                    raise RuntimeError('inconsistent ntick/period in %s' % arch)
                for ip, (s, e) in enumerate(split_planes(chans)):
                    for i in range(s, e):
                        L = glen.get(int(chans[i]))
                        if L is None:
                            continue
                        nseen += 1
                        w = frame[i]
                        if not usable(w):
                            continue
                        nused += 1
                        key = (ip, bin_index(ip, L))
                        amp = np.abs(np.fft.rfft(w - np.median(w))) * lsb
                        if key in acc:
                            acc[key] += amp
                            cnt[key] += 1
                            lensum[key] += L
                        else:
                            acc[key] = amp.copy()
                            cnt[key] = 1
                            lensum[key] = L
        print('  %-6s  ntick=%d period=%.0fns  kept %d / %d channel-events (%.0f%%)'
              % (region, ntick, period, nused, nseen, 100.0 * nused / max(nseen, 1)))
        for ip in range(3):
            for ib in range(nbins(ip)):
                key = (ip, ib)
                if key not in acc:
                    continue
                result[(region, ip, ib)] = dict(
                    freqs=freqs, amp=acc[key] / cnt[key], n=cnt[key],
                    meanlen=lensum[key] / cnt[key], ntick=ntick, period=period)
                print('    %s %s  bin%d  len~%6.0fmm  nch-evt=%5d'
                      % (region, PLANES[ip], ib,
                         result[(region, ip, ib)]['meanlen'], cnt[key]))
    return result


def save_npz(src, result):
    arrays = {}
    for (region, ip, ib), d in result.items():
        pre = '%s_%s_%d' % (region, PLANES[ip], ib)
        arrays[pre + '_freq'] = d['freqs']
        arrays[pre + '_amp'] = d['amp']
        arrays[pre + '_meta'] = np.array([d['n'], d['meanlen'],
                                          d['ntick'], d['period']])
    out = os.path.join(OUTDIR, 'noise_spectrum_%s.npz' % src)
    np.savez(out, **arrays)
    print('  wrote %s' % out)


def plot_region(src, region, result):
    """One figure per region: U/V/W panels, strip-length bins overlaid."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle('PDVD noise amplitude spectrum  --  %s drift  --  %s'
                 % (region, SOURCES[src]['label']), fontsize=12)
    for ip, ax in enumerate(axes):
        keys = sorted(ib for (r, p, ib) in result if r == region and p == ip)
        cmap = plt.cm.viridis(np.linspace(0, 0.9, max(len(keys), 1)))
        for j, ib in enumerate(keys):
            d = result[(region, ip, ib)]
            ax.plot(d['freqs'] * 1e3, d['amp'], lw=0.9, color=cmap[j],
                    label='%.0f mm (n=%d)' % (d['meanlen'], d['n']))
        ax.set_title('%s plane' % PLANES[ip], fontsize=10)
        ax.set_xlabel('frequency [MHz]')
        ax.set_ylabel('mean |FFT| amplitude [internal V]')
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, title='strip length')
    plt.tight_layout()
    out = os.path.join(OUTDIR, 'noise_spectrum_%s_%s.png' % (src, region))
    plt.savefig(out, dpi=130)
    plt.close()
    print('  wrote %s' % out)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', required=True, choices=sorted(SOURCES),
                    help='which input to analyze: data or sim')
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    result = extract(args.source)
    save_npz(args.source, result)
    for region in REGIONS:
        plot_region(args.source, region, result)
    print('done.')


if __name__ == '__main__':
    main()
