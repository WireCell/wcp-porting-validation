#!/usr/bin/env python3
"""
Region-restricted noise amplitude spectrum for ProtoDUNE-VD, anode 0 only.

Instead of scanning all channels/events and rejecting signal with a 6-sigma
cut, this picks a hand-chosen, signal-free rectangle in (channel, tick) space
and Fourier-transforms only that.  Per plane:

    U : channel-IDs 200-300    (wirelen ~1720.5 mm)
    V : channel-IDs 1150-1300  (wirelen ~1720.5 mm)
    W : channel-IDs 2100-2350  (wirelen ~1679.0 mm)
    tick window: 3500-6000  (2500 ticks)

Because the window is chosen to be signal-free, the 6-sigma signal cut is
dropped.  Only the dead-channel check (robust RMS ~ 0) is kept.

All channels in each selected range are required to share one wire length;
if a range mixes lengths, only the dominant-length channels are kept (the
others are reported and dropped) so the spectrum stays length-consistent --
matching how EmpiricalNoiseModel picks a spectrum by wire length.

The amplitude formula is identical to noise_spectrum.py:
    amp = |rfft(w - median(w))| * lsb,   lsb = SPAN * units::volt / NADC
giving the mean |FFT| amplitude in WireCell internal voltage units (the
EmpiricalNoiseModel `amps`).

Usage:
  ./noise_spectrum_region.py            # anode0, evt_0
"""
import os
import io
import json
import bz2
import argparse
import tarfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPTDIR = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(SCRIPTDIR, 'noise_spectrum_region')
WCD = '/nfs/data/1/xqian/toolkit-dev/wire-cell-data/'
WIRES_FILE = WCD + 'protodunevd-wires-larsoft-v3.json.bz2'

# anode 0 is a "bottom" drift region -> ADC fullscale span 1.4 V
DATAFILE = os.path.join(SCRIPTDIR, '..', '..', '..', 'data', 'vd', 'run039324',
                        'evt_0', 'protodune-sp-frames-raw-anode0.tar.bz2')
REGION = 'bottom'

PLANES = ['U', 'V', 'W']

# selected region: per-plane channel-ID range (inclusive) + tick window.
# channel-IDs are the PNG y-axis values (NOT array row indices).
CHAN_RANGES = {           # plane -> (chid_lo, chid_hi) inclusive
    'U': (200, 300),
    'V': (1150, 1300),
    'W': (2100, 2350),
}
TICK_LO, TICK_HI = 3500, 6000     # tick window [lo, hi)

# --- ADC count -> WireCell internal voltage units (see noise_spectrum.py) ---
UNIT_VOLT = 1.0e-6
NADC = 16384
SPAN = {'bottom': 1.4, 'top': 2.0}

# dead-channel threshold on robust rms (internal-V baseline units, i.e. ADC)
DEAD_RMS = 1.0e-6


def lsb_internal(region):
    return SPAN[region] * UNIT_VOLT / NADC


def geom_lengths():
    """Per-channel summed wire-segment length (mm), keyed by channel-ID."""
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


def robust_rms(wave):
    p16, p84 = np.percentile(wave, [16, 84])
    return 0.5 * (p84 - p16)


def select_rows(chans, glen, chid_lo, chid_hi):
    """Row indices whose channel-ID is in [lo,hi] AND that share the dominant
    wire length in that selection.  Returns (rows, kept_len, dropped)."""
    rows = np.where((chans >= chid_lo) & (chans <= chid_hi))[0]
    rows = np.array([r for r in rows if int(chans[r]) in glen])
    if rows.size == 0:
        return rows, None, 0
    lens = np.array([round(glen[int(chans[r])], 1) for r in rows])
    vals, counts = np.unique(lens, return_counts=True)
    dom = vals[np.argmax(counts)]                 # dominant length
    keep = rows[lens == dom]
    dropped = rows.size - keep.size
    return keep, float(dom), dropped


def extract():
    glen = geom_lengths()
    lsb = lsb_internal(REGION)
    frame, chans, period = load_frame(DATAFILE)
    ntick = frame.shape[1]
    if not (0 <= TICK_LO < TICK_HI <= ntick):
        raise RuntimeError('tick window %d-%d outside frame ntick=%d'
                           % (TICK_LO, TICK_HI, ntick))
    nwin = TICK_HI - TICK_LO
    freqs = np.fft.rfftfreq(nwin, d=period)        # 1/ns
    print('=== region noise spectrum: anode0, evt_0 (%s) ===' % REGION)
    print('  ntick=%d period=%.0fns  tick window %d-%d (%d ticks)'
          % (ntick, period, TICK_LO, TICK_HI, nwin))

    result = {}
    for ip, plane in enumerate(PLANES):
        lo, hi = CHAN_RANGES[plane]
        rows, klen, dropped = select_rows(chans, glen, lo, hi)
        if rows.size == 0:
            print('  %s chID %d-%d: no channels found, skipped' % (plane, lo, hi))
            continue
        if dropped:
            print('  %s: dropped %d channel(s) with non-dominant wire length'
                  % (plane, dropped))

        acc = np.zeros(freqs.size)
        nused = ndead = 0
        for r in rows:
            w = frame[r, TICK_LO:TICK_HI]
            if robust_rms(w) <= DEAD_RMS:          # dead / NF-zeroed
                ndead += 1
                continue
            acc += np.abs(np.fft.rfft(w - np.median(w))) * lsb
            nused += 1
        if nused == 0:
            print('  %s chID %d-%d: all %d channels dead, skipped'
                  % (plane, lo, hi, rows.size))
            continue
        amp = acc / nused
        result[plane] = dict(freqs=freqs, amp=amp, n=nused, wirelen=klen,
                             chid_lo=lo, chid_hi=hi, ntick=nwin, period=period)
        print('    %s chID %d-%d  len=%7.1fmm  used %d  dead %d'
              % (plane, lo, hi, klen, nused, ndead))
    return result


def save_npz(result):
    arrays = {}
    for plane, d in result.items():
        arrays['%s_freq' % plane] = d['freqs']
        arrays['%s_amp' % plane] = d['amp']
        arrays['%s_meta' % plane] = np.array(
            [d['n'], d['wirelen'], d['chid_lo'], d['chid_hi'],
             d['ntick'], d['period']])
    out = os.path.join(OUTDIR, 'noise_spectrum_region_anode0.npz')
    np.savez(out, **arrays)
    print('  wrote %s' % out)


def plot(result):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle('PDVD region noise amplitude spectrum -- anode0, evt_0 '
                 '(%s, ticks %d-%d)' % (REGION, TICK_LO, TICK_HI), fontsize=12)
    for ip, (plane, ax) in enumerate(zip(PLANES, axes)):
        if plane in result:
            d = result[plane]
            ax.plot(d['freqs'] * 1e3, d['amp'], lw=0.9, color='C0',
                    label='chID %d-%d\n%.0f mm (n=%d)'
                    % (d['chid_lo'], d['chid_hi'], d['wirelen'], d['n']))
            ax.legend(fontsize=8)
        ax.set_title('%s plane' % plane, fontsize=10)
        ax.set_xlabel('frequency [MHz]')
        ax.set_ylabel('mean |FFT| amplitude [internal V]')
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUTDIR, 'noise_spectrum_region_anode0.png')
    plt.savefig(out, dpi=130)
    plt.close()
    print('  wrote %s' % out)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    result = extract()
    if not result:
        print('no spectra extracted.')
        return
    save_npz(result)
    plot(result)
    print('done.')


if __name__ == '__main__':
    main()
