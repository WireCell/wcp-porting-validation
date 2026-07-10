#!/usr/bin/env python3
"""Dump the per-event light time zero from PDVD raw light ROOT files.

The PDVD light chain (pdvd/wct-light-reco.jsonnet via PDVDOpWaveformSource)
anchors flash time t=0 to the EARLIEST record start over all channels of
the event, where a self-trigger snippet (nsamp <= 1024) STARTS 64 samples
(1.024 us) before its `timestamp` (the timestamp marks the trigger sample):

    start_us = timestamp - (1.024 if snippet else 0)
    chain_t0_us = min over all records

in microseconds on the DAQ 16 ns DTS clock (absolute, 40-bit masked).
This script dumps that anchor per event.  RESOLVED 2026-07-10: the charge
window starts are now available per event (and per CRATE -- TDE/BDE open up
to ~32 us apart) in the rawwf trigoff/trigger_offset tree, and
run_light_evt.sh computes and stamps

    offset_{bot,top}_us(event) = chain_t0_us - charge_{bde,tde}_us

into the opflash archive metadata.  This script remains as an independent
cross-check of the chain_t0 half.  See pdvd/docs/pdvd-ql-pending.md.

Also prints, per event, the spread of the 16 cathode full-stream record
starts (expected 0.00 us -- they share one window) and the record length,
as coherence checks on the light DAQ.

Usage:
    python3 dump_light_t0.py ../input_data_light/*rawwf.root \
        [--csv light_t0.csv]
"""
import argparse
import csv
import sys

import numpy as np
import uproot


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('files', nargs='+', help='np02vd_raw_*_rawwf.root')
    ap.add_argument('--csv', metavar='PATH', help='also write a CSV table')
    args = ap.parse_args()

    rows = []
    for fn in args.files:
        f = uproot.open(fn)
        try:
            t = f['rawdump/raw_waveform']   # newer nested extraction layout
        except KeyError:
            t = f['raw_waveform']
        a = t.arrays(['run', 'event', 'opchannel', 'nsamp', 'timestamp'],
                     library='np')
        for evt in np.unique(a['event']):
            sel = a['event'] == evt
            ts = a['timestamp'][sel]
            # chain anchor: snippet records start 64 samples before their
            # timestamp (PDVDOpWaveformSource tick rule, exact in ticks)
            ticks = (np.round(ts * 62.5)
                     - np.where(a['nsamp'][sel] <= 1024, 64, 0))
            t0 = float(ticks.min() * 0.016)
            # cathode full streams: the longest records of the event
            nsamp = a['nsamp'][sel]
            full = nsamp == nsamp.max()
            spread = float(ts[full].max() - ts[full].min())
            rows.append(dict(run=int(a['run'][sel][0]), event=int(evt),
                             light_t0_us=t0, nrec=int(sel.sum()),
                             fullstream_spread_us=spread,
                             fullstream_us=float(nsamp.max()) * 0.016))
    rows.sort(key=lambda r: (r['run'], r['event']))
    print(f'{"run":>6} {"event":>8} {"light_t0_us":>18} {"nrec":>5} '
          f'{"fs_spread_us":>12} {"fs_len_us":>10}')
    for r in rows:
        print(f'{r["run"]:>6} {r["event"]:>8} {r["light_t0_us"]:>18.3f} '
              f'{r["nrec"]:>5} {r["fullstream_spread_us"]:>12.2f} '
              f'{r["fullstream_us"]:>10.1f}')
        if r['fullstream_spread_us'] > 1.0:
            print(f'  WARNING: cathode full-stream starts spread by '
                  f'{r["fullstream_spread_us"]:.2f} us -- light window not '
                  f'internally coherent for this event', file=sys.stderr)

    if args.csv:
        with open(args.csv, 'w', newline='') as fp:
            w = csv.DictWriter(fp, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f'wrote {args.csv}')


if __name__ == '__main__':
    main()
