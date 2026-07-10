#!/usr/bin/env python3
"""Dump the per-event light time zero from PDVD raw light ROOT files.

The PDVD light chain (pdvd/wct-light.jsonnet via PDVDOpWaveformSource)
anchors flash time t=0 to the EARLIEST record start over all channels of
the event -- the `timestamp` branch of the rawwf `raw_waveform` tree, in
microseconds on the DAQ 16 ns DTS clock (absolute). This script dumps that
anchor per event so it can be combined with the charge readout-window
start (same clock; NOT yet available -- the charge frame extraction writes
tickinfo time=0) into the deterministic per-event trigger offset:

    offset_us(event) = charge_window_start_us - light_t0_us

which is the PDVD analogue of PDHD's trigoff/trigger_offset tree
(offset_us = 249.808 constant there, measured per event from DAQ
timestamps). See fit_trigger_offset.py's docstring and
pdvd/docs/pdvd-ql-pending.md for why the statistical route is not enough.

Also prints, per event, the spread of the 16 cathode full-stream record
starts (expected 0.00 us -- they share one window) and the record length,
as coherence checks on the light DAQ.

Usage:
    python3 dump_light_t0.py /nfs/data/1/jjo/data/PDVD/*rawwf.root \
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
        t = uproot.open(fn)['raw_waveform']
        a = t.arrays(['run', 'event', 'opchannel', 'nsamp', 'timestamp'],
                     library='np')
        for evt in np.unique(a['event']):
            sel = a['event'] == evt
            ts = a['timestamp'][sel]
            t0 = float(ts.min())
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
