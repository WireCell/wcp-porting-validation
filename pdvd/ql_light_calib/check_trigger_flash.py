#!/usr/bin/env python3
"""Beam-trigger flash closure check for the PDVD light<->charge offsets.

THE decisive validation of the per-event per-crate trigger offsets
(pdvd-ql-pending.md sec 1): the runs are CTB beam-triggered, so every event
should contain a bright flash AT the trigger time.  With the offsets folded
into the calib-dump flash times (f['time'] is on the BDE charge axis), that
flash must sit at

    t_expect = tc_us - charge_bde_us     (per event, from jjo's table)

Flash times are ~1 us sharp, vs ~50 us mean flash spacing (>=150 PE), so the
accidental probability of a flash within +-1 us is only ~4%/event: a pile-up
of nearest-flash residuals at ~-1 us across most events is a many-sigma
validation of the whole mapping (chain light-t0 anchoring, sign, per-crate
charge window starts, metadata stamping, matcher folding).

Result 2026-07-10 (120 events, PDVD_QL_DIAG=2 dumps): 99/120 events have a
flash (median ~2000 PE) within +-5 us of t_expect, residual median -0.9 us
(inside the 1 us OpFlash binning).  The remaining events plausibly lack a
visible in-volume beam flash at the diag minPE=100 floor.  By contrast the
crosser-anchor fit (fit_trigger_offset.py) remains statistically BLIND at
this flash density -- its scan peak moves with event-mixing, so do not use
it to (in)validate the offsets.

Usage:
    python3 check_trigger_flash.py ../work/0*_*/calib-evt*.json \
        [--table ../data/jjo_triglight_offsets.txt] [--win 5]
"""
import argparse
import glob
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('dumps', nargs='+', help='calib-evt*.json (offsets folded)')
    ap.add_argument('--table', default='../data/jjo_triglight_offsets.txt',
                    help='jjo per-event timestamp table (tc_us, charge_bde_us)')
    ap.add_argument('--win', type=float, default=5.0,
                    help='closure acceptance half-window, us [5]')
    args = ap.parse_args()

    tbl = {}
    for line in open(args.table):
        if line.startswith('#') or not line.strip():
            continue
        f = line.split()
        tbl[(int(f[0]), int(f[2]))] = (float(f[6]), float(f[8]))  # tc, charge_bde

    res_all = {}
    for p in args.dumps:
        d = json.load(open(p))
        run = int(p.replace('\\', '/').split('/')[-2].split('_')[0])
        evt = d['charge_ident']
        if (run, evt) not in tbl:
            print(f'  [skip] run {run} evt {evt}: not in {args.table}')
            continue
        tc, cbde = tbl[(run, evt)]
        t_expect = tc - cbde
        ft = np.array([f['time'] for f in d['flashes']])
        pe = np.array([f['total_PE'] for f in d['flashes']])
        if not len(ft):
            continue
        i = np.abs(ft - t_expect).argmin()
        res_all.setdefault(run, []).append((ft[i] - t_expect, pe[i], evt))

    for run, v in sorted(res_all.items()):
        r = np.array([x[0] for x in v])
        pe = np.array([x[1] for x in v])
        close = np.abs(r) < args.win
        print(f'run {run:06d}: {close.sum()}/{len(r)} events with a flash '
              f'within +-{args.win:g} us of the trigger; residual median '
              f'{np.median(r[close]) if close.any() else float("nan"):+.2f} us, '
              f'PE median {np.median(pe[close]) if close.any() else 0:.0f}')
        far = [(x[2], x[0]) for x in v if abs(x[0]) >= args.win]
        if far:
            print(f'   no-close-flash events (evt, nearest dt us): '
                  f'{[(e, round(dt, 1)) for e, dt in far]}')


if __name__ == '__main__':
    main()
