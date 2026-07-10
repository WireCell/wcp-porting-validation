#!/usr/bin/env python3
"""Fit the PDVD light-vs-charge trigger offset from Q/L calib dumps.

Anchor: a cosmic that crosses the FULL drift of one volume has an apparent
x-extent equal to the drift length u_cathode regardless of its (unknown) T0,
so its absolute drift time is pinned by geometry alone (midpoint form, robust
against symmetric edge erosion; matches x_true = x_raw + sign_offset*(t+T)*v,
the matcher's flash_x_offset convention):

    t_true = t_flash + T = (x_center_raw - volume_center_x) / (sign_offset * v)

Method (two stages):
 1. SCAN -- histogram T = t_true - t_flash over every (crosser, flash) pair
    (flash floor --pe-min, default 150 PE: the true flash of a full-drift
    muon is sometimes only a few hundred PE, a higher floor drops it and the
    peak fragments) and take the +-win window holding the most DISTINCT
    (event, cluster) contributors as the run's candidate T0.
 2. REFINE -- for every crosser take the flash NEAREST t_true - T0; the
    residuals of true pairs pile within ~ +-15 us (cluster-edge resolution)
    while accidentals spread uniformly; sigma-clip and take the median.

Also reports per-event coverage and OUTLIER events (>= 2 crossers, no flash
within the acceptance).

*** STATUS 2026-07-10: SUPERSEDED as a calibrator AND as a validator.       ***
The offsets are now MEASURED per event and per crate from the rawwf
trigoff/trigger_offset tree (stamped into the opflash metadata by
run_light_evt.sh; see pdvd/docs/pdvd-ql-pending.md sec 1) and VALIDATED by
the beam-trigger flash closure (check_trigger_flash.py: 99/120 events show
a ~2000 PE flash within +-5 us of tc_us - charge_bde_us on the folded axis,
residual median -0.9 us).  This script's scan CANNOT confirm or refute the
offsets: on offset-folded PDVD_QL_DIAG=2 dumps it reports T = +2031/+2328 us
(039252/039253) with a deceptively tight MAD -- but the peak height equals
the event-mixed accidental maximum at every T, i.e. it is pure accidental
structure, exactly the trap the paragraph below documents.  Kept for the
method record and for detectors with sparse flash lists.

*** ORIGINAL STATUS 2026-07 (18+18+40 diag events, 47/47/55 anchors):       ***
*** a per-run constant is NEITHER CONFIRMED NOR EXCLUDED by this sample.    ***
The anchor resolution is ~ +-20 us (measured from bottom-vs-top t_mid of
gold two-volume through-goers) while flashes >=150 PE are ~50 us apart, so
the accidental probability of a nearest flash within the acceptance is
~60% per anchor: even a TRUE constant would only reach ~2-3 sigma here --
which is exactly what the best windows on the [-7.5ms, +5ms] grid show.
Do NOT trust "N/M events validate T" checks: nearest-flash residuals pile
near 0 by construction at this flash density. The most discriminating
subsample (7 gold two-volume anchors in run 039252) favors T ~ -2855 us at
3.2 sigma over event-mixed accidentals -- suggestive, not conclusive; and
run 039349 evt 19549 (6 anchors) sits ~4 ms from every run-level candidate,
so occasional per-event shifts are possible. The charge side is internally
coherent (gold-pair t_mid agree within +-30 us) and the light records are
event-aligned (all 16 cathode full streams share one start per event).
The decisive measurement is deterministic, not statistical: per-event DAQ
timestamps, as PDHD's extraction provided (trigoff/trigger_offset tree ->
offset_us). The light side is already recoverable from the rawwf
`timestamp` branch (see dump_light_t0.py); only the charge readout-window
start timestamp per event (same DTS clock) is missing from the charge
extraction. Once it exists, offset_us(event) = charge_window_start -
light_record_start, and this script becomes the physics cross-check.
See pdvd/docs/pdvd-ql-pending.md for the full follow-up list.

Run the inputs in DIAGNOSTIC mode (PDVD_QL_DIAG=1 ./run_clus_evt.sh -calib
<run> <idx>) so the dumps are produced with trigger_offset forced to 0 --
then the fitted T is directly the value for pdvd/data/ql_trigger_offset.txt
(all PDVD light archives carry metadata offset_us = 0; the script refuses
dumps with a non-zero trigger_offset).

Usage:
    python3 fit_trigger_offset.py work/0392*_*/calib-evt*.json \
        [--pe-min 150] [--win 30] [--write ../data/ql_trigger_offset.txt]
"""
import argparse
import json
import re
import sys
from collections import defaultdict

import numpy as np


def crossers(dump, ext_lo=25.0, ext_hi=8.0, npoints_min=300):
    """Full-drift-crossing clusters: apparent x-extent ~ u_cathode.

    Yields (apa, uid, q, t_drift_mid, t_drift_anode) where t_drift_* is the
    implied t_flash + T from the midpoint / anode-edge anchor.
    """
    v = dump['drift_speed']
    for apa, g in dump['geometry'].items():
        uc = g['u_cathode']
        so = g['sign_offset']
        vol_center = 0.5 * (g['anode_x'] + g['cathode_x'])
        for c in dump['clusters']:
            if str(c['apa']) != apa or c['npoints'] < npoints_min:
                continue
            xmin, xmax = min(c['x']), max(c['x'])
            ext = xmax - xmin
            if not (uc - ext_lo < ext < uc + ext_hi):
                continue
            x_mid = 0.5 * (xmin + xmax)
            # correction: x_true = x_raw + sign_offset*(t_flash+T)*v
            t_mid = (vol_center - x_mid) / (so * v)
            x_anode_edge = xmin if g['anode_x'] < g['cathode_x'] else xmax
            t_anode = (g['anode_x'] - x_anode_edge) / (so * v)
            yield int(apa), c['uid'], sum(c['q']), t_mid, t_anode


def scan_candidate(pairs, half_width):
    """Stage 1: densest +-half_width T window over (T, evt, uid) pairs, scored
    by distinct (evt, uid) contributors. Returns the window's median T."""
    ps = sorted(pairs)
    best = (0, 0, 0.0)
    j = 0
    for i in range(len(ps)):
        if j < i + 1:
            j = i + 1
        while j < len(ps) and ps[j][0] - ps[i][0] <= 2 * half_width:
            j += 1
        mem = ps[i:j]
        score = len(set((p[1], p[2]) for p in mem))
        if score > best[0] or (score == best[0] and len(mem) > best[1]):
            best = (score, len(mem), mem[len(mem) // 2][0])
    return best[2], best[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('dumps', nargs='+', help='calib-evt*.json paths (diag runs)')
    ap.add_argument('--pe-min', type=float, default=150.0,
                    help='min flash total_PE [150; higher floors drop true '
                         'flashes of dim full-drift muons and fragment the peak]')
    ap.add_argument('--npoints-min', type=int, default=300)
    ap.add_argument('--win', type=float, default=30.0,
                    help='scan/acceptance half-window in us [30]')
    ap.add_argument('--write', metavar='PATH',
                    help='write per-run "RUN OFFSET_US" table to PATH')
    args = ap.parse_args()

    by_run = defaultdict(list)   # run -> [(path, dump), ...]
    for path in args.dumps:
        m = re.search(r'(\d{5,6})_\d+/calib-evt(\d+)\.json', path.replace('\\', '/'))
        run = int(m.group(1)) if m else -1
        dump = json.load(open(path))
        if dump.get('trigger_offset', 0.0) != 0.0:
            sys.exit(f'{path}: trigger_offset={dump["trigger_offset"]} != 0 -- '
                     'rerun in diagnostic mode (PDVD_QL_DIAG=1)')
        by_run[run].append((path, dump))

    table = {}
    for run in sorted(by_run):
        # stage 1: candidate from the pair histogram
        pairs = []
        for path, d in by_run[run]:
            fl = [f for f in d['flashes'] if f['total_PE'] >= args.pe_min]
            for apa, uid, q, tm, ta in crossers(d, npoints_min=args.npoints_min):
                pairs += [(tm - f['time'], d['charge_ident'], uid) for f in fl]
        if not pairs:
            print(f'run {run:06d}: no (crosser, flash) pairs -- skipped')
            continue
        t0, score = scan_candidate(pairs, args.win)
        # stage 2: nearest-flash residuals, sigma-clipped median
        dts, outliers, nev_hit = [], [], 0
        for path, d in by_run[run]:
            ft = np.array(sorted(f['time'] for f in d['flashes']
                                 if f['total_PE'] >= args.pe_min))
            evt_dts = []
            xers = list(crossers(d, npoints_min=args.npoints_min))
            for apa, uid, q, tm, ta in xers:
                if not len(ft):
                    continue
                dT = tm - ft[np.abs(ft - (tm - t0)).argmin()] - t0
                if abs(dT) <= args.win + 10:
                    evt_dts.append(dT)
            dts += evt_dts
            if evt_dts:
                nev_hit += 1
            elif len(xers) >= 2:
                outliers.append((d['charge_ident'], len(xers)))
        dts = np.array(dts)
        med = np.median(dts)
        core = dts[np.abs(dts - med) < 20]
        T = t0 + float(np.median(core))
        mad = float(np.median(np.abs(core - np.median(core))))
        table[run] = T
        nev = len(by_run[run])
        print(f'run {run:06d}: T = {T:+.1f} us  (scan candidate {t0:+.1f}, '
              f'{score} distinct anchors; refine core {len(core)}/{len(dts)}, '
              f'MAD {mad:.1f} us; {nev_hit}/{nev} events contribute)')
        if outliers:
            print(f'  OUTLIER events (>=2 crossers, no in-window flash -- '
                  f'shifted DAQ light window?): {outliers}')

    if args.write and table:
        with open(args.write, 'w') as fp:
            fp.write('# run  trigger_offset_us  (light-vs-charge time base; '
                     'fit_trigger_offset.py from QL diag calib dumps)\n')
            for run, t in sorted(table.items()):
                fp.write(f'{run:06d} {t:.1f}\n')
        print(f'\nwrote {args.write}')


if __name__ == '__main__':
    main()
