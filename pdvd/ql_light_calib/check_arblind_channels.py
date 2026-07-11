#!/usr/bin/env python3
"""Are the Ar-blind channels dark in data?  (Ar 128 nm vs Xe 175 nm test.)

The static QL ch_mask holds two very different classes:

  - dead-in-readout {24, 27, 28, 34}: not in the DAPHNE data at all -> their
    reconstructed PE is identically zero (control sample for "absent");
  - Ar-blind {13 (membrane XA no PTP), 29/39 (PEN+Q PMTs), 32 (uncoated
    PMT)}: present in the readout but with eff_Ar = 0 at 128 nm.  If the runs
    were Xe-doped these channels MUST light up with the big flashes (Xe 175 nm
    passes their windows/WLS); if pure argon they stay at noise level.

So the per-channel response in the RAW opflash archives (no QL mask applied)
directly answers the "Ar or Xe?" library question from the data side.

Reads work/*_light*/opflash_pdvd-wct.tar.gz (tensor 0 = [nflash, 1+40]:
col 0 flash time ns, cols 1..40 per-OpDet PE).  For every channel reports,
per run: total PE, per-event max PE, and the median response in "big" flashes
(total PE >= --big).  Ar-blind channels are judged against their same-type
live peers.

Usage:  python3 check_arblind_channels.py [--workdir ../work] [--big 1000]
"""
import argparse
import glob
import io
import json
import os
import re
import tarfile

import numpy as np

DEAD = [24, 27, 28, 34]              # absent from DAPHNE readout (control)
ARBLIND = [13, 29, 32, 39]           # in readout, eff_Ar = 0 at 128 nm
# same-type live peers to compare against
PEERS = {13: [12, 18, 19],           # membrane XA (bottom volume)
         29: [25, 26, 30, 31],       # PEN bottom PMTs
         39: [35, 36, 37, 38],
         32: [30, 31, 33]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--workdir', default=os.path.join(os.path.dirname(__file__), '..', 'work'))
    ap.add_argument('--big', type=float, default=1000.0,
                    help='total-PE floor for the "big flash" response test')
    args = ap.parse_args()

    runs = {}
    for tgz in sorted(glob.glob(os.path.join(args.workdir, '*_light*', 'opflash_pdvd-wct.tar.gz'))):
        d = os.path.basename(os.path.dirname(tgz))
        m = re.match(r'^(\d{6})_light(\d+)$', d)   # skip _p2 / _trigofftest variants
        if not m:
            continue
        run = m.group(1)
        with tarfile.open(tgz) as tf:
            name = [n for n in tf.getnames() if n.endswith('_0_array.npy')][0]
            arr = np.load(io.BytesIO(tf.extractfile(name).read()))
        runs.setdefault(run, []).append(arr[:, 1:])   # drop the time column

    for run in sorted(runs):
        evs = runs[run]
        allf = np.vstack(evs)
        tot = allf.sum(axis=1)
        big = allf[tot >= args.big]
        print(f'\n=== run {run}: {len(evs)} events, {len(allf)} flashes, '
              f'{len(big)} big (>= {args.big:g} PE) ===')
        print(f'{"ch":>3} {"class":>9} {"totPE":>10} {"evt-max med":>11} '
              f'{"big mean":>9} {"big f>5":>8}  peers(big mean | f>5)')
        for ch in ARBLIND + DEAD:
            cls = 'Ar-blind' if ch in ARBLIND else 'dead-RO'
            evmax = np.median([e[:, ch].max() if len(e) else 0.0 for e in evs])
            bmean = float(big[:, ch].mean()) if len(big) else 0.0
            bfrac = float((big[:, ch] > 5).mean()) if len(big) else 0.0
            peers = PEERS.get(ch, [])
            pstr = ' '.join(f'{p}:{big[:, p].mean():.1f}|{(big[:, p] > 5).mean():.2f}'
                            for p in peers) if len(big) else ''
            print(f'{ch:>3} {cls:>9} {allf[:, ch].sum():>10.1f} {evmax:>11.2f} '
                  f'{bmean:>9.2f} {bfrac:>8.2f}  {pstr}')
        # correlation of each Ar-blind channel with the total flash PE
        # (a live channel tracks flash brightness; noise does not)
        print('corr(ch PE, total PE) over all flashes: ', end='')
        for ch in ARBLIND:
            c = np.corrcoef(allf[:, ch], tot)[0, 1] if allf[:, ch].std() > 0 else 0.0
            print(f'ch{ch}:{c:+.2f}', end='  ')
        print()


if __name__ == '__main__':
    main()
