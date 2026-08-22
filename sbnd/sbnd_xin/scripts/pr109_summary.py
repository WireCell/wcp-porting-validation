#!/usr/bin/env python3
"""doc pr/109 -- apply the pre-registered decision rule to pr109_2d_resid.py's TSVs.

Test unit: ONE entry per (event, region), U pooled over the three planes.
Regions from the two anchor arms are merged when their anchors sit within
--link cm of each other (the vertex moves between arms; it is the same region).

For each entry, over the variants v = {anchor arm} x {box size}:
    dU_v = U_ON(v) - U_OFF(v)          (negative = exclusion ON explains more charge)
    dU   = median_v dU_v
    Bsys = (max_v dU_v - min_v dU_v)/2
Verdict: ON-better if dU < -Bsys, ON-worse if dU > +Bsys, else equivalent.
Entries where BOTH arms have no prediction at all in the box (U == 1, B == -1)
are excluded and listed -- the saved fit covers nothing there, so the entry
carries no information about either arm.
"""
import argparse, csv, collections
import numpy as np


def load(path):
    rows = [r for r in csv.DictReader(open(path), delimiter='\t') if r['plane'] == 'ALL']
    for r in rows:
        for k in ('ax', 'ay', 'az', 'box', 'U', 'B', 'sum_y', 'chi2_per_N'):
            r[k] = float(r[k])
        for k in ('N', 'ncol', 'nshared', 'ndead'):
            r[k] = int(r[k])
    return rows


def regionize(rows, link):
    """Merge anchors of one tag that sit within `link` cm into one region id."""
    out = {}
    for tag in sorted({r['tag'] for r in rows}):
        pts, ids = [], {}
        for r in rows:
            if r['tag'] != tag:
                continue
            p = np.array([r['ax'], r['ay'], r['az']])
            hit = None
            for i, q in enumerate(pts):
                if np.linalg.norm(p - q) < link:
                    hit = i
                    break
            if hit is None:
                pts.append(p)
                hit = len(pts) - 1
            ids[(r['ax'], r['ay'], r['az'])] = hit
        out[tag] = (ids, pts)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tsv', nargs='+')
    ap.add_argument('--link', type=float, default=2.0)
    args = ap.parse_args()

    for path in args.tsv:
        rows = load(path)
        reg = regionize(rows, args.link)
        acc = collections.defaultdict(dict)     # (tag, rid) -> variant -> {arm: row}
        for r in rows:
            rid = reg[r['tag']][0][(r['ax'], r['ay'], r['az'])]
            acc[(r['tag'], rid)].setdefault((r['anchor_arm'], r['box']), {})[r['arm']] = r

        print('=== %s' % path)
        print('%-14s %3s %8s %8s %8s %8s %8s %6s %6s  %s' %
              ('tag', 'reg', 'U_ON', 'U_OFF', 'dU', 'Bsys', 'B_ON-OFF', 'N', 'dNcol', 'verdict'))
        tally = collections.Counter()
        dead = []
        for key in sorted(acc):
            vs = acc[key]
            dU, uon, uoff, dB, ncol, nn = [], [], [], [], [], []
            nopred = True
            for v, d in vs.items():
                if 'ON' not in d or 'OFF' not in d:
                    continue
                dU.append(d['ON']['U'] - d['OFF']['U'])
                uon.append(d['ON']['U']); uoff.append(d['OFF']['U'])
                dB.append(d['ON']['B'] - d['OFF']['B'])
                ncol.append(d['ON']['ncol'] - d['OFF']['ncol'])
                nn.append(d['OFF']['N'])
                if not (abs(d['ON']['U'] - 1) < 1e-9 and abs(d['OFF']['U'] - 1) < 1e-9):
                    nopred = False
            if not dU:
                continue
            if nopred:
                dead.append(key)
                continue
            m = float(np.median(dU)); bs = (max(dU) - min(dU)) / 2.0
            verdict = 'ON-better' if m < -bs else ('ON-worse' if m > bs else 'equivalent')
            tally[verdict] += 1
            print('%-14s %3d %8.4f %8.4f %8.4f %8.4f %8.4f %6d %6d  %s' %
                  (key[0], key[1], np.median(uon), np.median(uoff), m, bs,
                   float(np.median(dB)), int(np.median(nn)), int(np.median(ncol)), verdict))
        n = sum(tally.values())
        print('  entries %d: ON-better %d, equivalent %d, ON-worse %d%s' %
              (n, tally['ON-better'], tally['equivalent'], tally['ON-worse'],
               ('; excluded (no prediction in either arm): %s' % dead) if dead else ''))
        # per-event roll-up: the sign of the median dU over that event's regions
        ev = collections.defaultdict(list)
        for key in sorted(acc):
            if key in dead:
                continue
            vs = acc[key]
            d = [x['ON']['U'] - x['OFF']['U'] for x in vs.values() if 'ON' in x and 'OFF' in x]
            if d:
                ev[key[0]].append(float(np.median(d)))
        nb = sum(1 for t in ev if np.median(ev[t]) < 0)
        nw = sum(1 for t in ev if np.median(ev[t]) > 0)
        print('  per-event sign test: ON-better %d/%d, ON-worse %d/%d' % (nb, len(ev), nw, len(ev)))
        print()


if __name__ == '__main__':
    main()
