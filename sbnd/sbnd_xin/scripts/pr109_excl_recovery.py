#!/usr/bin/env python3
"""doc pr/109 sec 9 -- recovery curves for two proposed relaxations of the
exclusion arbitration, computed offline from the WCT_EXCL_DUMP decision dump.

update_association currently drops a 2-D cell from a segment iff some sibling
segment is at least as close:   drop  <=>  min_other <= min_dis_track,
with an unconditional keep below 0.3 cm.  Two relaxations have been proposed:

  TIE MARGIN m   drop only when a sibling beats this segment by more than m:
                 drop <=> min_dis_track - min_other > m.   m = 0 already
                 rescues exact ties (the current rule drops those from BOTH
                 segments); m -> infinity is identical to fit_exclusion=false.

  KEEP FLOOR f   raise the unconditional-keep radius from 0.3 cm to f.
                 f -> infinity is likewise identical to fit_exclusion=false.

Both therefore interpolate between exclusion ON and exclusion OFF, so the
charge either can recover is bounded above by the ON-vs-OFF gap already
measured in sec 9.  This script computes where on that interval each lands,
WITHOUT building either knob.

A cell is counted as lost only if EVERY segment that considered it dropped it
(a cell kept by any segment still supports the trajectory), so the unit is the
unique readout cell, not the per-segment decision, and charge is not
double-counted.  Dead-region fillers (flag 0) and non-positive charge are
excluded, matching the sec 4 metric's cell selection.

Usage:
  pr109_excl_recovery.py --dump <WCT_EXCL_DUMP file> --root <tracking root>
                         [--box-cm 3.0] [--label NAME]
"""
import argparse, collections
import numpy as np
import uproot

MARGINS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0]
FLOORS  = [0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.8, 1.0, 2.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', required=True)
    ap.add_argument('--root', required=True)
    ap.add_argument('--box-cm', type=float, default=3.0)
    ap.add_argument('--label', default='')
    args = ap.parse_args()

    g = uproot.open(args.root)
    k = g['T_kine'].arrays(['kine_nu_x_corr', 'kine_nu_y_corr', 'kine_nu_z_corr'], library='np')
    V = np.array([k['kine_nu_x_corr'][0], k['kine_nu_y_corr'][0], k['kine_nu_z_corr'][0]], float)

    # cell -> [charge, kept_any, min delta over dropping decisions, min min_dis over dropping]
    cells = {}
    near = set()
    nline = 0
    for line in open(args.dump):
        if line[0] == '#':
            continue
        p = line.split()
        if len(p) < 17:
            continue
        nline += 1
        plane, apa, face, wire, time = int(p[2]), int(p[3]), int(p[4]), int(p[5]), int(p[6])
        q, flag = float(p[7]), int(p[8])
        mind, mino = float(p[9]), float(p[10])
        kept = int(p[11])
        px, py, pz = float(p[14]), float(p[15]), float(p[16])
        if q <= 0 or flag == 0:
            continue
        key = (apa, face, plane, wire, time)
        e = cells.get(key)
        if e is None:
            e = cells[key] = [q, 0, 1e9, 1e9]
        if kept:
            e[1] = 1
        else:
            e[2] = min(e[2], mind - mino)      # how badly it lost
            e[3] = min(e[3], mind)             # how close it was to its own segment
        if np.linalg.norm(np.array([px, py, pz]) - V) < args.box_cm:
            near.add(key)

    def curve(keys, name):
        tot = sum(cells[k][0] for k in keys)
        lost = [k for k in keys if cells[k][1] == 0]
        qlost = sum(cells[k][0] for k in lost)
        print('  %-14s cells %6d  charge %.4g   LOST to exclusion: %d cells, %.4g (%.1f%%)'
              % (name, len(keys), tot, len(lost), qlost, 100 * qlost / tot if tot else 0))
        if not lost:
            return
        print('    tie margin m (cm) :', '  '.join('%5g' % m for m in MARGINS))
        rec = [sum(cells[k][0] for k in lost if cells[k][2] <= m) for m in MARGINS]
        print('    charge recovered %%:', '  '.join('%5.1f' % (100 * r / qlost) for r in rec))
        print('    keep floor f (cm) :', '  '.join('%5g' % f for f in FLOORS))
        rec = [sum(cells[k][0] for k in lost if cells[k][3] < f) for f in FLOORS]
        print('    charge recovered %%:', '  '.join('%5.1f' % (100 * r / qlost) for r in rec))

    print('# %s   %d decision lines, %d unique live cells' % (args.label, nline, len(cells)))
    curve(set(cells), 'all cells')
    curve(near, 'within %.1f cm' % args.box_cm)


if __name__ == '__main__':
    main()
