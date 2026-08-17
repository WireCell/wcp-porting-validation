#!/usr/bin/env python3
"""Draw the beam_cathode_split.py hits, in RAW image coordinates.

  grey   the whole raw image (t0 = 0), i.e. what the owner reads off Bee img-global
  blue   the in-beam bundle MAIN -- exactly what pattern recognition receives
  red    the other-TPC charge that continues the same line and is NOT in the main

Because the beam flash sits at t ~ 0 (window [0.2, 2.2) us => |dx| <= 0.34 cm), no
shift is applied to anything: blue and red are already in one consistent frame, and a
true hit is a single straight track with blue stopping dead at the dashed cathode.

Repro:
  cd .../sbnd/sbnd_xin
  python3 scripts/analysis/cathode/beam_cathode_split_plot.py \
      -w work-mcp1k-cb0805 -t products/beam-cathode-split-mcp1k.tsv \
      -o pics/beamsplit-mcp1k.png
"""
import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cathode_halfmatch import load_event  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-w', '--work-root', required=True, action='append')
    ap.add_argument('-t', '--tsv', required=True, action='append')
    ap.add_argument('-o', '--output', required=True)
    ap.add_argument('--kind', default='MISSING')
    a = ap.parse_args()

    rows = []
    for t in a.tsv:
        rows += [r for r in csv.DictReader(open(t), delimiter='\t')
                 if r['kind'] == a.kind]
    best = {}
    for r in sorted(rows, key=lambda r: -float(r['p_ext'])):
        best.setdefault(r['event'], r)
    top = sorted(best.values(), key=lambda r: -float(r['p_ext']))

    ncol = 4
    nrow = int(np.ceil(2 * len(top) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow))
    axes = np.atleast_2d(axes).ravel()
    for i, r in enumerate(top):
        ql = next(os.path.join(w, 'ql_evt%s' % r['event']) for w in a.work_root
                  if os.path.isdir(os.path.join(w, 'ql_evt%s' % r['event'])))
        img, clus, op, cid2apa = load_event(ql)
        ix = np.array(img['x'], float)
        iy = np.array(img['y'], float)
        iz = np.array(img['z'], float)
        icid = np.array(img['cluster_id'], int)
        ccid = np.array(clus['cluster_id'], int)
        m = ccid == int(r['main_id'])
        K = np.c_[np.array(clus['x'], float)[m], np.array(clus['y'], float)[m],
                  np.array(clus['z'], float)[m]]
        pm = icid == int(r['p_cid'])
        P = np.c_[ix[pm], iy[pm], iz[pm]]
        for j, (xa, ya, la, lb) in enumerate(((0, 1, 'x', 'y'), (0, 2, 'x', 'z'))):
            ax = axes[2 * i + j]
            ax.scatter(ix, iy if ya == 1 else iz, s=0.3, c='0.86')
            ax.scatter(K[:, xa], K[:, ya], s=1.4, c='tab:blue')
            ax.scatter(P[:, xa], P[:, ya], s=1.4, c='tab:red')
            ax.axvline(0, color='k', lw=0.8, ls='--')
            ax.set_xlabel('%s [cm]' % la)
            ax.set_ylabel('%s [cm]' % lb)
            ax.tick_params(labelsize=8)
            if j == 0:
                ax.set_title('evt%s  %s\nmissing %.0f cm, gap %.1f cm, %.1f deg'
                             % (r['event'], r['label'], float(r['p_ext']),
                                float(r['gap2d']), float(r['angle'])), fontsize=9)
            else:
                ax.set_title('beam flash %.2f us;  other half on %s'
                             % (float(r['flash_t']),
                                'the BEAM flash, different cluster'
                                if r['p_beam'] == '1' else 'a COSMIC flash'),
                             fontsize=9)
    for k in range(2 * len(top), len(axes)):
        axes[k].axis('off')
    fig.legend(handles=[
        plt.Line2D([], [], marker='o', ls='', color='0.86', label='raw image (all)'),
        plt.Line2D([], [], marker='o', ls='', color='tab:blue', label='in-beam main = what PR sees'),
        plt.Line2D([], [], marker='o', ls='', color='tab:red', label='continuation PR never got')],
        loc='lower center', ncol=3, fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(a.output, dpi=100)
    print('wrote %s (%d events)' % (a.output, len(top)))


if __name__ == '__main__':
    main()
