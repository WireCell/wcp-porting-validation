#!/usr/bin/env python3
"""Draw the cathode_halfmatch.py candidates: matched half, partner as reconstructed,
partner re-placed under the matched half's T0.

Three point sets per panel:
  blue   the matched all-APA cluster, T0-corrected -- what the reconstruction has
  grey   the partner img cluster where the reconstruction actually put it, i.e.
         x_raw + dx_partner -- NOT the raw image position, which for an
         'other-flash' partner is a third, meaningless place
  red    the same partner shifted by -dx_K, i.e. given the matched half's flash T0

A true one-side-only match shows blue stopping dead at x=0 and red continuing the
identical straight line, with grey parked tens to hundreds of cm away.

Repro:
  cd .../sbnd/sbnd_xin
  python3 scripts/analysis/cathode/cathode_halfmatch_plot.py \
      -w work-mcp1k-cb0805 -t products/cathode-halfmatch-mcp1k.tsv \
      -o /home/xqian/tmp/cathhalf-mcp1k-top10.png -n 10
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
import cathode_halfmatch as C            # noqa: E402


def event_sets(work_root, row):
    ql = os.path.join(work_root, 'ql_evt%s' % row['event'])
    img, clus, op, cid2apa = C.load_event(ql)
    ix = np.array(img['x'], float)
    iy = np.array(img['y'], float)
    iz = np.array(img['z'], float)
    iq = np.array(img['q'], float)
    icid = np.array(img['cluster_id'], int)
    ccid = np.array(clus['cluster_id'], int)
    m = ccid == int(row['gid'])
    K = np.c_[np.array(clus['x'], float)[m], np.array(clus['y'], float)[m],
              np.array(clus['z'], float)[m]]
    pm = icid == int(row['p_cid'])
    raw = np.c_[ix[pm], iy[pm], iz[pm]]
    reco = raw.copy()
    reco[:, 0] += float(row['p_dx'])     # where the reconstruction actually put it
    fix = raw.copy()
    fix[:, 0] -= float(row['dx'])        # where the matched half's T0 puts it
    return K, reco, fix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-w', '--work-root', required=True)
    ap.add_argument('-t', '--tsv', required=True)
    ap.add_argument('-o', '--output', required=True)
    ap.add_argument('-n', '--top', type=int, default=10)
    ap.add_argument('--kinds', default='other-flash,UNMATCHED')
    ap.add_argument('--min-disp', type=float, default=20.0,
                    help='cm; how far the partner must sit from where the matched '
                         'half\'s T0 puts it.  Below ~10 cm the charge is nearly in '
                         'place and nothing looks missing.')
    a = ap.parse_args()

    kinds = set(a.kinds.split(','))
    rows = [r for r in csv.DictReader(open(a.tsv), delimiter='\t')
            if r['kind'] in kinds and float(r['disp']) >= a.min_disp]
    best = {}
    for r in sorted(rows, key=lambda r: -float(r['p_ext'])):
        best.setdefault(r['event'], r)
    top = sorted(best.values(), key=lambda r: -float(r['p_ext']))[:a.top]

    ncol = 4
    nrow = int(np.ceil(2 * len(top) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow))
    axes = np.atleast_2d(axes).ravel()
    for i, r in enumerate(top):
        K, reco, fix = event_sets(a.work_root, r)
        for j, (xa, ya, la, lb) in enumerate(((0, 1, 'x', 'y'), (0, 2, 'x', 'z'))):
            ax = axes[2 * i + j]
            ax.scatter(reco[:, xa], reco[:, ya], s=0.6, c='0.78')
            ax.scatter(K[:, xa], K[:, ya], s=0.6, c='tab:blue')
            ax.scatter(fix[:, xa], fix[:, ya], s=0.6, c='tab:red')
            ax.axvline(0, color='k', lw=0.7, ls='--')
            ax.set_xlabel('%s [cm]' % la)
            ax.set_ylabel('%s [cm]' % lb)
            ax.tick_params(labelsize=8)
            if j == 0:
                ax.set_title('evt%s  %s\nmissing %.0f cm, gap %.1f cm, %.1f deg'
                             % (r['event'], r['kind'], float(r['p_ext']),
                                float(r['gap2d']), float(r['angle'])), fontsize=9)
            else:
                ax.set_title('t0(matched)=%.0f us;  partner displaced %.0f cm'
                             % (float(r['t0']), float(r['disp'])), fontsize=9)
    for k in range(2 * len(top), len(axes)):
        axes[k].axis('off')
    fig.legend(handles=[
        plt.Line2D([], [], marker='o', ls='', color='tab:blue', label='matched half (T0-corrected)'),
        plt.Line2D([], [], marker='o', ls='', color='0.78', label='partner as reconstructed'),
        plt.Line2D([], [], marker='o', ls='', color='tab:red', label="partner under matched half's T0")],
        loc='lower center', ncol=3, fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(a.output, dpi=100)
    print('wrote %s (%d events)' % (a.output, len(top)))


if __name__ == '__main__':
    main()
