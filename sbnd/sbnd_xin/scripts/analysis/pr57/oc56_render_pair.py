#!/usr/bin/env python3
"""doc pr/57 round 2: render one PNG per S6 candidate pair, off-screen.

Same content as the port-5018/5019 Bokeh viewer, drawn with matplotlib so a
pair can be inspected without a browser: row 1 the three 3-D projections of the
two components (plus the rest of the cluster in grey, the candidate edge, and
the reconstructed neutrino vertex), row 2 the U/V/W wire-vs-slice panels built
from the dump's OWN fired/dead/seed cells -- i.e. the pixels that actually
decided the S6 verdict, not a Python re-derivation of them.

Usage:
    oc56_render_pair.py --arm work-pr57r2-scan395 --first 50 --out <dir> \
        [--only bad] [--min-lmin 10] [--max 40]
"""
import argparse
import collections
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import oc56_autoscan as A  # noqa: E402

PLANE_NAME = {0: 'U', 1: 'V', 2: 'W'}
ZOOM_CM = 40.0


def draw_proj(ax, ptsA, ptsB, other, p1, p2, vtx, ia, ib, labels):
    for P in other:
        if len(P):
            ax.plot(P[:, ia], P[:, ib], '.', ms=0.7, color='0.78', zorder=1)
    ax.plot(ptsA[:, ia], ptsA[:, ib], '.', ms=1.6, color='tab:blue', zorder=3)
    ax.plot(ptsB[:, ia], ptsB[:, ib], '.', ms=1.6, color='tab:red', zorder=3)
    ax.plot([p1[ia], p2[ia]], [p1[ib], p2[ib]], '-', lw=1.4, color='k', zorder=5)
    ax.plot([p1[ia], p2[ia]], [p1[ib], p2[ib]], 'o', ms=3.5, color='k', zorder=5)
    if vtx is not None:
        ax.plot([vtx[ia]], [vtx[ib]], '*', ms=13, color='tab:green',
                mec='k', mew=0.5, zorder=6)
    mid = (np.asarray(p1) + np.asarray(p2)) / 2.0
    ax.set_xlim(mid[ia] - ZOOM_CM, mid[ia] + ZOOM_CM)
    ax.set_ylim(mid[ib] - ZOOM_CM, mid[ib] + ZOOM_CM)
    ax.set_xlabel(labels[0] + ' [cm]', fontsize=7)
    ax.set_ylabel(labels[1] + ' [cm]', fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_aspect('equal', adjustable='box')


def draw_plane(ax, pl, rec, step):
    wlo, whi, slo, shi = pl['win']
    for wind, lo, hi in pl['dead']:
        ax.add_patch(Rectangle((wind - 0.5, lo), 1.0, max(hi - lo, step),
                               color='tab:orange', alpha=0.25, lw=0, zorder=1))
    if pl['fired']:
        f = np.asarray(pl['fired'], dtype=float)
        ax.plot(f[:, 0], f[:, 1], 's', ms=1.6, color='0.35', zorder=2)
    for seeds, col in ((pl['seeds_a'], 'tab:blue'), (pl['seeds_b'], 'tab:red')):
        if seeds:
            s = np.asarray(seeds, dtype=float)
            ax.plot(s[:, 0], s[:, 1], 's', ms=2.6, color=col, zorder=3)
    ax.add_patch(Rectangle((wlo, slo), whi - wlo, shi - slo, fill=False,
                           ec='k', lw=0.6, ls=':', zorder=4))
    ax.set_xlim(wlo - 2, whi + 2)
    ax.set_ylim(slo - 2 * step, shi + 2 * step)
    p = pl['plane']
    ax.set_title('%s  gap=%d  matrix=%s' % (PLANE_NAME[p], rec['gap'][p],
                                            rec['matrix'][p] or '-'),
                 fontsize=7)
    ax.set_xlabel('wire index', fontsize=7)
    ax.set_ylabel('slice', fontsize=7)
    ax.tick_params(labelsize=6)


def render(arm, evt, path, pair, prm, outdir):
    comps, _ = A.load_event(path)
    vtx = A.nu_vertex(arm, evt)
    call, j, k = pair['call'], pair['j'], pair['k']
    Apts, Bpts = comps[(call, j)], comps[(call, k)]
    other = [P for (c, i), P in comps.items() if c == call and i not in (j, k)]
    rec = min(pair['edges'], key=lambda r: r['dis'])
    verdict, cause, rule, conf = A.classify(pair, prm)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.4))
    for ax, (ia, ib, lab) in zip(axes[0], [(2, 1, ('Z', 'Y')), (2, 0, ('Z', 'X')),
                                           (0, 1, ('X', 'Y'))]):
        draw_proj(ax, Apts, Bpts, other, rec['p1'], rec['p2'], vtx, ia, ib, lab)
    step = rec['slice_step']
    for ax, pl in zip(axes[1], sorted(rec['planes'], key=lambda p: p['plane'])):
        draw_plane(ax, pl, rec, step)

    fig.suptitle(
        'evt%s call%d  comp %d(blue,%dpts) vs %d(red,%dpts)   %s [%s]\n'
        'Lmin=%.1f Lmax=%.1f Tmax=%.1f npmin=%d ang=%.0f  gapUVW=%s excuse=%s '
        'wdeadX=%d dis=%.2f dens=%.0f dvtx=%.1f nedge=%d blk=%s'
        % (evt, call, j, len(Apts), k, len(Bpts), verdict.upper(), rule,
           pair['Lmin'], pair['Lmax'], pair['Tmax'], pair['npmin'],
           pair['angle'] if pair['angle'] is not None else -1,
           ''.join(str(int(g)) for g in rec['gap']),
           ''.join(str(int(e)) for e in rec['excuse']),
           pair['wdeadX'], pair['dis'], pair['dens'], pair['dvtx'],
           len(pair['edges']), rec['blk']), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    name = 'pair-evt%s-c%d-%d_%d-%s.png' % (evt, call, j, k, verdict)
    fig.savefig(os.path.join(outdir, name), dpi=95)
    plt.close(fig)
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', action='append', required=True)
    ap.add_argument('--first', type=int, default=0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--only', default='')
    ap.add_argument('--min-lmin', type=float, default=-1.0)
    ap.add_argument('--max', type=int, default=0)
    ap.add_argument('--params', default='')
    args = ap.parse_args()
    prm = A.parse_params(args.params)
    os.makedirs(args.out, exist_ok=True)
    n = 0
    for arm in args.arm:
        for evt, path in A.select_events(arm, args.first, ''):
            for pk, p in sorted(A.pair_table(arm, evt, path).items()):
                v = A.classify(p, prm)[0]
                if args.only and v != args.only:
                    continue
                if p['Lmin'] < args.min_lmin:
                    continue
                print(render(arm, evt, path, p, prm, args.out))
                n += 1
                if args.max and n >= args.max:
                    return
    print('rendered %d panels -> %s' % (n, args.out), file=sys.stderr)


if __name__ == '__main__':
    main()
