#!/usr/bin/env python3
'''
doc pr/78 round 3 -- pre-DL diagnosis of the candidate-missing taxonomy class.

For every labeled event whose truth vertex has NO scoreboard-row candidate
within tolerance (the class no amount of net training or re-ranking can fix),
measure where the truth sits relative to the PR graph:

  d_row   truth -> nearest scoreboard row candidate      (taxonomy's d_cand)
  d_vtx   truth -> nearest PR-graph vertex fit point     (candidate SOURCE)
  d_seg   truth -> nearest segment fit point (any, incl. interior)
  d_cloud truth -> nearest DL input-cloud point

Buckets (tol = --tol, default 1 cm; "near" = within --near, default 2 cm):
  admission-gap : a PR-graph VERTEX exists near truth but no row candidate
                  -> the candidate list, not the graph, dropped it
  on-track      : truth is near segment points but no vertex there
                  -> vertex proposal / segment breaking territory
  off-graph     : truth is far from every graph point -> pr/51 graph work

Usage: python3 pre_dl_diag.py --taxonomy runs/taxonomy-20260815.tsv \
           --tsv runs/pre-dl-diag-20260815.tsv
'''
import argparse
import csv
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scn_vtx import io as vio


def graph_distances(calib, truth):
    vtx = []
    for v in calib.get('vertices', []):
        f = v.get('fit') or {}
        if f.get('x') is not None:
            vtx.append([f['x'], f['y'], f['z']])
    seg = []
    for s in calib.get('segments', []):
        for p in s.get('points', []):
            if p.get('x') is not None:
                seg.append([p['x'], p['y'], p['z']])
    xyz, q, _ = vio.rebuild_cloud(calib)
    def near(arr):
        if not len(arr):
            return float('inf')
        return float(np.min(np.linalg.norm(np.asarray(arr, dtype=float)
                                           - np.asarray(truth), axis=1)))
    return near(vtx), near(seg), near(xyz if len(xyz) else [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--taxonomy', required=True)
    ap.add_argument('--sbnd-root', default=vio.default_sbnd_root())
    ap.add_argument('--tol', type=float, default=1.0)
    ap.add_argument('--near', type=float, default=2.0)
    ap.add_argument('--cls', default='candidate-missing')
    ap.add_argument('--tsv', default=None)
    args = ap.parse_args()

    with open(args.taxonomy) as fh:
        tax = {int(r['evt']): r for r in csv.DictReader(fh, delimiter='\t')}
    wanted = {e for e, r in tax.items() if r['cls'] == args.cls}

    labels = {}
    for lab in vio.iter_labels(args.sbnd_root, [
            'vtxscan-prod0813', 'vtxscan-prod0813-ncpi0',
            'vtxscan-prod0813-mcp1k']):
        if lab['eventNo'] in wanted:
            labels[lab['eventNo']] = lab

    recs = []
    for evt in sorted(wanted):
        lab = labels[evt]
        calib = vio.load_calib(vio.calib_path_for_label(args.sbnd_root, lab))
        d_vtx, d_seg, d_cloud = graph_distances(calib, lab['truth_xyz'])
        d_row = float(tax[evt]['d_cand'])
        if d_vtx <= args.near:
            bucket = 'admission-gap'
        elif d_seg <= args.near:
            bucket = 'on-track'
        else:
            bucket = 'off-graph'
        recs.append(dict(evt=evt, sample=tax[evt]['sample'],
                         route=tax[evt]['route'], d_row='%.2f' % d_row,
                         d_vtx='%.2f' % d_vtx, d_seg='%.2f' % d_seg,
                         d_cloud='%.2f' % d_cloud, bucket=bucket))
        print('  evt%-8d %-7s d_row=%7s d_vtx=%7s d_seg=%7s d_cloud=%7s  %s'
              % (evt, recs[-1]['sample'], recs[-1]['d_row'], recs[-1]['d_vtx'],
                 recs[-1]['d_seg'], recs[-1]['d_cloud'], bucket))

    print('\n== %s (%d events, near=%.1f cm) ==' % (args.cls, len(recs), args.near))
    from collections import Counter
    for (b, s), n in sorted(Counter((r['bucket'], r['sample'])
                                    for r in recs).items()):
        print('  %-14s %-7s %3d' % (b, s, n))
    print('  totals:', dict(Counter(r['bucket'] for r in recs)))

    if args.tsv:
        cols = ['evt', 'sample', 'route', 'd_row', 'd_vtx', 'd_seg',
                'd_cloud', 'bucket']
        with open(args.tsv, 'w') as fh:
            fh.write('\t'.join(cols) + '\n')
            for r in recs:
                fh.write('\t'.join(str(r[c]) for c in cols) + '\n')
        print('wrote', args.tsv)
    return 0


if __name__ == '__main__':
    sys.exit(main())
