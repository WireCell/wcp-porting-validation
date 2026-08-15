#!/usr/bin/env python3
'''
doc pr/79 step 6 (case campaign round 1) -- CASE CENSUS over the k20
selection-wrong events.  Read-only over the recorded k20 arm calibs.

Input: runs/ab-ma10k20-marg-20260815.tsv (cls_new/route_new -- the live k20
taxonomy; NOT taxonomy-20260815.tsv, which is the prod0813/k5 arm) and
runs/rankfit-k20-20260815.tsv (total_argmax_ok -- acceptance-wrong vs
chooser-wrong triage).

Per event: where truth sits on the scoreboard (nearest usable row), where the
final vertex sits, whether the traditional stage-1 scorer ever saw the truth
vertex, whether the miss crosses clusters, and the acceptance margin.

Usage:
  python3 case_census.py \
      --marg runs/ab-ma10k20-marg-20260815.tsv \
      --rankfit runs/rankfit-k20-20260815.tsv \
      --out ../docs/pr/79_case_census.tsv
'''
import argparse
import csv
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scn_vtx import io as vio
from taxonomy import ALL_TAGS
from rank_sim import manifest_sets

ARMS = ['vtxscan-prod0813=work-nuecc48-ma10-k20',
        'vtxscan-prod0813-ncpi0=work-ncpi0-ma10-k20',
        'vtxscan-prod0813-mcp1k=work-mcp1k-ma10-k20']

COLS = ['evt', 'sample', 'route', 'd_final', 'acc_wrong', 'margin',
        'n_rows', 'n_usable', 'n_trad_scored',
        't_vid', 't_clus', 't_d', 't_rank', 't_dl', 't_snap', 't_total',
        't_trad_scored', 't_trad_score', 't_trad_winner',
        'f_vid', 'f_clus', 'f_in_rows', 'f_trad_scored', 'f_trad_score',
        'f_dl_snapped', 'f_total',
        'cluster_swap', 'truth_unscored', 'truth_scored_lost', 'far_miss']


def census_event(label, calib, route, argmax_ok, sample):
    sb = calib.get('vertex_scoreboard') or {}
    rows = sb.get('rows') or []
    usable = [r for r in rows
              if r.get('dl_snapped') and not r.get('skipped_by_swap_guard')]
    truth = np.asarray(label['truth_xyz'], np.float64)
    mv = calib.get('main_vertex') or {}
    fin = np.array([mv.get('x', np.nan), mv.get('y', np.nan),
                    mv.get('z', np.nan)], np.float64)
    d_final = float(np.linalg.norm(fin - truth))

    def d_row(r):
        return float(np.linalg.norm(
            np.array([r['x'], r['y'], r['z']]) - truth))

    t = min(usable, key=d_row) if usable else None
    f_vid = sb.get('final_vertex_id', -1)
    f = next((r for r in rows if r.get('vertex_id') == f_vid), None)
    best_total = max((float(r['total']) for r in usable), default=float('nan'))
    min_acc = float(sb.get('dl_min_accept_score', float('nan')))

    rec = dict(evt=label['eventNo'], sample=sample, route=route,
               d_final=round(d_final, 2), acc_wrong=int(argmax_ok),
               margin=round(min_acc - best_total, 3),
               n_rows=len(rows), n_usable=len(usable),
               n_trad_scored=sum(1 for r in rows if r.get('trad_scored')))
    if t is not None:
        rec.update(t_vid=t['vertex_id'], t_clus=t['cluster_id'],
                   t_d=round(d_row(t), 2), t_rank=t['voxel_rank'],
                   t_dl=round(float(t['dl_score']), 3),
                   t_snap=round(float(t['snap_dis']), 2),
                   t_total=round(float(t['total']), 3),
                   t_trad_scored=int(bool(t.get('trad_scored'))),
                   t_trad_score=round(float(t.get('trad_score', 0.0)), 3),
                   t_trad_winner=int(bool(t.get('trad_winner'))))
    else:
        rec.update({k: '' for k in COLS[11:19]})
    if f is not None:
        rec.update(f_vid=f_vid, f_clus=f['cluster_id'], f_in_rows=1,
                   f_trad_scored=int(bool(f.get('trad_scored'))),
                   f_trad_score=round(float(f.get('trad_score', 0.0)), 3),
                   f_dl_snapped=int(bool(f.get('dl_snapped'))),
                   f_total=round(float(f.get('total', 0.0)), 3))
    else:
        rec.update(f_vid=f_vid, f_clus='', f_in_rows=0, f_trad_scored='',
                   f_trad_score='', f_dl_snapped='', f_total='')
    rec['cluster_swap'] = (int(t['cluster_id'] != f['cluster_id'])
                           if (t is not None and f is not None) else '')
    rec['truth_unscored'] = (int(not t.get('trad_scored'))
                             if t is not None else '')
    rec['truth_scored_lost'] = (int(bool(t.get('trad_scored'))
                                    and not t.get('trad_winner'))
                                if t is not None else '')
    rec['far_miss'] = int(d_final > 20.0)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--marg', default='runs/ab-ma10k20-marg-20260815.tsv')
    ap.add_argument('--rankfit', default='runs/rankfit-k20-20260815.tsv')
    ap.add_argument('--out', default='../docs/pr/79_case_census.tsv')
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)

    sel = {}
    with open(args.marg) as fh:
        for row in csv.DictReader(fh, delimiter='\t'):
            if row['cls_new'] == 'selection-wrong':
                sel[int(row['evt'])] = row['route_new']
    argmax_ok = {}
    with open(args.rankfit) as fh:
        for row in csv.DictReader(fh, delimiter='\t'):
            if row['evt'].startswith('#') or row.get('total_argmax_ok') is None:
                continue   # trailing "# weights:" comment line
            argmax_ok[int(row['evt'])] = int(row['total_argmax_ok'])

    numu, _ = manifest_sets(os.path.join(here, 'data/full473/manifest.tsv'))
    roots = vio.parse_arm_roots(ARMS, vio.default_sbnd_root())
    recs = []
    for label in vio.iter_labels(vio.default_sbnd_root(), ALL_TAGS):
        evt = label['eventNo']
        if evt not in sel:
            continue
        path = vio.calib_path_in_roots(roots, label)
        calib = vio.load_calib(path)
        recs.append(census_event(label, calib, sel[evt],
                                 argmax_ok.get(evt, 0),
                                 vio.sample_of_label(label, numu, 'numu')))
    recs.sort(key=lambda r: (r['route'], -r['acc_wrong'], r['evt']))

    with open(args.out, 'w') as fh:
        fh.write('\t'.join(COLS) + '\n')
        for r in recs:
            fh.write('\t'.join(str(r[c]) for c in COLS) + '\n')
    print('wrote %s  (%d events)' % (args.out, len(recs)))

    rej = [r for r in recs if r['route'] == 'dl-rerank-reject']
    acc = [r for r in recs if r['route'] == 'dl-rerank-accept']
    print('\nreject-route %d | accept-route %d' % (len(rej), len(acc)))
    for name, sub in (('reject', rej), ('accept', acc)):
        n = len(sub)
        agg = {
            'acc_wrong (argmax=truth, gate rejected)':
                sum(r['acc_wrong'] for r in sub),
            'cluster_swap (final on other cluster than truth row)':
                sum(1 for r in sub if r['cluster_swap'] == 1),
            'truth_unscored (stage-1 never scored truth vertex)':
                sum(1 for r in sub if r['truth_unscored'] == 1),
            'truth_scored_lost (scored, lost trad argmax)':
                sum(1 for r in sub if r['truth_scored_lost'] == 1),
            'truth_trad_winner (trad picked truth, then lost it!)':
                sum(1 for r in sub if r['t_trad_winner'] == 1),
            'final_not_in_rows':
                sum(1 for r in sub if r['f_in_rows'] == 0),
            'far_miss (>20cm)': sum(r['far_miss'] for r in sub),
        }
        print('-- %s (%d):' % (name, n))
        for k, v in agg.items():
            print('   %-52s %3d' % (k, v))
    return 0


if __name__ == '__main__':
    sys.exit(main())
