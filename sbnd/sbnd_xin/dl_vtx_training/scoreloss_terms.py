#!/usr/bin/env python3
'''
doc pr/79 §9 (case campaign round 1) -- per-term decomposition of the
stage-1 compare_main_vertices score for the SCORE-LOSS class: truth row and
winner row on the same cluster, truth scored but lost.  Uses the harvest
arm's hv_* fields.

Reconstructable terms (compare_main_vertices blocks, NeutrinoVertexFinder.cxx):
  proton  = -(in-out)/4            [in>out]
          = -(in-out)/4 + (in+out)/8   [else]
  z       = -hv_z_prior
  segs    = 0.25*hv_n_tracks + 0.125*hv_n_showers   (base part)
  fv      = 0.5 if hv_in_fv
  confl   = -hv_conflicts/4
  bonus   = trad_score - (proton+z+segs+fv+confl)
            = the unrecorded direction/clear-proton/long-muon/daughter-shower
              bonuses, lumped.
'''
import os
import sys
import csv
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scn_vtx import io as vio

HERE = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(HERE)

EVENTS = {  # evt -> (t_vid,) from the census; winner = trad_winner row on t_clus
    351749: 14000, 285467: 18000, 411460: 6000, 315849: 11000, 402880: 9004,
    166804: 9003, 404684: 9002, 56211: 11001, 291064: 14001, 63359: 3001,
}
ARMS = {'mcp1k': 'work-mcp1k-ma10k20-harv2', 'ncpi0': 'work-ncpi0-ma10k20-harv2',
        'nuecc': 'work-nuecc48-ma10k20-harv2'}


def terms(r):
    pin, pout = r['hv_n_proton_in'], r['hv_n_proton_out']
    if pin > pout:
        proton = -(pin - pout) / 4.0
    else:
        proton = -(pin - pout) / 4.0 + (pin + pout) / 8.0
    z = -r['hv_z_prior']
    segs = 0.25 * r['hv_n_tracks'] + 0.125 * r['hv_n_showers']
    fv = 0.5 if r['hv_in_fv'] else 0.0
    confl = -r['hv_conflicts'] / 4.0 if r['hv_conflicts'] >= 0 else 0.0
    bonus = r['trad_score'] - (proton + z + segs + fv + confl)
    return dict(proton=proton, z=z, segs=segs, fv=fv, confl=confl,
                bonus=bonus, total=r['trad_score'])


def main():
    cen = {int(r['evt']): r for r in csv.DictReader(
        open(os.path.join(SX, 'docs/pr/79_case_census.tsv')), delimiter='\t')}
    for evt, tvid in EVENTS.items():
        r = cen[evt]
        arm = ARMS.get(r['sample'], ARMS['mcp1k'])
        path = os.path.join(SX, arm, 'pr_evt%d' % evt, 'calib-pr-evt%d.json' % evt)
        if not os.path.exists(path):
            print('evt%-8d (pending harvest)' % evt)
            continue
        sb = vio.load_calib(path)['vertex_scoreboard']
        rows = {x['vertex_id']: x for x in sb['rows']}
        t = rows.get(tvid)
        tclus = tvid // 1000
        win = next((x for x in sb['rows']
                    if x['cluster_id'] == tclus and x.get('trad_winner')), None)
        if not t or not win or not t.get('hv_filled') or not win.get('hv_filled'):
            print('evt%-8d rows missing hv (t=%s win=%s)' % (evt, bool(t), bool(win)))
            continue
        tt, wt = terms(t), terms(win)
        gap = wt['total'] - tt['total']
        drivers = sorted(((k, wt[k] - tt[k]) for k in
                          ('proton', 'z', 'segs', 'fv', 'confl', 'bonus')),
                         key=lambda kv: -abs(kv[1]))
        print('evt%-8d truth v%-6d %.3f  vs winner v%-6d %.3f  gap %+0.3f | drivers: %s'
              % (evt, tvid, tt['total'], win['vertex_id'], wt['total'], gap,
                 '  '.join('%s %+0.3f' % kv for kv in drivers if abs(kv[1]) > 1e-9)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
