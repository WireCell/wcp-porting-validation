#!/usr/bin/env python3
'''Refit-drift base-rate probe (scratchpad, read-only): on accept-route
events, compare the ACCEPTED row's snapped position vs the RECORDED final
vertex (post snap_main_vertex_to_kink / improve_vertex).  Counts:
  harm  = row within tol, final outside tol   (refit drifted off truth)
  help  = row outside tol, final within tol   (refit recovered truth)
  moved = |final - row| distribution, split by harm/help/neutral
Decides whether a deterministic drift guard (revert refit if it moved the
vertex > X from the accepted candidate) has a viable base rate.
'''
import os
import sys
import numpy as np

HERE = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin/dl_vtx_training'
sys.path.insert(0, HERE)
from scn_vtx import io as vio
from taxonomy import ALL_TAGS
from rank_sim import load_events, manifest_sets

ARMS = ['vtxscan-prod0813=work-nuecc48-ma10-k20',
        'vtxscan-prod0813-ncpi0=work-ncpi0-ma10-k20',
        'vtxscan-prod0813-mcp1k=work-mcp1k-ma10-k20']
TOL = 1.0


def main():
    numu, _ = manifest_sets(os.path.join(HERE, 'data/full473/manifest.tsv'))
    roots = vio.parse_arm_roots(ARMS, vio.default_sbnd_root())
    evs = load_events(vio.default_sbnd_root(), ALL_TAGS, roots, TOL, numu)
    rows = []
    for e in evs:
        if 'F' not in e or not e['rec_accept'] or e['rec_winner_i'] is None:
            continue
        row = e['pos'][e['rec_winner_i']]
        moved = float(np.linalg.norm(e['rec'] - row))
        row_ok = bool(np.linalg.norm(row - e['truth']) <= TOL)
        rows.append((e['evt'], e['sample'], moved, row_ok, e['ok_rec']))
    harm = [r for r in rows if r[3] and not r[4]]
    help_ = [r for r in rows if not r[3] and r[4]]
    both = [r for r in rows if r[3] and r[4]]
    neither = [r for r in rows if not r[3] and not r[4]]
    print('accept-route events with recorded winner: %d' % len(rows))
    print('  row ok & final ok   : %3d' % len(both))
    print('  HARM (row ok, final wrong): %3d  %s'
          % (len(harm), [(r[0], '%.2fcm' % r[2]) for r in harm]))
    print('  HELP (row wrong, final ok): %3d  %s'
          % (len(help_), [(r[0], '%.2fcm' % r[2]) for r in help_]))
    print('  both wrong          : %3d' % len(neither))
    mv = np.array([r[2] for r in rows])
    print('moved dist: median %.3f  p90 %.3f  max %.3f cm'
          % (np.median(mv), np.percentile(mv, 90), mv.max()))
    # threshold scan: guard = revert to row when moved > X
    print('\nguard "revert refit if moved > X cm": net = harm_undone - help_undone')
    for x in (0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
        hu = sum(1 for r in harm if r[2] > x)
        pu = sum(1 for r in help_ if r[2] > x)
        n_act = sum(1 for r in rows if r[2] > x)
        print('  X=%.1f: acts on %3d evts, undoes %d harm, undoes %d help, net %+d'
              % (x, n_act, hu, pu, hu - pu))
    return 0


if __name__ == '__main__':
    sys.exit(main())
