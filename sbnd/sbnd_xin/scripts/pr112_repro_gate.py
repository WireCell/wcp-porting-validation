#!/usr/bin/env python3
'''doc pr/112 -- reproduction gate for the arms regenerated after the
2026-08-23 retire campaign.

pr/106 sec 9/10 and all of pr/111 were measured on work-vtx105-* /
work-vtx106-*.  The retire campaign deleted 380 arms; only vtx105-base and
four vtx106 nueCC48 arms survived, and EVERY DL-off arm is gone.  The owner
authorised regenerating what this round needs (2026-08-23).

Before any regenerated arm is used interchangeably with the retired one, it
has to be shown to BE the retired one.  This compares work-pr112-harv-<s>
against the surviving work-vtx106-harv-base-<s> on the quantities every pr112
script actually reads:

  * the harvested SCN input cloud  (hv_cloud x/y/z/q, exact float compare) --
    pr/79 sec 10 states offline voxelization from these floats reproduces the
    live network input bit-exactly, so any drift here invalidates everything;
  * the candidate set (n_vertex_rows, vertex_ids, in order);
  * the live route and the DL winner;
  * the final main_vertex position.

A binary difference would show up first in the cloud.  Note the installed
libWireCellClus.so predates HEAD b5c9f43a by 8 minutes; b5c9f43a is the
env-gated WCT_EXCL_DUMP debug dump (no code path when unset), so it cannot
change these outputs -- this gate is the empirical check of that claim.

Usage: ./pr112_repro_gate.py --sample nuecc48
'''
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load(arm, evt):
    p = os.path.join(ROOT, arm, 'pr_evt%s' % evt, 'calib-pr-evt%s.json' % evt)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', default='nuecc48')
    ap.add_argument('--new', default='work-pr112-harv-%s')
    ap.add_argument('--old', default='work-vtx106-harv-base-%s')
    a = ap.parse_args()
    new, old = a.new % a.sample, a.old % a.sample
    if not os.path.isdir(os.path.join(ROOT, old)):
        print('reference arm %s is GONE -- gate cannot run' % old)
        return 2
    evts = sorted(d.replace('pr_evt', '')
                  for d in os.listdir(os.path.join(ROOT, new))
                  if d.startswith('pr_evt'))
    n = same = 0
    diffs = []
    for e in evts:
        A, B = load(new, e), load(old, e)
        if A is None or B is None:
            continue
        n += 1
        sa = (A.get('vertex_scoreboard') or {})
        sb = (B.get('vertex_scoreboard') or {})
        ca, cb = sa.get('hv_cloud'), sb.get('hv_cloud')
        why = []
        if not ca or not cb:
            why.append('missing hv_cloud')
        else:
            for k in ('x', 'y', 'z', 'q'):
                va = np.asarray(ca[k], np.float32)
                vb = np.asarray(cb[k], np.float32)
                if va.shape != vb.shape:
                    why.append('cloud %s len %d vs %d' % (k, va.size, vb.size)); break
                if not np.array_equal(va, vb):
                    why.append('cloud %s max|d|=%.3e' % (k, float(np.abs(va - vb).max())))
            if ca.get('n_vertex_rows') != cb.get('n_vertex_rows'):
                why.append('n_vertex_rows %s vs %s'
                           % (ca.get('n_vertex_rows'), cb.get('n_vertex_rows')))
            elif ca.get('vertex_ids') != cb.get('vertex_ids'):
                why.append('vertex_ids differ')
        if sa.get('route') != sb.get('route'):
            why.append('route %s vs %s' % (sa.get('route'), sb.get('route')))
        wa = next((r['vertex_id'] for r in (sa.get('rows') or []) if r.get('dl_winner')), None)
        wb = next((r['vertex_id'] for r in (sb.get('rows') or []) if r.get('dl_winner')), None)
        if wa != wb:
            why.append('dl_winner %s vs %s' % (wa, wb))
        ma, mb = (A.get('main_vertex') or {}), (B.get('main_vertex') or {})
        if ma.get('x') is not None and mb.get('x') is not None:
            d = float(np.linalg.norm(np.array([ma['x'], ma['y'], ma['z']])
                                     - np.array([mb['x'], mb['y'], mb['z']])))
            if d > 1e-6:
                why.append('main_vertex moved %.4f cm' % d)
        if why:
            diffs.append((e, '; '.join(why)))
        else:
            same += 1
    print('GATE %s vs %s : %d/%d events identical' % (new, old, same, n))
    for e, w in diffs:
        print('   evt %-8s %s' % (e, w))
    return 0 if same == n and n else 1


if __name__ == '__main__':
    sys.exit(main())
