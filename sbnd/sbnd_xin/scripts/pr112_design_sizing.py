#!/usr/bin/env python3
'''doc pr/112 sec 5.7 -- sizing the TRUE dual-chain design against the source.

Owner, 2026-08-23: "Can you design this approach and add the information into
the md file?  We will implement it later today."  Sec 5.7 is that design; this
script produces its four measured numbers.  Everything else in 5.7 is a source
read (file+line cited in the doc), not a measurement.

  --cost      the PR-wall multiplier of running a second, exclusion-free pass.
              Parsed from the per-stage "TaggerCheckNeutrino timing: ... took
              ... ms" lines already present in every arm's wct_pr_evt*.log.
              The OFF pass is priced through :2372 (it must run the vertex
              REFINEMENT block -- pr112_dualchain_sim.py reads the OFF arm's
              SHIPPED main_vertex, which is post-improve_vertex/mvga/stitch, so
              a design that stopped at determine_overall_main_vertex would not
              be the design that was measured).

  --xcluster  does the transfer land on a different cluster than production's
              own pick, and what does forbidding that cost?  vertex_id // 1000
              is the cluster id (verified against the scoreboard rows' own
              cluster_id field).

  --refine    how far the post-choice refinement block (:2310-2372) moves the
              vertex.  This is what makes 35/42 and 36/42 INJECTION-POINT
              numbers rather than predicted shipped numbers: the target metric
              is refinement-immune (it scores candidate identity) but the
              shipped POSITION is not.

TRAP: --cost accumulates by log pattern.  On a real dual-chain arm every timer
line appears TWICE and this will silently double-count.  Key on the pass, or
have the OFF pass prefix its timing lines, before re-using it there.

Usage: ./pr112_design_sizing.py --cost --xcluster --refine
'''
import argparse
import glob
import os
import re
import statistics as st
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(os.path.dirname(HERE), 'dl_vtx_training')
sys.path.insert(0, TRAIN)
from scn_vtx import io as vio      # noqa: E402

ON_ARM = 'work-vtx106-harv-base-%s'
OFF_ARM = 'work-vtx106-harv-nofitx-%s'
TAGS = {'nuecc48': ['vtxscan-harv3-nuecc48'], 'ncpi0': ['vtxscan-harv3-ncpi0'],
        'mcp1k': ['vtxscan-harv3-mcp1k']}

# The stages the OFF pass must run before determine_overall_main_vertex returns.
UPTO = ['preload_clusters', 'main_cluster initial PR', 'other_clusters PR',
        'deghosting', 'overall main vertex']
# The vertex-refinement block (snap_to_kink .. stitch_disconnected).  The timer
# spans :2339-2413, so it OVER-counts by the trailing clustering_points +
# reassociate_cluster_orphans -- which makes the projection an upper-middle.
REFINE_TIMER = 'improve_vertex + examine_direction'


def scan_timing(root, arm):
    out = {}
    for d in sorted(glob.glob(os.path.join(root, arm, 'pr_evt*'))):
        evt = os.path.basename(d)[6:]
        lg = os.path.join(d, 'wct_pr_evt%s.log' % evt)
        if not os.path.exists(lg):
            continue
        r = {'tcn': 0.0, 'mabc': 0.0, 'upto': 0.0, 'refine': 0.0, 'passes': 0}
        for line in open(lg, errors='replace'):
            m = re.search(r'MABC timing: TaggerCheckNeutrino:pr took ([0-9.]+) ms', line)
            if m:
                r['tcn'] = float(m.group(1))
            m = re.search(r'MABC timing: done took [0-9.]+ ms \(cumulative ([0-9.]+) ms\)', line)
            if m:
                r['mabc'] = float(m.group(1))
            if 'TaggerCheckNeutrino timing:' not in line:
                continue
            for pat in UPTO:
                m = re.search(re.escape(pat) + r' took ([0-9.]+) ms', line)
                if m:
                    r['upto'] += float(m.group(1))
            m = re.search(re.escape(REFINE_TIMER) + r' took ([0-9.]+) ms', line)
            if m:
                r['refine'] += float(m.group(1))
            if 'visit() TOTAL' in line:
                r['passes'] += 1
        if r['mabc']:
            out[evt] = r
    return out


def cost(root, sample):
    on = scan_timing(root, ON_ARM % sample)
    off = scan_timing(root, OFF_ARM % sample)
    ev = sorted(set(on) & set(off))
    if not ev:
        print('cost: no timing data'); return
    print('=== sec 5.7.6 cost, n=%d (%s) ===' % (len(ev), sample))
    for nm, a in (('production   ', on), ('exclusion-free', off)):
        print('  %s TCN visit median %7.0f ms   PR job median %6.2f s   arm total %6.1f s'
              % (nm, st.median([a[e]['tcn'] for e in ev]),
                 st.median([a[e]['mabc'] for e in ev]) / 1000,
                 sum(a[e]['mabc'] for e in ev) / 1000))
    print('  OFF visit / ON visit, per event: median %.2fx'
          % st.median([off[e]['tcn'] / on[e]['tcn'] for e in ev]))
    print('  OFF up-to-vertex share of its own visit: median %.1f%%'
          % (st.median([off[e]['upto'] / off[e]['tcn'] for e in ev]) * 100))
    mid = [(on[e]['mabc'] + off[e]['upto'] + off[e]['refine']) / on[e]['mabc'] for e in ev]
    hi = [(on[e]['mabc'] + off[e]['tcn']) / on[e]['mabc'] for e in ev]
    tcn = [(on[e]['tcn'] + off[e]['upto'] + off[e]['refine']) / on[e]['tcn'] for e in ev]
    print('  PROJECTED dual chain (OFF runs :2145-2372):')
    print('    PR-job wall  median %.2fx  mean %.2fx  max %.2fx' % (st.median(mid), st.mean(mid), max(mid)))
    print('    TCN stage    median %.2fx  mean %.2fx  max %.2fx' % (st.median(tcn), st.mean(tcn), max(tcn)))
    tot_on = sum(on[e]['mabc'] for e in ev) / 1000
    tot_du = sum(on[e]['mabc'] + off[e]['upto'] + off[e]['refine'] for e in ev) / 1000
    print('    arm total    %.0f s -> %.0f s  (%.2fx on totals)' % (tot_on, tot_du, tot_du / tot_on))
    print('    upper bound (OFF runs the whole visit): median %.2fx mean %.2fx max %.2fx'
          % (st.median(hi), st.mean(hi), max(hi)))
    np_ = [on[e]['passes'] for e in ev]
    print('  candidate PR passes per event (production): %s'
          % {k: np_.count(k) for k in sorted(set(np_))})


def _board(sb):
    c = sb.get('hv_cloud')
    if not c or not int(c.get('n_vertex_rows', 0)):
        return None, None
    n = int(c['n_vertex_rows'])
    return c['vertex_ids'][:n], np.array([c['x'][:n], c['y'][:n], c['z'][:n]], float).T


def _final(j):
    mv = j.get('main_vertex') or {}
    if mv.get('x') is not None:
        return np.array([mv['x'], mv['y'], mv['z']], float)
    sb = j.get('vertex_scoreboard') or {}
    if sb.get('filled'):
        return np.array([sb['final_x'], sb['final_y'], sb['final_z']], float)
    return None


def _pairs(root, sample):
    for lab in vio.iter_labels(root, TAGS[sample]):
        e = int(lab['eventNo'])
        pon = os.path.join(root, ON_ARM % sample, 'pr_evt%d' % e, 'calib-pr-evt%d.json' % e)
        pof = os.path.join(root, OFF_ARM % sample, 'pr_evt%d' % e, 'calib-pr-evt%d.json' % e)
        if not (os.path.exists(pon) and os.path.exists(pof)):
            continue
        yield e, lab, vio.load_calib(pon), vio.load_calib(pof)


def xcluster(root, sample):
    rows = []
    for e, lab, jon, jof in _pairs(root, sample):
        ids, xyz = _board(jon.get('vertex_scoreboard') or {})
        ido, _ = _board(jof.get('vertex_scoreboard') or {})
        if ids is None or ido is None:
            continue
        p_on, p_of = _final(jon), _final(jof)
        if p_on is None or p_of is None:
            continue
        tr = np.asarray(lab['truth_xyz'], float)
        target = ids[int(np.argmin(np.linalg.norm(xyz - tr, axis=1)))]
        pick_prod = ids[int(np.argmin(np.linalg.norm(xyz - p_on, axis=1)))]
        d = np.linalg.norm(xyz - p_of, axis=1)
        pick_dual = ids[int(np.argmin(d))]
        cl = pick_prod // 1000
        m = np.array([i // 1000 == cl for i in ids])
        if m.any():
            dd = np.where(m, d, np.inf)
            i = int(np.argmin(dd)); pick_rest, d_rest = ids[i], float(dd[i])
        else:
            pick_rest, d_rest = pick_prod, 0.0
        rows.append(dict(evt=e, target=target, prod=pick_prod, dual=pick_dual,
                         rest=pick_rest, cl_prod=cl, cl_dual=pick_dual // 1000,
                         d=float(d.min()), d_rest=d_rest))
    n = len(rows)
    print('\n=== sec 5.7.8 cross-cluster transfer, n=%d (%s) ===' % (n, sample))
    xc = [r for r in rows if r['cl_prod'] != r['cl_dual']]
    print('  transfer lands on a DIFFERENT cluster on %d/%d events' % (len(xc), n))
    for r in xc:
        print('    evt %-7d prod cluster %d -> %d at %.2f cm   prod %s / dual %s'
              % (r['evt'], r['cl_prod'], r['cl_dual'], r['d'],
                 'RIGHT' if r['prod'] == r['target'] else 'wrong',
                 'RIGHT' if r['dual'] == r['target'] else 'wrong'))
    for nm, k in (('production', 'prod'), ('dual, any cluster', 'dual'),
                  ('dual, same-cluster only', 'rest')):
        print('  %-26s %d/%d' % (nm, sum(r[k] == r['target'] for r in rows), n))
    print('  guard sweep, same-cluster-restricted:')
    for D in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 99):
        h = x = f = b = 0
        for r in rows:
            use = r['d_rest'] <= D
            pick = r['rest'] if use else r['prod']
            h += pick == r['target']; x += use
            if use and pick == r['target'] and r['prod'] != r['target']:
                f += 1
            if use and pick != r['target'] and r['prod'] == r['target']:
                b += 1
        print('    D=%-5s hit %2d/%d  xfer %2d  fixed %d broken %d' % (D, h, n, x, f, b))


def refine(root, sample):
    print('\n=== sec 5.7.7(b) post-choice refinement displacement (%s) ===' % sample)
    for arm, lbl in ((ON_ARM, 'production ON'), (OFF_ARM, 'exclusion-free')):
        ds = []
        for lab in vio.iter_labels(root, TAGS[sample]):
            e = int(lab['eventNo'])
            p = os.path.join(root, arm % sample, 'pr_evt%d' % e, 'calib-pr-evt%d.json' % e)
            if not os.path.exists(p):
                continue
            j = vio.load_calib(p)
            ids, xyz = _board(j.get('vertex_scoreboard') or {})
            f = _final(j)
            if ids is None or f is None:
                continue
            ds.append(float(np.min(np.linalg.norm(xyz - f, axis=1))))
        ds = np.array(ds)
        print('  %-15s n=%d  median %.3f  p90 %.2f  max %.2f cm  ; >1 cm on %d'
              % (lbl, len(ds), np.median(ds), np.percentile(ds, 90), ds.max(), int((ds > 1).sum())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', default='nuecc48')
    ap.add_argument('--cost', action='store_true')
    ap.add_argument('--xcluster', action='store_true')
    ap.add_argument('--refine', action='store_true')
    a = ap.parse_args()
    if not (a.cost or a.xcluster or a.refine):
        a.cost = a.xcluster = a.refine = True
    root = vio.default_sbnd_root()
    if a.cost:
        cost(root, a.sample)
    if a.xcluster:
        xcluster(root, a.sample)
    if a.refine:
        refine(root, a.sample)


if __name__ == '__main__':
    main()
