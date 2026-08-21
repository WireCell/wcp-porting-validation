#!/usr/bin/env python3
'''doc pr/106 -- target-anchored evaluation of the DL main-vertex SELECTION.

Owner (2026-08-21): the hand-scan labels were taken on fit_exclusion-OFF
reconstructions, so a 1 cm match of the CURRENT fitted vertex to the click
(pr/105's ruler) is biased against the current sample.  Instead: on the
current topology define the TARGET = the pre-DL candidate vertex closest to
the click (no cut), and score a selection method by whether it PICKS the
target.  Pure selection metric, immune to the fit epoch.

Candidate set = the dl_vtx_harvest cloud's vertex rows (hv_cloud, the EXACT
live SCN input at DL time: doc pr/79 sec 10), NOT vertices[] (post-refit;
ids renumber, 5/124 missing on nueCC48 163543).  pr75 id = cluster*1000+idx.

Consistency contract (owner): the offline decision space is exactly the live
one -- only dl_snapped rows, the dumped s_* terms (s_dl = dl_score*scale and
s_topo = w*(frac-center) are the two knob-linear terms), the threshold rule,
the pr/48 protected veto as dumped.  A REJECT resolves to the live fallback
vertex of an arm that rejected on that event (trad arm = dl-not-run covers
every event); an uncovered decision counts WRONG.  --closure must be 0 on
every live arm before any number is quoted.

Repro (doc pr/106 sec 0):
  ./vtx_target_eval.py --carried-tags vtxscan-vtx105-{nuecc48,ncpi0,mcp1k,delta,mcp2k,mcp2k-auto,mcp2k-ragree} \\
      --orig-tags vtxscan-harv3-{nuecc48,ncpi0,mcp1k,delta} vtxscan-mcp2k vtxscan-mcp2k-auto vtxscan-mcp2k-ragree \\
      --exclude-events runs/vtx106/lockbox.txt --closure --table --events-tsv out.tsv
'''
import argparse
import collections
import csv
import json
import math
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from vtx_strategy_table import load_truth, read_event_file  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, '..'))
GEO = ['s_snap', 's_fwd_z', 's_clen', 's_isol', 's_main', 's_fv']
PROD = dict(w_snap=1.0, w_fwd_z=1.0, w_clen=1.0, w_isol=1.0, w_main=1.0,
            w_fv=1.0, w_topo=0.0, center=0.0, min_accept=10.0, scale=1000.0)
TOPO3 = dict(PROD, w_topo=3.0, center=0.0)
ACCEPT = ('dl-rerank-accept', 'dl-legacy-accept')
VETO = 'dl-veto-protected'
DL_VTX_CUT_CM = 2.5
SAMPLES = ('nuecc48', 'ncpi0', 'mcp1k', 'mcp2k')
PICK_TOL_CM = 1.5   # vertex-identity tolerance when a final id left the cloud
DEFAULT_LIVE = ['base=work-vtx105-base-{sample}', 'topo3=work-vtx105-topo3-{sample}',
                'dlonly=work-vtx105-dlonly-{sample}', 'ma4=work-vtx105-ma4-{sample}',
                'topk10=work-vtx105-topk10-{sample}', 'trad=work-vtx105-trad-{sample}']
SWEEP = dict(
    w_snap=[0, 0.5, 1, 1.5, 2, 3], w_fwd_z=[0, 0.5, 1, 1.5, 2, 3],
    w_clen=[0, 0.5, 1, 1.5, 2, 3], w_isol=[0, 0.5, 1, 1.5, 2, 3],
    w_main=[0, 0.5, 1, 1.5, 2, 3], w_fv=[0, 0.5, 1, 1.5, 2, 3],
    w_topo=[0, 1, 2, 3, 5], center=[0, 0.25, 0.5],
    min_accept=[0, 4, 6, 8, 10, 12, 15, 20], scale=[500, 750, 1000, 1500, 2000])


def calib(arm_dir, sample, evt):
    p = os.path.join(ROOT, arm_dir.format(sample=sample), 'pr_evt%d' % evt,
                     'calib-pr-evt%d.json' % evt)
    return json.load(open(p)) if os.path.exists(p) else None


def board_rows(sb):
    rows = []
    for r in sb.get('rows', []):
        if not r.get('dl_snapped'):
            continue
        rows.append(dict(
            vid=r['vertex_id'], dl=float(r['dl_score']),
            geo={t: float(r.get(t) or 0.0) for t in GEO},
            frac=float(r.get('topo_frac', -1.0)), votes=int(r.get('topo_votes') or 0),
            voxel_rank=int(r['voxel_rank']), snap_dis=float(r.get('snap_dis') or 0.0),
            total=float(r.get('total') or 0.0), xyz=(r['x'], r['y'], r['z'])))
    return rows


PICK_TOL_LOOSE_CM = 3.0   # second tier: accepted only if unambiguous (next candidate > 1 cm farther)
RES = collections.Counter()   # resolution tiers, printed once


def resolve_pick(sb, cloud_ids, cloud_xyz):
    '''live final vertex -> pre-DL cloud vertex id.  Accept routes pass the exact
    dl_winner id; otherwise final_vertex_id if still a cloud id (post-refit
    renumbering is common), else nearest cloud vertex by position: <= 1.5 cm,
    or <= 3 cm when the runner-up is > 1 cm farther (improve_vertex moves the
    fallback vertex by a few cm).  None = unresolved (counts as a miss).'''
    if sb['route'] in ACCEPT:
        win = next((r['vertex_id'] for r in sb.get('rows', []) if r.get('dl_winner')), None)
        if win is not None and win in set(cloud_ids):
            RES['accept-id'] += 1
            return win, 0.0
    fid = sb.get('final_vertex_id')
    if fid in set(cloud_ids):
        RES['final-id'] += 1
        return fid, 0.0
    f = np.array([sb['final_x'], sb['final_y'], sb['final_z']])
    d = np.linalg.norm(cloud_xyz - f, axis=1)
    order = np.argsort(d)
    i = int(order[0])
    d2 = float(d[order[1]]) if len(order) > 1 else 1e9
    ids = list(cloud_ids)
    if d[i] <= PICK_TOL_CM:
        RES['pos<=1.5'] += 1
        return ids[i], float(d[i])
    if d[i] <= PICK_TOL_LOOSE_CM and d2 > d[i] + 1.0:
        RES['pos<=3-unambiguous'] += 1
        return ids[i], float(d[i])
    RES['unresolved'] += 1
    return None, float(d[i])


class Event:
    pass


def load_events(args):
    truth = load_truth(args.sbnd_root, args.carried_tags, args.orig_tags)
    excl = read_event_file(args.exclude_events) or set()
    only = read_event_file(args.only_events)
    ipw = {}
    if args.ipw_tsv and os.path.exists(args.ipw_tsv):
        for r in csv.DictReader(open(args.ipw_tsv), delimiter='\t'):
            ipw[int(r['evt'])] = float(r['weight'])
    live = [a.split('=', 1) for a in args.live_arms]
    events, skipped = [], collections.Counter()
    for e in sorted(truth):
        if e in excl or (only is not None and e not in only):
            continue
        t = truth[e]
        s = 'nuecc48' if t['sample'] == 'nuecc' else t['sample']
        hb = calib(args.harv_base, s, e)
        hr = calib(args.harv_rows, s, e)
        if hb is None or hr is None:
            skipped['no-harvest-dump'] += 1
            continue
        sb = hb['vertex_scoreboard']
        c = sb['hv_cloud']
        nv = c['n_vertex_rows']
        if nv == 0:                       # net never ran (empty cloud): no candidate set
            skipped['empty-cloud'] += 1
            continue
        ids = c['vertex_ids'][:nv]
        xyz = np.array([c['x'][:nv], c['y'][:nv], c['z'][:nv]], dtype=float).T
        ev = Event()
        ev.evt, ev.sample, ev.src, ev.carried = e, s, t['label_source'], t['carried']
        ev.w = ipw.get(e, 1.0)
        ev.cloud_ids, ev.cloud_xyz = ids, xyz
        ev.idset = ids
        tr = np.array(t['truth'], dtype=float)
        d = np.linalg.norm(xyz - tr, axis=1)
        i = int(np.argmin(d))
        ev.target, ev.d_target = ids[i], float(d[i])
        ev.rows = board_rows(hr['vertex_scoreboard'])
        rb = {r['vid']: r for r in board_rows(sb)}
        # rows closure: same snapped set and same dl_score in both harvest arms
        ev.rows_mismatch = (set(rb) != set(r['vid'] for r in ev.rows) or
                            any(abs(rb[r['vid']]['dl'] - r['dl']) > 1e-9 for r in ev.rows))
        ev.admitted5 = ev.target in set(r['vid'] for r in ev.rows)
        ev.veto = sb['route'] == VETO
        ev.veto_pick = resolve_pick(sb, ev.idset, xyz)[0] if ev.veto else None
        # live arms: route, winner, resolved pick
        ev.live = {}
        ev.live['harv-base'] = live_rec(sb, ev)
        ev.live['harv-topo3'] = live_rec(hr['vertex_scoreboard'], ev)
        for name, d_ in live:
            cj = calib(d_, s, e)
            if cj is not None:
                ev.live[name] = live_rec(cj['vertex_scoreboard'], ev)
                if name == 'topk10':
                    ev.rows10 = board_rows(cj['vertex_scoreboard'])
        ev.admitted10 = ev.target in set(r['vid'] for r in getattr(ev, 'rows10', []))
        # reject outcome: any live arm that ended in the traditional fallback
        ev.reject_pick = None
        for name in ('base', 'harv-base', 'topo3', 'harv-topo3', 'ma4', 'dlonly', 'topk10', 'trad', 'ma20'):
            L = ev.live.get(name)
            if L and L['route'] not in ACCEPT and L['route'] != VETO and L['pick'] is not None:
                ev.reject_pick = L['pick']
                ev.reject_src = name
                break
        events.append(ev)
    return events, skipped


def live_rec(sb, ev):
    pick, dres = resolve_pick(sb, ev.idset, ev.cloud_xyz)
    win = next((r['vertex_id'] for r in sb.get('rows', []) if r.get('dl_winner')), None)
    return dict(route=sb['route'], win=win, pick=pick, dres=dres,
                final=(sb['final_x'], sb['final_y'], sb['final_z']))


def decide(rows, th):
    if th.get('mode') == 'dlonly':
        if not rows:
            return None
        r = min(rows, key=lambda r: r['voxel_rank'])
        return ('accept', r['vid']) if r['snap_dis'] <= DL_VTX_CUT_CM else ('reject',)
    best, bvid = -1e18, None
    for r in rows:
        tot = r['dl'] * th['scale'] + sum(r['geo'][t] * th['w_' + t[2:]] for t in GEO)
        if th['w_topo'] != 0.0 and r['votes'] >= 1:
            tot += th['w_topo'] * (r['frac'] - th['center'])
        if tot > best:
            best, bvid = tot, r['vid']
    if bvid is None:
        return None
    return ('accept', bvid) if best >= th['min_accept'] else ('reject',)


def outcome(ev, dec):
    '''decision -> (picked cloud vid or None, class)'''
    if dec is None:                      # no snapped candidate: DL has no say
        return ev.reject_pick, 'no-cand'
    if dec[0] == 'accept':
        if ev.veto:
            return ev.veto_pick, 'veto'
        return dec[1], 'accept'
    return ev.reject_pick, 'reject'


def score(events, th, rows_attr='rows'):
    res = dict(n=0, hit=0, w=0.0, wh=0.0, per=collections.Counter(), cls=collections.Counter())
    for ev in events:
        dec = decide(getattr(ev, rows_attr), th)
        pick, cl = outcome(ev, dec)
        hit = int(pick is not None and pick == ev.target)
        res['n'] += 1
        res['hit'] += hit
        res['w'] += ev.w
        res['wh'] += ev.w * hit
        res['per'][(ev.sample, hit)] += 1
        res['per'][(ev.src, hit)] += 1
        if hit:
            res['cls']['hit'] += 1
        elif not ev.admitted5 and cl == 'accept':
            res['cls']['not-admitted'] += 1
        elif cl == 'accept':
            res['cls']['wrong-accept'] += 1
        elif pick is None:
            res['cls']['uncovered'] += 1
        else:
            res['cls'][cl + '-to-wrong'] += 1
    return res


def score_live(events, name):
    res = dict(n=0, hit=0, w=0.0, wh=0.0, per=collections.Counter(), cls=collections.Counter())
    for ev in events:
        L = ev.live.get(name)
        if L is None:
            continue
        hit = int(L['pick'] is not None and L['pick'] == ev.target)
        res['n'] += 1
        res['hit'] += hit
        res['w'] += ev.w
        res['wh'] += ev.w * hit
        res['per'][(ev.sample, hit)] += 1
        res['per'][(ev.src, hit)] += 1
        res['cls']['hit' if hit else ('unresolved' if L['pick'] is None else 'miss')] += 1
    return res


def fmt_row(label, r):
    cells = [label, '%d/%d (%.1f%%)' % (r['hit'], r['n'], 100.0 * r['hit'] / max(1, r['n']))]
    for s in SAMPLES:
        h, m = r['per'][(s, 1)], r['per'][(s, 0)]
        cells.append('%d/%d' % (h, h + m))
    cells.append('%.1f%%' % (100.0 * r['wh'] / max(1e-9, r['w'])))
    for s in ('human', 'ai-scanner'):
        h, m = r['per'][(s, 1)], r['per'][(s, 0)]
        cells.append('%d/%d' % (h, h + m))
    return '| ' + ' | '.join(cells) + ' |'


HDR = ('| method | ALL | nueCC48 | NCpi0 | mcp1k | mcp2k | IPW | human | AI |\n'
       '|---|---|---|---|---|---|---|---|---|')


def closure(events):
    '''offline decision vs live (route, dl_winner) on every arm; returns {arm: [bad evts]}'''
    checks = [('harv-base', PROD, 'rows'), ('base', PROD, 'rows'),
              ('harv-topo3', TOPO3, 'rows'), ('topo3', TOPO3, 'rows'),
              ('ma4', dict(PROD, min_accept=4.0), 'rows'),
              ('topk10', PROD, 'rows10'), ('dlonly', dict(PROD, mode='dlonly'), 'rows'),
              ('ma20', dict(PROD, min_accept=20.0), 'rows')]   # pr/106 sec 6 live validation arm
    out = {}
    for name, th, ra in checks:
        bad = []
        for ev in events:
            L = ev.live.get(name)
            if L is None or not hasattr(ev, ra):
                continue
            dec = decide(getattr(ev, ra), th)
            live_acc = L['route'] in ACCEPT or L['route'] == VETO
            if dec is None:
                ok = not live_acc
            elif dec[0] == 'accept':
                ok = live_acc and dec[1] == L['win']
            else:
                ok = not live_acc
            if not ok:
                bad.append((ev.evt, L['route'], L['win'], dec))
        out[name] = bad
    return out


def parse_theta(kv):
    th = dict(PROD)
    for item in kv or []:
        k, v = item.split('=')
        th[k] = v if k == 'mode' else float(v)
    return th


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sbnd-root', default=ROOT)
    ap.add_argument('--carried-tags', nargs='*', default=[])
    ap.add_argument('--orig-tags', nargs='*', default=[])
    ap.add_argument('--harv-base', default='work-vtx106-harv-base-{sample}')
    ap.add_argument('--harv-rows', default='work-vtx106-harv-topo3-{sample}')
    ap.add_argument('--live-arms', nargs='*', default=DEFAULT_LIVE, metavar='name=dir-{sample}')
    ap.add_argument('--ipw-tsv', default=os.path.join(HERE, 'runs/ipw-vtx100-20260820.tsv'))
    ap.add_argument('--exclude-events')
    ap.add_argument('--only-events')
    ap.add_argument('--dmax', type=float, default=3.0, help='primary universe: d_target <= dmax cm')
    ap.add_argument('--dmax2', type=float, default=10.0, help='secondary band upper edge')
    ap.add_argument('--closure', action='store_true')
    ap.add_argument('--table', action='store_true')
    ap.add_argument('--eval', nargs='*', metavar='K=V')
    ap.add_argument('--sweep', help='write 1-D sweeps from PROD to this tsv')
    ap.add_argument('--search', action='store_true', help='coordinate ascent + random restarts')
    ap.add_argument('--rows-attr', default='rows', choices=['rows', 'rows10'],
                    help='candidate rows for --eval/--sweep/--search: top-5 (rows) or the topk10 arm rows')
    ap.add_argument('--guard', default='nuecc48', help='sample that may not regress in --search')
    ap.add_argument('--events-tsv')
    ap.add_argument('--miss-ledger', help='per-term ledger of M3 wrong-accepts (tsv)')
    args = ap.parse_args()

    events, skipped = load_events(args)
    nmis = sum(ev.rows_mismatch for ev in events)
    print('events %d (carried %d, frozen %d) skipped %s; harv-base/topo3 snapped-row mismatch %d'
          % (len(events), sum(ev.carried for ev in events), sum(not ev.carried for ev in events),
             dict(skipped), nmis))
    if nmis:
        sys.exit('rows mismatch between harvest arms -- stop')
    print('live-pick resolution onto the pre-DL cloud: %s' % dict(RES))
    # d_target distribution
    bands = [(0, 1), (1, args.dmax), (args.dmax, args.dmax2), (args.dmax2, 1e9)]
    print('\nd_target (click -> nearest pre-DL candidate), per sample [<=1 | 1-%g | %g-%g | >%g]:'
          % (args.dmax, args.dmax, args.dmax2, args.dmax2))
    for s in SAMPLES + ('human', 'ai-scanner', 'ALL'):
        sel = [ev for ev in events if s == 'ALL' or ev.sample == s or ev.src == s]
        cnt = [sum(1 for ev in sel if lo < ev.d_target <= hi or (lo == 0 and ev.d_target <= hi)) for lo, hi in bands]
        print('  %-10s n=%4d  %s' % (s, len(sel), '  '.join('%4d' % c for c in cnt)))
    prim = [ev for ev in events if ev.d_target <= args.dmax]
    sec = [ev for ev in events if args.dmax < ev.d_target <= args.dmax2]
    print('\nprimary universe (d_target <= %g): %d ; secondary band: %d ; candidate-missing: %d'
          % (args.dmax, len(prim), len(sec), len(events) - len(prim) - len(sec)))
    print('admission (target among snapped rows): top5 %d/%d, top10 %d/%d ; reject-outcome coverage %d/%d'
          % (sum(ev.admitted5 for ev in prim), len(prim), sum(ev.admitted10 for ev in prim), len(prim),
             sum(ev.reject_pick is not None for ev in prim), len(prim)))

    if args.closure:
        cl = closure(events)
        print('\nclosure (offline decision == live route/winner), all label events:')
        for k, bad in cl.items():
            print('  %-11s mismatches %d' % (k, len(bad)) + ('' if not bad else '  e.g. ' + str(bad[:5])))
        if any(cl[k] for k in cl if k != 'dlonly'):
            sys.exit('CLOSURE FAILED -- stop')

    if args.table:
        for title, sel in (('primary universe', prim), ('secondary band %g-%g cm' % (args.dmax, args.dmax2), sec)):
            print('\n### target-hit table, %s (n=%d)\n' % (title, len(sel)))
            print(HDR)
            print(fmt_row('M1 DL alone (offline, legacy rule)', score(sel, dict(PROD, mode='dlonly'))))
            print(fmt_row('M1 DL alone (live dlonly arm)', score_live(sel, 'dlonly')))
            print(fmt_row('M2 re-rank + topo (w=3)', score(sel, TOPO3)))
            print(fmt_row('M3 re-rank, no topo = PRODUCTION', score(sel, PROD)))
            print(fmt_row('production (live base arm)', score_live(sel, 'base')))
            print(fmt_row('min_accept 4 (live ma4)', score_live(sel, 'ma4')))
            print(fmt_row('top_k 10 (live topk10)', score_live(sel, 'topk10')))
            if any('ma20' in ev.live for ev in sel):
                print(fmt_row('min_accept 20 (offline)', score(sel, dict(PROD, min_accept=20.0))))
                print(fmt_row('min_accept 20 (live ma20 arm)', score_live(sel, 'ma20')))
            print(fmt_row('no DL, traditional (live trad)', score_live(sel, 'trad')))
            r = score(sel, PROD)
            print('\nM3 classes: %s' % dict(r['cls']))

    if args.eval is not None:
        th = parse_theta(args.eval)
        r = score(prim, th, args.rows_attr)
        print('\neval %s\n%s\n%s\nclasses %s' % (th, HDR, fmt_row('theta', r), dict(r['cls'])))

    if args.sweep:
        with open(args.sweep, 'w') as fh:
            fh.write('param\tvalue\thit\tn\t' + '\t'.join(SAMPLES) + '\n')
            for k, vals in SWEEP.items():
                for v in vals:
                    th = dict(PROD, **{k: float(v)})
                    if k == 'center':
                        th['w_topo'] = 3.0
                    r = score(prim, th, args.rows_attr)
                    fh.write('%s\t%g\t%d\t%d\t%s\n' % (k, v, r['hit'], r['n'],
                             '\t'.join('%d' % r['per'][(s, 1)] for s in SAMPLES)))
        print('\nsweep written: %s' % args.sweep)

    if args.search:
        def obj(th):
            r = score(prim, th, args.rows_attr)
            return r['hit'], r['per'][(args.guard, 1)]
        base_hit, base_g = obj(PROD)
        print('\ncoordinate ascent from PROD: hit %d, %s %d' % (base_hit, args.guard, base_g))

        def ascend(th0, tag):
            th, cur = dict(th0), obj(th0)
            improved = True
            while improved:
                improved = False
                for k, vals in SWEEP.items():
                    for v in vals:
                        if th.get(k) == float(v):
                            continue
                        t2 = dict(th, **{k: float(v)})
                        h, g = obj(t2)
                        if h > cur[0] and g >= base_g:
                            th, cur, improved = t2, (h, g), True
                            print('  [%s] %s=%g -> hit %d (%s %d)' % (tag, k, v, h, args.guard, g))
            return th, cur
        best_th, best = ascend(PROD, 'asc')
        rng = random.Random(106)
        for i in range(8):
            th0 = {k: float(rng.choice(v)) for k, v in SWEEP.items()}
            th_i, cur_i = ascend(th0, 'rs%d' % i)
            print('  restart %d from %s -> hit %d' % (i, th0, cur_i[0]))
            if cur_i > best:
                best_th, best = th_i, cur_i
        print('best theta %s -> hit %d/%d (%s %d); PROD %d' % (best_th, best[0], len(prim), args.guard, best[1], base_hit))
        print(HDR)
        print(fmt_row('best theta', score(prim, best_th, args.rows_attr)))

    if args.events_tsv:
        with open(args.events_tsv, 'w') as fh:
            fh.write('evt\tsample\tsrc\tcarried\ttarget\td_target\tadm5\tadm10\tveto\treject_pick\t'
                     'M1\tM2\tM3\tlive_base\tlive_topo3\tlive_dlonly\tlive_trad\thit_M3\n')
            for ev in events:
                o = {m: outcome(ev, decide(ev.rows, th))[0] for m, th in
                     (('M1', dict(PROD, mode='dlonly')), ('M2', TOPO3), ('M3', PROD))}
                fh.write('%d\t%s\t%s\t%d\t%d\t%.2f\t%d\t%d\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%d\n' % (
                    ev.evt, ev.sample, ev.src, ev.carried, ev.target, ev.d_target, ev.admitted5,
                    ev.admitted10, ev.veto, ev.reject_pick, o['M1'], o['M2'], o['M3'],
                    ev.live.get('base', {}).get('pick'), ev.live.get('topo3', {}).get('pick'),
                    ev.live.get('dlonly', {}).get('pick'), ev.live.get('trad', {}).get('pick'),
                    int(o['M3'] == ev.target)))
        print('events tsv: %s' % args.events_tsv)

    if args.miss_ledger:
        with open(args.miss_ledger, 'w') as fh:
            fh.write('evt\tsample\ttarget\twinner\td_target\tdelta_total\tdelta_s_dl\t' +
                     '\t'.join('delta_' + t for t in GEO) + '\tdecisive\n')
            for ev in prim:
                dec = decide(ev.rows, PROD)
                if dec is None or dec[0] != 'accept' or dec[1] == ev.target or not ev.admitted5:
                    continue
                rt = next(r for r in ev.rows if r['vid'] == ev.target)
                rw = next(r for r in ev.rows if r['vid'] == dec[1])
                dd = {'s_dl': (rt['dl'] - rw['dl']) * PROD['scale']}
                for t in GEO:
                    dd[t] = rt['geo'][t] - rw['geo'][t]
                tot = sum(dd.values())
                decisive = min(dd, key=lambda k: dd[k])   # most negative term for the target
                fh.write('%d\t%s\t%d\t%d\t%.2f\t%.2f\t%.2f\t%s\t%s\n' % (
                    ev.evt, ev.sample, ev.target, dec[1], ev.d_target, tot, dd['s_dl'],
                    '\t'.join('%.2f' % dd[t] for t in GEO), decisive))
        print('miss ledger: %s' % args.miss_ledger)


if __name__ == '__main__':
    main()
