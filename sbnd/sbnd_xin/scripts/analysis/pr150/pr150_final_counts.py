#!/usr/bin/env python3
"""doc sbnd_xin/pr/150 sec 8 -- the full-sample answer on all 3067 events: the nueCC (nue BDT > 7.0) and numuCC
(numu BDT > 0.9) selections, the pi0 count, and the neutrino vertex, per cell vs s0.

Inputs (read-only): docs/pr/150_figs/metrics/pr150<cell>-<sample>.tsv (pr150_metrics.py extract) and the calib
dumps work-<sample>-pr150<cell>/pr_evt*/calib-pr-evt*.json (showers[].pio_id), the vtx105 labels.

Conventions (the sec 8 text states them):
  * a working point needs nu_evaluated == 1 (pr150_metrics.wp, as in secs 2.1 / 2.2);
  * the paired decomposition vs s0 splits every change into: lost because the event is no longer evaluated
    (de-evaluated), lost while still evaluated, gained from a newly evaluated event, gained while evaluated;
  * pi0 is counted three ways, kept apart: (P1) events with >= 1 ACCEPTED reco pi0 group (showers sharing
    pio_id >= 0 in the calib dump = the winner loop's mass-windowed pairs, doc pr/126), (P2) the kine_pio BDT
    feature in the (100,170) MeV window (the census proxy of secs 2.1 / 2.2; kine_pio_flag == 1 alone is NOT a
    pi0, d86_video_picks.py), (P3) the position-anchored hand pi0 pairs (sec 4.1, geo_rescore.py; quoted);
  * vertex: the vtx105 labels, keyed by (run, subrun, event); the physical event 18255-1-69314 sits in both nuecc48
    and mcp2k, its label was made on nuecc48, so the mcp2k copy is dropped from the labelled set (878 distinct);
    a > 10 cm mover follows pr150_vtx_movers.py (moved > 10 cm or one side without a vertex; away if dB > dA).
  * the ladder levels r1..r4 ran on the 626-event ladder manifest only; they get a separate table on that manifest.

Usage: pr150_final_counts.py [--out docs/pr/150_figs/150_final_counts.txt] [--tsv docs/pr/150_figs/150_final_counts.tsv]
"""
import argparse, glob, json, math, os, sys
from concurrent.futures import ProcessPoolExecutor

SX = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'
sys.path.insert(0, os.path.join(SX, 'scripts/analysis/pr150'))
sys.path.insert(0, os.path.join(SX, 'vtx_rules'))
import pr150_metrics as M   # noqa: E402
import vtx_io               # noqa: E402

S = ['nuecc48', 'ncpi0', 'mcp1k', 'mcp2k']
CELLS = ['s0', 'cs', 'p3bw', 'csp3bw', 'tfull']
LADDER = ['r1', 'r2', 'r3', 'r4']
DUP = ('mcp2k', 69314)          # the same physical event as ('nuecc48', 69314); label made on nuecc48
HAND_PI0 = dict(s0=28, cs=21, p3bw=22, csp3bw=18, tfull=20)   # sec 4.1 table, 65 events / 66 hand pi0


def pi0_groups(path):
    try:
        d = json.load(open(path))
    except (OSError, ValueError):
        return path, None
    g = {}
    for s in d.get('showers') or ():
        pid = int(s.get('pio_id', -1))
        if pid >= 0:
            g.setdefault(pid, 0)
            g[pid] += 1
    return path, len(g)


def load_groups(arms):
    jobs = []
    for arm in arms:
        for s in S:
            jobs += glob.glob(os.path.join(SX, f'work-{s}-{arm}/pr_evt*/calib-pr-evt*.json'))
    with ProcessPoolExecutor(16) as ex:
        res = dict(ex.map(pi0_groups, jobs, chunksize=20))
    G = {}
    for p, n in res.items():
        parts = p.split('/')
        arm = parts[-3].split('-', 2)[2]
        s = parts[-3].split('-')[1]
        ev = int(parts[-2][len('pr_evt'):])
        if n is None:
            sys.exit(f'unreadable calib dump {p}')
        G[(arm, s, ev)] = n
    return G


def sign_p(lost, gained):
    """two-sided exact sign test on lost vs gained (the net change against the churn)."""
    n = lost + gained
    if n == 0:
        return 1.0
    k = min(lost, gained)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def sel(r):
    w = M.wp(r)
    return dict(nue=w['nue7'], numu=w['numu09'], ev=r['nu_evaluated'] == '1')


def vtx(r):
    if r is None or r.get('has_calib') != '1':
        return None
    v = (M.fnum(r['vx']), M.fnum(r['vy']), M.fnum(r['vz']))
    return v if all(math.isfinite(c) for c in v) else None


def decomp(A, B, keys, f):
    """A = s0 rows, B = cell rows, f(row) -> bool (requires nu_evaluated inside)."""
    o = dict(A=0, B=0, both=0, lost_deval=0, lost_eval=0, gain_neweval=0, gain_eval=0)
    for k in keys:
        a, b = f(A[k]), f(B[k])
        o['A'] += a; o['B'] += b; o['both'] += a and b
        if a and not b:
            o['lost_deval' if B[k]['nu_evaluated'] != '1' else 'lost_eval'] += 1
        if b and not a:
            o['gain_neweval' if A[k]['nu_evaluated'] != '1' else 'gain_eval'] += 1
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(M.FIGS, '150_final_counts.txt'))
    ap.add_argument('--tsv', default=os.path.join(M.FIGS, '150_final_counts.tsv'))
    a = ap.parse_args()
    D = {c: M.load('pr150' + c, S) for c in CELLS}
    L = {c: M.load('pr150' + c, S) for c in LADDER}
    keys = sorted(D['s0'])
    for c in CELLS:
        assert sorted(D[c]) == keys, c
        assert all(r['rc'] == '0' for r in D[c].values()), c
    man = sorted(L['r1'])
    for c in LADDER:
        assert sorted(L[c]) == man, c
    G = load_groups(['pr150' + c for c in CELLS + LADDER])
    for c in CELLS:           # a calib dump exists exactly on the evaluated events
        ev = {k for k in keys if D[c][k]['nu_evaluated'] == '1'}
        gd = {(s, e) for (arm, s, e) in G if arm == 'pr150' + c}
        assert ev == gd, (c, len(ev), len(gd))
    grp = lambda c, k: G.get(('pr150' + c, k[0], k[1]), 0)

    out, tsv = [], ['section\tcell\tsample\tquantity\tvalue']
    P = out.append
    P('# doc pr/150 sec 8 -- full-sample counts on all 3067 events (pr150_final_counts.py)')
    P(f'# events {len(keys)} per cell (every cell rc 0 on every event); the physical event 18255-1-69314 is in both nuecc48 and mcp2k')
    P('')
    P('## A. selections per sample (nu_evaluated == 1 required): nueCC = nue BDT > 7.0, numuCC = numu BDT > 0.9, both = passes both')
    hdr = f"{'cell':7s} " + '  '.join(f'{s:>22s}' for s in S + ['ALL'])
    P(hdr)
    P(f"{'':7s} " + '  '.join(f'{"eval nue numu both":>22s}' for _ in S + ['ALL']))
    for c in CELLS:
        cols = []
        for s in S + ['ALL']:
            ks = [k for k in keys if s == 'ALL' or k[0] == s]
            ev = sum(D[c][k]['nu_evaluated'] == '1' for k in ks)
            ne = sum(sel(D[c][k])['nue'] for k in ks)
            nm = sum(sel(D[c][k])['numu'] for k in ks)
            bo = sum(sel(D[c][k])['nue'] and sel(D[c][k])['numu'] for k in ks)
            cols.append(f'{ev:>5d} {ne:>4d} {nm:>4d} {bo:>4d}   ')
            for q, v in (('nu_evaluated', ev), ('nueCC', ne), ('numuCC', nm), ('both', bo)):
                tsv.append(f'A\t{c}\t{s}\t{q}\t{v}')
        P(f'{c:7s} ' + '  '.join(f'{x:>22s}' for x in cols))
    P('')
    P('## B. paired decomposition vs s0, all 3067 events: s0 -> cell = both + lost(de-evaluated, still evaluated) + gained(newly evaluated, already evaluated)')
    for name, f in (('nu_evaluated', lambda r: r['nu_evaluated'] == '1'),
                    ('nueCC', lambda r: sel(r)['nue']), ('numuCC', lambda r: sel(r)['numu']),
                    ('pi0 accepted group (P1)', None)):
        P(f'  [{name}]')
        for c in CELLS[1:]:
            if f is None:
                ff = None
                o = dict(A=0, B=0, both=0, lost_deval=0, lost_eval=0, gain_neweval=0, gain_eval=0)
                for k in keys:
                    x, y = grp('s0', k) > 0, grp(c, k) > 0
                    o['A'] += x; o['B'] += y; o['both'] += x and y
                    if x and not y:
                        o['lost_deval' if D[c][k]['nu_evaluated'] != '1' else 'lost_eval'] += 1
                    if y and not x:
                        o['gain_neweval' if D['s0'][k]['nu_evaluated'] != '1' else 'gain_eval'] += 1
            else:
                o = decomp(D['s0'], D[c], keys, f)
            lo, ga = o['lost_deval'] + o['lost_eval'], o['gain_neweval'] + o['gain_eval']
            o['sign_p'] = round(sign_p(lo, ga), 3)
            P(f"    {c:7s} s0 {o['A']:5d} -> {o['B']:5d} (net {o['B'] - o['A']:+d}); both {o['both']}; lost {lo} "
              f"(de-evaluated {o['lost_deval']}, still evaluated {o['lost_eval']}); gained {ga} "
              f"(newly evaluated {o['gain_neweval']}, already evaluated {o['gain_eval']}); sign test p {o['sign_p']:.2f}")
            for q, v in o.items():
                tsv.append(f'B\t{c}\t{name}\t{q}\t{v}')
    P('')
    P('  common evaluated set (evaluated in both s0 and the cell): counts on the SAME events')
    for c in CELLS[1:]:
        ce = [k for k in keys if D['s0'][k]['nu_evaluated'] == '1' and D[c][k]['nu_evaluated'] == '1']
        r = []
        for name, f in (('nueCC', lambda r_: sel(r_)['nue']), ('numuCC', lambda r_: sel(r_)['numu'])):
            r.append(f"{name} {sum(f(D['s0'][k]) for k in ce)} -> {sum(f(D[c][k]) for k in ce)}")
        r.append(f"pi0 groups(P1) {sum(grp('s0', k) > 0 for k in ce)} -> {sum(grp(c, k) > 0 for k in ce)}")
        P(f'    {c:7s} n={len(ce)}: ' + '; '.join(r))
        tsv.append(f'B\t{c}\tALL\tcommon_evaluated\t{len(ce)}')
    P('')
    P('  who leaves / enters the evaluated set (attribution from the per-bundle tgm,stm,fc of nusel-evt*.tsv):')
    for c in CELLS[1:]:
        att = {}
        for k in keys:
            ea, eb = D['s0'][k]['nu_evaluated'] == '1', D[c][k]['nu_evaluated'] == '1'
            if ea == eb:
                continue
            ba = json.loads(D['s0'][k]['bundles'] or '{}'); bb = json.loads(D[c][k]['bundles'] or '{}')
            why = set()
            for bk in set(ba) & set(bb):
                ta, tb = ba[bk].split(','), bb[bk].split(',')
                for i, nm in enumerate(('tgm', 'stm', 'fc')):
                    if ta[i] != tb[i]:
                        why.add(f'{nm} {ta[i]}->{tb[i]}')
            if set(ba) != set(bb):
                why.add('bundle set changed')
            lab = ('lost ' if ea else 'gained ') + ('stm' if any(w.startswith('stm') for w in why) else
                                                    'tgm' if any(w.startswith('tgm') for w in why) else
                                                    'bundle-set' if 'bundle set changed' in why else
                                                    'fc' if any(w.startswith('fc') for w in why) else
                                                    'other (cosmic-tagged in both, same bundle flags)')
            att[lab] = att.get(lab, 0) + 1
        P(f'    {c:7s} ' + ', '.join(f'{k} {v}' for k, v in sorted(att.items())))
    P('')
    P('## C. pi0, three quantities kept apart (per sample and ALL)')
    P('   P1 = events with >= 1 accepted reco pi0 group (calib showers[].pio_id >= 0); P1g = accepted groups;')
    P('   P2 = kine_pio BDT feature with mass in (100,170) MeV (NOT a pi0 finding; census proxy of secs 2.1/2.2);')
    P('   P3 = hand pi0 pairs recovered by position (sec 4.1, 65 events / 66 hand pi0; quoted, not recomputed here)')
    for c in CELLS:
        cols = []
        for s in S + ['ALL']:
            ks = [k for k in keys if s == 'ALL' or k[0] == s]
            p1 = sum(grp(c, k) > 0 for k in ks); p1g = sum(grp(c, k) for k in ks)
            p2 = sum(D[c][k]['pio_flag'] == '1' and 100 < M.fnum(D[c][k]['pio_mass']) < 170 for k in ks)
            cols.append(f'{s} P1 {p1} (P1g {p1g}) P2 {p2}')
            for q, v in (('P1_events', p1), ('P1_groups', p1g), ('P2_kine_window', p2)):
                tsv.append(f'C\t{c}\t{s}\t{q}\t{v}')
        P(f'  {c:7s} ' + ' | '.join(cols) + f' | P3 {HAND_PI0[c]}')
        tsv.append(f'C\t{c}\tpi0scan\tP3_hand_pairs\t{HAND_PI0[c]}')
    P('')
    P('## D. neutrino vertex')
    labs = {}
    for Ld in vtx_io.load_labels(tags=vtx_io.TAGS_VTX105):
        if Ld['truth'] is not None:
            labs[(int(Ld['runNo']), int(Ld['subRunNo']), int(Ld['eventNo']))] = tuple(Ld['truth'])
    sub = {}
    for s in S:
        for ln in open(os.path.join(SX, f'work-{s}-pr150s0/nusel-events.tsv')):
            f_ = ln.split()
            if f_ and f_[0].isdigit():
                sub[(s, int(f_[2]))] = (int(f_[0]), int(f_[1]), int(f_[2]))
    lk = [k for k in keys if k != DUP and sub.get(k) in labs]
    P(f'  labelled (vtx105, keyed run/subrun/event; mcp2k copy of 18255-1-69314 dropped): {len(lk)} '
      + '(' + ', '.join(f'{s} {sum(k[0] == s for k in lk)}' for s in S) + ')')
    tsv.append(f'D\tall\tALL\tlabelled\t{len(lk)}')
    base = {k: vtx(D['s0'][k]) for k in keys}
    for c in CELLS:
        V = {k: vtx(D[c][k]) for k in keys}
        d = {k: (math.dist(V[k], labs[sub[k]]) if V[k] else math.inf) for k in lk}
        d0 = {k: (math.dist(base[k], labs[sub[k]]) if base[k] else math.inf) for k in lk}
        n1, n3, n10 = (sum(x <= t for x in d.values()) for t in (1, 3, 10))
        nov = sum(math.isinf(x) for x in d.values())
        away = tow = 0
        for k in lk:
            mv = math.inf if (V[k] is None or base[k] is None) else math.dist(V[k], base[k])
            if (V[k] is None) and (base[k] is None):
                continue
            if mv > 10:
                if d[k] > d0[k]:
                    away += 1
                else:
                    tow += 1
        st = [k for k in lk if k[0] in ('mcp1k', 'mcp2k')]
        s3 = sum(d[k] <= 3 for k in st)
        sh = dict(same=0, lt1=0, lt3=0, lt10=0, gt10=0, lost=0, gained=0)
        for k in keys:
            x, y = base[k], V[k]
            if x is None and y is None:
                continue
            if x is None:
                sh['gained'] += 1; continue
            if y is None:
                sh['lost'] += 1; continue
            m = math.dist(x, y)
            sh['same' if m < 0.1 else 'lt1' if m < 1 else 'lt3' if m < 3 else 'lt10' if m < 10 else 'gt10'] += 1
        P(f'  {c:7s} labelled <= 1 / 3 / 10 cm: {n1} / {n3} / {n10}; no vertex {nov}; > 10 cm movers vs s0 away {away} / toward {tow}; '
          f'[numu-only <= 3 cm {s3} of {len(st)}]')
        P(f'          all events, main vertex vs s0: unchanged (< 0.1 cm) {sh["same"]}, 0.1-1 cm {sh["lt1"]}, 1-3 cm {sh["lt3"]}, '
          f'3-10 cm {sh["lt10"]}, > 10 cm {sh["gt10"]}, vertex lost {sh["lost"]}, gained {sh["gained"]}')
        for q, v in (('le1', n1), ('le3', n3), ('le10', n10), ('novtx', nov), ('away10', away), ('toward10', tow), ('numu_le3', s3)):
            tsv.append(f'D\t{c}\tlabelled\t{q}\t{v}')
        for q, v in sh.items():
            tsv.append(f'D\t{c}\tall_shift\t{q}\t{v}')
    P('')
    P(f'## E. the retune ladder, on its 626-event manifest only ({len(man)} events: Stage 1 + the pr/149 Stage-2 manifest)')
    lkm = [k for k in man if k != DUP and sub.get(k) in labs]
    for c in ['s0', 'csp3bw'] + LADDER:
        R = D[c] if c in D else L[c]
        ne = sum(sel(R[k])['nue'] for k in man); nm = sum(sel(R[k])['numu'] for k in man)
        ev = sum(R[k]['nu_evaluated'] == '1' for k in man)
        p1 = sum(grp(c, k) > 0 for k in man)
        n3 = sum((math.dist(vtx(R[k]), labs[sub[k]]) if vtx(R[k]) else math.inf) <= 3 for k in lkm)
        P(f'  {c:7s} evaluated {ev}; nueCC {ne}; numuCC {nm}; pi0 P1 {p1}; vertex <= 3 cm {n3} of {len(lkm)}')
        for q, v in (('nu_evaluated', ev), ('nueCC', ne), ('numuCC', nm), ('P1_events', p1), ('vtx_le3', n3)):
            tsv.append(f'E\t{c}\tmanifest626\t{q}\t{v}')
    open(a.out, 'w').write('\n'.join(out) + '\n')
    open(a.tsv, 'w').write('\n'.join(tsv) + '\n')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
