#!/usr/bin/env python3
"""doc pr/139 sec 9.1 -- item 4, the k >= 3 cap, graded on the owner's boundaries.

A FORK of the tape/agreement machinery in pr139_scan_analysis.py (M10; that
script stays byte-untouched as the record of sec 6).  Two things are added:
`--arm` so two arms can be compared, and the split of the agreement table by
the OWNER's k rather than by the C++'s, which is the whole question -- sec 6.1
read the k>=3 mean of 0.800 as `max_parts=2` refusing the third cut.

    python3 scripts/pr140_k3.py --arm work-pr140r1-onk3 --base work-pr140r1-onrh15

`--base` is any arm running the shipped `max_parts = 2` with the debug tape on;
work-pr140r1-onrh15 qualifies because the re-home acts on an orphan daughter
AFTER the peel, so the `SHOWER_SPLIT part` lines it writes are the unmodified
max_parts=2 boundaries on the same flipped production config.
"""
import argparse, collections, glob, itertools, json, os, re, statistics, sys

SX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAG = 'splitscan-0902-pi0'
POS = lambda v: bool(v) and v.startswith('SPLIT')
P = re.compile(r'SHOWER_SPLIT part shower=(\d+) part=(\d+) nseg=(\d+) q=([\d.eE+-]+) segs=([\d,]+)')


def owner_labels(tag=TAG):
    verd, part, kk = {}, {}, {}
    for f in sorted(glob.glob(os.path.join(SX, 'em_labels', tag, 'labels-evt*.json'))):
        d = json.load(open(f))
        ev = int(d['event'][3:])
        for n, v in (d.get('split_labels') or {}).items():
            k = (ev, int(n))
            verd[k] = v.get('verdict')
            kk[k] = int(v.get('n_parts') or 1)
            part[k] = {int(s): int(g) for s, g in (v.get('groups') or {}).items() if int(g) >= 0}
    return verd, part, kk


def tape_parts(arm):
    part = collections.defaultdict(dict)
    qmap = collections.defaultdict(dict)
    logs = sorted(glob.glob(os.path.join(SX, arm) + '-*/pr_evt*/stdout.log'))
    if not logs:
        sys.exit('no tape under %s-*/pr_evt*/stdout.log' % arm)
    for lg in logs:
        ev = int(re.search(r'pr_evt(\d+)/', lg).group(1))
        for line in open(lg, errors='replace'):
            m = P.search(line)
            if not m:
                continue
            node, g, q = int(m.group(1)), int(m.group(2)), float(m.group(4))
            segs = [int(x) for x in m.group(5).split(',') if x]
            for s in segs:
                part[(ev, node)][s] = g
                qmap[(ev, node)][s] = q / max(len(segs), 1)
    return part, qmap


def agreement(prop, own, qm):
    """charge-weighted best-label-permutation agreement -- doc pr/138 sec A5.6's
    metric, transcribed from pr138_kernel_k.py:220 so the numbers compare."""
    segs = [s for s in prop if s in own]
    if not segs:
        return None
    pg = sorted({prop[s] for s in segs})
    og = sorted({own[s] for s in segs})
    Qt = sum(qm.get(s, 0.0) for s in segs) or 1.0
    best = 0.0
    for perm in itertools.permutations(pg, min(len(pg), len(og))):
        m = dict(zip(perm, og))
        best = max(best, sum(qm.get(s, 0.0) for s in segs if m.get(prop[s]) == own[s]) / Qt)
    return best


def score(arm, verd, opart, kk):
    cpart, qmap = tape_parts(arm)
    rows = []
    for k, v in verd.items():
        if not POS(v) or k not in cpart:
            continue
        a = agreement(cpart[k], opart.get(k, {}), qmap[k])
        if a is None:
            continue
        rows.append(dict(key=k, verdict=v, owner_k=kk[k],
                         cxx_k=len(set(cpart[k].values())), agree=a))
    return rows


def table(name, rows):
    def stat(sel):
        v = [r['agree'] for r in rows if sel(r)]
        if not v:
            return '   -        -      -     '
        ex = sum(1 for x in v if x >= 0.9995)
        return '%5d   %.3f  %.3f  %d/%d' % (len(v), statistics.median(v),
                                            statistics.fmean(v), ex, len(v))
    print('  %-26s %s' % ('%s  SPLIT2' % name, stat(lambda r: r['owner_k'] == 2)))
    print('  %-26s %s' % ('%s  k>=3 (owner)' % name, stat(lambda r: r['owner_k'] >= 3)))
    print('  %-26s %s' % ('%s  ALL' % name, stat(lambda r: True)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arm', required=True)
    ap.add_argument('--base')
    a = ap.parse_args()
    verd, opart, kk = owner_labels()
    A = score(a.arm, verd, opart, kk)
    print('=== boundary agreement vs the owner 2026-09-01 per-part labels ===')
    print('  %-26s %5s   %6s %6s  %s' % ('', 'n', 'median', 'mean', 'exact'))
    table(a.arm.split('-')[-1], A)
    if a.base:
        B = score(a.base, verd, opart, kk)
        print()
        table(a.base.split('-')[-1] + ' (max_parts=2)', B)
        bm = {r['key']: r for r in B}
        print('\n  per-object movement (owner k >= 3 first):')
        print('  %-8s %-7s %-8s %5s %6s %6s %8s %8s' %
              ('event', 'node', 'verdict', 'own_k', 'k_base', 'k_arm', 'a_base', 'a_arm'))
        for r in sorted(A, key=lambda r: (-r['owner_k'], r['key'])):
            b = bm.get(r['key'])
            if not b:
                continue
            mark = '' if abs(r['agree'] - b['agree']) < 1e-9 else \
                   ('   UP' if r['agree'] > b['agree'] else '   DOWN')
            print('  %-8d %-7d %-8s %5d %6d %6d %8.3f %8.3f%s'
                  % (r['key'][0], r['key'][1], r['verdict'], r['owner_k'],
                     b['cxx_k'], r['cxx_k'], b['agree'], r['agree'], mark))
        up = [r for r in A if bm.get(r['key']) and r['agree'] > bm[r['key']]['agree'] + 1e-9]
        dn = [r for r in A if bm.get(r['key']) and r['agree'] < bm[r['key']]['agree'] - 1e-9]
        print('\n  moved UP %d, moved DOWN %d, unchanged %d'
              % (len(up), len(dn), len(A) - len(up) - len(dn)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
