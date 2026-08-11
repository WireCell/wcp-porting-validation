#!/usr/bin/env python3
"""doc pr/57 round 6: collect the owner's full hand scan into one truth table.

The owner scanned all three displays (5018 root, 5019 claude-scan50,
5020 claude-scan223) on 2026-08-10.  This script joins every label against the
round-4 dump arms and writes ONE row per component pair -- the unit the owner
judges ("what matters is the separation of two clusters, not the individual
edge") -- with:

  * the owner verdict (bad dominates, then good, then OK -- a pair can carry
    several edge labels and bad anywhere means the pair must not be broken),
  * whether the owner FLIPPED a machine label (verdict != the `auto <v>` token
    in the comment; only 5019/5020 have machine comments),
  * the round-4 pair token recomputed from the connectivity records
    (SEP / dir / via N) -- the 5018/5019 label comments predate the token, so
    it is recomputed for everyone and cross-checked against the 5020 comments,
  * every pair feature the round-3 classifier used.

Gates (all must pass, printed at the end):
  * zero orphan labels (every label key resolves to a dumped edge),
  * recomputed pair token == comment token wherever a comment carries one,
  * owner label dirs are read-only here by construction (this script only
    ever opens them for reading).

Usage:
    oc56_truth.py --out docs/pr/pr57r6-truth.tsv
      # defaults: the four work-pr57r4-scan* arms, all three label dirs
"""
import argparse
import collections
import json
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import oc56_autoscan as A  # noqa: E402

SBND = A.SBND

# doc pr/64 round 4 HARNESS FIX: the round-4 arms (work-pr57r4-scan*) were
# generated with the retired 'relaxed_strict_img_2d_wfloor' flavor and predate
# the pr/62 S7 production flip -- their connectivity records contain MST edges
# that no longer exist in production (demonstrated: evt174224's phantom 82.7cm
# comp1-3 edge, doc pr/64 round 3 caveat 2), so any REPLAY over them scores
# against a graph that is not today's.  Defaults now point at post-pr/62 bare
# production dumps: work-pr64r4-scan48/19 (regenerated 2026-08-11) and
# work-pr64-scan1k (the full 1000-event PR-data pool, superset of the old
# scan50+scan395 events).  The archived docs/pr/pr57r6-truth.tsv was built
# from the r4 arms and stays as-is (scientific record); pass --arm to rebuild
# against any arm set explicitly.
DEFAULT_ARMS = ['work-pr64r4-scan48', 'work-pr64r4-scan19',
                'work-pr64-scan1k']
DEFAULT_LABEL_DIRS = ['overclustering_labels',
                      'overclustering_labels/claude-scan50',
                      'overclustering_labels/claude-scan223']
POPULATION = {'work-pr57r4-scan48': 'nueCC48',
              'work-pr57r4-scan19': 'NCpi0',
              'work-pr57r4-scan50': 'PR-data',
              'work-pr57r4-scan395': 'PR-data',
              'work-pr64r4-scan48': 'nueCC48',
              'work-pr64r4-scan19': 'NCpi0',
              'work-pr64-scan1k': 'PR-data'}

COLS = ['population', 'arm', 'tag', 'evt', 'call', 'j', 'k', 'nedge', 'nlab',
        'verdict', 'flip', 'pair', 'Lmin', 'Lmax', 'Tmax', 'npmin', 'angle',
        'gw', 'wdeadX', 'dis', 'dens', 'dvtx', 'blks', 'killed', 'gaps',
        'excuses']


def load_labels(dirs):
    """{edge_key: (verdict, auto_token_or_None, comment_pair_token_or_None,
                   tag)} over every label dir."""
    out = {}
    for d in dirs:
        tag = os.path.basename(d.rstrip('/'))
        if tag == 'overclustering_labels':
            tag = 'root'
        for f in sorted(glob.glob(os.path.join(SBND, d, 'labels-evt*.json'))):
            for k, v in json.load(open(f))['labels'].items():
                c = v.get('comment', '')
                m = re.match(r'auto (\w+) ', c)
                mt = re.search(r'pair=(\S+)', c)
                out[k] = (v['verdict'], m.group(1) if m else None,
                          mt.group(1) if mt else None, tag)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', action='append', default=None)
    ap.add_argument('--labels', action='append', default=None)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    arms = args.arm or [os.path.join(SBND, a) for a in DEFAULT_ARMS]
    ldirs = args.labels or DEFAULT_LABEL_DIRS

    truth = load_labels(ldirs)
    print('labels loaded: %d  by tag: %s' % (
        len(truth),
        dict(collections.Counter(t[3] for t in truth.values()))))

    rows, joined_keys = [], set()
    token_mismatch = []
    for arm in arms:
        base = os.path.basename(arm.rstrip('/'))
        pop = POPULATION.get(base, base)
        for evt, path in A.arm_events(arm):
            # want_shown_only=False: the owner labelled a handful of edges that
            # do NOT pass the viewer's default filter (surviving candidates,
            # dis > 5) by toggling the display filter -- 15 root labels sit on
            # such edges.  The truth join must see every dumped edge.
            for pk, p in sorted(A.pair_table(arm, evt, path,
                                             want_shown_only=False).items()):
                labs = []
                for r in p['edges']:
                    kk = A.edge_key(evt, r)
                    if kk in truth:
                        labs.append((kk, truth[kk]))
                        joined_keys.add(kk)
                if not labs:
                    continue
                verds = [t[1][0] for t in labs]
                verdict = ('bad' if 'bad' in verds else
                           'good' if 'good' in verds else 'OK')
                flip = any(t[1][1] is not None and t[1][1] != t[1][0]
                           for t in labs)
                tok = p['sep']['token'].replace(' ', '')
                for kk, (v, auto, ctok, tag) in labs:
                    if ctok is not None and ctok != tok:
                        token_mismatch.append((kk, ctok, tok))
                tags = sorted({t[1][3] for t in labs})
                recs = sorted(p['edges'], key=lambda r: r['dis'])
                rows.append([
                    pop, base, '+'.join(tags), p['evt'], p['call'], p['j'],
                    p['k'], len(p['edges']), len(labs), verdict, int(flip),
                    tok, '%.2f' % p['Lmin'], '%.2f' % p['Lmax'],
                    '%.2f' % p['Tmax'], p['npmin'],
                    '%.1f' % (p['angle'] if p['angle'] is not None else -1),
                    int(p['gw']), p['wdeadX'], '%.2f' % p['dis'],
                    '%.0f' % p['dens'], '%.1f' % p['dvtx'],
                    '|'.join(r['blk'] for r in recs),
                    '|'.join(str(int(r['killed'])) for r in recs),
                    '|'.join(''.join(str(int(g)) for g in r['gap'])
                             for r in recs),
                    '|'.join(''.join(str(int(e)) for e in r['excuse'])
                             for r in recs)])

    orphans = sorted(set(truth) - joined_keys)
    with open(args.out, 'w') as fh:
        fh.write('\t'.join(COLS) + '\n')
        for r in rows:
            fh.write('\t'.join(str(x) for x in r) + '\n')

    print('pairs written: %d -> %s' % (len(rows), args.out))
    per = collections.defaultdict(collections.Counter)
    for r in rows:
        per[r[0]][r[9]] += 1
    for pop in sorted(per):
        c = per[pop]
        print('  %-8s pairs=%3d  %s' % (pop, sum(c.values()), dict(c)))
    sep = collections.defaultdict(collections.Counter)
    for r in rows:
        sep[r[9]][r[11]] += 1
    for v in ('bad', 'good', 'OK'):
        print('  verdict=%-4s by pair token: %s' % (v, dict(sep[v])))

    print('GATE orphan labels: %d %s' % (
        len(orphans), 'PASS' if not orphans else 'FAIL'))
    for k in orphans[:5]:
        print('  ORPHAN %s (%s)' % (k, truth[k][3]))
    print('GATE comment-token cross-check: %d mismatches %s' % (
        len(token_mismatch), 'PASS' if not token_mismatch else 'FAIL'))
    for k, ctok, tok in token_mismatch[:5]:
        print('  TOKEN %s comment=%s recomputed=%s' % (k, ctok, tok))
    if orphans or token_mismatch:
        sys.exit(1)


if __name__ == '__main__':
    main()
