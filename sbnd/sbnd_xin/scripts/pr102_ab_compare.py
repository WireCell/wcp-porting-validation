#!/usr/bin/env python3
"""doc pr/102: join two pr102_region_census.py --summary-tsv files (epoch A =
before, epoch B = after) on common scored events and report, per class, the
events fixed / persisting / new, plus the event-set difference (events scored
in only one epoch -- reported, never silently dropped).

Usage: pr102_ab_compare.py <events-A.tsv> <events-B.tsv> [--label-a S] [--label-b S]
"""
import sys, csv

CLASSES = ['NEARVTX', 'HAD-A5', 'HAD-ADJ', 'EM-INT', 'TRK-COR', 'OTHER', 'UNTYPED']
SCORED = ('OK', 'NO_CALIB', 'FRAME_MISMATCH')

def load(path):
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f, delimiter='\t'):
            out[r['evt']] = r
    return out

def main(argv):
    args, la, lb, i = [], 'A', 'B', 1
    while i < len(argv):
        if argv[i] == '--label-a':   la = argv[i+1]; i += 2
        elif argv[i] == '--label-b': lb = argv[i+1]; i += 2
        else:                        args.append(argv[i]); i += 1
    if len(args) != 2:
        print(__doc__); return 2
    A, B = load(args[0]), load(args[1])
    sa = {e for e, r in A.items() if r['status'] in SCORED}
    sb = {e for e, r in B.items() if r['status'] in SCORED}
    common = sorted(sa & sb, key=int)
    print(f'[{la}: {len(sa)} scored, {lb}: {len(sb)} scored, common {len(common)}; '
          f'only-{la} {len(sa - sb)}, only-{lb} {len(sb - sa)}]')
    if sa - sb:
        print(f'  only-{la}: {" ".join(sorted(sa - sb, key=int)[:20])}'
              + (' ...' if len(sa - sb) > 20 else ''))
    if sb - sa:
        print(f'  only-{lb}: {" ".join(sorted(sb - sa, key=int)[:20])}'
              + (' ...' if len(sb - sa) > 20 else ''))

    def hit(r, c):
        # EM-INT is reported (never flagged); use presence for it
        key = c if c == 'EM-INT' else c + '_flag'
        try:
            return int(r.get(key, 0) or 0) > 0
        except ValueError:
            return False

    print(f'\n{"class":8s} {la+" evts":>8s} {lb+" evts":>8s} {"fixed":>6s} '
          f'{"persist":>7s} {"new":>5s}')
    for c in CLASSES:
        ea = {e for e in common if hit(A[e], c)}
        eb = {e for e in common if hit(B[e], c)}
        print(f'{c:8s} {len(ea):8d} {len(eb):8d} {len(ea - eb):6d} '
              f'{len(ea & eb):7d} {len(eb - ea):5d}')
        for tag, s in (('fixed', ea - eb), ('new', eb - ea)):
            if s:
                print(f'    {tag:5s}: {" ".join(sorted(s, key=int))}')

    fa = {e for e in common if int(A[e].get('flagged', 0) or 0) > 0}
    fb = {e for e in common if int(B[e].get('flagged', 0) or 0) > 0}
    print(f'\n[any-flag: {la} {len(fa)}, {lb} {len(fb)}, fixed {len(fa - fb)}, '
          f'persist {len(fa & fb)}, new {len(fb - fa)}]')
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
