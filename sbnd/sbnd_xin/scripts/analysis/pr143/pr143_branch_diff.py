#!/usr/bin/env python3
"""doc sbnd_xin/docs/pr/143 -- name the LEAVES that move between two SBND PR arms.

A SAME/DIFF verdict on tracking-pr.root says something changed; this says what.
Compares every branch of the named trees (default T_tagger, T_kine) event by
event and reports, per branch, how many events differ -- so a predicted
mechanism (sec 4: shw_sp_pio_2_v_* / flag_pi0_2 via cluster_acc_length) can be
confirmed or refuted instead of assumed.

Usage: pr143_branch_diff.py <armA> <armB> [--events-file F] [--trees T_tagger,T_kine] [--jobs N]
"""
import sys, os, argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor
try:
    import uproot
except ImportError:
    sys.exit('needs uproot')

def eq(a, b):
    """NaN-aware equality.  The trap this walked into first time round: a
    scalar branch comes back as numpy.float32, NOT an ndarray, so an
    isinstance(ndarray) guard falls through to `a == b` and `nan == nan` is
    False -- which reported T_kine:kine_pio_angle as "moved" on an event whose
    value is NaN in BOTH arms.  Normalise to an array first, then treat
    NaN==NaN as equal."""
    try:
        aa, bb = np.atleast_1d(np.asarray(a)), np.atleast_1d(np.asarray(b))
    except Exception:
        return repr(a) == repr(b)
    if aa.shape != bb.shape:
        return False
    if aa.dtype.kind == 'f' and bb.dtype.kind == 'f':
        return bool(np.all((aa == bb) | (np.isnan(aa) & np.isnan(bb))))
    if aa.dtype.kind in 'OSU' or bb.dtype.kind in 'OSU':
        return repr(aa.tolist()) == repr(bb.tolist())
    return bool(np.array_equal(aa, bb))

def one(args):
    pa, pb, evt, trees = args
    if not (os.path.isfile(pa) and os.path.isfile(pb)):
        return evt, None, 'MISSING'
    moved = []
    try:
        with uproot.open(pa) as fa, uproot.open(pb) as fb:
            for tn in trees:
                if tn not in fa or tn not in fb:
                    if (tn in fa) != (tn in fb): moved.append(tn + ':<tree presence>')
                    continue
                ta, tb = fa[tn].arrays(library='np'), fb[tn].arrays(library='np')
                ka, kb = set(ta), set(tb)
                for k in sorted(ka ^ kb): moved.append('%s:%s<branch presence>' % (tn, k))
                for k in sorted(ka & kb):
                    va, vb = ta[k], tb[k]
                    same = True
                    if len(va) != len(vb): same = False
                    else:
                        for i in range(len(va)):
                            if not eq(va[i], vb[i]): same = False; break
                    if not same: moved.append('%s:%s' % (tn, k))
    except Exception as e:
        return evt, None, 'ERROR:%s' % e
    return evt, moved, 'OK'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('armA'); ap.add_argument('armB')
    ap.add_argument('--events-file'); ap.add_argument('--trees', default='T_tagger,T_kine')
    ap.add_argument('--jobs', type=int, default=8); ap.add_argument('--out')
    a = ap.parse_args()
    trees = a.trees.split(',')
    ev = lambda r: set(d[len('pr_evt'):] for d in os.listdir(r) if d.startswith('pr_evt'))
    common = ev(a.armA) & ev(a.armB)
    if a.events_file:
        common &= set(open(a.events_file).read().split())
    common = sorted(common, key=int)
    print('branch diff over %d events, trees %s' % (len(common), trees))
    tasks = [(os.path.join(a.armA, 'pr_evt'+e, 'tracking-pr.root'),
              os.path.join(a.armB, 'pr_evt'+e, 'tracking-pr.root'), e, trees) for e in common]
    per_branch, bad, movers = {}, [], []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for evt, moved, st in ex.map(one, tasks, chunksize=4):
            if st != 'OK': bad.append((evt, st)); continue
            if moved:
                movers.append(evt)
                for m in moved: per_branch.setdefault(m, []).append(evt)
    print('events with >=1 moved leaf: %d / %d' % (len(movers), len(common)))
    if bad: print('problems on %d events: %s' % (len(bad), bad[:5]))
    for b, evs in sorted(per_branch.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        print('  %-52s %4d events  %s%s' % (b, len(evs), ' '.join(sorted(evs, key=int)[:12]),
                                            ' ...' if len(evs) > 12 else ''))
    if a.out:
        with open(a.out, 'w') as f:
            f.write('branch\tn_events\tevents\n')
            for b, evs in sorted(per_branch.items(), key=lambda kv: (-len(kv[1]), kv[0])):
                f.write('%s\t%d\t%s\n' % (b, len(evs), ' '.join(sorted(evs, key=int))))
        print('wrote', a.out)
    if movers:
        open(os.path.join(os.path.dirname(a.out or '.'), 'pr143_movers.txt') if a.out else '/dev/null', 'w').write('\n'.join(sorted(movers, key=int)) + '\n')

if __name__ == '__main__':
    main()
