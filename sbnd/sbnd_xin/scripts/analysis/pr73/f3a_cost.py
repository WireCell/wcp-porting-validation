#!/usr/bin/env python3
"""doc sbnd_xin/docs/pr/73 round 2: the runtime/RSS cost checkpoint for F3a.

The owner's standing requirement for anything that builds or queries a derived
graph (set when pr/51 round 5's steiner_graph_gap was approved): prove the cost
is negligible BEFORE the validation census is believed, with wall and peak-RSS
deltas over a full manifest, not an estimate.

F3a queries a SECOND flavor's shortest_path per gap-flavor do_rough_path call.
That flavor's GraphAlgorithms LRU is otherwise cold in production, so the
routing step roughly doubles -- it restores the pre-round-5 base-flavor cost
*in addition to* the gap cost.  On top sits an O(n_gap * n_base) distance loop,
short-circuited whenever the two routes agree (the common case).

Two deltas, and they answer different questions:

  off vs production  -- the code RESTRUCTURE alone.  Both are behaviourally
                        identical, so any difference here is noise and this
                        pair calibrates what "noise" looks like on this box.
  on  vs off         -- the SHIPPED cost.  Conflates the extra routing with
                        the changed downstream work a re-routed event does,
                        which is the honest production number.

Reads each event's .time.meta (rc / wall_s / maxrss_kb), written by
abtest/timecmd.py under setarch -R.  Read-only.

Usage:  f3a_cost.py <ARM_A> <ARM_B> [<ARM_A2> <ARM_B2> ...]
        (arms are compared pairwise, A = baseline, B = new)
"""
import sys
import os
import glob

SB = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


def meta(arm):
    out = {}
    for d in sorted(glob.glob(os.path.join(SB, arm, 'pr_evt*'))):
        p = os.path.join(d, '.time.meta')
        if not os.path.exists(p):
            continue
        kv = {}
        for line in open(p):
            if '=' in line:
                k, v = line.strip().split('=', 1)
                kv[k] = v
        try:
            out[os.path.basename(d)[6:]] = (int(kv['wall_s']), int(kv['maxrss_kb']),
                                            int(kv.get('rc', 0)))
        except (KeyError, ValueError):
            pass
    return out


def med(v):
    s = sorted(v)
    n = len(s)
    return 0.0 if not n else (s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2]))


def compare(a, b):
    A, B = meta(a), meta(b)
    ev = sorted(set(A) & set(B))
    if not ev:
        print('  %-22s vs %-22s : no common events' % (a, b))
        return
    dw = [B[e][0] - A[e][0] for e in ev]
    dr = [B[e][1] - A[e][1] for e in ev]
    wa = sum(A[e][0] for e in ev) / len(ev)
    wb = sum(B[e][0] for e in ev) / len(ev)
    ra = sum(A[e][1] for e in ev) / len(ev)
    rb = sum(B[e][1] for e in ev) / len(ev)
    bad = [e for e in ev if A[e][2] or B[e][2]]
    print('  %-22s -> %-22s  n=%d' % (a, b, len(ev)))
    print('      wall_s    mean %7.2f -> %7.2f   delta %+6.2f s (%+5.2f %%)  median delta %+.1f s'
          % (wa, wb, wb - wa, 100.0 * (wb - wa) / wa if wa else 0, med(dw)))
    print('      maxrss_kb mean %7.0f -> %7.0f   delta %+6.0f kB (%+5.2f %%) median delta %+.0f kB'
          % (ra, rb, rb - ra, 100.0 * (rb - ra) / ra if ra else 0, med(dr)))
    if bad:
        print('      NONZERO rc on: %s' % ' '.join(bad))


def main():
    argv = sys.argv[1:]
    if len(argv) < 2 or len(argv) % 2:
        sys.exit(__doc__)
    print('=== F3a cost checkpoint (wall + peak RSS, per manifest) ===')
    for i in range(0, len(argv), 2):
        compare(argv[i], argv[i + 1])
    print('\n  Disqualifying (pr/51 round 5 bar): mean wall delta > 1 %% with a')
    print('  CONSISTENT SIGN across all three manifests, or mean RSS delta > 1 %%.')
    print('  Round 5 was accepted at +0.17 / -0.21 / +0.56 s -- sign flipping,')
    print('  i.e. noise -- and <= 0.13 %% on RSS.')


if __name__ == '__main__':
    main()
