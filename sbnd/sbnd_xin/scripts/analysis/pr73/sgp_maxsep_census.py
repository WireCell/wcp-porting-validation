#!/usr/bin/env python3
"""doc sbnd_xin/docs/pr/73 sec 6 F3a: how big is the excursion guard's footprint?

Sizes the proposed `sgp_max_sep` guard BEFORE it exists, using the shipped
default-OFF `sgp_edge_probe` sentinel -- which already routes both the
penalized "steiner_graph_gap" flavor and the untouched "steiner_graph" base
flavor and already logs the very quantity the guard would threshold:

    sgp path: cluster C first=(...) last=(...) same=S n_gap=N n_base=M
              gap_on_gap=.. gap_on_base=.. base_on_gap=.. base_on_base=..
              diverge_at=K maxsep=..

`maxsep` is the one-sided (gap->base) Hausdorff distance sampled at route
VERTICES, in cm.  It is the exact expression F3a would evaluate, so the
would-fire counts below transfer verbatim to the guard.

This differs from sgp_path_map.py in the one way that matters for a census:
NO single-cluster filter.  Every call in every cluster of every event is kept.

Read-only.  Usage:
    sgp_maxsep_census.py <ARM> [ARM ...] [--tsv OUT.tsv]
                         [--thresh 2,3,4,5,6] [--evt N] [--quiet]

    <ARM> is a work-* directory under sbnd_xin (or any path holding
    pr_evt<ID>/wct_pr_evt<ID>.log).  --evt restricts to one event id and
    prints its per-call table (the form used for the three named events).
"""
import sys
import os
import re
import glob

SB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

RXP = re.compile(
    r"sgp path: cluster (\d+) first=\(([-\d.]+),([-\d.]+),([-\d.]+)\) "
    r"last=\(([-\d.]+),([-\d.]+),([-\d.]+)\) same=(\d+) n_gap=(\d+) n_base=(\d+) "
    r"gap_on_gap=([-\d.]+) gap_on_base=([-\d.]+) base_on_gap=([-\d.]+) "
    r"base_on_base=([-\d.]+) diverge_at=(\d+) maxsep=([-\d.]+)")

COLS = ("arm evt cluster call same n_gap n_base gap_on_gap gap_on_base "
        "base_on_gap base_on_base detour_cm detour_pct tax diverge_at maxsep").split()


def parse_arm(arm):
    """One row per 'sgp path:' line.  `call` is the per-EVENT call ordinal, in
    log order -- the same numbering sgp_fix_design.py's tables use."""
    root = arm if os.path.isdir(arm) else os.path.join(SB, arm)
    label = os.path.basename(root.rstrip('/'))
    rows = []
    for d in sorted(glob.glob(os.path.join(root, 'pr_evt*'))):
        evt = os.path.basename(d)[6:]
        logs = glob.glob(os.path.join(d, 'wct_pr_evt*.log'))
        if not logs:
            continue
        call = 0
        for line in open(logs[0], errors='replace'):
            m = RXP.search(line)
            if not m:
                continue
            g = m.groups()
            bb = float(g[13])
            gb = float(g[11])
            detour = gb - bb
            rows.append(dict(
                arm=label, evt=evt, cluster=int(g[0]), call=call,
                same=int(g[7]), n_gap=int(g[8]), n_base=int(g[9]),
                gap_on_gap=float(g[10]), gap_on_base=gb,
                base_on_gap=float(g[12]), base_on_base=bb,
                detour_cm=detour,
                detour_pct=(100.0 * detour / bb) if bb > 0 else 0.0,
                tax=float(g[12]) - float(g[10]),
                diverge_at=int(g[14]), maxsep=float(g[15])))
            call += 1
    return rows


def pct(vals, q):
    if not vals:
        return float('nan')
    s = sorted(vals)
    k = min(len(s) - 1, int(round(q * (len(s) - 1))))
    return s[k]


def main():
    argv = sys.argv[1:]
    arms, tsv, evt_only, quiet = [], None, None, False
    thresh = [2.0, 3.0, 4.0, 5.0, 6.0]
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == '--tsv':
            tsv = argv[i + 1]; i += 2; continue
        if a == '--thresh':
            thresh = [float(x) for x in argv[i + 1].split(',')]; i += 2; continue
        if a == '--evt':
            evt_only = argv[i + 1]; i += 2; continue
        if a == '--quiet':
            quiet = True; i += 1; continue
        if a.startswith('--'):
            sys.exit('unknown flag %s\n\n%s' % (a, __doc__))
        arms.append(a); i += 1
    if not arms:
        sys.exit(__doc__)

    rows = []
    for a in arms:
        r = parse_arm(a)
        if not quiet:
            nev = len({x['evt'] for x in r})
            print('# %-24s %6d calls over %3d events' % (a, len(r), nev))
        rows += r
    if not rows:
        sys.exit('no "sgp path:" lines found -- was SBND_SGP_EDGE_PROBE=true set?')

    if tsv:
        with open(tsv, 'w') as fh:
            fh.write('\t'.join(COLS) + '\n')
            for r in rows:
                fh.write('\t'.join(
                    ('%.3f' % r[c]) if isinstance(r[c], float) else str(r[c])
                    for c in COLS) + '\n')
        print('# wrote %s (%d rows)' % (tsv, len(rows)))

    if evt_only:
        sel = [r for r in rows if r['evt'] == str(evt_only)]
        print('\n=== event %s: every do_rough_path call ===' % evt_only)
        print('%5s %8s %7s %6s %10s %10s %9s %9s' % (
            'call', 'cluster', 'same', 'n_gap', 'base_cm', 'detour_cm',
            'detour_%', 'maxsep'))
        for r in sel:
            print('%5d %8d %7d %6d %10.2f %10.3f %9.2f %9.3f%s' % (
                r['call'], r['cluster'], r['same'], r['n_gap'],
                r['base_on_base'], r['detour_cm'], r['detour_pct'], r['maxsep'],
                '  <-- moved' if not r['same'] else ''))
        moved = [r for r in sel if not r['same']]
        if moved:
            mx = max(r['maxsep'] for r in moved)
            print('  %d calls, %d moved, MAX maxsep over moved calls = %.3f cm'
                  % (len(sel), len(moved), mx))
            for t in thresh:
                print('    margin to %.1f cm cap: %+.3f cm  (%s)'
                      % (t, t - mx, 'no fire' if mx <= t else 'FIRES'))
        else:
            print('  %d calls, 0 moved -- the penalty changed no route here' % len(sel))
        return

    # ---- distribution, moved calls only -------------------------------------
    print('\n=== maxsep distribution (cm) ===')
    print('%-24s %8s %8s %8s %8s %8s %8s' % (
        'arm', 'calls', 'moved', 'p50', 'p90', 'p99', 'max'))
    for a in sorted({r['arm'] for r in rows}) + ['POOLED']:
        sub = rows if a == 'POOLED' else [r for r in rows if r['arm'] == a]
        mv = [r['maxsep'] for r in sub if not r['same']]
        print('%-24s %8d %8d %8.3f %8.3f %8.3f %8.3f' % (
            a, len(sub), len(mv), pct(mv, .5), pct(mv, .9), pct(mv, .99),
            max(mv) if mv else 0.0))
    print('  (percentiles over MOVED calls only -- same=1 calls have maxsep=0\n'
          '   by construction and would dilute every number.)')

    # ---- would-fire ---------------------------------------------------------
    nev_tot = len({(r['arm'], r['evt']) for r in rows})
    print('\n=== would-fire, by threshold ===')
    hdr = '%8s %10s %10s %14s' % ('cap cm', 'fire calls', 'of moved', 'events w/ fire')
    per_arm = sorted({r['arm'] for r in rows})
    hdr += ''.join('%14s' % a[-8:] for a in per_arm)
    print(hdr)
    nmoved = len([r for r in rows if not r['same']])
    for t in thresh:
        fire = [r for r in rows if not r['same'] and r['maxsep'] > t]
        ev = {(r['arm'], r['evt']) for r in fire}
        line = '%8.1f %10d %9.1f%% %8d/%-5d' % (
            t, len(fire), 100.0 * len(fire) / nmoved if nmoved else 0,
            len(ev), nev_tot)
        for a in per_arm:
            n_a = len({r['evt'] for r in fire if r['arm'] == a})
            tot_a = len({r['evt'] for r in rows if r['arm'] == a})
            line += '%14s' % ('%d/%d' % (n_a, tot_a))
        print(line)
    print('  "events w/ fire" is the guard FOOTPRINT: events whose output could\n'
          '  change.  It is the number that has to be hand-adjudicable.')


if __name__ == '__main__':
    main()
