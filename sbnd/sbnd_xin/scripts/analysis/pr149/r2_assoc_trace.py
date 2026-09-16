#!/usr/bin/env python3
"""doc pr/149 round 2: what the track fit's pixel ASSOCIATION looks like on isochronous fit points,
per arm, from the WCT_TRAJ_ASSOC_DEBUG trace (doc pdvd/111, TrackFitting.cxx trajectory_fit).

Line format (stdout.log):
    TRAJASSOC <tag> <method> <call> <i> <cluster> <apa> <face> <kept> <skip> <rev> <nan>
              <init xyz> <sol xyz> <final xyz> <noU xyz> <noV xyz> <noW xyz>            (cm)
              then per plane U, V, W: <ncell> <sum_q> <wcen> <tcen> <wsol> <quantity>
              <tsol>
For the event's main cluster (main_cid of the arm's metrics table) and its largest trajectory_fit
call of the highest charge-division method (the whole-cluster fit; per-segment refits print a handful), a fit point is isochronous when the local chord through its neighbours i-1, i+1 (solved
positions) makes >= 75 deg with the drift axis.  Per arm, over those points:
    ncell per plane (median; share with 0 cells on any plane),
    |wcen - wsol| per plane (wire units): how far the charge-weighted association centroid sits
        from the solved point, in the wire direction,
    |tcen - tsol| (ticks, the plane-averaged centroid of the three planes).

Usage: r2_assoc_trace.py --arms TAG[:METRICS_ARM] ... --samples mcp1k mcp2k --events-manifest DIR
       (TAG = the work-<s>-<TAG> trace arm; METRICS_ARM defaults to TAG)
"""
import argparse
import csv
import math
import os

SX = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'
M = os.path.join(SX, 'docs/pr/149_figs/metrics')
COS75 = math.cos(math.radians(75))


def median(v):
    v = sorted(x for x in v if math.isfinite(x))
    if not v:
        return float('nan')
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def main_cids(arm, sample):
    p = os.path.join(M, f'{arm}-{sample}.tsv')
    out = {}
    if os.path.exists(p):
        for r in csv.DictReader(open(p), delimiter='\t'):
            if r['has_calib'] == '1' and r['main_cid'] not in ('', 'nan'):
                out[int(r['event'])] = int(float(r['main_cid']))
    return out


def parse(stdout_path, cid):
    """Rows of the LAST call that fitted cluster `cid`."""
    calls = {}
    with open(stdout_path, errors='replace') as fh:
        for line in fh:
            if not line.startswith('TRAJASSOC '):
                continue
            t = line.split()
            if len(t) != 49 or int(t[5]) != cid:
                continue
            call = (t[1], t[2], int(t[3]))
            row = dict(i=int(t[4]), kept=int(t[8]),
                       sol=tuple(map(float, t[15:18])),
                       planes=[tuple(map(float, t[30 + 6 * p:36 + 6 * p])) for p in range(3)],
                       tsol=float(t[48]))
            calls.setdefault(call, []).append(row)
    if not calls:
        return []
    # The whole-cluster fit prints every point of every segment in ONE call; per-segment refits
    # print a handful.  Take the largest call of the highest charge-division method, latest on a tie.
    top_method = max(c[1] for c in calls)
    pick = max((c for c in calls if c[1] == top_method), key=lambda c: (len(calls[c]), c[2]))
    return sorted(calls[pick], key=lambda r: r['i'])


def iso_rows(rows):
    out = []
    for k in range(1, len(rows) - 1):
        a, b = rows[k - 1]['sol'], rows[k + 1]['sol']
        d = [b[j] - a[j] for j in range(3)]
        n = math.sqrt(sum(x * x for x in d))
        if n <= 0 or not math.isfinite(n):
            continue
        if abs(d[0]) / n <= COS75:
            out.append(rows[k])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', nargs='+', required=True)
    ap.add_argument('--samples', nargs='+', default=['mcp1k', 'mcp2k'])
    ap.add_argument('--events-manifest', required=True)
    a = ap.parse_args()
    print('# doc pr/149 round 2 association trace on ISO fit points (main cluster, largest trajectory_fit call of the top method)')
    print('# per plane U/V/W: median ncell | median |wcen-wsol| (wires); median |tcen-tsol| (ticks, 3-plane mean); '
          'share of ISO points with 0 cells on any plane')
    pooled = {}
    for spec in a.arms:
        tag, _, marm = spec.partition(':')
        marm = marm or tag
        for s in a.samples:
            mf = os.path.join(a.events_manifest, f'{s}.txt')
            if not os.path.exists(mf):
                continue
            evts = [int(x.split()[0]) for x in open(mf) if x.strip() and not x.startswith('#')]
            cids = main_cids(marm, s)
            for e in evts:
                so = os.path.join(SX, f'work-{s}-{tag}', f'pr_evt{e}', 'stdout.log')
                if e not in cids or not os.path.exists(so):
                    print(f'{tag:14s} {s} {e}: no main cluster / no stdout')
                    continue
                allrows = parse(so, cids[e])
                rows = iso_rows(allrows)
                P = pooled.setdefault(tag, dict(n=0, nc=[[], [], []], dw=[[], [], []], dt=[], zero=0))
                nc = [[r['planes'][p][0] for r in rows] for p in range(3)]
                dw = [[abs(r['planes'][p][2] - r['planes'][p][4]) for r in rows] for p in range(3)]
                dt = [abs(sum(r['planes'][p][3] for p in range(3)) / 3 - r['tsol']) for r in rows]
                zero = sum(1 for r in rows if any(r['planes'][p][0] == 0 for p in range(3)))
                P['n'] += len(rows)
                P['zero'] += zero
                for p in range(3):
                    P['nc'][p] += nc[p]
                    P['dw'][p] += dw[p]
                P['dt'] += dt
                print(f'{tag:14s} {s} {e} cid {cids[e]}: fit pts {len(allrows):4d} iso pts {len(rows):4d}  ncell '
                      + '/'.join(f'{median(nc[p]):.0f}' for p in range(3))
                      + '  |wcen-wsol| ' + '/'.join(f'{median(dw[p]):.3f}' for p in range(3))
                      + f'  |tcen-tsol| {median(dt):.3f}  zero-cell share {zero / len(rows) if rows else float("nan"):.3f}')
    print('# pooled')
    for tag, P in pooled.items():
        print(f'{tag:14s} iso pts {P["n"]:5d}  ncell ' + '/'.join(f'{median(P["nc"][p]):.0f}' for p in range(3))
              + '  |wcen-wsol| ' + '/'.join(f'{median(P["dw"][p]):.3f}' for p in range(3))
              + f'  |tcen-tsol| {median(P["dt"]):.3f}  zero-cell share {P["zero"] / P["n"] if P["n"] else float("nan"):.3f}')


if __name__ == '__main__':
    main()
