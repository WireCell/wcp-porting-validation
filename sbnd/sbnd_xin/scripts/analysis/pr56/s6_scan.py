#!/usr/bin/env python3
"""doc pr/56 round 3: offline S6 operating-point scan.

Parses the single self-contained OC56CENSUS-2D "edge" line
(connect_graph_relaxed_strict.cxx, round 3) out of one or more
wct_pr_evt<N>.log files produced with WCT_RELAXED_EDGE_CENSUS=1
SBND_PROTECT_GRAPH=relaxed_strict_img_2d, and re-derives the S6 verdict at any
candidate per-plane (dw, ds) operating point + distance floor from the logged
dw={1..4} x ds={1..4} connectivity matrix embedded in that line -- no rerun
needed. (A first cut of this script/log format split the matrix across a
separate line per plane, correlated by (blk,j,k) with the edge line that
followed it; that broke silently whenever ClusteringProtectBundle processes
clusters concurrently and their log lines interleave. Fixed by putting
everything on one line per edge -- nothing left to correlate.)

Usage:
    s6_scan.py <arm_dir> [<arm_dir> ...]
      # summary: kill count at the shipped baseline (1,1)/(1,1)/(1,1), floor=0,
      # for every pr_evt*/wct_pr_evt*.log found under each arm_dir.

    s6_scan.py <arm_dir> [<arm_dir> ...] --candidates "2,1,1,1,0" "1,2,1,1,0.5"
      # each candidate is dw_uv,ds_uv,dw_w,ds_w,floor_cm -- report kill counts and
      # the diff (edges that flip verdict vs baseline) for each.

    s6_scan.py <arm_dir> --near-touching 3.0
      # list every S6-evaluated edge with dis < 3.0cm that is killed at the
      # baseline operating point -- the class the 71372 j=12/k=13 (0.76cm)
      # control belongs to, and the natural place to look for more of the
      # same isochronous/prolonged-signal false positive.

No silent caps: --candidates prints exactly how many edges changed verdict
vs. baseline (capped display only, count never truncated).
"""
import argparse
import glob
import os
import re

# doc pr/56 round 3 (2nd cut): ONE self-contained log line per edge, all
# three planes' matrices embedded directly -- the original two-line-per-plane
# design correlated by (blk,j,k) across sequential log lines, which silently
# mis-paired whenever ClusteringProtectBundle processes clusters concurrently
# and their log lines interleave (component indices j,k restart per cluster,
# so (blk,j,k) collides across clusters in the same event).  A single line
# has nothing to correlate, so that failure mode is gone by construction.
EDGE_RE = re.compile(
    r'OC56CENSUS-2D edge blk=(\S+) j=(\d+) k=(\d+) dis=([-\d.]+)cm apa=(\d+) face=(\d+) '
    r'p1=\(([-\d.]+),([-\d.]+),([-\d.]+)\) p2=\(([-\d.]+),([-\d.]+),([-\d.]+)\) slice_step=(\d+) '
    r'gap_u=(\w+) gap_v=(\w+) gap_w=(\w+) excuse_u=(\w+) excuse_v=(\w+) '
    r'budget_u=(\w+) budget_v=(\w+) budget_w=(\w+) '
    r'matrix_u=([01]{16}) matrix_v=([01]{16}) matrix_w=([01]{16}) killed=(\w+)')

PLANE_NAME = {0: 'u', 1: 'v', 2: 'w'}


def b(s):
    return s == 'true'


def parse_log(path):
    """Yield dicts, one per S6-evaluated edge."""
    with open(path, errors='replace') as f:
        for line in f:
            me = EDGE_RE.search(line)
            if not me:
                continue
            blk, j, k = me.group(1), int(me.group(2)), int(me.group(3))
            gaps = (b(me.group(14)), b(me.group(15)), b(me.group(16)))
            excuse_u, excuse_v = b(me.group(17)), b(me.group(18))
            budgets = (b(me.group(19)), b(me.group(20)), b(me.group(21)))
            matrices = (me.group(22), me.group(23), me.group(24))
            killed = b(me.group(25))
            # A plane with no seed data on either side never gets a matrix
            # bit set (shipped code: gap[]=false, no matrix loop entered) --
            # here that shows as an all-'0' matrix (never returned by a real
            # BFS: dw=ds=1 in a real matrix is the SAME query as gap[plane],
            # so an all-'0' matrix with gap=false is exactly that no-data
            # case; treat it as "no plane data" for candidate-point lookups.
            planes = {}
            for pidx, m in enumerate(matrices):
                if m == '0' * 16 and not gaps[pidx]:
                    continue  # no seed data this plane -- not evaluated
                planes[pidx] = dict(matrix=m, budget_hit=budgets[pidx])
            rec = dict(
                blk=blk, j=j, k=k, dis=float(me.group(4)),
                apa=int(me.group(5)), face=int(me.group(6)),
                p1=tuple(float(me.group(i)) for i in (7, 8, 9)),
                p2=tuple(float(me.group(i)) for i in (10, 11, 12)),
                slice_step=int(me.group(13)),
                gap_u=gaps[0], gap_v=gaps[1], gap_w=gaps[2],
                excuse_u=excuse_u, excuse_v=excuse_v,
                killed=killed,
                planes=planes,
                log=path,
            )
            yield rec


def matrix_connected(planes, plane_idx, dw, ds):
    """Look up the logged dw x ds connectivity matrix for one plane. Returns
    None if this plane had no seed data at all (both components empty --
    the shipped code never votes a gap in that case either)."""
    if plane_idx not in planes:
        return None
    m = planes[plane_idx]['matrix']
    if not (1 <= dw <= 4 and 1 <= ds <= 4):
        raise ValueError(f'matrix only covers dw,ds in 1..4, got {dw},{ds}')
    return m[(dw - 1) * 4 + (ds - 1)] == '1'


def verdict_at(rec, dw_uv, ds_uv, dw_w, ds_w, floor_cm):
    """Re-derive the S6 kill verdict at a candidate operating point from one
    parsed edge record. Mirrors two_d_connectivity_bad() exactly."""
    if rec['dis'] < floor_cm:
        return False
    cu = matrix_connected(rec['planes'], 0, dw_uv, ds_uv)
    cv = matrix_connected(rec['planes'], 1, dw_uv, ds_uv)
    cw = matrix_connected(rec['planes'], 2, dw_w, ds_w)
    gap_u = (cu is not None) and (not cu)
    gap_v = (cv is not None) and (not cv)
    gap_w = (cw is not None) and (not cw)
    return (gap_u and not rec['excuse_u']) or (gap_v and not rec['excuse_v']) or gap_w


def load_arm(arm_dir):
    recs = []
    for logf in sorted(glob.glob(os.path.join(arm_dir, 'pr_evt*', 'wct_pr_evt*.log'))):
        evt = re.search(r'wct_pr_evt(\d+)\.log', logf).group(1)
        for rec in parse_log(logf):
            rec['evt'] = evt
            recs.append(rec)
    return recs


def summarize(recs, label):
    n = len(recs)
    nk = sum(1 for r in recs if r['killed'])
    events = sorted(set(r['evt'] for r in recs))
    nev_killed = len(set(r['evt'] for r in recs if r['killed']))
    print(f'{label}: {n} S6-evaluated edges across {len(events)} events '
          f'({nev_killed} events with >=1 kill), {nk} killed ({100.0*nk/n:.1f}%)' if n
          else f'{label}: 0 S6-evaluated edges')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arm_dirs', nargs='+')
    ap.add_argument('--candidates', nargs='*', default=[],
                     help='dw_uv,ds_uv,dw_w,ds_w,floor_cm')
    ap.add_argument('--near-touching', type=float, default=None,
                     help='list baseline-killed edges with dis below this (cm)')
    args = ap.parse_args()

    recs = []
    for d in args.arm_dirs:
        recs.extend(load_arm(d))
    summarize(recs, 'baseline (shipped, dw=ds=1 all planes, floor=0)')

    if args.near_touching is not None:
        hits = [r for r in recs if r['killed'] and r['dis'] < args.near_touching]
        print(f'\n-- near-touching baseline kills under {args.near_touching}cm: {len(hits)} --')
        for r in sorted(hits, key=lambda r: r['dis']):
            print(f"  evt={r['evt']:>7} blk={r['blk']:<8} j={r['j']:<3} k={r['k']:<3} "
                  f"dis={r['dis']:5.2f}cm gap=u{int(r['gap_u'])}v{int(r['gap_v'])}w{int(r['gap_w'])} "
                  f"ex=u{int(r['excuse_u'])}v{int(r['excuse_v'])} "
                  f"p1={r['p1']} p2={r['p2']} log={r['log']}")

    for cspec in args.candidates:
        dw_uv, ds_uv, dw_w, ds_w, floor_cm = [float(x) for x in cspec.split(',')]
        dw_uv, ds_uv, dw_w, ds_w = int(dw_uv), int(ds_uv), int(dw_w), int(ds_w)
        changed = []
        nk = 0
        for r in recs:
            v = verdict_at(r, dw_uv, ds_uv, dw_w, ds_w, floor_cm)
            if v:
                nk += 1
            if v != r['killed']:
                changed.append((r, v))
        print(f'\n== candidate dw_uv={dw_uv} ds_uv={ds_uv} dw_w={dw_w} ds_w={ds_w} '
              f'floor={floor_cm}cm ==')
        print(f'   killed: {nk}/{len(recs)} ({100.0*nk/len(recs):.1f}%)  '
              f'changed vs baseline: {len(changed)}')
        for r, v in sorted(changed, key=lambda x: x[0]['dis'])[:40]:
            print(f"    evt={r['evt']:>7} blk={r['blk']:<8} j={r['j']:<3} k={r['k']:<3} "
                  f"dis={r['dis']:5.2f}cm baseline_killed={r['killed']} -> {v}")
        if len(changed) > 40:
            print(f'    ... {len(changed)-40} more (not shown, not dropped from the count above)')


if __name__ == '__main__':
    main()
