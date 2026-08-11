#!/usr/bin/env python3
"""doc pr/64: mine WCT_OC56_SCAN_DUMP jsonl for prolonged (dashed) signal
candidates, stratified by wire plane (U/V/W) and by whether the S6/S7
topology excuse fires -- following up doc pr/63's induction-only search
after evt 314507 turned out to be a COLLECTION-plane (W) case.

This is a standalone read/analyze script, not a fork of oc56_autoscan.py's
owner-label-calibrated `bad`/`good` classifier (that classifier answers a
different question -- pair-level match to the owner's hand scan -- and does
not touch plane identity). It reads the raw dump records directly, the same
schema oc56_autoscan.pair_table() and oc56_render_pair.render() consume, so
those two ARE reused (imported, not duplicated) for the render step.

Definitions (see doc pr/64 for the derivation):
  - "prolonged on plane p" = the edge's own seed-cell cloud (`seeds_a`/
    `seeds_b`, the exact cells fed to the S6/S7 corridor search) has
    tick-span / wire-span >= RATIO_MIN on BOTH sides of the candidate pair.
    This is a CLOUD metric.
  - "excused" = `excuse_u`/`excuse_v` from the same edge record, computed by
    the C++ from the p1->p2 DISPLACEMENT direction (connect_graph_relaxed_
    strict.cxx:1636-1649), not from the cloud. Plane W has no excuse channel
    at all (excuse[2] does not exist; always treated as False here).
  - A candidate only matters if it is also GAPPED on that plane (`gap[p]`) --
    prolongation without a gap on the same plane isn't what killed anything.

Usage:
    mine_pr64.py <arm_dir> [--out out.json] [--ratio-min 6.0]
        scans <arm_dir>/pr_evt*/oc56scan-evt*.jsonl (S6/S7 dump, <30cm only --
        >=30cm S7 candidates are dump-blind by construction, see doc pr/64).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

PLANE_NAME = {0: 'U', 1: 'V', 2: 'W'}


def load_dump(path):
    comps, edges, conns = [], [], []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d['type'] == 'component':
            comps.append(d)
        elif d['type'] == 'edge':
            edges.append(d)
        elif d['type'] == 'connectivity':
            conns.append(d)
    return comps, edges, conns


def seed_ratio(seeds):
    """tick-span / wire-span for one side's seed-cell cloud; None if <3 pts."""
    if not seeds or len(seeds) < 3:
        return None
    a = np.asarray(seeds, dtype=float)
    w = float(np.ptp(a[:, 0]))
    t = float(np.ptp(a[:, 1]))
    return t / max(w, 1.0), t


def pca_linearity(points):
    """fraction of point-cloud SVD variance along the principal axis."""
    p = np.asarray(points, dtype=float)
    if len(p) < 3:
        return 1.0
    p = p - p.mean(axis=0)
    s = np.linalg.svd(p, full_matrices=False, compute_uv=False)
    return float(s[0] / max(s.sum(), 1e-9))


def pca_length(points):
    p = np.asarray(points, dtype=float)
    if len(p) < 3:
        return float(np.linalg.norm(p[-1] - p[0])) if len(p) == 2 else 0.0
    p = p - p.mean(axis=0)
    _, _, vt = np.linalg.svd(p, full_matrices=False)
    proj = p @ vt[0]
    return float(proj.max() - proj.min())


def mine_arm(arm_dir, ratio_min=6.0):
    """Returns (rows, strat) where rows is per-candidate detail and strat is
    the plane x excusal stratification table (n, killed)."""
    rows = []
    strat = {}  # (plane_name, 'EXC'|'NOT') -> [n, killed]
    files = sorted(glob.glob(os.path.join(arm_dir, 'pr_evt*', 'oc56scan-evt*.jsonl')))
    n_nonempty = 0
    for f in files:
        if os.path.getsize(f) == 0:
            continue
        n_nonempty += 1
        evt = os.path.basename(f)[len('oc56scan-evt'):-len('.jsonl')]
        comps, edges, conns = load_dump(f)
        comp_by = {(c['graph_call'], c['comp']): c for c in comps}
        final_by_call = {c['graph_call']: c['final'] for c in conns}
        for e in edges:
            call = e['graph_call']
            j, k = e['j'], e['k']
            fin = final_by_call.get(call)
            together = None
            if fin and j < len(fin) and k < len(fin):
                together = bool(fin[j] == fin[k])
            for pl in e.get('planes', []):
                p = pl['plane']
                if not e['gap'][p]:
                    continue
                ra = seed_ratio(pl.get('seeds_a'))
                rb = seed_ratio(pl.get('seeds_b'))
                if ra is None or rb is None:
                    continue
                ratio = min(ra[0], rb[0])
                if ratio < ratio_min:
                    continue
                excused = bool(e['excuse'][p]) if p < 2 else False
                key = (PLANE_NAME[p], 'EXC' if excused else 'NOT')
                strat.setdefault(key, [0, 0])
                strat[key][0] += 1
                if e['killed']:
                    strat[key][1] += 1
                ca = comp_by.get((call, j))
                cb = comp_by.get((call, k))
                lin_a = pca_linearity(ca['points']) if ca else None
                lin_b = pca_linearity(cb['points']) if cb else None
                len_a = pca_length(ca['points']) if ca else None
                len_b = pca_length(cb['points']) if cb else None
                rows.append(dict(
                    evt=evt, call=call, j=j, k=k, blk=e['blk'], dis=e['dis'],
                    plane=PLANE_NAME[p], ratio=round(ratio, 1),
                    tick=round(max(ra[1], rb[1]), 0),
                    excused=excused, killed=e['killed'],
                    killed_pre_rescue=e.get('killed_pre_rescue'),
                    rescued=e.get('rescued'), together=together,
                    npA=ca['npts'] if ca else None, npB=cb['npts'] if cb else None,
                    linA=round(lin_a, 2) if lin_a is not None else None,
                    linB=round(lin_b, 2) if lin_b is not None else None,
                    lenA=round(len_a, 1) if len_a is not None else None,
                    lenB=round(len_b, 1) if len_b is not None else None,
                ))
    return rows, strat, n_nonempty, len(files)


def print_strat(strat, label):
    print(f'--- stratification ({label}) ---')
    print('  plane  excusal              n  killed  kill%')
    for pl in ('U', 'V', 'W'):
        for e, lbl in (('EXC', 'excused'), ('NOT', 'NOT excused')):
            if pl == 'W' and e == 'EXC':
                continue  # no excuse channel exists for W
            n, k = strat.get((pl, e), [0, 0])
            elbl = lbl if pl != 'W' else 'no excuse channel exists'
            if n:
                print(f'    {pl}   {elbl:<24s} {n:4d}   {k:4d}   {100.0*k/n:5.0f}%')


def track_shortlist(rows, lin_min=0.75, len_min=15.0):
    out = []
    for r in rows:
        if r['linA'] is None or r['linB'] is None:
            continue
        if r['linA'] < lin_min or r['linB'] < lin_min:
            continue
        if max(r['lenA'] or 0, r['lenB'] or 0) < len_min:
            continue
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arm_dir')
    ap.add_argument('--out', default=None)
    ap.add_argument('--ratio-min', type=float, default=6.0)
    ap.add_argument('--track-only', action='store_true',
                     help='apply the doc pr/63 PCA-linearity track-vs-shower filter')
    args = ap.parse_args()

    rows, strat, n_nonempty, n_total = mine_arm(args.arm_dir, args.ratio_min)
    print(f'arm={args.arm_dir}  events_scanned={n_total}  non-empty dumps={n_nonempty}  '
          f'candidate rows={len(rows)}', file=sys.stderr)
    print_strat(strat, args.arm_dir)

    if args.track_only:
        rows = track_shortlist(rows)
        print(f'\n{len(rows)} rows survive the track-shape filter (linearity>=0.75, len>=15cm)',
              file=sys.stderr)

    rows.sort(key=lambda r: (r['plane'], -r['ratio']))
    for r in rows[:60]:
        print('evt%-8s %2d-%-2d %-8s %s dis=%6.2f ratio=%5.1f tick=%4.0f exc=%s killed=%s '
              'rescued=%s together=%s linA=%s linB=%s lenA=%s lenB=%s'
              % (r['evt'], r['j'], r['k'], r['blk'], r['plane'], r['dis'], r['ratio'],
                 r['tick'], r['excused'], r['killed'], r['rescued'], r['together'],
                 r['linA'], r['linB'], r['lenA'], r['lenB']))

    if args.out:
        json.dump(rows, open(args.out, 'w'), indent=1)
        print(f'wrote {args.out} ({len(rows)} rows)', file=sys.stderr)


if __name__ == '__main__':
    main()
