#!/usr/bin/env python3
"""doc pr/20 Part II, gate A-2: enumerate the cluster merges the A1/A2 relaxation
actually produced, event by event.

The doc's design-stage prediction came from re-evaluating the connector's own
`[feat]` dump offline.  This measures the OUTCOME instead of the intent: for
every event it compares the Bee `clustering-global` partition of the OFF and ON
arms and reports each ON cluster that covers more than one OFF cluster.  That is
robust to cluster renumbering (labels are never compared across arms, only the
induced partition) and it catches a merge however it arose, including one the
connector's tracer would not print.

A merge is reported as CATHODE when the two OFF parts approach each other across
x = cathode_x -- i.e. their closest inter-part pair straddles the plane with both
tips inside `--tip-cut` -- which is the signature A1/A2 target.

  ./pr20_edge_census.py --base work-mcp1kall-cathA12off --on work-mcp1kall-cathA12on \
      --out /home/xqian/tmp/pr20exec/edges_mcp1k.tsv --jobs 8
"""
import argparse
import json
import os
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.spatial import cKDTree

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
SRC = 'ql_evt{evt}/mabc-all-apa.zip'
MEMBER = 'data/0/0-clustering-global.json'


def load(arm, evt):
    path = os.path.join(arm, SRC.format(evt=evt))
    if not os.path.exists(path):
        return None
    with zipfile.ZipFile(path) as z:
        d = json.loads(z.read(MEMBER))
    P = np.stack([np.asarray(d['x'], float), np.asarray(d['y'], float),
                  np.asarray(d['z'], float)], 1)
    rc = np.asarray(d['real_cluster_id'], int)
    return P, rc


def one(args):
    A, B, evt, tip_cut, cathode_x, min_pts = args
    a, b = load(A, evt), load(B, evt)
    if a is None or b is None:
        return evt, None, []
    PA, rcA = a
    PB, rcB = b

    # Map every ON point to its OFF cluster.  Same imaging and same upstream
    # passes => the point sets agree; fall back to nearest-neighbour when a
    # length mismatch says otherwise, and record that we did.
    exact = (len(PA) == len(PB)) and np.allclose(PA, PB)
    if exact:
        offof = rcA
    else:
        t = cKDTree(PA)
        offof = rcA[t.query(PB)[1]]

    out = []
    for c in np.unique(rcB):
        m = rcB == c
        parts, counts = np.unique(offof[m], return_counts=True)
        keep = parts[counts >= min_pts]
        if len(keep) < 2:
            continue
        # geometry of the merge: closest approach between the two largest parts
        order = np.argsort(-counts[np.isin(parts, keep)])
        big = keep[order][:2]
        Q1 = PB[m][offof[m] == big[0]]
        Q2 = PB[m][offof[m] == big[1]]
        d, j = cKDTree(Q1).query(Q2, k=1)
        k = int(np.argmin(d))
        p1, p2 = Q1[j[k]], Q2[k]
        straddle = ((p1[0] - cathode_x) * (p2[0] - cathode_x) < 0
                    and abs(p1[0] - cathode_x) < tip_cut
                    and abs(p2[0] - cathode_x) < tip_cut)
        out.append(dict(evt=evt, on_cluster=int(c), off_parts=[int(x) for x in keep],
                        npts=[int(x) for x in counts[np.isin(parts, keep)]],
                        dis=float(d[k]), p1=[round(float(v), 2) for v in p1],
                        p2=[round(float(v), 2) for v in p2],
                        cathode=bool(straddle), exact=bool(exact)))
    return evt, exact, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--on', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--tip-cut', type=float, default=6.0, help='cm from the cathode')
    ap.add_argument('--cathode-x', type=float, default=0.0, help='cm')
    ap.add_argument('--min-pts', type=int, default=20,
                    help='ignore OFF parts smaller than this (blob-level noise)')
    args = ap.parse_args()

    A = args.base if os.path.isabs(args.base) else os.path.join(BASE, args.base)
    B = args.on if os.path.isabs(args.on) else os.path.join(BASE, args.on)

    evA = {os.path.basename(d)[6:] for d in
           __import__('glob').glob(os.path.join(A, 'ql_evt*'))}
    evB = {os.path.basename(d)[6:] for d in
           __import__('glob').glob(os.path.join(B, 'ql_evt*'))}
    evts = sorted(evA & evB, key=int)
    print(f'base = {A}\non   = {B}\ncommon events = {len(evts)}', file=sys.stderr)

    work = [(A, B, e, args.tip_cut, args.cathode_x, args.min_pts) for e in evts]
    merges, n_inexact, n_missing = [], 0, 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for evt, exact, out in ex.map(one, work, chunksize=4):
            if exact is None:
                n_missing += 1
                continue
            if not exact:
                n_inexact += 1
            merges.extend(out)

    cath = [m for m in merges if m['cathode']]
    with open(args.out, 'w') as f:
        f.write('# evt\ton_cluster\toff_parts\tnpts\tdis_cm\tp1\tp2\tcathode\n')
        for m in sorted(merges, key=lambda m: (not m['cathode'], int(m['evt']))):
            f.write(f"{m['evt']}\t{m['on_cluster']}\t{','.join(map(str,m['off_parts']))}\t"
                    f"{','.join(map(str,m['npts']))}\t{m['dis']:.2f}\t"
                    f"{','.join(map(str,m['p1']))}\t{','.join(map(str,m['p2']))}\t"
                    f"{int(m['cathode'])}\n")

    print(f'\nevents missing an arm: {n_missing}   '
          f'events whose point sets did not match exactly: {n_inexact}')
    print(f'MERGES (ON cluster covering >1 OFF cluster): {len(merges)}   '
          f'of which cathode-straddling: {len(cath)}')
    print(f'\ncathode merges, enumerated (the gate A-2 list):')
    print(f"{'evt':>8} {'dis':>6}  {'npts':>13}  p1 / p2")
    for m in sorted(cath, key=lambda m: m['dis']):
        print(f"{m['evt']:>8} {m['dis']:6.2f}  {str(m['npts']):>13}  "
              f"{m['p1']} / {m['p2']}")
    other = [m for m in merges if not m['cathode']]
    if other:
        print(f'\nnon-cathode merges ({len(other)}) -- must be explained, '
              f'A1/A2 can only act across the cathode:')
        for m in sorted(other, key=lambda m: m['dis'])[:40]:
            print(f"{m['evt']:>8} {m['dis']:6.2f}  {str(m['npts']):>13}  "
                  f"{m['p1']} / {m['p2']}")
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
