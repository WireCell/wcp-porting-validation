#!/usr/bin/env python3
"""doc pr/64 round 5 -- read-only probe for evt 18259-18625.

Reproduces every number in the round-5 section: rules out the four literal
readings of "missing clustering points" at the owner-reported coordinate, then
characterizes the one real anomaly (the S6 W-sole-voter split of the neutrino
cluster into PR clusters 11/126) and its consequence (energy included, not
lost).  No production run, no write under work-*/bee/ -- reads existing
arms/zips only.

Usage:
  python3 probe_18625.py --arm work-pr64r4-scan19/pr_evt18625
  python3 probe_18625.py --arm work-pr64r4-on19/pr_evt18625
  python3 probe_18625.py --bee bee/prod0811/ncpi0-prod0811.zip
"""
import argparse
import collections
import json
import math
import os
import sys
import zipfile

import numpy as np

TARGET = np.array([142.1, 78.3, 176.5])
# the S6-killed bridge edge (oc56scan-evt18625.jsonl, j=0 k=1, blk=closest,
# graph_call=0): p1 lands in cluster 126, p2 in cluster 11.
P1 = np.array([144.01621090127335, 78.91833269331426, 173.62967439511405])
P2 = np.array([145.89181090127335, 78.91833269331426, 173.62967439511402])


def load_layer(zpath, layer, required=True):
    with zipfile.ZipFile(zpath) as z:
        name = f'data/0/0-{layer}.json'
        if name not in z.namelist():
            if required:
                raise KeyError(f'{zpath} has no {name}')
            return None
        d = json.loads(z.read(name))
    return d


def pts(d):
    return np.stack([d['x'], d['y'], d['z']], 1)


def report(zpath):
    print(f'=== {zpath}')
    cl = load_layer(zpath, 'clustering-global')
    im = load_layer(zpath, 'img-global', required=False)
    st = load_layer(zpath, 'shower_track-global')
    tf = load_layer(zpath, 'track_fit-global')

    Pc, Cc = pts(cl), np.array(cl['cluster_id'])
    Ps = pts(st)
    Pf, Rf = pts(tf), np.array(tf['real_cluster_id'])

    from scipy.spatial import cKDTree
    tc = cKDTree(Pc)
    dc = np.linalg.norm(Pc - TARGET, axis=1)

    if im is not None:
        Pi = pts(im)
        # (1) img charge with no clustering point nearby -- literal "hole" reading
        d_img, _ = tc.query(Pi)
        near = np.linalg.norm(Pi - TARGET, axis=1) < 10
        print(f'  [1] img points within 10cm of target: {near.sum()}, '
              f'of which orphaned (no clustering pt <1.5cm): {(near & (d_img > 1.5)).sum()}')
        di = np.linalg.norm(Pi - TARGET, axis=1)
        print(f'  [2] nearest clustering point to target: {dc.min():.3f} cm; nearest img point: {di.min():.3f} cm')
    else:
        print('  [1,2] img-global not in this zip (PR-job mabc-pr.zip does not carry it, only Bee uploads do); '
              f'nearest clustering point to target: {dc.min():.3f} cm')

    # (3) fitted trajectory with no charge under it (pr/61 phantom class)
    dcf, _ = tc.query(Pf)
    print(f'  [3] track_fit points unsupported by clustering (>2cm): {(dcf > 2).sum()} / {len(Pf)}')

    # (4) PF segments with zero associated (shower_track) points
    fit = collections.defaultdict(list)
    for x, y, z, r in zip(tf['x'], tf['y'], tf['z'], tf['real_cluster_id']):
        if r != -1:
            fit[r].append((x, y, z))
    assoc = collections.Counter(r for r in st['real_cluster_id'])
    zero = [r for r in fit if assoc.get(r, 0) == 0]
    print(f'  [4] segments with fit pts but zero associated pts: {len(zero)} / {len(fit)} '
          f'({100.0*len(zero)/len(fit):.1f}%)')

    # the split: which cluster owns the target, and the two endpoints of the S6 edge
    m = dc < 0.4
    print(f'  target-cluster: {sorted(set(Cc[m].tolist()))}  (nearest cid {Cc[np.argmin(dc)]})')
    for name, p in (('p1', P1), ('p2', P2)):
        mm = np.linalg.norm(Pc - p, axis=1) < 0.4
        print(f'  {name} cluster membership within 0.4cm: {sorted(set(Cc[mm].tolist()))}')

    sizes = collections.Counter(Cc.tolist())
    for cid in (11, 126):
        if cid in sizes:
            print(f'  cluster {cid}: {sizes[cid]} clustering pts')

    # PF-association holes inside the largest (neutrino-flash-matched) cluster
    if 11 in sizes and sizes[11] > 1000:
        ts = cKDTree(Ps)
        Q = Pc[Cc == 11]
        dq, _ = ts.query(Q)
        orph = (dq > 2).sum()
        print(f'  cluster 11 (main): {len(Q)} pts, {orph} with no PF-association pt within 2cm')


def report_kine(root_path):
    import uproot
    f = uproot.open(root_path)
    t = f['T_kine'].arrays(library='np')
    print(f'=== {root_path}  T_kine')
    print('  kine_reco_Enu           =', t['kine_reco_Enu'][0])
    print('  kine_energy_particle    =', t['kine_energy_particle'][0])
    print('  kine_particle_type      =', t['kine_particle_type'][0])
    print('  kine_energy_included    =', t['kine_energy_included'][0])


def report_edge_dump(jsonl_path):
    print(f'=== {jsonl_path}  S6/pr64 edge record (j=0,k=1,graph_call=0,blk=closest)')
    for line in open(jsonl_path):
        r = json.loads(line)
        if r.get('type') == 'edge' and r.get('j') == 0 and r.get('k') == 1 \
                and r.get('graph_call') == 0 and r.get('blk') == 'closest':
            keep = {k: v for k, v in r.items() if k not in ('matrix', 'planes')}
            print(' ', json.dumps(keep, sort_keys=True))


def report_label(label_path):
    print(f'=== {label_path}')
    d = json.load(open(label_path))
    for k, v in d['labels'].items():
        if abs(v.get('dis', -1) - 1.8756000259399414) < 1e-3 or abs(v.get('dis', -1) - 1.8756000000000086) < 1e-3 \
                or abs(v.get('dis', -1) - 5.047560509586226) < 1e-3:
            print(f"  {k}: killed={v['killed']} verdict={v['verdict']!r} blk={v['blk']}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', action='append', default=[],
                     help='pr_evt18625 directory (e.g. work-pr64r4-scan19/pr_evt18625)')
    ap.add_argument('--bee', action='append', default=[],
                     help='Bee zip path (e.g. bee/prod0811/ncpi0-prod0811.zip)')
    args = ap.parse_args()

    for arm in args.arm:
        report(os.path.join(arm, 'mabc-pr.zip'))
        report_kine(os.path.join(arm, 'tracking-pr.root'))
        report_edge_dump(os.path.join(arm, 'oc56scan-evt18625.jsonl'))
    for bee in args.bee:
        report(bee)

    lbl = 'overclustering_labels/labels-evt18625.json'
    if os.path.exists(lbl):
        report_label(lbl)
