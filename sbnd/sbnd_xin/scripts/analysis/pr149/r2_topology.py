#!/usr/bin/env python3
"""doc pr/149 round 2: does the resample change the OBJECT set, not just the points?

The resample (ClusteringResampleLive) keeps every cluster and blob, but it runs before
switch_scope, whose in-volume split reads the new points (extra points can only turn a blob's
filter fail -> pass), and before every later stage that splits or merges.  So the per-event metric
join (main cluster, segments) may be on a different object set.  This counts it, per event, from
tracking-pr.root T_cluster: number of clusters, number in scope, and the in-scope main cluster's
length.

Usage: r2_topology.py --base ARM --arms ARM [ARM ...] --samples S [S ...] [--manifest DIR]
       (ARM = work-<sample>-<ARM>, e.g. pr149s2r2s0)
"""
import argparse
import math
import os

import uproot

SX = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'


def per_event(arm, sample):
    root = os.path.join(SX, f'work-{sample}-{arm}')
    out = {}
    if not os.path.isdir(root):
        raise SystemExit(f'missing arm {root}')
    for d in sorted(os.listdir(root)):
        if not d.startswith('pr_evt'):
            continue
        f = os.path.join(root, d, 'tracking-pr.root')
        evt = int(d[len('pr_evt'):])
        if not os.path.exists(f):
            out[evt] = None
            continue
        t = uproot.open(f)['T_cluster'].arrays(['cluster_id', 'in_scope', 'is_main', 'length_cm'], library='np')
        n = len(t['cluster_id'])
        nin = int((t['in_scope'] != 0).sum())
        mains = [float(t['length_cm'][i]) for i in range(n) if t['in_scope'][i] and t['is_main'][i]]
        out[evt] = dict(n=n, nin=nin, nmain=len(mains), main_len=max(mains) if mains else float('nan'))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--arms', nargs='+', required=True)
    ap.add_argument('--samples', nargs='+', required=True)
    a = ap.parse_args()
    print(f'# doc pr/149 round 2 topology vs {a.base} (tracking-pr.root T_cluster)')
    for s in a.samples:
        B = per_event(a.base, s)
        for arm in a.arms:
            X = per_event(arm, s)
            keys = sorted(k for k in B if k in X and B[k] is not None and X[k] is not None)
            dn = sum(1 for k in keys if X[k]['n'] != B[k]['n'])
            dnin = sum(1 for k in keys if X[k]['nin'] != B[k]['nin'])
            dmain = sum(1 for k in keys if X[k]['nmain'] != B[k]['nmain'])
            dlen = sum(1 for k in keys if not (math.isnan(X[k]['main_len']) and math.isnan(B[k]['main_len']))
                       and not abs(X[k]['main_len'] - B[k]['main_len']) <= 1.0)
            dn_sum = sum(X[k]['n'] - B[k]['n'] for k in keys)
            print(f'{s:8s} {arm:18s} events {len(keys)}  n_clusters differs {dn} (net {dn_sum:+d})  '
                  f'n_in_scope differs {dnin}  n_main differs {dmain}  main length moved >1 cm {dlen}')


if __name__ == '__main__':
    main()
