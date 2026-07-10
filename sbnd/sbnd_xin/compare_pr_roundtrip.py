#!/usr/bin/env python3
"""Round-trip gate for the PR job (Milestone 2 of sbnd-pattern-recognition.md).

Usage: python3 compare_pr_roundtrip.py <EVT_ID> [bee|tar|all]

Two gates, needing two different PR runs (both against work/ql_evt<ID>):
 - 'tar' after a pass-through run (./run_pr_evt.sh <mode> <idx>, empty
   pipeline): the re-saved tarball's member content hashes must equal the
   input tarball's (tar.gz bytes are never comparable; members are).
 - 'bee' after a switch_scope run (./run_pr_evt.sh <mode> -p switch_scope
   <idx>): the 'clustering' Bee layer of mabc-pr.zip must match
   mabc-all-apa.zip (per-point multisets of x/y/z/q/cluster_id;
   real_cluster_id reported separately -- display-only, known not to persist
   for flash-merged clusters), and the dead-area layers must be
   byte-identical.  A switch_scope run cannot pass 'tar': the visitor
   legitimately rebuilds clusters and drops the stale isolated/perblob
   arrays.
"""
import hashlib
import json
import sys
import tarfile
import zipfile
from collections import Counter


def bee_layer(zf, member):
    return json.loads(zipfile.ZipFile(zf).read(member))


def compare_clustering(qlz, prz):
    a = bee_layer(qlz, 'data/0/0-clustering-global.json')
    b = bee_layer(prz, 'data/0/0-clustering-global.json')
    ok = True
    for k in ('runNo', 'subRunNo', 'eventNo', 'geom', 'type'):
        if a[k] != b[k]:
            print(f'  [FAIL] clustering header {k}: {a[k]!r} vs {b[k]!r}')
            ok = False
    # y,z,q,cluster_id must be exact.  x (= x_t0cor) is allowed a bounded
    # difference on flash-merged clusters only: the file carries the per-SUB-
    # cluster t0 correction from before examine_bundles' flash merge, while the
    # PR re-run of switch_scope applies the merged cluster's single t0.  The
    # bound is flash_group_window x drift speed: 80 ns x 1.563 mm/us = 0.0125
    # cm (checked with margin below).
    X_TOL = 0.02  # cm
    pa = Counter(zip(a['y'], a['z'], a['q'], a['cluster_id']))
    pb = Counter(zip(b['y'], b['z'], b['q'], b['cluster_id']))
    if pa != pb:
        da, db = pa - pb, pb - pa
        print(f'  [FAIL] clustering (y,z,q,cluster_id) differ: ql-only {sum(da.values())}, pr-only {sum(db.values())}')
        ok = False
    else:
        xs_a, xs_b = {}, {}
        for d, xs in ((a, xs_a), (b, xs_b)):
            for i, c in enumerate(d['cluster_id']):
                xs.setdefault(c, []).append(d['x'][i])
        worst, nshift = 0.0, 0
        for c in xs_a:
            for xa, xb in zip(sorted(xs_a[c]), sorted(xs_b[c])):
                dx = abs(xa - xb)
                if dx > 0:
                    nshift += 1
                    worst = max(worst, dx)
        if worst == 0:
            print(f'  [PASS] clustering points identical (x,y,z,q,cluster_id): {len(a["x"])} points')
        elif worst <= X_TOL:
            print(f'  [PASS] clustering points: y/z/q/cluster_id exact; {nshift} pts with '
                  f'|dx| <= {worst:.4g} cm (flash-merge t0 re-correction, bound {X_TOL} cm)')
        else:
            print(f'  [FAIL] clustering x differs beyond tolerance: {nshift} pts, worst {worst:.4g} cm')
            ok = False
    ra = Counter(zip(a['cluster_id'], a['real_cluster_id']))
    rb = Counter(zip(b['cluster_id'], b['real_cluster_id']))
    if ra == rb:
        print('  [PASS] real_cluster_id identical too')
    else:
        diff_cl = sorted({c for c, _ in (ra - rb)} | {c for c, _ in (rb - ra)})
        print(f'  [info] real_cluster_id differs for clusters {diff_cl} '
              '(expected: per-cluster-only pcarray, not persisted)')
    return ok


def compare_deadarea(qlz, prz):
    za, zb = zipfile.ZipFile(qlz), zipfile.ZipFile(prz)
    names = [n for n in zb.namelist() if 'deadarea' in n]
    ok = True
    for n in names:
        same = za.read(n) == zb.read(n)
        print(f'  [{"PASS" if same else "FAIL"}] dead-area {n} byte-identical')
        ok = ok and same
    return ok


def member_hashes(fname):
    out = {}
    with tarfile.open(fname) as tf:
        for m in tf.getmembers():
            out[m.name] = hashlib.sha256(tf.extractfile(m).read()).hexdigest()
    return out


def compare_tarballs(intar, outtar):
    ha, hb = member_hashes(intar), member_hashes(outtar)
    if ha == hb:
        print(f'  [PASS] tarball member hashes identical ({len(ha)} members)')
        return True
    only_a = set(ha) - set(hb)
    only_b = set(hb) - set(ha)
    differ = [n for n in set(ha) & set(hb) if ha[n] != hb[n]]
    print(f'  [FAIL] tarballs differ: {len(only_a)} only-in, {len(only_b)} only-out, {len(differ)} changed')
    for n in sorted(differ)[:10]:
        print(f'    changed: {n}')
    for n in sorted(only_b)[:10]:
        print(f'    only-out: {n}')
    return False


def main(evt, mode='all'):
    qlz = f'work/ql_evt{evt}/mabc-all-apa.zip'
    prz = f'work/pr_evt{evt}/mabc-pr.zip'
    intar = f'work/ql_evt{evt}/pctree-evt{evt}.tar.gz'
    outtar = f'work/pr_evt{evt}/pctree-pr-evt{evt}.tar.gz'
    print(f'=== evt {evt} ({mode})')
    ok = True
    if mode in ('bee', 'all'):
        ok = compare_clustering(qlz, prz) and ok
        ok = compare_deadarea(qlz, prz) and ok
    if mode in ('tar', 'all'):
        ok = compare_tarballs(intar, outtar) and ok
    print('OVERALL:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'all'))
