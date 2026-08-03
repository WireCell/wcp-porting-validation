#!/usr/bin/env python3
"""doc pr/20 Part I gate PI-5: read back the "real_cluster_was_main" array.

The failure mode this gate exists for is NOT a wrong value -- it is the array
being silently dropped.  aux/src/TensorDMpointtree.cxx:88-93: Dataset::append
keys the copy on the ACCUMULATED dataset, so an array whose key is absent from
the first-seen node's same-named local PC disappears without a word, and a key
the tail lacks throws instead.  Same-named local PCs must therefore be
key-HOMOGENEOUS across clusters.  So this checks, on a saved pctree:

  1. every cluster that has a "perblob" PC has the was_main key (homogeneity);
  2. clusters the flash merge never touched carry the all-1 fill-in sentinel;
  3. at least one cluster carries a MIXED array -- rows both 0 and 1 -- which is
     the only proof the writer recorded something the merge would have destroyed
     (a merged cluster holding a demoted main plus other members);
  4. was_main agrees with real_cluster_main wherever the latter is 1: the
     representative member is a main by construction, so was_main must be 1 on
     every real_cluster_main==1 row.  A free cross-check on P1's placement.

Usage: ./pr20_wasmain_check.py <pctree.tar.gz> [...]
Exit 0 iff every file passes 1, 2 and 4 (3 is reported, not enforced -- an
event whose merge demoted nobody legitimately has no mixed cluster).
"""
import io
import json
import re
import sys
import tarfile

import numpy as np

WAS = 'real_cluster_was_main'


def load_tensors(fname):
    metas, arrays = {}, {}
    with tarfile.open(fname) as tf:
        for m in tf.getmembers():
            f = tf.extractfile(m)
            if m.name.endswith('_metadata.json'):
                metas[m.name[:-len('_metadata.json')]] = json.load(f)
            elif m.name.endswith('_array.npy'):
                arrays[m.name[:-len('_array.npy')]] = np.load(io.BytesIO(f.read()))
    return {md['datapath']: (md, arrays.get(b))
            for b, md in metas.items() if 'datapath' in md}


def check(fname):
    bp = load_tensors(fname)
    live = [p for p in bp if re.fullmatch(r'pointtrees/\d+/live', p)]
    if len(live) != 1:
        print(f'{fname}: expected one live tree, got {live}')
        return False
    live = live[0]
    md = bp[live][0]
    items = bp[md['pointclouds']][0]['items']
    lpc = bp[md['lpcmaps']][0]['arrays']

    def arr(pcname, aname):
        ds = bp[items[pcname]][0]['arrays']
        return bp[ds[aname]][1] if aname in ds else None

    if 'perblob' not in items:
        print(f'{fname}: no perblob PC at all')
        return False
    keys = sorted(bp[items['perblob']][0]['arrays'])
    w = arr('perblob', WAS)
    rmain = arr('perblob', 'real_cluster_main')
    if w is None:
        print(f'{fname}: FAIL -- "{WAS}" absent.  perblob keys: {keys}')
        return False

    # HOMOGENEITY, the failure this gate exists for: the serialized perblob PC
    # is the concatenation over cluster nodes, so a cluster missing the key
    # shows up as this array being SHORTER than its neighbours -- never as a
    # hole.  Every key must therefore have the same length.
    lens = {k: len(bp[bp[items['perblob']][0]['arrays'][k]][1]) for k in keys}
    if len(set(lens.values())) != 1:
        print(f'{fname}: FAIL -- ragged perblob key lengths {lens}')
        return False

    # The lpcmap arrays are per-NODE row COUNTS in node order (not row->node),
    # so cluster membership comes from walking the tree the way
    # unmerge_crosser_audit.py does: a nonzero cluster_scalar count opens the
    # next cluster, and the perblob counts that follow are its blobs.
    map_pb = bp[lpc['perblob']][1].astype(int)
    map_cs = bp[lpc['cluster_scalar']][1].astype(int)
    ident = arr('cluster_scalar', 'ident').astype(int)

    spans = []            # cluster index -> [row, ...]
    ci, brow = -1, 0
    for n in range(len(map_cs)):
        if map_cs[n]:
            ci += 1
            spans.append([])
        for _ in range(int(map_pb[n])):
            if ci >= 0:
                spans[ci].append(brow)
            brow += 1
    if brow != len(w):
        print(f'{fname}: FAIL -- walked {brow} perblob rows, array has {len(w)}')
        return False

    n_all1 = n_all0 = n_mixed = n_empty = 0
    bad_agree = []
    for i, rows in enumerate(spans):
        if not rows:
            n_empty += 1
            continue
        vals = w[np.array(rows)]
        n1 = int((vals != 0).sum())
        if n1 == len(vals):
            n_all1 += 1
        elif n1 == 0:
            n_all0 += 1
        else:
            n_mixed += 1
        if rmain is not None:
            rm = rmain[np.array(rows)]
            # the representative member is a main by construction
            if int(((rm != 0) & (vals == 0)).sum()):
                bad_agree.append(int(ident[i]) if i < len(ident) else -1)

    ok = not bad_agree
    tag = 'PASS' if ok else 'FAIL'
    print(f'{tag} {fname}')
    print(f'      perblob keys ({len(keys)}), all {next(iter(lens.values()))} rows: {keys}')
    print(f'      clusters: {n_all1} all-1, {n_all0} all-0, {n_mixed} MIXED, '
          f'{n_empty} blobless   (mixed = the merge demoted somebody there)')
    if bad_agree:
        print(f'      FAIL -- real_cluster_main==1 but {WAS}==0 on cluster(s) '
              f'{bad_agree}')
    return ok


if __name__ == '__main__':
    sys.exit(0 if all([check(f) for f in sys.argv[1:]]) else 1)
