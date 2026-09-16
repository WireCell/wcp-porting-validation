#!/usr/bin/env python3
"""doc pr/149: full per-event identity of two PR arms.

pr85_hash_gate.py compares only mabc-pr.zip and the pctree archive.  This also
compares, for every event both arms hold:
  - mabc-pr.zip / pctree-pr-evt*.tar.gz  member-content rollup (abtest/hash_archive.py)
  - calib-pr-evt*.json                   parsed, with wall-clock '*_ms' keys dropped
  - nusel-evt*.tsv                       bytes
  - tracking-pr.root                     every branch of every TTree (NaN-aware),
                                         over the UNION of tree/branch names
                                         (feedback_flip_proof_matches_measured_arm #3)
Allowed differences are named with --allow TREE.BRANCH (e.g. Trun.cfg_tree for an
arm that ran on a PR_CFG_TREE overlay); they are reported, never silently skipped.

Usage: arm_identity.py <armA> <armB> [--allow T.B ...]
Exit 0 = identical apart from the allowed branches; 1 otherwise.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np
import uproot

sys.path.insert(0, '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/abtest')
import hash_archive  # noqa: E402


def rollup(path):
    r = hashlib.sha256()
    n = 0
    for name, payload in hash_archive.members(path):
        r.update(hashlib.sha256(name.encode() + payload).hexdigest().encode())
        n += 1
    return r.hexdigest(), n


def strip_ms(o):
    if isinstance(o, dict):
        return {k: strip_ms(v) for k, v in o.items() if not k.endswith('_ms')}
    if isinstance(o, list):
        return [strip_ms(v) for v in o]
    return o


def same(x, y):
    """Recursive NaN-aware equality for scalars, numeric arrays and jagged object arrays."""
    xa, ya = np.asarray(x), np.asarray(y)
    if xa.dtype == object or ya.dtype == object:
        if xa.shape != ya.shape:
            return False
        return all(same(p, q) for p, q in zip(xa.ravel(), ya.ravel()))
    if xa.shape != ya.shape:
        return False
    if xa.dtype.kind in 'fc' or ya.dtype.kind in 'fc':
        return bool(np.array_equal(xa, ya, equal_nan=True))
    return bool(np.array_equal(xa, ya))


def root_diff(fa, fb, allow):
    out = []
    ta = uproot.open(fa)
    tb = uproot.open(fb)
    trees_a = {k.split(';')[0] for k, v in ta.classnames().items() if v == 'TTree'}
    trees_b = {k.split(';')[0] for k, v in tb.classnames().items() if v == 'TTree'}
    for t in sorted(trees_a ^ trees_b):
        out.append(f'tree only in {"A" if t in trees_a else "B"}: {t}')
    for t in sorted(trees_a & trees_b):
        A, B = ta[t], tb[t]
        ka, kb = set(A.keys()), set(B.keys())
        for k in sorted(ka ^ kb):
            out.append(f'branch only in {"A" if k in ka else "B"}: {t}.{k}')
        for k in sorted(ka & kb):
            if f'{t}.{k}' in allow:
                continue
            try:
                va = A[k].array(library='np')
                vb = B[k].array(library='np')
            except Exception as e:  # noqa: BLE001
                out.append(f'{t}.{k}: unreadable ({e.__class__.__name__})')
                continue
            if len(va) != len(vb):
                out.append(f'{t}.{k}: entries {len(va)} vs {len(vb)}')
                continue
            if not same(va, vb):
                out.append(f'{t}.{k}: values differ')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arm_a')
    ap.add_argument('arm_b')
    ap.add_argument('--allow', nargs='*', default=[])
    a = ap.parse_args()
    allow = set(a.allow)
    ev = lambda arm: {re.sub(r'.*pr_evt', '', d.rstrip('/')) for d in glob.glob(f'{arm}/pr_evt*/')}
    ea, eb = ev(a.arm_a), ev(a.arm_b)
    common = sorted(ea & eb, key=int)
    bad = 0
    nfile = 0
    print(f'# A={a.arm_a} ({len(ea)} evt)  B={a.arm_b} ({len(eb)} evt)  common={len(common)}  '
          f'only A={len(ea-eb)} only B={len(eb-ea)}  allow={sorted(allow)}')
    for e in common:
        da, db = f'{a.arm_a}/pr_evt{e}', f'{a.arm_b}/pr_evt{e}'
        diffs = []
        for pat in ('mabc-pr.zip', f'pctree-pr-evt{e}.tar.gz'):
            pa, pb = f'{da}/{pat}', f'{db}/{pat}'
            if os.path.exists(pa) != os.path.exists(pb):
                diffs.append(f'{pat}: present in one arm only')
            elif os.path.exists(pa):
                nfile += 1
                if rollup(pa) != rollup(pb):
                    diffs.append(f'{pat}: members differ')
        pa, pb = f'{da}/calib-pr-evt{e}.json', f'{db}/calib-pr-evt{e}.json'
        if os.path.exists(pa) != os.path.exists(pb):
            diffs.append('calib: present in one arm only')
        elif os.path.exists(pa):
            nfile += 1
            if strip_ms(json.load(open(pa))) != strip_ms(json.load(open(pb))):
                diffs.append('calib: differs')
        pa, pb = f'{da}/nusel-evt{e}.tsv', f'{db}/nusel-evt{e}.tsv'
        if os.path.exists(pa) and os.path.exists(pb):
            nfile += 1
            if open(pa, 'rb').read() != open(pb, 'rb').read():
                diffs.append('nusel: differs')
        pa, pb = f'{da}/tracking-pr.root', f'{db}/tracking-pr.root'
        if os.path.exists(pa) and os.path.exists(pb):
            nfile += 1
            diffs += root_diff(pa, pb, allow)
        if diffs:
            bad += 1
            print(f'evt {e}: ' + '; '.join(diffs))
    print(f'# {len(common) - bad}/{len(common)} events identical over {nfile} files'
          + ('' if not bad else f'  -- {bad} DIFFER'))
    sys.exit(1 if bad or (ea ^ eb) else 0)


if __name__ == '__main__':
    main()
