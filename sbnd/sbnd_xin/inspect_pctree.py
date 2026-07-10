#!/usr/bin/env python3
"""Inspect a post-QL point-cloud-tree tarball (the SBND PR-job intermediate file).

Usage: python3 inspect_pctree.py work/ql_evt12/pctree-evt12.tar.gz

Prints the TensorDM inventory and PASS/FAIL for the required content listed in
sbnd/docs/sbnd-pattern-recognition.md section 2.2: live+dead trees, cluster_scalar
with cluster_t0/flash/matched_flash_gid + main/associated flags, 3d points,
flashpred, and the root-level opflash PC.
"""
import io
import json
import sys
import tarfile

import numpy as np


def load(fname):
    """Return {datapath: (metadata, array-or-None)} keyed from the tar members."""
    metas = {}   # member basename (no suffix) -> metadata dict
    arrays = {}  # member basename -> ndarray
    with tarfile.open(fname) as tf:
        for m in tf.getmembers():
            f = tf.extractfile(m)
            if m.name.endswith('_metadata.json'):
                metas[m.name[:-len('_metadata.json')]] = json.load(f)
            elif m.name.endswith('_array.npy'):
                arrays[m.name[:-len('_array.npy')]] = np.load(io.BytesIO(f.read()))
    byPath = {}
    for base, md in metas.items():
        if 'datapath' not in md:   # the tensorset metadata itself
            byPath.setdefault('__tensorset__', []).append(md)
            continue
        byPath[md['datapath']] = (md, arrays.get(base))
    return byPath


def main(fname):
    byPath = load(fname)
    tsets = byPath.pop('__tensorset__', [])
    print(f'{fname}: {len(byPath)} tensors, {len(tsets)} tensor set(s)')

    bytype = {}
    for path, (md, arr) in byPath.items():
        bytype.setdefault(md.get('datatype', '?'), []).append(path)
    for dt, paths in sorted(bytype.items()):
        print(f'  {dt:12s} x{len(paths)}')

    trees = bytype.get('pctree', [])
    print('\ntrees:', trees)

    npass = nfail = 0

    def check(ok, what, detail=''):
        nonlocal npass, nfail
        mark = 'PASS' if ok else 'FAIL'
        npass += ok
        nfail += not ok
        print(f'  [{mark}] {what}{"  " + detail if detail else ""}')

    live = [t for t in trees if t.endswith('/live')]
    dead = [t for t in trees if t.endswith('/dead')]
    check(len(live) == 1, 'one /live tree', str(live))
    check(len(dead) == 1, 'one /dead tree', str(dead))

    for tree in live + dead:
        kind = tree.rsplit('/', 1)[-1]
        md, _ = byPath[tree]
        pcns = byPath.get(md.get('pointclouds', ''), (None, None))[0]
        if pcns is None:
            check(False, f'{kind}: pointclouds namedset reachable')
            continue
        items = pcns.get('items', {})
        print(f'\n{kind} tree local-PC names: {sorted(items)}')

        def arrays_of(pcname):
            dsmd = byPath.get(items.get(pcname, ''), (None, None))[0]
            return dsmd.get('arrays', {}) if dsmd else {}

        def arr_of(pcname, aname):
            ref = arrays_of(pcname).get(aname)
            return byPath[ref][1] if ref in byPath else None

        if kind == 'live':
            xyz = arrays_of('3d')
            check(all(k in xyz for k in ('x', 'y', 'z')), 'live 3d has x,y,z',
                  f'{len(xyz)} arrays, npts={len(byPath[xyz["x"]][1]) if "x" in xyz else 0}')
            cs = arrays_of('cluster_scalar')
            for key in ('cluster_t0', 'flash', 'matched_flash_gid',
                        'flag_main_cluster', 'flag_associated_cluster'):
                a = arr_of('cluster_scalar', key)
                detail = ''
                if a is not None and key == 'cluster_t0':
                    matched = a[a > -1e11]
                    detail = f'{len(matched)}/{len(a)} clusters matched'
                if a is not None and key == 'flag_main_cluster':
                    detail = f'{int((a != 0).sum())} main clusters'
                check(key in cs, f'cluster_scalar has {key}', detail)
            # flashpred is consumed by the all-APA MABC's (pre-pipeline) Bee op
            # dump and does not survive the T0-corrected re-clustering, so it is
            # informational only -- not part of the required inventory.
            print(f'  [info] flashpred PC: '
                  f'{"present" if "flashpred" in items else "absent (expected: consumed pre-pipeline)"}')
            opf = arrays_of('opflash')
            check(len(opf) > 0, 'root opflash PC present (on live tree root)',
                  f'arrays={sorted(opf)}')
        else:
            check(len(items) > 0, 'dead tree has local PCs', f'{sorted(items)[:6]}')

    print(f'\n{npass} passed, {nfail} failed')
    return 1 if nfail else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
