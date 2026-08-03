#!/usr/bin/env python3
"""doc pr/14 sec 7.4: OFF-vs-ON no-regression comparison over the sweep arms.

Per event: member-content hashes (hash_archive) of the QL zip, the pctree
tarball and the nusel PR zip, OFF arm vs ON arm; plus the rescue firing log.
Verdict per event: IDENTICAL (no firing, all hashes equal), FIRED (rescue log
lines present; differences expected), or MISMATCH (no firing but something
differs -- a regression).
"""
import argparse
import glob
import hashlib
import os
import re
import subprocess
import sys
import tarfile
import zipfile

AB = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/abtest'
sys.path.insert(0, AB)

def hash_archive(path):
    """Member-content hash, same definition as abtest/hash_archive.py."""
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    if path.endswith('.zip'):
        with zipfile.ZipFile(path) as z:
            for name in sorted(z.namelist()):
                h.update(name.encode())
                h.update(z.read(name))
    else:
        with tarfile.open(path) as t:
            members = sorted(t.getmembers(), key=lambda m: m.name)
            for m in members:
                h.update(m.name.encode())
                if m.isfile():
                    h.update(t.extractfile(m).read())
    return h.hexdigest()[:16]


def events_of(root):
    return sorted(int(d.split('ql_evt')[1]) for d in glob.glob(os.path.join(root, 'ql_evt*')))


def firing(root, evt):
    log = os.path.join(root, f'ql_evt{evt}', f'wct_ql_evt{evt}.log')
    if not os.path.exists(log):
        return []
    out = []
    for line in open(log, errors='replace'):
        if 'rescue round' in line:
            out.append(line.split('rescue round', 1)[1].strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--off', required=True)
    ap.add_argument('--on', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    evts_off = set(events_of(args.off))
    evts_on = set(events_of(args.on))
    both = sorted(evts_off & evts_on)
    only = sorted(evts_off ^ evts_on)
    if only:
        print(f'WARNING: {len(only)} events in one arm only: {only[:10]}...')

    n_ident = n_fired = n_mismatch = 0
    fired_rows = []
    mismatch_rows = []
    with open(args.out, 'w') as out:
        out.write('event\tverdict\tql_zip\tpctree\tpr_zip\tn_moves\n')
        for evt in both:
            fl = firing(args.on, evt)
            cols = []
            same = True
            for rel in (f'ql_evt{evt}/mabc-all-apa.zip',
                        f'ql_evt{evt}/pctree-evt{evt}.tar.gz',
                        f'nusel_evt{evt}/mabc-pr.zip'):
                ha = hash_archive(os.path.join(args.off, rel))
                hb = hash_archive(os.path.join(args.on, rel))
                eq = (ha is not None and ha == hb)
                cols.append('=' if eq else ('miss' if ha is None or hb is None else 'DIFF'))
                if not eq:
                    same = False
            if fl:
                verdict = 'FIRED'
                n_fired += 1
                fired_rows.append((evt, fl))
            elif same:
                verdict = 'IDENTICAL'
                n_ident += 1
            else:
                verdict = 'MISMATCH'
                n_mismatch += 1
                mismatch_rows.append((evt, cols))
            out.write(f'{evt}\t{verdict}\t' + '\t'.join(cols) + f'\t{len(fl)}\n')

    print(f'events compared: {len(both)}')
    print(f'  IDENTICAL (no firing, all hashes equal): {n_ident}')
    print(f'  FIRED (rescue applied, diffs expected):  {n_fired}')
    print(f'  MISMATCH (no firing but output differs): {n_mismatch}')
    for evt, fl in fired_rows:
        print(f'  fired evt {evt}: {len(fl)} move(s)')
        for l in fl:
            print(f'      {l}')
    for evt, cols in mismatch_rows:
        print(f'  MISMATCH evt {evt}: {cols}')


if __name__ == '__main__':
    main()
