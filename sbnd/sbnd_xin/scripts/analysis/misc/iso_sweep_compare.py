#!/usr/bin/env python3
"""doc pr/18: neutrino-stage iso-band-guard no-regression comparison over sweep arms.

Per event: member-content hashes (hash_archive definition) of the QL zip and
the pctree tarball, baseline arm vs veto-ON arm; plus the guard firing log
("Neutrino iso_band_guard:" cout lines, found in the harness stdout logs --
see firing_map).  Verdict per event:
IDENTICAL (no firing, QL hashes equal), FIRED (veto log lines present;
differences expected), or MISMATCH (no firing but QL output differs -- a
regression, or the evt-286191-style QL nondeterminism, re-run to tell).

The nusel PR zip is deliberately NOT compared: the veto-ON arm is produced by
a binary that also carries 45dae9d0 (nu_skip_cosmic_bundle, PR-stage only,
SBND default ON), so cross-arm PR diffs are expected on events my change never
touched.  PR labels of FIRED events are inspected separately.
"""
import argparse
import concurrent.futures
import glob
import hashlib
import os
import sys
import tarfile
import zipfile


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


def firing_map(root):
    """evt -> veto lines.  The marker is a cout line: wire-cell's -l "<log>"
    captures spdlog only, so the marker lands in the harness stdout capture
    (.log_e<entry>.log from run_full1k_nusel.sh, .batch_ql_evt<ID>.log from
    run_ql_evt.sh batch mode, or .batch_nusel_evt<ID>.log from run_nusel_evt.sh
    batch mode), NOT in ql_evt*/wct_ql_evt*.log."""
    out = {}
    for log in glob.glob(os.path.join(root, '.log_e*.log')) + \
               glob.glob(os.path.join(root, '.batch_ql_evt*.log')) + \
               glob.glob(os.path.join(root, '.batch_nusel_evt*.log')):
        evt = None
        # .batch_*_evt<ID>.log carries the event id in its name.
        base = os.path.basename(log)
        if '_evt' in base:
            try:
                evt = int(base.split('_evt', 1)[1].split('.')[0])
            except ValueError:
                pass
        lines = []
        for line in open(log, errors='replace'):
            if evt is None and '[evt ' in line:
                try:
                    evt = int(line.split('[evt ', 1)[1].split(']')[0])
                except ValueError:
                    pass
            if 'Neutrino iso_band_guard:' in line:
                lines.append(line.split('Neutrino iso_band_guard:', 1)[1].strip())
        if evt is not None and lines:
            out.setdefault(evt, []).extend(lines)
    return out


def compare_event(base, on, evt):
    """Hash the QL products of one event in both arms; safe to run in a pool."""
    cols = []
    same = True
    for rel in (f'ql_evt{evt}/mabc-all-apa.zip',
                f'ql_evt{evt}/pctree-evt{evt}.tar.gz'):
        ha = hash_archive(os.path.join(base, rel))
        hb = hash_archive(os.path.join(on, rel))
        eq = (ha is not None and ha == hb)
        cols.append('=' if eq else ('miss' if ha is None or hb is None else 'DIFF'))
        if not eq:
            same = False
    return evt, cols, same


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True, help='baseline arm (veto off / pre-veto binary)')
    ap.add_argument('--on', required=True, help='veto-ON arm')
    ap.add_argument('--out', required=True)
    ap.add_argument('--jobs', type=int, default=12,
                    help='hash-worker processes (hashing is CPU-bound decompress+sha256)')
    args = ap.parse_args()

    evts_base = set(events_of(args.base))
    evts_on = set(events_of(args.on))
    both = sorted(evts_base & evts_on)
    only = sorted(evts_base ^ evts_on)
    if only:
        print(f'WARNING: {len(only)} events in one arm only: {only[:10]}...')

    fmap = firing_map(args.on)
    n_ident = n_fired = n_mismatch = 0
    fired_rows = []
    mismatch_rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results = dict()
        for evt, cols, same in pool.map(compare_event,
                                        [args.base] * len(both),
                                        [args.on] * len(both), both,
                                        chunksize=8):
            results[evt] = (cols, same)
    with open(args.out, 'w') as out:
        out.write('event\tverdict\tql_zip\tpctree\tn_vetoes\n')
        for evt in both:
            fl = fmap.get(evt, [])
            cols, same = results[evt]
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
    print(f'  IDENTICAL (no firing, QL hashes equal):  {n_ident}')
    print(f'  FIRED (guard fired, diffs expected):    {n_fired}')
    print(f'  MISMATCH (no firing but QL differs):     {n_mismatch}')
    for evt, fl in fired_rows:
        print(f'  fired evt {evt}: {len(fl)} veto(es)')
        for l in fl:
            print(f'      {l}')
    for evt, cols in mismatch_rows:
        print(f'  MISMATCH evt {evt}: {cols}')


if __name__ == '__main__':
    main()
