#!/usr/bin/env python3
"""doc pr/64 round 4: per-event byte-identity gate between two PR arms.

Compares, for every event present in BOTH arms:
  * mabc-pr.zip           -- member-content hash via abtest/hash_archive.py
  * pctree-pr-evt*.tar.gz -- member-content hash
  * nusel-evt*.tsv        -- exact bytes

Never compares raw archive bytes (tar/zip embed timestamps -- CLAUDE.md M2).

Usage:
    pr64_gate.py <armA> <armB> [--verbose]
Exit 0 iff every compared event matches on every product.
"""
import argparse
import glob
import hashlib
import os
import sys
import tarfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
SBND = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
AB = os.path.join(os.path.dirname(os.path.dirname(SBND)), 'abtest')


def member_hash(path):
    """sha256 over sorted (member_name + payload) -- mirrors
    abtest/hash_archive.py's roll-up so gate labels stay comparable."""
    h = hashlib.sha256()
    if path.endswith('.zip'):
        with zipfile.ZipFile(path) as z:
            for name in sorted(z.namelist()):
                h.update(name.encode())
                h.update(z.read(name))
    else:
        members = {}
        with tarfile.open(path) as t:
            for m in t.getmembers():
                if not m.isfile():
                    continue
                members[m.name] = t.extractfile(m).read()
        for name in sorted(members):
            h.update(name.encode())
            h.update(members[name])
    return h.hexdigest()


def event_products(arm, evt):
    d = os.path.join(arm, 'pr_evt%s' % evt)
    out = {}
    for pat, kind in (('mabc-pr.zip', 'zip'), ('pctree-pr-evt*.tar.gz', 'tar'),
                      ('nusel-evt*.tsv', 'raw')):
        hits = sorted(glob.glob(os.path.join(d, pat)))
        for hh in hits:
            out[os.path.basename(hh)] = (hh, kind)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('armA')
    ap.add_argument('armB')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()
    armA = args.armA if os.path.isdir(args.armA) else os.path.join(SBND, args.armA)
    armB = args.armB if os.path.isdir(args.armB) else os.path.join(SBND, args.armB)

    evA = {os.path.basename(d)[len('pr_evt'):] for d in glob.glob(os.path.join(armA, 'pr_evt*'))}
    evB = {os.path.basename(d)[len('pr_evt'):] for d in glob.glob(os.path.join(armB, 'pr_evt*'))}
    both = sorted(evA & evB)
    print('events: A=%d B=%d compared=%d  (A-only=%d B-only=%d)'
          % (len(evA), len(evB), len(both), len(evA - evB), len(evB - evA)))

    movers, n_ok, n_missing = [], 0, 0
    for evt in both:
        pa = event_products(armA, evt)
        pb = event_products(armB, evt)
        diffs = []
        for name in sorted(set(pa) | set(pb)):
            if name not in pa or name not in pb:
                diffs.append(name + ':MISSING')
                n_missing += 1
                continue
            fa, kind = pa[name]
            fb, _ = pb[name]
            if kind == 'raw':
                same = open(fa, 'rb').read() == open(fb, 'rb').read()
            else:
                same = member_hash(fa) == member_hash(fb)
            if not same:
                diffs.append(name)
        if diffs:
            movers.append((evt, diffs))
        else:
            n_ok += 1
        if args.verbose and diffs:
            print('  MOVER evt%s: %s' % (evt, ' '.join(diffs)))
    print('identical events: %d / %d   movers: %d' % (n_ok, len(both), len(movers)))
    for evt, diffs in movers:
        print('  MOVER evt%s: %s' % (evt, ' '.join(diffs)))
    sys.exit(0 if not movers and not n_missing else 1)


if __name__ == '__main__':
    main()
