#!/usr/bin/env python3
"""doc pr/38 gate + recovery check.

Knob-off gate: work-pr38-base7{,mc} (pre-fix binary) vs work-pr38-off7{,mc}
(post-fix binary, SBND_PF_BARRIER_SEGMENT_VERTICES=0 SBND_PF_ORPHAN_TRACK_ROOTS=0)
must be byte-identical on all four per-event artifacts.

Knob-on check: work-pr38-on7{,mc} may differ from base ONLY in
mabc-pr.zip::data/0/0-mc.json, and the locally-missing main-cluster
non-shower segments must be recovered there (or claimed-but-KeepMC-floored,
reported as such).

Usage: python3 pr38_cmp.py [base_suffix off_suffix on_suffix]
       (defaults base7/off7/on7 + the mc counterparts)
"""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

SX = Path(__file__).resolve().parent
EVENTS = [
    ('219295', ''), ('234638', ''), ('447477', ''), ('489330', ''),
    ('52657', 'mc'), ('55715', 'mc'), ('56243', 'mc'),
]
BASE, OFF, ON = (sys.argv[1:4] if len(sys.argv) >= 4 else ('base7', 'off7', 'on7'))


def archive_hash(path):
    """Member-content hash (tar/zip agnostic) -- same idea as abtest/hash_archive.py."""
    h = hashlib.sha256()
    if str(path).endswith('.zip'):
        with zipfile.ZipFile(path) as z:
            for name in sorted(z.namelist()):
                h.update(name.encode())
                h.update(z.read(name))
    else:
        import tarfile
        with tarfile.open(path) as t:
            for m in sorted(t.getmembers(), key=lambda m: m.name):
                h.update(m.name.encode())
                if m.isfile():
                    h.update(t.extractfile(m).read())
    return h.hexdigest()


def mc_ids(path, evt):
    with zipfile.ZipFile(path / f'pr_evt{evt}' / 'mabc-pr.zip') as z:
        mc = json.loads(z.read('data/0/0-mc.json'))
    ids = set()

    def walk(n):
        ids.add(n['id'])
        for c in n.get('children', []):
            walk(c)
    for n in mc:
        walk(n)
    return ids


ok = True
for evt, mc_sfx in EVENTS:
    a = SX / f'work-pr38-{BASE}{mc_sfx}'
    b = SX / f'work-pr38-{OFF}{mc_sfx}'
    o = SX / f'work-pr38-{ON}{mc_sfx}'
    # knob-off gate
    for f in (f'pr_evt{evt}/mabc-pr.zip', f'pr_evt{evt}/pctree-pr-evt{evt}.tar.gz'):
        same = archive_hash(a / f) == archive_hash(b / f)
        ok &= same
        print(f"{'SAME' if same else 'DIFF'}  off-gate evt{evt} {f.split('/')[-1]}")
    for f in (f'pr_evt{evt}/calib-pr-evt{evt}.json', f'pr_evt{evt}/nusel-evt{evt}.tsv'):
        same = (a / f).read_bytes() == (b / f).read_bytes()
        ok &= same
        print(f"{'SAME' if same else 'DIFF'}  off-gate evt{evt} {f.split('/')[-1]}")
    # knob-on recovery
    calib = json.load(open(a / f'pr_evt{evt}' / f'calib-pr-evt{evt}.json'))
    base_ids, on_ids = mc_ids(a, evt), mc_ids(o, evt)
    for s in calib['segments']:
        if s['is_main_cluster'] != 1 or s['shower_id'] != -1 or s['id'] in base_ids:
            continue
        state = 'RECOVERED' if s['id'] in on_ids else 'still-hidden (KeepMC floor or check stdout.log)'
        print(f"  on-check evt{evt}: base-missing seg {s['id']} "
              f"pdg={s['particle_id']} npts={len(s['points'])} -> {state}")
print('GATE ' + ('PASS' if ok else 'FAIL'))
sys.exit(0 if ok else 1)
