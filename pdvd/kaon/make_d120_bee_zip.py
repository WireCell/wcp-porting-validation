#!/usr/bin/env python3
"""doc pdvd/120: pack the run-39305 kaon events' mabc-pr.zip of one PR tag (the
-beam chain: clustering + track_fit/shower_track/vertices/mc rooted at the beam
entry) into one Bee set, renumbering data/0/0-<name>.json -> data/<i>/<i>-<name>.json.
Events without a beam bundle still ship (their PR layers are empty by
construction), so the set index matches doc 118's order.
Usage: make_d120_bee_zip.py <tag> <out.zip> [evt ...]   (default: the doc-118 six)"""
import os, sys, zipfile
PDVD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = [157312, 317673, 245576, 191916, 2001, 36591]
tag, dst = sys.argv[1:3]
order = [int(a) for a in sys.argv[3:]] or DEFAULT
out = zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED)
for idx, e in enumerate(order):
    z = zipfile.ZipFile(f'{PDVD}/work/039305_{e}_{tag}/mabc-pr.zip')
    for n in z.namelist():
        parts = n.split('/')   # data/0/0-name.json
        assert parts[0] == 'data' and parts[1] == '0', n
        out.writestr(f"data/{idx}/{idx}-{parts[2].split('-', 1)[1]}", z.read(n))
out.close()
z = zipfile.ZipFile(dst)
names = sorted(set(n.split('/')[2].split('-', 1)[1] for n in z.namelist()))
print(dst, len(z.namelist()), 'members over', len(order), 'events:', names)
