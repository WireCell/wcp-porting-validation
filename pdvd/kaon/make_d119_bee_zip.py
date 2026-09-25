#!/usr/bin/env python3
"""doc pdvd/119: pack the 10 run-39305 kaon events' mabc-all-apa.zip of one clustering
tag into one Bee set (doc-118 order first, then the other five), renumbering
data/0/0-<name>.json -> data/<i>/<i>-<name>.json.
Usage: make_d119_bee_zip.py <tag> <out.zip>   (e.g. d119sidescratch)"""
import os, sys, zipfile
PDVD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER = [157312, 317673, 245576, 2001, 408552, 36591, 69596, 191916, 326459, 351293]
tag, dst = sys.argv[1:3]
out = zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED)
for idx, e in enumerate(ORDER):
    z = zipfile.ZipFile(f'{PDVD}/work/039305_{e}_{tag}/mabc-all-apa.zip')
    for n in z.namelist():
        parts = n.split('/')   # data/0/0-name.json
        assert parts[0] == 'data' and parts[1] == '0', n
        out.writestr(f"data/{idx}/{idx}-{parts[2].split('-', 1)[1]}", z.read(n))
out.close()
z = zipfile.ZipFile(dst)
print(dst, len(z.namelist()), 'members', sorted(set(n.split('/')[2].split('-', 1)[1] for n in z.namelist())))
