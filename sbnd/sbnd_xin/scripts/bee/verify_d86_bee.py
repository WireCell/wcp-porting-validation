#!/usr/bin/env python3
"""doc 86 -- prove each built Bee zip carries the events the manifest names.

The pick lists are event ids only, and make_pr_bee.py takes the FIRST -q/-p
root that has each id, so an id shared by two samples could silently ship the
wrong event.  This re-reads every zip and checks, per Bee index:

  runNo / subRunNo / eventNo  ==  the d86-final.tsv row for that (set, index)

and that the four PR layers plus img-global are actually present.
"""
import json
import os
import sys
import zipfile
from collections import defaultdict

SX = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
FINAL = os.path.join(SX, 'docs', '86_video', 'd86-final.tsv')
NEED = ['img-global', 'track_fit-global', 'shower_track-global',
        'vertices-global', 'mc']

want = defaultdict(dict)
with open(FINAL) as f:
    hdr = f.readline().rstrip('\n').split('\t')
    for line in f:
        d = dict(zip(hdr, line.rstrip('\n').split('\t')))
        want[d['set']][int(d['bee_index'])] = (
            int(d['run']), int(d['subrun']), int(d['event']))

bad = 0
for setname in want:
    zp = os.path.join(SX, 'bee', 'd86', f'd86-{setname}.zip')
    if not os.path.isfile(zp):
        print(f'MISSING zip {zp}')
        bad += 1
        continue
    with zipfile.ZipFile(zp) as zf:
        names = zf.namelist()
        idxs = sorted({int(n.split('/')[1]) for n in names
                       if n.startswith('data/') and n.count('/') >= 2})
        if idxs != sorted(want[setname]):
            print(f'{setname}: INDEX MISMATCH zip={idxs} want={sorted(want[setname])}')
            bad += 1
        for i in idxs:
            layers = {os.path.basename(n)[len(f'{i}-'):-len('.json')]
                      for n in names if os.path.basename(n).startswith(f'{i}-')}
            miss = [l for l in NEED if l not in layers]
            hdrn = f'data/{i}/{i}-img-global.json'
            if hdrn not in names:
                print(f'{setname}[{i}]: no img-global')
                bad += 1
                continue
            d = json.loads(zf.read(hdrn))
            got = (int(d['runNo']), int(d['subRunNo']), int(d['eventNo']))
            exp = want[setname].get(i)
            ok = got == exp
            bad += 0 if ok and not miss else 1
            print(f"{'OK ' if ok and not miss else 'BAD'} {setname}[{i}] "
                  f"run/sub/evt={got} want={exp}"
                  + (f' MISSING LAYERS {miss}' if miss else ''))

print('\nFAILURES:', bad)
sys.exit(1 if bad else 0)
