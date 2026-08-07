#!/usr/bin/env python3
"""doc pr/47 sec 8: compare work-pr47f-on1k (cathode_wide_kink_angle=25 ON)
vs work-pr46-m1konb (production baseline at the same HEAD, knob absent) --
archive movers, per-mover cathode-band vertex diff, nusel diffs (both
granularities; nusel tsvs are WHITESPACE-separated despite the extension)."""
import sys, os, glob, json, zipfile
sys.path.insert(0, '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/abtest')
import hash_archive as ha

SB = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'
BASE = os.path.join(SB, 'work-pr46-m1konb')
NEW = os.path.join(SB, 'work-pr47f-on1k')

base_evts = sorted(int(os.path.basename(p).replace('pr_evt', ''))
                   for p in glob.glob(os.path.join(BASE, 'pr_evt*')))
new_evts = sorted(int(os.path.basename(p).replace('pr_evt', ''))
                  for p in glob.glob(os.path.join(NEW, 'pr_evt*')))
common = sorted(set(base_evts) & set(new_evts))
print('base events:', len(base_evts), 'new events:', len(new_evts), 'common:', len(common))

movers = []
for evt in common:
    bp = os.path.join(BASE, 'pr_evt%d' % evt, 'mabc-pr.zip')
    np_ = os.path.join(NEW, 'pr_evt%d' % evt, 'mabc-pr.zip')
    if not (os.path.exists(bp) and os.path.exists(np_)):
        print('  MISSING artifact evt', evt)
        continue
    a = dict(ha.members(bp))
    b = dict(ha.members(np_))
    dm = [k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
    if dm:
        movers.append((evt, dm))

print('\nARCHIVE-LEVEL: %d/%d events differ' % (len(movers), len(common)))
for evt, dm in movers:
    print('  %d  %s' % (evt, dm))

print('\n--- per-mover vertex diff (cathode band |x|<5cm highlighted) ---')
for evt, dm in movers:
    if not any('vertices-global' in m for m in dm):
        print('  evt %d: no vertices-global change (%s)' % (evt, dm))
        continue
    bp = os.path.join(BASE, 'pr_evt%d' % evt, 'mabc-pr.zip')
    np_ = os.path.join(NEW, 'pr_evt%d' % evt, 'mabc-pr.zip')
    with zipfile.ZipFile(bp) as z:
        vb = json.loads(z.read('data/0/0-vertices-global.json'))
    with zipfile.ZipFile(np_) as z:
        vn = json.loads(z.read('data/0/0-vertices-global.json'))
    xb = set(round(x, 3) for x in vb.get('x', []))
    xn = set(round(x, 3) for x in vn.get('x', []))
    new_x = xn - xb
    removed_x = xb - xn
    new_cathode = sorted(x for x in new_x if abs(x) < 5)
    print('  evt %d: nvtx %d->%d  new_x_in_cathode_band(<5cm)=%s  removed=%d added=%d' % (
        evt, len(xb), len(xn), new_cathode, len(removed_x), len(new_x)))

def load_nusel(root, fname):
    rows = {}
    path = os.path.join(root, fname)
    with open(path) as f:
        header = f.readline().split()
        for line in f:
            parts = line.split()
            d = dict(zip(header, parts))
            rows[int(d.get('event', d.get('evt', -1)))] = d
    return rows

for fname in ('nusel-events.tsv', 'nusel-table.tsv'):
    nb = load_nusel(BASE, fname)
    nn = load_nusel(NEW, fname)
    ndiff = 0
    for evt in common:
        if evt in nb and evt in nn and nb[evt] != nn[evt]:
            ndiff += 1
            dk = [k for k in nb[evt] if nb[evt].get(k) != nn[evt].get(k)]
            print('  evt %d %s diff: %s' % (evt, fname,
                  {k: (nb[evt][k], nn[evt][k]) for k in dk}))
    print('%s diffs: %d/%d' % (fname, ndiff, len(common)))
