#!/usr/bin/env python3
"""doc pr/48 sec 9.6: per-event examination of every TEB-BREAK mover.

For each mover whose knobs-ON log carries a BROKE line, print:
  - the scan diagnostics (arms, rises, scores, route) from the log,
  - the base vs new MAIN vertex (q==15000 in vertices-global.json),
  - whether the main vertex moved, and the distance break-point -> new main.
Verdict guidance: a genuine back-to-back case shows both-end rise >= the route
floor and either (a) the main vertex relocated onto/near the break point, or
(b) selection declined the break (main vertex unmoved) and the structure healed
-- both are designed outcomes; anything else needs eyes.

Usage: python3 teb_case_exam.py [BASE_LABEL NEW_LABEL]
"""
import sys, os, glob, json, zipfile, re, math
sys.path.insert(0, '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/abtest')
import hash_archive as ha

SB = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'
BASE = os.path.join(SB, sys.argv[1] if len(sys.argv) > 2 else 'work-pr47f-on1k')
NEW = os.path.join(SB, sys.argv[2] if len(sys.argv) > 2 else 'work-pr48-on1kc')

scanned = set()
with open(os.path.join(SB, 'docs/pr/mcp1k-50-cb0805.index.txt')) as f:
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) == 2:
            scanned.add(int(parts[1]))

def main_vtx(zpath):
    z = zipfile.ZipFile(zpath)
    v = json.loads(z.read('data/0/0-vertices-global.json'))
    for x, y, zz, q in zip(v['x'], v['y'], v['z'], v['q']):
        if q == 15000.0:
            return (x, y, zz)
    return None

def dist(a, b):
    if a is None or b is None:
        return float('nan')
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))

evts = sorted(set(int(os.path.basename(p).replace('pr_evt', ''))
                  for p in glob.glob(os.path.join(BASE, 'pr_evt*'))) &
              set(int(os.path.basename(p).replace('pr_evt', ''))
                  for p in glob.glob(os.path.join(NEW, 'pr_evt*'))))

n_sel, n_decl = 0, 0
for evt in evts:
    bp = os.path.join(BASE, 'pr_evt%d' % evt, 'mabc-pr.zip')
    np_ = os.path.join(NEW, 'pr_evt%d' % evt, 'mabc-pr.zip')
    if not (os.path.exists(bp) and os.path.exists(np_)):
        continue
    if dict(ha.members(bp)) == dict(ha.members(np_)):
        continue
    log = os.path.join(NEW, 'pr_evt%d' % evt, 'wct_pr_evt%d.log' % evt)
    text = open(log, errors='replace').read() if os.path.exists(log) else ''
    broke = re.findall(r'qdx: BROKE cluster \d+ at fit idx (\d+) \(([-\d.]+),([-\d.]+),([-\d.]+)\)cm', text)
    if not re.search(r'qdx: BROKE cluster \d+ at fit idx \d+', text):
        continue
    diag = re.findall(r'break_two_end_dqdx: cluster \d+ seg len [^\n]*found=true', text)
    tag = ' [OWNER-SCANNED]' if evt in scanned else ' [NEW CASE]'
    print('=== evt %d%s' % (evt, tag))
    for d in diag[:2]:
        print('    ', d)
    bpt = None
    if broke:
        bpt = tuple(float(broke[0][i]) for i in (1, 2, 3))
    else:
        print('     (BROKE line torn -- break point unavailable from log)')
    mb, mn = main_vtx(bp), main_vtx(np_)
    moved = dist(mb, mn)
    tob = dist(bpt, mn) if bpt else float('nan')
    if moved > 0.01:
        n_sel += 1
        print('     main vertex MOVED %.2fcm: %s -> %s   (new main is %.2fcm from break pt)'
              % (moved, mb, mn, tob))
    else:
        n_decl += 1
        print('     main vertex unmoved at %s   (break pt %.2fcm away; selection declined, designed outcome)'
              % (mn, tob))
print()
print('summary: %d breaks moved the main vertex, %d declined-and-healed' % (n_sel, n_decl))
