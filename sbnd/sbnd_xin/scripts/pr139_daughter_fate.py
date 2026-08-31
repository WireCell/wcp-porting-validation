#!/usr/bin/env python3
# doc pr/139 sec 2.2 -- what actually happens to a peeled daughter?
"""Size the re-home front before building it.

doc pr/138 sec B4 argued from the owner's comments that a cut leaving an orphan
has not finished the job.  That is his requirement and it stands.  What was NOT
measured is the SIZE of it, and the first attempt at this number was wrong: a
lookup by the daughter's new start-segment id missed most of them and reported
"0 of 51 daughters are used by the pi0 finder".  Joining the other way -- iterate
the dump's showers[] and ask which are daughters -- gives the opposite answer.

The honest sizing is below: daughters are paired at ABOVE the base rate, so the
re-home front is not "the finder cannot see them at all".  It is narrower: the
EM-typed daughters above the pairing floor that still did not pair.

Repro:
    python3 scripts/pr139_daughter_fate.py
"""
import os, re, sys, json, glob, csv, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pr137_lib as L

ARM = 'work-pr138r2-c90on'
PEEL = re.compile(r'SHOWER_SPLIT peel shower=(-?\d+) part=(\d+) new_start=(-?\d+) '
                  r'nseg=(\d+) conn=(\d+)')
PART = re.compile(r'SHOWER_SPLIT part shower=(-?\d+) part=(\d+) nseg=(\d+) q=(\S+)')

peel = collections.defaultdict(list)
partq = collections.defaultdict(dict)
for log in sorted(glob.glob(ARM + '-*/pr_evt*/stdout.log')):
    ev = int(re.search(r'pr_evt(\d+)', log).group(1))
    for ln in open(log, errors='replace'):
        if 'SHOWER_SPLIT' not in ln:
            continue
        m = PEEL.search(ln)
        if m:
            peel[ev].append(dict(parent=int(m.group(1)), part=int(m.group(2)),
                                 start=int(m.group(3)), nseg=int(m.group(4)),
                                 conn=int(m.group(5))))
            continue
        m = PART.search(ln)
        if m:
            partq[(ev, int(m.group(1)))][int(m.group(2))] = float(m.group(4))


def dump(ev):
    for s in ('mcp1k', 'mcp2k', 'ncpi0', 'nuecc48'):
        p = '%s-%s/pr_evt%d/calib-pr-evt%d.json' % (ARM, s, ev, ev)
        if os.path.exists(p):
            return json.load(open(p))
    return None


rows, others = [], []
for ev, pl in sorted(peel.items()):
    d = dump(ev)
    if d is None:
        continue
    SR = L.shower_recs(d)
    starts = {p['start']: p for p in pl}
    parents = {p['parent'] for p in pl}
    for nid, rec in sorted(SR.items()):
        pid = rec.get('pio_id', -1)
        pid = -1 if pid is None else int(pid)
        r = dict(event=ev, node=nid, pdg=int(rec.get('particle_id') or 0),
                 kine=float(rec.get('kine_charge') or 0), paired=int(pid >= 0))
        if nid in starts:
            r.update(kind='daughter', conn=starts[nid]['conn'], nseg=starts[nid]['nseg'])
            rows.append(r)
        elif nid in parents:
            r.update(kind='parent', conn=-1, nseg=-1)
            rows.append(r)
        elif abs(r['pdg']) == 11:
            others.append(r)

dau = [r for r in rows if r['kind'] == 'daughter']
par = [r for r in rows if r['kind'] == 'parent']
print("doc pr/139 sec 2.2 -- the fate of every peeled daughter on the SHIPPED baseline")
print("arm %s-*, %d events carrying a peel" % (ARM, len(peel)))


def line(nm, xs):
    if not xs:
        return
    print("  %-34s n=%3d  EM-typed %3d  paired into a pi0 %3d (%3.0f%%)  median kine %6.1f MeV"
          % (nm, len(xs), sum(1 for r in xs if abs(r['pdg']) == 11),
             sum(r['paired'] for r in xs), 100 * sum(r['paired'] for r in xs) / len(xs),
             np.median([r['kine'] for r in xs])))


print()
line("the peeled DAUGHTERS", dau)
line("their PARENTS (post-cut)", par)
line("every other EM shower, same events", others)
print()
print("  READ THIS BEFORE PROPOSING A RE-HOME: the daughters pair at ABOVE the base")
print("  rate, so the finder is NOT blind to them.  The re-home front is the")
print("  narrower set below -- plausible gammas that still did not pair.")
print()
for floor in (10, 20, 30):
    cand = [r for r in dau if abs(r['pdg']) == 11 and r['kine'] > floor]
    unp = [r for r in cand if not r['paired']]
    print("  daughters EM-typed and above %2d MeV: %2d, of which UNPAIRED %2d"
          % (floor, len(cand), len(unp)))
print()
print("  conn type of the daughters:",
      dict(collections.Counter(r['conn'] for r in dau)))
print("  pdg of the daughters      :",
      dict(collections.Counter(r['pdg'] for r in dau)))
print("\n  the unpaired EM daughters above 20 MeV, largest first:")
for r in sorted([r for r in dau if abs(r['pdg']) == 11 and r['kine'] > 20 and not r['paired']],
                key=lambda r: -r['kine'])[:15]:
    print("    evt%-8d node%-8d %8.1f MeV  nseg %2d  conn %d"
          % (r['event'], r['node'], r['kine'], r['nseg'], r['conn']))

with open('docs/pr/pr139-daughter-fate.tsv', 'w') as f:
    w = csv.writer(f, delimiter='\t')
    f.write("# doc pr/139 sec 2.2 -- every peeled daughter and its parent, shipped baseline\n")
    w.writerow(['event', 'node', 'kind', 'pdg', 'kine_MeV', 'paired', 'conn', 'nseg'])
    for r in rows:
        w.writerow([r['event'], r['node'], r['kind'], r['pdg'], '%.2f' % r['kine'],
                    r['paired'], r['conn'], r['nseg']])
print("\nwrote docs/pr/pr139-daughter-fate.tsv")
