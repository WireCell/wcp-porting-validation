#!/usr/bin/env python3
"""doc pr/48 sec 9.6: classify every archive-level mover of the knobs-ON 1k
arm against the production baseline, and dump the per-event evidence for the
individual examination the owner asked for.

Classes:
  TEB-BREAK   the two-end break fired (BROKE line in the arm's log); the
              designed new back-to-back population.  The break diagnostics
              (route, scores, rises, arms) and the main-vertex shift are
              printed for each.
  F3-PROTECT  a Bragg-hot C4/A0 kink break was protected from
              examine_vertices_4 absorption ("protected kink-break" log).
  F2-WALK     neither of the above: the mover can only come from
              kink_walk_dqdx_stop changing a Bragg-hot C4 walk (no direct
              log line; verified by knob attribution arms).
Also marks whether the event is in the owner-scanned first-50 Bee subset
(docs/pr/mcp1k-50-cb0805.index.txt).

Usage: python3 mover_census.py [BASE_LABEL NEW_LABEL]
       (defaults: work-pr47f-on1k vs work-pr48-on1kb)
"""
import sys, os, glob, json, zipfile, re
sys.path.insert(0, '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/abtest')
import hash_archive as ha

SB = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'
BASE = os.path.join(SB, sys.argv[1] if len(sys.argv) > 2 else 'work-pr47f-on1k')
NEW = os.path.join(SB, sys.argv[2] if len(sys.argv) > 2 else 'work-pr48-on1kb')

scanned = set()
with open(os.path.join(SB, 'docs/pr/mcp1k-50-cb0805.index.txt')) as f:
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) == 2:
            scanned.add(int(parts[1]))

evts = sorted(set(int(os.path.basename(p).replace('pr_evt', ''))
                  for p in glob.glob(os.path.join(BASE, 'pr_evt*'))) &
              set(int(os.path.basename(p).replace('pr_evt', ''))
                  for p in glob.glob(os.path.join(NEW, 'pr_evt*'))))

movers = []
for evt in evts:
    bp = os.path.join(BASE, 'pr_evt%d' % evt, 'mabc-pr.zip')
    np_ = os.path.join(NEW, 'pr_evt%d' % evt, 'mabc-pr.zip')
    if not (os.path.exists(bp) and os.path.exists(np_)):
        continue
    if dict(ha.members(bp)) != dict(ha.members(np_)):
        movers.append(evt)

counts = {'TEB-BREAK': 0, 'F3-PROTECT': 0, 'F2-WALK': 0}
rows = []
for evt in movers:
    log = os.path.join(NEW, 'pr_evt%d' % evt, 'wct_pr_evt%d.log' % evt)
    text = open(log, errors='replace').read() if os.path.exists(log) else ''
    # WCT log lines can tear mid-write (documented tearing), so the match must
    # not require the tail of the line (coordinates / route) to be intact.
    broke = re.findall(r'qdx: BROKE cluster \d+ at fit idx \d+[^\n]*', text)
    diag = re.findall(r'break_two_end_dqdx: cluster \d+ seg len [^\n]*found=true', text)
    prot = re.findall(r'protected kink-break vertex at [^\n]*', text)
    if broke:
        cls = 'TEB-BREAK'
    elif prot:
        cls = 'F3-PROTECT'
    else:
        cls = 'F2-WALK'
    counts[cls] += 1
    rows.append((evt, cls, broke, diag, prot))

print('movers: %d/%d   (owner-scanned first-50 subset among them: %d)' % (
    len(movers), len(evts), sum(1 for e in movers if e in scanned)))
print('classes:', counts)
print()
for evt, cls, broke, diag, prot in rows:
    tag = ' [OWNER-SCANNED]' if evt in scanned else ''
    print('=== evt %d  %s%s' % (evt, cls, tag))
    for d in diag[:2]:
        print('    ', d)
    for b in broke[:2]:
        print('    ', b)
    for p in prot[:2]:
        print('    ', p)
