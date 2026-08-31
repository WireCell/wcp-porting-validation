#!/usr/bin/env python3
# doc pr/138 sec A1.5 + A1.6 -- the two censuses over the owner's 50 scan objects.
"""Why the split display looked monochrome, and what the 50 objects actually are.

Two questions, one pass over the curated set:

  A1.5  how many groups does the proposal pre-fill?   ("no red vs. blue")
  A1.6  what particle does the arm think each object is?  ("I only see the
        major track in it, not any of the EM shower")

Read-only.  Prints the tables quoted in doc pr/138; no file is written.

    python3 scripts/pr138_scanset_census.py [--set docs/pr/pr137-curated-set.tsv]
"""
import os, sys, json, glob, argparse, collections

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(ROOT, 'split_display'))
sys.path.insert(0, HERE)
os.chdir(ROOT)
import split_model as SM                                    # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--set', default='docs/pr/pr137-curated-set.tsv')
ap.add_argument('--arm', default='onV1c90')
ap.add_argument('--all', action='store_true',
                help='every row, not only owner_scan=1')
args = ap.parse_args()

rows, hdr = [], None
for line in open(args.set):
    if line.startswith('#'):
        continue
    f = line.rstrip('\n').split('\t')
    if hdr is None:
        hdr = f
        continue
    d = dict(zip(hdr, f))
    if not args.all and d.get('owner_scan') != '1':
        continue
    rows.append((int(d['event']), int(d['node']), float(d['Q']), d.get('stratum', ''),
                 d.get('proxy_cls', '')))
rows.sort(key=lambda t: -t[2])          # the viewer's own order, so indices match

NAMES = {11: 'e', -11: 'e', 13: 'mu', -13: 'mu', 211: 'pi', -211: 'pi', 2212: 'p'}
ng_c, pid_c, tracks, singles = collections.Counter(), collections.Counter(), [], []

print("doc pr/138 -- census over %d objects from %s\n" % (len(rows), args.set))
print("%-4s %-9s %-9s %-4s %-8s %-6s %-9s %s"
      % ('idx', 'event', 'node', 'ng', 'stratum', 'pid', 'len(cm)', 'proposal'))
for i, (ev, nd, Q, st, px) in enumerate(rows):
    r = SM.load_object(ev, nd, arm=args.arm)
    if r is None:
        ng_c['missing'] += 1
        continue
    p = SM.object_payload(r)
    ng = len({s['group'] for s in p['segs'] if s['group'] != SM.JUNK})
    ng_c[ng] += 1
    rec = r.get('rec') or {}
    pid = rec.get('particle_id')
    pid_c[pid] += 1
    if pid is not None and abs(int(pid)) != 11:
        tracks.append((ev, nd, pid, rec.get('total_length', float('nan')),
                       rec.get('kine_best', float('nan'))))
    if ng == 1:
        singles.append((ev, nd))
    print("%-4d %-9d %-9d %-4d %-8s %-6s %-9.0f %s"
          % (i, ev, nd, ng, st, NAMES.get(pid, pid),
             rec.get('total_length', float('nan')), p['reason'][:64]))

print("\nA1.5  groups proposed:", dict(ng_c),
      " -- %d of %d objects are single-group" % (len(singles), len(rows)))
print("A1.6  particle_id     :", {NAMES.get(k, k): v for k, v in sorted(
    pid_c.items(), key=lambda t: -t[1])})
print("\nA1.6  the %d track-typed objects (excluded from the §A4 denominators):"
      % len(tracks))
for ev, nd, pid, ln, ke in tracks:
    print("   evt%-8d node%-8d pid=%-5s %6.0f cm  kine_best %6.0f MeV"
          % (ev, nd, pid, ln, ke))
