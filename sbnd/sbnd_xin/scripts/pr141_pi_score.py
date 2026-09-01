#!/usr/bin/env python3
# doc pr/141 rec 2 -- score the owner's pi-typed verdicts.  READ-ONLY.
"""Does the SHIPPED splitter fire on the pi-typed objects the owner says to cut?

doc pr/139 sec 22.5 proposed restricting the splitter to EM-typed objects on the
strength of "not EM-typed: 8 candidates, 1 confirmed cut, 1 fire, purity 0.000".
The owner's 2026-08-31 scan of the whole pi-typed population (7 objects, all at
confidence `high`) says 4 of 7 are confirmed cuts.  This script asks the second
half of the question: does the shipped kernel FIRE on them, and where does it
put the boundary?

Reuses the kernel and the charge-weighted boundary metric from
scripts/pr141_kernel_recurse.py (itself a fork of pr138_kernel_k.py), so the
numbers are comparable to every splitter scan of this campaign.

    python3 scripts/pr141_pi_score.py --tsv docs/pr/pr141-pi-score.tsv
"""
import os, sys, json, glob, csv, argparse, itertools, collections
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'split_display'))
import numpy as np
import pr137_lib as L
import split_model as SM
import importlib.util
_s = importlib.util.spec_from_file_location(
    "kr", os.path.join(HERE, "pr141_kernel_recurse.py"))

ap = argparse.ArgumentParser()
ap.add_argument('--tag', default='pisplit-0905-owner')
ap.add_argument('--set', default='docs/pr/pr141-pi-scanset.tsv')
ap.add_argument('--tsv', default='docs/pr/pr141-pi-score.tsv')
args = ap.parse_args()

SEP_SCALE, MAX_SEEDS, V_ACCEPT, F_ACCEPT = 1.6, 4, 0.95, 0.03


def maxima(pts, q, v):
    if pts is None or len(pts) < 8:
        return None
    return L.angular_maxima(pts, q, v, L.profile_sigma_fn(),
                            sep_scale=SEP_SCALE, max_seeds=MAX_SEEDS)


def accept_pair(M):
    if M is None or len(M['dirs']) < 2:
        return []
    V, fr, best = M['valley'], M['frac'], None
    for i in range(len(M['dirs'])):
        for j in range(i + 1, len(M['dirs'])):
            if min(fr[i], fr[j]) < F_ACCEPT:
                continue
            if best is None or V[i, j] < best[0]:
                best = (float(V[i, j]), i, j)
    return [] if (best is None or best[0] > V_ACCEPT) else [best[1], best[2]]


def assign(row, M, acc):
    P, ms, segs = row['P'], row['ms'], sorted(row['segs'])
    out = {s: 0 for s in segs}
    if not acc:
        return out
    D = M['dirs'][acc]
    for b in SM.connected_bundles(P, segs, ms, gap=4.0):
        bp, bq, _ = L.pack(P, b)
        if bp is None:
            continue
        c = L.qw_centroid(bp, bq) - row['v']
        n = np.linalg.norm(c)
        if n <= 0:
            continue
        g = int(np.argmax(D @ (c / n)))
        for s in b:
            out[s] = g
    return out


def agreement(prop, own, qmap):
    segs = [s for s in prop if s in own]
    if not segs:
        return None
    pg, og = sorted({prop[s] for s in segs}), sorted({own[s] for s in segs})
    Qt = sum(qmap.get(s, 0.0) for s in segs) or 1.0
    best = 0.0
    for perm in itertools.permutations(pg, min(len(pg), len(og))):
        m = dict(zip(perm, og))
        best = max(best, sum(qmap.get(s, 0.0) for s in segs
                             if m.get(prop[s]) == own[s]) / Qt)
    return best


OWN = {}
for f in sorted(glob.glob('em_labels/%s/labels-evt*.json' % args.tag)):
    j = json.load(open(f))
    ev = int(str(j.get('event', '')).replace('evt', ''))
    for nd, r in (j.get('split_labels') or {}).items():
        OWN[(ev, int(nd))] = r

SET = [r for r in csv.DictReader(
    (l for l in open(args.set) if not l.startswith('#')), delimiter='\t')]

rows = []
print("doc pr/141 rec 2 -- the owner's pi-typed verdicts vs the SHIPPED kernel\n")
for d in SET:
    ev, nd = int(d['event']), int(d['node'])
    rec = OWN.get((ev, nd))
    if rec is None:
        print("evt %-7d node %-7d  NOT SCANNED" % (ev, nd))
        continue
    verdict = rec.get('verdict')
    own = {int(a): int(b) for a, b in (rec.get('groups') or {}).items()}
    row = SM.load_object(ev, nd)
    fires, k, agree = False, 1, None
    if row is not None:
        pts, q, _ = L.pack(row['P'], sorted(row['segs']))
        M = maxima(pts, q, row['v'])
        acc = accept_pair(M)
        fires = bool(acc)
        prop = assign(row, M, acc)
        k = len(set(prop.values()))
        if own:
            qmap = {s: max(row['ms'].get(s, 0.0), 0.0) for s in row['segs']}
            agree = agreement(prop, own, qmap)
    n_own = len(set(own.values())) if own else 0
    print("evt %-7d node %-7d owner=%-7s (k=%d)   shipped kernel: %-9s k=%d   "
          "boundary agreement %s"
          % (ev, nd, verdict, n_own, "FIRES" if fires else "no fire", k,
             "%.3f" % agree if agree is not None else "-"))
    rows.append(dict(event=ev, node=nd, E=d.get('E'), length=d.get('length'),
                     nseg=d.get('nseg'), owner=verdict, n_own=n_own,
                     fires=fires, k_kernel=k,
                     agreement=("%.4f" % agree) if agree is not None else ""))

cut = [r for r in rows if r['owner'] in ('SPLIT2', 'SPLIT3', 'SPLIT4+')]
keep = [r for r in rows if r['owner'] == 'KEEP']
print("\n=== the EM-only restriction, decided ===")
print("  pi-typed objects scanned          : %d" % len(rows))
print("  owner CONFIRMED CUTS              : %d" % len(cut))
print("  owner KEEP                        : %d" % len(keep))
print("  confirmed cuts the kernel FIRES on: %d" % sum(1 for r in cut if r['fires']))
print("  KEEPs the kernel fires on (false) : %d" % sum(1 for r in keep if r['fires']))
if cut:
    ag = [float(r['agreement']) for r in cut if r['agreement'] and r['fires']]
    if ag:
        print("  boundary agreement on fired cuts  : median %.3f  mean %.3f  (n=%d)"
              % (sorted(ag)[len(ag) // 2], sum(ag) / len(ag), len(ag)))

with open(args.tsv, 'w') as fh:
    hdr = ["event", "node", "E", "length", "nseg", "owner", "n_own",
           "fires", "k_kernel", "agreement"]
    fh.write("\t".join(hdr) + "\n")
    for r in rows:
        fh.write("\t".join(str(r[h]) for h in hdr) + "\n")
print("\nwrote %s (%d rows)" % (args.tsv, len(rows)))
