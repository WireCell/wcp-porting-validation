#!/usr/bin/env python3
# doc pr/138 -- PRE-REGISTERED prediction of the completeness deltas, from the tape.
"""Predict what the splitter does to q_miss and q_extra BEFORE reading the arms.

WHY.  "Did the numbers move?" is a weak test.  The metric's mechanics fix the
SIGN in advance, so the strong test is to predict the two deltas from the probe
tape and then see whether the arms agree.  A mismatch is a join bug, not a
physics result -- which is exactly the failure mode a bare before/after cannot
see.

THE MECHANICS.  em117_score scores each hand-marked shower as
`target = (members | marked-in) - marked-out`, charge-weighted, against the
segments the arm gave that shower; the dump's `segments[].shower_id` is
SINGLE-VALUED, so after a split the daughter's segments belong to the DAUGHTER.
Relative to the marked parent:

  * a CORRECT cut removes charge that was scoring as q_extra  -> q_extra FALLS
  * a WRONG   cut removes charge that was scoring as q_comp   -> q_miss  RISES

so the pass/fail rule of doc pr/136 sec 11.2 must be INVERTED for a splitter:
that rule ("fails if q_extra rises by more than q_miss falls") grades a
COMPLETENESS change like the pass-4 escape.  For a splitter the rule is

        FAILS IF q_miss RISES BY MORE THAN q_extra FALLS.

WHAT THIS PRINTS.  For every fire on the tape, the charge the splitter would
detach, split by whether the object carries an owner SPLIT label (the cut is
right), a KEEP/TRIM label (the cut is wrong), or no label at all -- and, the
number that bounds the whole exercise, how many fires land on one of the 90
HAND-MARKED showers the completeness instrument actually scores.

Repro:
    python3 scripts/pr138_predict_delta.py --tape 'work-pr138r2-pon-*'
"""
import os, sys, re, json, glob, csv, argparse, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import pr137_lib as L

ap = argparse.ArgumentParser()
ap.add_argument('--tape', default='work-pr138r2-pon-*')
ap.add_argument('--owner-tag', default='splitscan-0901-owner')
ap.add_argument('--mark-tags', default='emscan-0827,emscan-0828-agent5')
ap.add_argument('--tsv', default='docs/pr/pr138-predict-delta.tsv')
args = ap.parse_args()
SPLITS = ('SPLIT2', 'SPLIT3', 'SPLIT4+')

CAND = re.compile(r'SHOWER_SPLIT cand shower=(-?\d+) pdg=(-?\d+) nseg=(\d+) npts=(\d+) '
                  r'Q=(\S+) .* nparts=(\d+) fired=(\d)')
PART = re.compile(r'SHOWER_SPLIT part shower=(-?\d+) part=(\d+) nseg=(\d+) q=(\S+) segs=(\S+)')
PEEL = re.compile(r'SHOWER_SPLIT peel shower=(-?\d+) part=(\d+) new_start=(-?\d+) nseg=(\d+)')

cand, part, peel = {}, collections.defaultdict(dict), collections.defaultdict(set)
for log in sorted(glob.glob(os.path.join(args.tape, 'pr_evt*', 'stdout.log'))):
    m = re.search(r'pr_evt(\d+)', log)
    if not m:
        continue
    ev = int(m.group(1))
    for ln in open(log, errors='replace'):
        if 'SHOWER_SPLIT' not in ln:
            continue
        c = CAND.search(ln)
        if c:
            cand[(ev, int(c.group(1)))] = dict(pdg=int(c.group(2)), Q=float(c.group(5)),
                                               nparts=int(c.group(6)), fired=int(c.group(7)))
            continue
        p = PART.search(ln)
        if p:
            part[(ev, int(p.group(1)))][int(p.group(2))] = (float(p.group(4)),
                                                            [int(x) for x in p.group(5).split(',')])
            continue
        q = PEEL.search(ln)
        if q:
            peel[(ev, int(q.group(1)))].add(int(q.group(2)))


def load_labels(tag):
    out = {}
    for f in sorted(glob.glob('em_labels/%s/labels-evt*.json' % tag)):
        j = json.load(open(f))
        try:
            ev = int(str(j.get('event', '')).replace('evt', ''))
        except Exception:
            continue
        for nd, r in (j.get('split_labels') or {}).items():
            out[(ev, int(nd))] = r
    return out


OWN = load_labels(args.owner_tag)

# the 90 hand-MARKED showers the completeness instrument scores -- a different
# population from the 172 split labels, and the intersection bounds how much
# signal that instrument can carry about this change at all.
marked = set()
for tag in args.mark_tags.split(','):
    for f in sorted(glob.glob('em_labels/%s/labels-evt*.json' % tag)):
        j = json.load(open(f))
        try:
            ev = int(str(j.get('event', '')).replace('evt', ''))
        except Exception:
            continue
        for key in ('marks_by_shower', 'em', 'showers'):
            blk = j.get(key)
            if isinstance(blk, dict) and key == 'marks_by_shower':
                for nd in blk:
                    marked.add((ev, int(nd)))
            elif isinstance(blk, dict) and isinstance(blk.get('marks_by_shower'), dict):
                for nd in blk['marks_by_shower']:
                    marked.add((ev, int(nd)))

fires = [(k, v) for k, v in sorted(cand.items()) if v['fired']]
print("doc pr/138 -- PRE-REGISTERED delta prediction   (tape %s)" % args.tape)
print("candidates %d, fires %d, hand-marked showers found in %s: %d"
      % (len(cand), len(fires), args.mark_tags, len(marked)))
print()
print("THE RULE, inverted for a splitter (doc pr/136 sec 11.2 grades the other")
print("direction):  FAILS IF q_miss RISES BY MORE THAN q_extra FALLS.")
print()

rows = []
tot = collections.defaultdict(float)
n = collections.Counter()
for k, v in fires:
    lab = OWN.get(k, {}).get('verdict')
    cls = ('CORRECT (owner SPLIT)' if lab in SPLITS else
           'WRONG   (owner KEEP/TRIM)' if lab else 'UNLABELLED')
    parts = part.get(k, {})
    if not parts:
        continue
    keep = max(parts, key=lambda g: parts[g][0]) if parts else None
    # the charge that LEAVES the parent = every part actually peeled; fall back
    # to "every part but the biggest" when the peel lines are absent (knob off).
    pl = peel.get(k) or {g for g in parts if g != keep}
    qdet = sum(parts[g][0] for g in pl if g in parts)
    tot[cls] += qdet
    n[cls] += 1
    rows.append(dict(event=k[0], node=k[1], verdict=lab or '', cls=cls,
                     Q=v['Q'], q_detached=qdet, frac=qdet / max(v['Q'], 1.0),
                     marked=int(k in marked)))

print("%-26s %5s %14s %14s" % ("class", "fires", "charge detached", "of parent Q"))
for cls in ('CORRECT (owner SPLIT)', 'WRONG   (owner KEEP/TRIM)', 'UNLABELLED'):
    q = tot[cls]
    Q = sum(r['Q'] for r in rows if r['cls'] == cls) or 1.0
    print("%-26s %5d %14.4g %13.1f%%" % (cls, n[cls], q, 100 * q / Q))
print("%-26s %5d %14.4g" % ("TOTAL", len(rows), sum(tot.values())))
print()
mk = [r for r in rows if r['marked']]
print("=== THE BOUND: how many fires land on a shower the completeness instrument scores? ===")
print("  fires on one of the %d hand-marked showers: %d" % (len(marked), len(mk)))
for r in mk:
    print("    evt%-8d node%-8d %-10s q_detached %.4g (%.0f%% of parent)"
          % (r['event'], r['node'], r['verdict'] or '(unlabelled)', r['q_detached'],
             100 * r['frac']))
print()
print("=== THE PREDICTION ===")
qc = tot['CORRECT (owner SPLIT)']
qw = tot['WRONG   (owner KEEP/TRIM)']
print("  On the LABELLED objects the tape says %d cuts are right and %d wrong,"
      % (n['CORRECT (owner SPLIT)'], n['WRONG   (owner KEEP/TRIM)']))
print("  carrying %.4g and %.4g of detached charge respectively (%.0f%% / %.0f%%)."
      % (qc, qw, 100 * qc / max(qc + qw, 1), 100 * qw / max(qc + qw, 1)))
print("  So the SIGN is pre-registered: q_extra should FALL and q_miss should")
print("  RISE, with the fall the larger of the two IF the labelled ratio carries")
print("  over to the marked population.  The bound above says how much of that")
print("  is even visible to the instrument -- if it is a handful of showers, the")
print("  completeness deltas will be small and the pi0 census is the real test.")

with open(args.tsv, 'w') as f:
    w = csv.writer(f, delimiter='\t')
    f.write("# doc pr/138 -- per-fire detached charge, pre-registered before the arms were read\n")
    w.writerow(['event', 'node', 'owner_verdict', 'class', 'parent_Q', 'q_detached',
                'frac_detached', 'on_hand_marked_shower'])
    for r in rows:
        w.writerow([r['event'], r['node'], r['verdict'], r['cls'], '%.6g' % r['Q'],
                    '%.6g' % r['q_detached'], '%.4f' % r['frac'], r['marked']])
print("\nwrote %s" % args.tsv)
