#!/usr/bin/env python3
# doc pr/138 Phase B -- the boundary the SHIPPED C++ drew, and the peel it performed.
"""Score the C++ partition against the owner's hand labels, from the probe tape.

The offline kernel score (pr138_kernel_k.py) says what the PYTHON would draw.
This says what the C++ actually drew, read out of its own WCT_SHOWER_SPLIT_DEBUG
`part` lines -- the partition is computed and taped whether or not the knob acts,
so the boundary can be scored from a knob-OFF arm and the knob-ON arm is then
only asked whether the surgery succeeded.

Two questions, two arms:
  --tape <arm>   the boundary: charge-weighted best-label-permutation agreement
                 with the owner's groups, exactly sec A5.6's metric.
  --on <arm>     the action: how many peels, with which start segment, and the
                 FORWARD check (does the daughter's charge sit downstream of the
                 start segment it was seeded on?).  A backwards daughter would
                 poison the pi0 finders, which run after the splitter and read
                 get_init_dir().

ONE CROSS-ARM JOIN, DISCLOSED.  The partition comes from the tape arm, but the
per-segment CHARGE used to weight the agreement comes from the scan's own arm
(`onV1c90`), because a no-pr_display arm writes no calib dump.  doc sec B1
measured what that costs: membership differs on 2 of 172 objects and the
reference vertex on 5, all in evt76346 and evt396222.  A segment absent from the
onV1c90 dump contributes zero weight rather than a wrong one, and neither of
those two objects is a fired true positive, so no scored row is affected -- but
the join is cross-arm and is named here rather than left implicit.

Repro:
    python3 scripts/pr138_smoke_split.py --tape 'work-pr138r1-dbg-*' \
        --on 'work-pr138r1-on2-*'
"""
import os, sys, re, json, glob, csv, argparse, collections, itertools
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'split_display'))
import pr137_lib as L

ap = argparse.ArgumentParser()
ap.add_argument('--tape', default='work-pr138r1-dbg-*')
ap.add_argument('--on', default='')
ap.add_argument('--owner-tag', default='splitscan-0901-owner')
ap.add_argument('--tsv', default='docs/pr/pr138-smoke-split.tsv')
args = ap.parse_args()
SPLITS = ('SPLIT2', 'SPLIT3', 'SPLIT4+')
TRACKS = {(99838, 14004), (389538, 19021), (292524, 9018), (176502, 109141),
          (286681, 72040), (122660, 54071), (415278, 23047), (278420, 18002)}

CAND = re.compile(r'SHOWER_SPLIT cand shower=(-?\d+) .* nacc=(\d+) nparts=(\d+) fired=(\d)')
PART = re.compile(r'SHOWER_SPLIT part shower=(-?\d+) part=(\d+) nseg=(\d+) q=(\S+) segs=(\S+)')
PEEL = re.compile(r'SHOWER_SPLIT peel shower=(-?\d+) part=(\d+) new_start=(-?\d+) nseg=(\d+) '
                  r'conn=(\d+) root_vtx_cm=(\S+) fwd=(\S+)')


def read(globpat):
    cand, part, peel = {}, collections.defaultdict(dict), collections.defaultdict(list)
    for log in sorted(glob.glob(os.path.join(globpat, 'pr_evt*', 'stdout.log'))):
        m = re.search(r'pr_evt(\d+)', log)
        if not m:
            continue
        ev = int(m.group(1))
        for ln in open(log, errors='replace'):
            if 'SHOWER_SPLIT' not in ln:
                continue
            c = CAND.search(ln)
            if c:
                cand[(ev, int(c.group(1)))] = dict(nacc=int(c.group(2)),
                                                   nparts=int(c.group(3)),
                                                   fired=int(c.group(4)))
                continue
            p = PART.search(ln)
            if p:
                part[(ev, int(p.group(1)))][int(p.group(2))] = \
                    [int(x) for x in p.group(5).split(',')]
                continue
            q = PEEL.search(ln)
            if q:
                peel[(ev, int(q.group(1)))].append(
                    dict(part=int(q.group(2)), start=int(q.group(3)), nseg=int(q.group(4)),
                         conn=int(q.group(5)), root_cm=float(q.group(6)), fwd=float(q.group(7))))
    return cand, part, peel


OWN = {}
for f in sorted(glob.glob('em_labels/%s/labels-evt*.json' % args.owner_tag)):
    j = json.load(open(f))
    try:
        ev = int(str(j.get('event', '')).replace('evt', ''))
    except Exception:
        continue
    for nd, r in (j.get('split_labels') or {}).items():
        OWN[(ev, int(nd))] = r

cand, part, peel = read(args.tape)
print("doc pr/138 -- the boundary the C++ drew   (tape arm: %s)" % args.tape)
print("  taped candidates %d, of which fired %d"
      % (len(cand), sum(1 for c in cand.values() if c['fired'])))

# per-event charge, straight from the arm's own dumps is unavailable on a
# no-pr_display arm, so charge comes from the scan's own arm -- the same numbers
# sec A5.6 weighted with.
QCACHE = {}


def qmap(ev):
    if ev not in QCACHE:
        d = L.dump(ev, 'onV1c90')
        P = L.seg_pts(d) if d else {}
        QCACHE[ev] = {s: float(max(A[:, 3].sum(), 0.0)) for s, A in P.items()}
    return QCACHE[ev]


rows = []
for k in sorted(OWN):
    if k not in cand:
        continue
    lab = OWN[k]
    hand = {int(a): int(b) for a, b in (lab.get('groups') or {}).items()}
    prop = {}
    for g, segs in part.get(k, {}).items():
        for s in segs:
            prop[s] = g
    q = qmap(k[0])
    segs = [s for s in prop if s in hand]
    agree = None
    if segs and prop:
        pg = sorted({prop[s] for s in segs})
        og = sorted({hand[s] for s in segs})
        Qt = sum(q.get(s, 0.0) for s in segs) or 1.0
        agree = max(sum(q.get(s, 0.0) for s in segs if dict(zip(perm, og)).get(prop[s]) == hand[s]) / Qt
                    for perm in itertools.permutations(pg, min(len(pg), len(og))))
    rows.append(dict(event=k[0], node=k[1], verdict=lab.get('verdict'),
                     track=k in TRACKS, fired=cand[k]['fired'], nparts=cand[k]['nparts'],
                     n_own=len(set(hand.values())), agree=agree,
                     npeel=len(peel.get(k, []))))

em = [r for r in rows if not r['track']]
fired = [r for r in em if r['fired']]
pos = [r for r in em if r['verdict'] in SPLITS]
tp = [r for r in fired if r['verdict'] in SPLITS]
print("\n=== the TRIGGER, on the 164 scanned EM objects ===")
print("  fires %d   efficiency %.3f   purity %.3f   (tp %d of %d positives)"
      % (len(fired), len(tp) / max(len(pos), 1), len(tp) / max(len(fired), 1), len(tp), len(pos)))
print("  owner verdicts of the fires:", dict(collections.Counter(r['verdict'] for r in fired)))

print("\n=== the BOUNDARY, C++ vs the owner, on the fired true positives ===")
for cls in ('SPLIT2', 'SPLIT3', 'SPLIT4+'):
    v = sorted(r['agree'] for r in tp if r['verdict'] == cls and r['agree'] is not None)
    if not v:
        continue
    print("  %-8s n=%2d  median %.3f  mean %.3f  >=0.90 %2d  ==1.000 %2d"
          % (cls, len(v), v[len(v) // 2], sum(v) / len(v),
             sum(1 for x in v if x >= 0.90), sum(1 for x in v if x > 0.9999)))
v3 = [r['agree'] for r in tp if r['verdict'] in ('SPLIT3', 'SPLIT4+') and r['agree'] is not None]
va = [r['agree'] for r in tp if r['agree'] is not None]
print("  %-8s n=%2d  %22s mean %.3f" % ('k>=3', len(v3), '', sum(v3) / max(len(v3), 1)))
print("  %-8s n=%2d  %22s mean %.3f" % ('ALL', len(va), '', sum(va) / max(len(va), 1)))
print("\n  the six worst:")
for r in sorted([r for r in tp if r['agree'] is not None], key=lambda r: r['agree'])[:6]:
    print("    evt%-8d node%-8d %-8s owner_k=%d cxx_k=%d  %.3f"
          % (r['event'], r['node'], r['verdict'], r['n_own'], r['nparts'], r['agree']))

if args.on:
    c2, p2, pl = read(args.on)
    print("\n=== the ACTION   (ON arm: %s) ===" % args.on)
    allpeel = [x for v in pl.values() for x in v]
    print("  candidates taped %d, fired %d, peels performed %d over %d parent(s)"
          % (len(c2), sum(1 for c in c2.values() if c['fired']), len(allpeel), len(pl)))
    if allpeel:
        fw = sorted(x['fwd'] for x in allpeel)
        print("  daughter start-vertex distance: median %.2f cm, max %.2f cm"
              % (sorted(x['root_cm'] for x in allpeel)[len(allpeel) // 2],
                 max(x['root_cm'] for x in allpeel)))
        print("  FORWARD check  cos(start ray, body ray): median %.3f, min %.3f; "
              "backwards (<0) on %d of %d"
              % (fw[len(fw) // 2], fw[0], sum(1 for x in fw if x < 0), len(fw)))
        print("  conn types:", dict(collections.Counter(x['conn'] for x in allpeel)))
        refused = sum(1 for k, c in c2.items() if c['fired'] and len(pl.get(k, [])) == 0)
        print("  fired but NO peel (detach refused, or long-muon skip): %d" % refused)

with open(args.tsv, 'w') as f:
    w = csv.writer(f, delimiter='\t')
    f.write("# doc pr/138 -- the C++ partition vs the owner labels (tape=%s)\n" % args.tape)
    w.writerow(['event', 'node', 'verdict', 'track_typed', 'fired', 'cxx_nparts',
                'owner_k', 'agreement', 'npeel'])
    for r in rows:
        w.writerow([r['event'], r['node'], r['verdict'], int(r['track']), r['fired'],
                    r['nparts'], r['n_own'],
                    '%.4f' % r['agree'] if r['agree'] is not None else '', r['npeel']])
print("\nwrote %s" % args.tsv)
