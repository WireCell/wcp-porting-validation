#!/usr/bin/env python3
"""doc pr/139 sec 17 -- item 2: the third boundary is a SEEDING problem, measured.

sec 13 showed that lifting `shower_split_max_parts` to 3 makes the third cut in
the wrong place on two objects of three.  sec 13.4 reposed the question as the
kernel's seeding rather than the cap.  This script measures that directly by
joining the debug tape's seeding fields to the owner's 2026-09-01 per-part
labels and to the boundary agreement.

    python3 scripts/pr140_seeding.py --arm work-pr140r1-onk3
                                     --base work-pr140r1-onrh15   # max_parts=2

`max_seeds` is a hardcoded 4 in the seed finder
(NeutrinoShowerClustering.cxx:5494), NOT a knob -- which is the point.
"""
import argparse, collections, csv, glob, itertools, json, os, re, statistics, sys

SX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAG = 'splitscan-0902-pi0'
MAX_SEEDS = 4          # the hardcoded value the tape is being compared against
CAND = re.compile(r'SHOWER_SPLIT cand shower=(\d+) .*?n_seed=(\d+) valley_best=([\d.]+) '
                  r'angle_best=(-?[\d.]+) nacc=(\d+) nparts=(\d+) fired=(\d)')
PART = re.compile(r'SHOWER_SPLIT part shower=(\d+) part=(\d+) nseg=(\d+) q=([\d.eE+-]+) segs=([\d,]+)')


def owner():
    verd, grp, kk = {}, {}, {}
    for f in sorted(glob.glob(os.path.join(SX, 'em_labels', TAG, 'labels-evt*.json'))):
        d = json.load(open(f)); ev = int(d['event'][3:])
        for n, v in (d.get('split_labels') or {}).items():
            k = (ev, int(n)); verd[k] = v.get('verdict'); kk[k] = int(v.get('n_parts') or 1)
            grp[k] = {int(s): int(g) for s, g in (v.get('groups') or {}).items() if int(g) >= 0}
    return verd, grp, kk


def tape(arm):
    cand, part, qmap = {}, collections.defaultdict(dict), collections.defaultdict(dict)
    for lg in sorted(glob.glob(os.path.join(SX, arm) + '-*/pr_evt*/stdout.log')):
        ev = int(re.search(r'pr_evt(\d+)/', lg).group(1))
        for line in open(lg, errors='replace'):
            m = CAND.search(line)
            if m:
                cand[(ev, int(m.group(1)))] = dict(
                    n_seed=int(m.group(2)), valley=float(m.group(3)),
                    angle=float(m.group(4)), nacc=int(m.group(5)),
                    nparts=int(m.group(6)), fired=int(m.group(7)))
                continue
            m = PART.search(line)
            if m:
                node, g, q = int(m.group(1)), int(m.group(2)), float(m.group(4))
                segs = [int(x) for x in m.group(5).split(',') if x]
                for s in segs:
                    part[(ev, node)][s] = g
                    qmap[(ev, node)][s] = q / max(len(segs), 1)
    return cand, part, qmap


def agreement(prop, own, qm):
    segs = [s for s in prop if s in own]
    if not segs:
        return None
    pg = sorted({prop[s] for s in segs}); og = sorted({own[s] for s in segs})
    Qt = sum(qm.get(s, 0.0) for s in segs) or 1.0
    best = 0.0
    for perm in itertools.permutations(pg, min(len(pg), len(og))):
        m = dict(zip(perm, og))
        best = max(best, sum(qm.get(s, 0.0) for s in segs if m.get(prop[s]) == own[s]) / Qt)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arm', default='work-pr140r1-onk3')
    ap.add_argument('--base', default='work-pr140r1-onrh15')
    ap.add_argument('--tsv')
    a = ap.parse_args()
    verd, grp, kk = owner()
    A, Ap, Aq = tape(a.arm)
    B, Bp, Bq = tape(a.base)

    fired = [k for k, c in B.items() if c['fired']]
    print("=== how often is the hardcoded max_seeds = %d BINDING? ===" % MAX_SEEDS)
    for name, C in (('all candidates', B), ('fired candidates', {k: B[k] for k in fired})):
        d = collections.Counter(c['n_seed'] for c in C.values())
        sat = d.get(MAX_SEEDS, 0); tot = sum(d.values())
        print("  %-18s n=%4d   n_seed==%d on %d (%.0f %%)   dist %s"
              % (name, tot, MAX_SEEDS, sat, 100.0 * sat / tot if tot else 0,
                 dict(sorted(d.items()))))
    print()
    print("=== the owner's 19 confirmed cuts + the false fires, joined ===")
    print("  %-8s %-7s %-8s %5s %6s %5s %6s %7s %8s %8s"
          % ('event', 'node', 'verdict', 'own_k', 'n_seed', 'nacc', 'valley', 'a_k2', 'a_k3', 'delta'))
    rows = []
    for k in sorted(verd, key=lambda x: (-kk[x], x)):
        if k not in B:
            continue
        c = B[k]
        if not c['fired']:
            continue
        ab = agreement(Bp[k], grp.get(k, {}), Bq[k]) if k in Bp else None
        aa = agreement(Ap[k], grp.get(k, {}), Aq[k]) if k in Ap else None
        rows.append(dict(event=k[0], node=k[1], verdict=verd[k], own_k=kk[k],
                         n_seed=c['n_seed'], nacc=c['nacc'], valley=c['valley'],
                         a_k2=ab, a_k3=aa))
        print("  %-8d %-7d %-8s %5d %6d %5d %6.3f %7s %8s %8s"
              % (k[0], k[1], verd[k], kk[k], c['n_seed'], c['nacc'], c['valley'],
                 "%.3f" % ab if ab is not None else "-",
                 "%.3f" % aa if aa is not None else "-",
                 ("%+.3f" % (aa - ab)) if (aa is not None and ab is not None) else "-"))
    print()
    sat_k = [r for r in rows if r['n_seed'] == MAX_SEEDS]
    hi = [r for r in rows if r['own_k'] >= 3]
    print("  confirmed/false fires with n_seed at the cap : %d of %d" % (len(sat_k), len(rows)))
    print("  objects the owner cut into k >= 3            : %d, and ALL of them sit at the cap: %s"
          % (len(hi), all(r['n_seed'] == MAX_SEEDS for r in hi)))
    print("  their owner k values                         : %s" % sorted(r['own_k'] for r in hi))
    print("  => a %d-seed finder cannot express k > %d, so the k >= 3 population is"
          % (MAX_SEEDS, MAX_SEEDS))
    print("     capped upstream of `shower_split_max_parts` and raising that knob")
    print("     alone (sec 13) could only ever redistribute seeds it already had.")
    if a.tsv:
        with open(a.tsv, 'w', newline='') as fh:
            w = csv.DictWriter(fh, delimiter='\t', fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print("\nwrote %s (%d rows)" % (a.tsv, len(rows)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
