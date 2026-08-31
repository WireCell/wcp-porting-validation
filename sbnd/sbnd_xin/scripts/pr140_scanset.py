#!/usr/bin/env python3
"""doc pr/139 sec 19 -- item 3: the wider per-part label set, built scan-ready.

sec 13.3 is the reason this exists.  Three instruments disagreed on the k>=3 arm
and the tie-breaker -- boundary agreement against hand per-part labels -- rests
on **19 confirmed cuts and 29 hand parts**.  sec 18 then found the last false
fire sits inside the confirmed-cut range on every tape feature, with three
"outlier" margins under 1 %.  Both are the same limit: the label set is too
small to resolve what is now being asked of it.

This selects the next ~30 objects, STRATIFIED, from the pr/137 curated set
(known loadable by split_model.load_object, which reads the pre-split onV1c90
arm) minus everything already labelled in splitscan-0902-pi0.

    python3 scripts/pr140_scanset.py --out docs/pr/pr140-scan-set.tsv

Strata, each tied to a measurement that is currently blocked:
  S1 seed-capped   n_seed == 4 -- sec 17: the cap binds on 76 % of fires and on
                   ALL FOUR k>=3 objects.  Grading a max_seeds arm needs labels
                   HERE and there are only 4 today.
  S2 bound-region  b in [15, 60] cm -- where the bound's 3 suppressions and the
                   last false fire live.  sec 18's margins were 0.43 cm.
  S3 unjudged fire the splitter peels these and nobody has said whether it should.
  S4 control       seeded-random over the rest, chosen INDEPENDENTLY of every
                   feature above, so the set is not selected by the hypotheses
                   it will be used to test ([[feedback_blind_the_scan_sheet]]).

READ-ONLY over em_labels/ (M13).  Writes only the new set TSV.
"""
import argparse, collections, csv, glob, json, os, random, re, sys

SX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CURATED = os.path.join(SX, 'docs', 'pr', 'pr137-curated-set.tsv')
DONE_TAG = 'splitscan-0902-pi0'
ARM = 'work-pr140r1-on'
CAND = re.compile(r'SHOWER_SPLIT cand shower=(\d+) .*?n_seed=(\d+) valley_best=([\d.]+) '
                  r'.*?nacc=(\d+) nparts=(\d+) fired=(\d) .*?b_cm=(-?[\d.]+) veto=(\d)')
PEEL = re.compile(r'SHOWER_SPLIT peel shower=(\d+) part=(\d+)')
SEED = 20260901


def curated():
    rows, hdr = [], None
    with open(CURATED) as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            f = line.rstrip('\n').split('\t')
            if hdr is None:
                hdr = f; continue
            rows.append(dict(zip(hdr, f)))
    return rows, hdr


def labelled():
    out = set()
    for f in sorted(glob.glob(os.path.join(SX, 'em_labels', DONE_TAG, 'labels-evt*.json'))):
        d = json.load(open(f)); ev = int(d['event'][3:])
        for n in (d.get('split_labels') or {}):
            out.add((ev, int(n)))
    return out


def tape():
    C, peeled = {}, set()
    for lg in sorted(glob.glob(os.path.join(SX, ARM) + '-*/pr_evt*/stdout.log')):
        ev = int(re.search(r'pr_evt(\d+)/', lg).group(1))
        for line in open(lg, errors='replace'):
            m = CAND.search(line)
            if m:
                C[(ev, int(m.group(1)))] = dict(n_seed=int(m.group(2)), valley=float(m.group(3)),
                                                nacc=int(m.group(4)), fired=int(m.group(6)),
                                                b=float(m.group(7)), veto=int(m.group(8)))
                continue
            m = PEEL.search(line)
            if m:
                peeled.add((ev, int(m.group(1))))
    return C, peeled


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=os.path.join(SX, 'docs', 'pr', 'pr140-scan-set.tsv'))
    ap.add_argument('--n-per-stratum', type=int, default=8)
    a = ap.parse_args()

    rows, hdr = curated()
    done = labelled()
    C, peeled = tape()
    pool = [r for r in rows if (int(r['event']), int(r['node'])) not in done]
    print("curated %d, already labelled %d, pool %d" % (len(rows), len(done), len(pool)))

    def key(r):
        return (int(r['event']), int(r['node']))

    picked, strat = [], {}

    def take(name, sel, n):
        got = 0
        for r in sorted(sel, key=lambda r: -float(r['Q'])):
            k = key(r)
            if k in strat:
                continue
            strat[k] = name; picked.append(r); got += 1
            if got >= n:
                break
        print("  %-16s selected %d of %d available" % (name, got, len(sel)))

    take('S3-unjudged-fire', [r for r in pool if key(r) in peeled], a.n_per_stratum)
    take('S1-seed-capped', [r for r in pool if C.get(key(r), {}).get('n_seed') == 4],
         a.n_per_stratum)
    take('S2-bound-region', [r for r in pool
                             if 15.0 <= C.get(key(r), {}).get('b', -1) <= 60.0],
         a.n_per_stratum)
    rest = [r for r in pool if key(r) not in strat]
    rng = random.Random(SEED)
    rng.shuffle(rest)
    for r in rest[:a.n_per_stratum]:
        strat[key(r)] = 'S4-control'; picked.append(r)
    print("  %-16s selected %d of %d available (seed %d)"
          % ('S4-control', min(a.n_per_stratum, len(rest)), len(rest), SEED))

    cols = ['event', 'node', 'stratum', 'owner_scan', 'proxy_cls', 'Q', 'nseg', 'npts',
            'n_seed', 'valley', 'b_cm', 'fired', 'peeled']
    with open(a.out, 'w', newline='') as fh:
        fh.write("# doc pr/139 sec 19 -- the NEXT per-part scan set, tag splitscan-0903-wide.\n")
        fh.write("# Stratified over the pr/137 curated set MINUS the %d objects already\n" % len(done))
        fh.write("# labelled in %s.  Strata are named in the `stratum` column and each is\n" % DONE_TAG)
        fh.write("# tied to a measurement sec 13.3/sec 17/sec 18 cannot currently resolve.\n")
        fh.write("# S4-control is seeded-random (seed %d) and chosen INDEPENDENTLY of every\n" % SEED)
        fh.write("# feature the other strata select on, so the set is not selected by the\n")
        fh.write("# hypotheses it will be used to test.\n")
        fh.write("# owner_scan=1 throughout: every object here is for the owner.\n")
        w = csv.writer(fh, delimiter='\t', lineterminator='\n')
        w.writerow(cols)
        for r in sorted(picked, key=lambda r: (strat[key(r)], -float(r['Q']))):
            k = key(r); c = C.get(k, {})
            w.writerow([r['event'], r['node'], strat[k], 1, r.get('proxy_cls', ''),
                        r['Q'], r.get('nseg', ''), r.get('npts', ''),
                        c.get('n_seed', ''), ('%.4f' % c['valley']) if 'valley' in c else '',
                        ('%.2f' % c['b']) if 'b' in c else '',
                        c.get('fired', ''), 1 if k in peeled else 0])
    print("\nwrote %s (%d objects)" % (a.out, len(picked)))
    print("counts by stratum: %s" % dict(collections.Counter(strat.values())))
    return 0


if __name__ == '__main__':
    sys.exit(main())
