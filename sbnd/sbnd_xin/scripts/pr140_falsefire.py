#!/usr/bin/env python3
"""doc pr/139 sec 18 -- item 4: the one false fire left at the P1.6 operating point.

At `skip_shared` + `max_impact = 30`, four of the five false fires are already
handled (54332, 174771, 393505 by the bound; 165157 by the guard).  **278420/61027
is the only one standing.**  This asks the only question worth asking about a
single object: is it an OUTLIER on any tape feature, relative to the 14
owner-confirmed cuts that fire at the same operating point?

It deliberately does NOT fit anything.  One negative against fourteen positives
cannot support a threshold -- doc pr/138 sec A5.4 spent the holdout on exactly
this mistake.  A feature only counts here if 278420 falls strictly OUTSIDE the
full range of the 14, which is a claim a single object can actually support.

    python3 scripts/pr140_falsefire.py --arm work-pr140r1-on
"""
import argparse, collections, glob, json, os, re, statistics, sys

SX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAG = 'splitscan-0902-pi0'
TARGET = (278420, 61027)
CAND = re.compile(
    r'SHOWER_SPLIT cand shower=(\d+) pdg=(-?\d+) nseg=(\d+) npts=(\d+) Q=([\d.eE+-]+) '
    r'n_seed=(\d+) valley_best=([\d.]+) angle_best=(-?[\d.]+) nacc=(\d+) nparts=(\d+) '
    r'fired=(\d) decim=(\d) vtx=\S+ vgap_cm=(-?[\d.]+) vchi2=([\d.]+) vdQ=([\d.eE+-]+) '
    r'vfit=(\d) b_cm=(-?[\d.]+) veto=(\d)')
PART = re.compile(r'SHOWER_SPLIT part shower=(\d+) part=(\d+) nseg=(\d+) q=([\d.eE+-]+)')
PEEL = re.compile(r'SHOWER_SPLIT peel shower=(\d+) part=(\d+) new_start=(\d+) nseg=(\d+) '
                  r'conn=(\d+) root_vtx_cm=(-?[\d.]+) fwd=(-?[\d.]+)')


def owner():
    v = {}
    for f in sorted(glob.glob(os.path.join(SX, 'em_labels', TAG, 'labels-evt*.json'))):
        d = json.load(open(f)); ev = int(d['event'][3:])
        for n, x in (d.get('split_labels') or {}).items():
            v[(ev, int(n))] = x.get('verdict')
    return v


def read(arm):
    F, P, L = {}, collections.defaultdict(list), collections.defaultdict(list)
    for lg in sorted(glob.glob(os.path.join(SX, arm) + '-*/pr_evt*/stdout.log')):
        ev = int(re.search(r'pr_evt(\d+)/', lg).group(1))
        for line in open(lg, errors='replace'):
            m = CAND.search(line)
            if m:
                g = m.groups()
                F[(ev, int(g[0]))] = dict(
                    nseg=int(g[2]), npts=int(g[3]), Q=float(g[4]), n_seed=int(g[5]),
                    valley=float(g[6]), angle=float(g[7]), nacc=int(g[8]),
                    nparts=int(g[9]), fired=int(g[10]), vgap=float(g[12]),
                    vchi2=float(g[13]), b=float(g[16]), veto=int(g[17]))
                continue
            m = PART.search(line)
            if m:
                P[(ev, int(m.group(1)))].append(float(m.group(4)))
                continue
            m = PEEL.search(line)
            if m:
                L[(ev, int(m.group(1)))].append(dict(conn=int(m.group(5)),
                                                     rvtx=float(m.group(6)),
                                                     fwd=float(m.group(7))))
    return F, P, L


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arm', default='work-pr140r1-on')
    a = ap.parse_args()
    verd = owner()
    F, P, L = read(a.arm)

    # the population that actually FIRES at the operating point
    live = [k for k, c in F.items() if k in verd and c['fired'] and not c['veto']]
    # skip_shared refusals are not fires
    shared = {(281485, 89095), (165157, 9000), (350354, 18092)}
    live = [k for k in live if k not in shared]
    pos = [k for k in live if verd[k].startswith('SPLIT')]
    neg = [k for k in live if not verd[k].startswith('SPLIT')]
    print("firing at skip_shared + b<=30 : %d  (%d owner-confirmed, %d false)"
          % (len(live), len(pos), len(neg)))
    print("false fires still standing    : %s" % sorted(neg))
    if TARGET not in F:
        sys.exit("278420/61027 not on this arm's tape")

    def feats(k):
        c = dict(F[k])
        qs = sorted(P.get(k, []), reverse=True)
        c['q_bal'] = (qs[1] / qs[0]) if len(qs) > 1 and qs[0] > 0 else float('nan')
        c['q_small'] = qs[-1] if qs else float('nan')
        c['q_small_frac'] = (qs[-1] / sum(qs)) if qs and sum(qs) > 0 else float('nan')
        pl = L.get(k, [])
        c['fwd_min'] = min((x['fwd'] for x in pl), default=float('nan'))
        c['rvtx_max'] = max((x['rvtx'] for x in pl), default=float('nan'))
        c['conn_max'] = max((x['conn'] for x in pl), default=float('nan'))
        return c

    T = feats(TARGET)
    POS = [feats(k) for k in pos]
    keys = ['nseg', 'npts', 'Q', 'n_seed', 'valley', 'angle', 'nacc', 'vgap', 'vchi2', 'b',
            'q_bal', 'q_small', 'q_small_frac', 'fwd_min', 'rvtx_max', 'conn_max']
    print()
    print("  %-13s %12s | %12s %12s %12s | %s"
          % ('feature', '278420', 'pos min', 'pos median', 'pos max', 'SEPARATES?'))
    hits = []
    for f in keys:
        t = T.get(f, float('nan'))
        v = [c[f] for c in POS if c.get(f) == c.get(f)]
        if not v or t != t:
            continue
        lo, hi = min(v), max(v)
        sep = ''
        if t < lo:
            sep = 'YES  below all 14 (margin %.4g)' % (lo - t)
        elif t > hi:
            sep = 'YES  above all 14 (margin %.4g)' % (t - hi)
        if sep:
            hits.append((f, t, lo, hi))
        print("  %-13s %12.4g | %12.4g %12.4g %12.4g | %s"
              % (f, t, lo, statistics.median(v), hi, sep))
    print()
    if hits:
        print("  %d feature(s) put 278420 strictly outside the confirmed cuts:" % len(hits))
        for f, t, lo, hi in hits:
            print("     %-13s 278420=%.4g   confirmed range [%.4g, %.4g]" % (f, t, lo, hi))
        print("  A single negative cannot set a threshold -- these are LEADS for a")
        print("  wider label set (sec 19), not a knob.")
    else:
        print("  NO feature separates it.  278420/61027 sits inside the confirmed-cut")
        print("  range on every quantity the tape carries.  It is not distinguishable")
        print("  by more of the same information, and the honest next move is more")
        print("  labels (sec 19), not another feature.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
