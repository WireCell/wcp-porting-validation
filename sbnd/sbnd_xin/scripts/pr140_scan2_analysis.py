#!/usr/bin/env python3
"""doc pr/139 sec 22 -- the owner's 2026-09-03 scan (tag splitscan-0903-wide), scored.

A FORK of pr139_scan_analysis.py (M10); that script stays byte-untouched as the
record of sec 6.  This one scores the 32-object stratified set sec 19 shipped,
and answers the three questions sec 19 said it existed to answer:

  * sec 17 -- does anything actually want k > 4?  (the max_seeds decision)
  * sec 13 -- does raising max_parts to 3 pay, on objects chosen blind?
  * sec 10 -- what does the b <= 30 bound cost on a bound-region sample that was
              NOT chosen by the census?

Outputs docs/pr/pr140-scan2-verdicts.tsv.

Two arms are read, because "does the splitter fire" has two answers:
  work-pr140r2-off    the SHIPPED production config
  work-pr140r2-on     the P1.6 operating point (skip_shared + max_impact 30)
"""
import csv, collections, glob, itertools, json, os, re, statistics, sys

SX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAG = 'splitscan-0903-wide'
OLD = 'splitscan-0902-pi0'
SET = os.path.join(SX, 'docs', 'pr', 'pr140-scan-set.tsv')
OUT = os.path.join(SX, 'docs', 'pr', 'pr140-scan2-verdicts.tsv')
POS = lambda v: bool(v) and v.startswith('SPLIT')
CAND = re.compile(r'SHOWER_SPLIT cand shower=(\d+) .*?n_seed=(\d+) valley_best=([\d.]+) '
                  r'.*?nacc=(\d+) nparts=(\d+) fired=(\d) .*?b_cm=(-?[\d.]+) veto=(\d)')
PART = re.compile(r'SHOWER_SPLIT part shower=(\d+) part=(\d+) nseg=(\d+) q=([\d.eE+-]+) segs=([\d,]+)')
SHARED = re.compile(r'SHOWER_SPLIT shared shower=(\d+) part=(\d+)')


def labels(tag):
    verd, grp, kk = {}, {}, {}
    for f in sorted(glob.glob(os.path.join(SX, 'em_labels', tag, 'labels-evt*.json'))):
        d = json.load(open(f)); ev = int(d['event'][3:])
        for n, v in (d.get('split_labels') or {}).items():
            k = (ev, int(n))
            verd[k] = v.get('verdict'); kk[k] = int(v.get('n_parts') or 1)
            grp[k] = {int(s): int(g) for s, g in (v.get('groups') or {}).items() if int(g) >= 0}
    return verd, grp, kk


def strata():
    out = {}
    hdr = None
    for line in open(SET):
        if line.startswith('#'):
            continue
        f = line.rstrip('\n').split('\t')
        if hdr is None:
            hdr = f; continue
        d = dict(zip(hdr, f))
        out[(int(d['event']), int(d['node']))] = d['stratum']
    return out


def tape(arm):
    cand, part, qmap, shared = {}, collections.defaultdict(dict), collections.defaultdict(dict), set()
    for lg in sorted(glob.glob(os.path.join(SX, arm) + '-*/pr_evt*/stdout.log')):
        ev = int(re.search(r'pr_evt(\d+)/', lg).group(1))
        for line in open(lg, errors='replace'):
            m = CAND.search(line)
            if m:
                cand[(ev, int(m.group(1)))] = dict(n_seed=int(m.group(2)), valley=float(m.group(3)),
                                                   nacc=int(m.group(4)), nparts=int(m.group(5)),
                                                   fired=int(m.group(6)), b=float(m.group(7)),
                                                   veto=int(m.group(8)))
                continue
            m = PART.search(line)
            if m:
                node, g, q = int(m.group(1)), int(m.group(2)), float(m.group(4))
                segs = [int(x) for x in m.group(5).split(',') if x]
                for s in segs:
                    part[(ev, node)][s] = g
                    qmap[(ev, node)][s] = q / max(len(segs), 1)
                continue
            m = SHARED.search(line)
            if m:
                shared.add((ev, int(m.group(1))))
    return cand, part, qmap, shared


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


verd, grp, kk = labels(TAG)
ST = strata()
Coff, Poff, Qoff, _ = tape('work-pr140r2-off')
Con, Pon, Qon, SHon = tape('work-pr140r2-on')

print("=== the 2026-09-03 scan, tag %s (%d objects, all high confidence) ===" % (TAG, len(verd)))
print("  verdicts : %s" % dict(collections.Counter(verd.values())))
print("  n_parts  : %s" % dict(sorted(collections.Counter(kk.values()).items())))
print()
print("=== A. by stratum -- the S4 control is the unbiased number ===")
print("  %-18s %5s %6s %7s %7s %7s   %s" % ('stratum', 'n', 'KEEP', 'SPLIT2', 'SPLIT3', 'SPLIT>=4', 'split rate'))
for st in sorted(set(ST.values())):
    ks = [k for k in verd if ST.get(k) == st]
    c = collections.Counter(verd[k] for k in ks)
    ns = sum(1 for k in ks if POS(verd[k]))
    print("  %-18s %5d %6d %7d %7d %7d   %.3f"
          % (st, len(ks), c.get('KEEP', 0), c.get('SPLIT2', 0), c.get('SPLIT3', 0),
             sum(v for x, v in c.items() if x not in ('KEEP', 'SPLIT2', 'SPLIT3')),
             ns / len(ks) if ks else float('nan')))
print()
print("=== B. THE max_seeds QUESTION (sec 17) ===")
seedcap = [k for k in verd if ST.get(k) == 'S1-seed-capped']
print("  S1 objects were chosen BECAUSE n_seed sits at the hardcoded cap of 4.")
print("  the owner's k on them : %s" % dict(sorted(collections.Counter(kk[k] for k in seedcap).items())))
print("  max k over ALL 32     : %d" % max(kk.values()))
print("  objects wanting k > 4 : %d" % sum(1 for k in verd if kk[k] > 4))
print("  objects wanting k > 3 : %d" % sum(1 for k in verd if kk[k] > 3))
oldv, oldg, oldk = labels(OLD)
print("  for contrast, the 0902 set had k up to %d (%s)"
      % (max(oldk.values()), sorted(collections.Counter(oldk.values()).items())))
print()
print("=== C. the trigger on a BLIND set ===")
for name, C, SH in (('production (work-pr140r2-off)', Coff, set()),
                    ('P1.6        (work-pr140r2-on)', Con, SHon)):
    fired = [k for k in verd if C.get(k, {}).get('fired') and not C[k]['veto'] and k not in SH]
    pos = [k for k in verd if POS(verd[k])]
    right = [k for k in fired if POS(verd[k])]
    print("  %-30s fires %2d  eff %.3f (%d/%d)  pur %.3f"
          % (name, len(fired), len(right) / len(pos) if pos else float('nan'),
             len(right), len(pos), len(right) / len(fired) if fired else float('nan')))
    miss = [k for k in pos if k not in fired]
    fp = [k for k in fired if not POS(verd[k])]
    print("      missed cuts (%d): %s" % (len(miss), sorted(miss)))
    print("      false fires (%d): %s" % (len(fp), sorted(fp)))
print()
print("=== D. boundary agreement on this set (production arm) ===")
rows = []
for k in sorted(verd):
    if not POS(verd[k]) or k not in Poff:
        continue
    a = agreement(Poff[k], grp.get(k, {}), Qoff[k])
    if a is None:
        continue
    rows.append((k, verd[k], kk[k], len(set(Poff[k].values())), a))
for tier, sel in (('SPLIT2', lambda r: r[2] == 2), ('k>=3', lambda r: r[2] >= 3)):
    v = [r[4] for r in rows if sel(r)]
    if v:
        print("  %-8s n=%2d  median %.3f  mean %.3f  exact %d/%d"
              % (tier, len(v), statistics.median(v), statistics.fmean(v),
                 sum(1 for x in v if x >= 0.9995), len(v)))
print("  %-8s %-7s %-8s %5s %6s %8s" % ('event', 'node', 'verdict', 'own_k', 'cxx_k', 'agree'))
for k, ve, ok, ck, a in sorted(rows, key=lambda r: r[4]):
    print("  %-8d %-7d %-8s %5d %6d %8.3f" % (k[0], k[1], ve, ok, ck, a))
print()
print("=== E. what the b bound costs on a sample not chosen by the census ===")
print("  %-6s %5s %5s %5s %6s %6s   %s" % ('b <=', 'fires', 'right', 'wrong', 'eff', 'pur', 'confirmed cuts suppressed'))
pos = [k for k in verd if POS(verd[k])]
base = [k for k in verd if Coff.get(k, {}).get('fired')]
for B in (12, 20, 25, 30, 40, 10 ** 9):
    fires = [k for k in base if Coff[k]['b'] <= B]
    right = [k for k in fires if POS(verd[k])]
    sup = [k for k in pos if k in base and Coff[k]['b'] > B]
    print("  %-6s %5d %5d %5d %6.3f %6.3f   %d of %d %s"
          % ('none' if B > 1e8 else B, len(fires), len(right), len(fires) - len(right),
             len(right) / len(pos) if pos else float('nan'),
             len(right) / len(fires) if fires else float('nan'),
             len(sup), len(pos), sorted(sup) if sup else ''))
print()
print("=== F. the label set, combined ===")
allv = dict(oldv); allv.update(verd)
allk = dict(oldk); allk.update(kk)
print("  objects labelled : %d (0902) + %d (0903) = %d"
      % (len(oldv), len(verd), len(allv)))
print("  confirmed cuts   : %d + %d = %d"
      % (sum(1 for v in oldv.values() if POS(v)), sum(1 for v in verd.values() if POS(v)),
         sum(1 for v in allv.values() if POS(v))))
print("  hand parts       : %d" % sum(allk[k] for k in allv if POS(allv[k])))
print("  k distribution   : %s" % dict(sorted(collections.Counter(allk.values()).items())))

with open(OUT, 'w', newline='') as fh:
    w = csv.writer(fh, delimiter='\t', lineterminator='\n')
    w.writerow(['event', 'node', 'stratum', 'owner_verdict', 'owner_k',
                'off_fired', 'off_b_cm', 'off_n_seed', 'off_nacc', 'off_valley',
                'on_fired', 'on_veto', 'on_shared', 'cxx_k', 'boundary_agree'])
    for k in sorted(verd):
        c, o = Coff.get(k, {}), Con.get(k, {})
        a = agreement(Poff[k], grp.get(k, {}), Qoff[k]) if k in Poff and POS(verd[k]) else None
        w.writerow([k[0], k[1], ST.get(k, ''), verd[k], kk[k],
                    c.get('fired', ''), ('%.2f' % c['b']) if 'b' in c else '',
                    c.get('n_seed', ''), c.get('nacc', ''),
                    ('%.4f' % c['valley']) if 'valley' in c else '',
                    o.get('fired', ''), o.get('veto', ''), 1 if k in SHon else 0,
                    len(set(Poff[k].values())) if k in Poff else '',
                    ('%.4f' % a) if a is not None else ''])
print("\nwrote %s" % OUT)

# ---------------------------------------------------------------- appendix
print()
print("=== G. did the splitter fire on the RANDOM CONTROL? ===")
s4 = [k for k in verd if ST.get(k) == 'S4-control']
f4 = [k for k in s4 if Coff.get(k, {}).get('fired')]
print("  S4 objects %d, owner SPLIT %d, splitter fires %d  -> false-fire rate on a"
      % (len(s4), sum(1 for k in s4 if POS(verd[k])), len(f4)))
print("     blind random sample: %d/%d = %.3f" % (len(f4), len(s4), len(f4) / len(s4)))
print()
print("=== H. the three cuts the b<=30 bound rejects here, and their margins ===")
for k in sorted(verd):
    if not POS(verd[k]) or not Coff.get(k, {}).get('fired'):
        continue
    b = Coff[k]['b']
    if b > 30:
        print("  evt%-8d %-7d %-8s b=%6.2f   margin over the bound %+.2f cm"
              % (k[0], k[1], verd[k], b, b - 30))
print("  (offline-vs-C++ b agrees to 0.5 cm on 383/390 objects -- sec 8.3)")
print()
print("=== I. the operating points, on ALL 71 labelled objects ===")
oldset = {}
for line in open(os.path.join(SX, 'docs', 'pr', 'pr139-scan-verdicts.tsv')):
    f = line.rstrip('\n').split('\t')
    if f[0] == 'event':
        continue
    oldset[(int(f[0]), int(f[1]))] = dict(verdict=f[2], fired=f[4] == '1',
                                          b=float(f[5]), shared=f[8] == '1')
new = {}
for k in verd:
    c = Coff.get(k, {})
    new[k] = dict(verdict=verd[k], fired=bool(c.get('fired')), b=c.get('b', -1.0),
                  shared=k in SHon)
comb = dict(oldset); comb.update(new)
P = [k for k, d in comb.items() if POS(d['verdict'])]


def score(name, keep):
    fires = [k for k, d in comb.items() if d['fired'] and keep(d)]
    right = [k for k in fires if POS(comb[k]['verdict'])]
    print("  %-34s fires %2d  eff %.3f (%2d/%2d)  pur %.3f  cuts suppressed %2d"
          % (name, len(fires), len(right) / len(P), len(right), len(P),
             len(right) / len(fires) if fires else float('nan'),
             len(P) - len(right)))


score('production (as shipped)', lambda d: True)
score('skip_shared only', lambda d: not d['shared'])
score('skip_shared + b <= 30 (P1.6)', lambda d: not d['shared'] and d['b'] <= 30)
score('skip_shared + b <= 40', lambda d: not d['shared'] and d['b'] <= 40)
score('b <= 30 only', lambda d: d['b'] <= 30)
