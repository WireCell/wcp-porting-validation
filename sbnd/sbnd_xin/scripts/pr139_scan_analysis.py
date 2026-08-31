#!/usr/bin/env python3
"""doc pr/139 §6 -- the owner's 2026-09-01 scan (tag splitscan-0902-pi0), scored.

This is the measurement that OVERTURNS this round's own recommendation.  §3bis
proposed `shower_split_max_impact = 12` and said in §2.1, before the scan, that
the bound was chosen after seeing eight movers and had to be priced on fresh
labels.  It now is, and it fails: at 12 cm the bound suppresses NINE of the
nineteen cuts the owner confirms.

Outputs docs/pr/pr139-scan-verdicts.tsv.
"""
import json, glob, re, csv, sys, itertools, collections, statistics

TAG = 'splitscan-0902-pi0'
TAPE_ARM = 'work-pr139r2-oncomb'          # fixed binary, probe on, b in cm
POS = lambda v: bool(v) and v.startswith('SPLIT')


def owner_labels(tag=TAG):
    verd, part = {}, {}
    for f in sorted(glob.glob(f'em_labels/{tag}/labels-evt*.json')):
        d = json.load(open(f))
        ev = int(d['event'][3:])
        for n, v in (d.get('split_labels') or {}).items():
            k = (ev, int(n))
            verd[k] = v.get('verdict')
            part[k] = {int(s): int(g) for s, g in (v.get('groups') or {}).items() if int(g) >= 0}
    return verd, part


def tape_b(arm=TAPE_ARM):
    """(event,node) -> {fired, b_cm}.  `fired` is the KERNEL decision: the tape
    is written before the veto, so a vetoed candidate is still on it."""
    out = {}
    C = re.compile(r'SHOWER_SPLIT cand shower=(\d+) .*?fired=(\d) .*?b_cm=(-?[\d.]+) veto=(\d)')
    for lg in sorted(glob.glob(f'{arm}-*/pr_evt*/stdout.log')):
        ev = int(re.search(r'pr_evt(\d+)/', lg).group(1))
        for line in open(lg, errors='replace'):
            m = C.search(line)
            if m:
                out[(ev, int(m.group(1)))] = dict(fired=int(m.group(2)),
                                                  b=float(m.group(3)))
    return out


def tape_parts(arm=TAPE_ARM):
    P = re.compile(r'SHOWER_SPLIT part shower=(\d+) part=(\d+) nseg=(\d+) q=([\d.eE+-]+) segs=([\d,]+)')
    part = collections.defaultdict(dict)
    qmap = collections.defaultdict(dict)
    for lg in sorted(glob.glob(f'{arm}-*/pr_evt*/stdout.log')):
        ev = int(re.search(r'pr_evt(\d+)/', lg).group(1))
        for line in open(lg, errors='replace'):
            m = P.search(line)
            if not m:
                continue
            node, g, q = int(m.group(1)), int(m.group(2)), float(m.group(4))
            segs = [int(x) for x in m.group(5).split(',') if x]
            for s in segs:
                part[(ev, node)][s] = g
                qmap[(ev, node)][s] = q / max(len(segs), 1)
    return part, qmap


def agreement(prop, own, qm):
    """charge-weighted best-label-permutation agreement -- doc pr/138 §A5.6's
    metric, transcribed from pr138_kernel_k.py:220 so the numbers compare."""
    segs = [s for s in prop if s in own]
    if not segs:
        return None
    pg = sorted({prop[s] for s in segs})
    og = sorted({own[s] for s in segs})
    Qt = sum(qm.get(s, 0.0) for s in segs) or 1.0
    best = 0.0
    for perm in itertools.permutations(pg, min(len(pg), len(og))):
        m = dict(zip(perm, og))
        best = max(best, sum(qm.get(s, 0.0) for s in segs if m.get(prop[s]) == own[s]) / Qt)
    return best


def shared_refused():
    """the events where P1.1 (skip_shared) refused a peel -- measured, not
    assumed, by diffing the daughter sets of the off and onshared arms."""
    import os

    def daus(arm):
        out = collections.defaultdict(set)
        for s in ('mcp1k', 'mcp2k', 'ncpi0', 'nuecc48'):
            base = f'work-pr138r2-c90off-{s}'
            for d in sorted(glob.glob(base + '/pr_evt*')):
                ev = int(d.rsplit('pr_evt', 1)[1])
                pa = f'{base}/pr_evt{ev}/calib-pr-evt{ev}.json'
                pb = f'{arm}-{s}/pr_evt{ev}/calib-pr-evt{ev}.json'
                if not (os.path.exists(pa) and os.path.exists(pb)):
                    continue
                O = {x['shower_id'] for x in json.load(open(pa))['showers']}
                for x in json.load(open(pb))['showers']:
                    if x['shower_id'] not in O:
                        out[ev].add(x['id'])
        return out
    A, B = daus('work-pr139r1-off'), daus('work-pr139r1-onshared')
    return {ev for ev in set(A) | set(B) if A[ev] - B[ev]}


verd, opart = owner_labels()
B = tape_b()
cpart, qmap = tape_parts()
J = {k: v for k, v in verd.items() if k in B}
P = [k for k in J if POS(J[k])]

print("=== the 2026-09-01 owner scan (tag %s) ===" % TAG)
print("  objects labelled                     : %d" % len(verd))
print("  ...also taped splitter candidates    : %d" % len(J))
print("  verdicts: %s" % dict(collections.Counter(verd.values())))

print("\n=== 1. the TRIGGER, on these labels (kernel decision, no bound) ===")
tp = [k for k in J if B[k]['fired'] and POS(J[k])]
fp = [k for k in J if B[k]['fired'] and not POS(J[k])]
print("  fires %d   correct %d   false %d   efficiency %.3f   purity %.3f"
      % (len(tp) + len(fp), len(tp), len(fp), len(tp) / len(P), len(tp) / max(len(tp) + len(fp), 1)))
print("  the false fires:")
for k in sorted(fp, key=lambda k: -B[k]['b']):
    print("    evt%-8d node=%-8d %-8s b=%.2f cm" % (k[0], k[1], J[k], B[k]['b']))

print("\n=== 2. what the b bound COSTS on these labels ===")
print("  %-8s %6s %6s %6s %8s %8s  %s" % ('b <=', 'fires', 'right', 'wrong', 'eff', 'pur', 'confirmed cuts suppressed'))
for T in (10, 12, 15, 20, 25, 30, 40, 1e9):
    ff = [k for k in J if B[k]['fired'] and B[k]['b'] <= T]
    rr = [k for k in ff if POS(J[k])]
    sup = len([k for k in tp if B[k]['b'] > T])
    print("  %-8s %6d %6d %6d %8.3f %8.3f  %d of %d"
          % (('%g' % T) if T < 1e8 else 'none', len(ff), len(rr), len(ff) - len(rr),
             len(rr) / len(P), len(rr) / max(len(ff), 1), sup, len(tp)))

SH = shared_refused()
print("\n=== 3. P1.1 (skip_shared) + a bound -- the operating point the scan supports ===")
print("  P1.1 refuses a peel in events: %s" % sorted(SH))
print("  %-8s %6s %6s %6s %8s %8s  %s" % ('b <=', 'fires', 'right', 'wrong', 'eff', 'pur', 'false fires left'))
for T in (12, 20, 25, 30, 40, 1e9):
    ff = [k for k in J if B[k]['fired'] and k[0] not in SH and B[k]['b'] <= T]
    rr = [k for k in ff if POS(J[k])]
    left = ['%d/%d' % k for k in ff if not POS(J[k])]
    print("  %-8s %6d %6d %6d %8.3f %8.3f  %s"
          % (('%g' % T) if T < 1e8 else 'none', len(ff), len(rr), len(ff) - len(rr),
             len(rr) / len(P), len(rr) / max(len(ff), 1), ",".join(left) or '-'))

print("\n=== 4. the KERNEL's boundary vs the owner's (doc pr/138 §A5.6 metric) ===")
rows = []
for k in sorted(J):
    if not POS(J[k]) or k not in cpart:
        continue
    a = agreement(cpart[k], opart[k], qmap[k])
    if a is not None:
        rows.append((k[0], k[1], J[k], len(set(opart[k].values())), len(set(cpart[k].values())), a))
print("  %-9s %-9s %-8s %7s %6s %8s" % ('event', 'node', 'owner', 'owner_k', 'cxx_k', 'agree'))
for r in sorted(rows, key=lambda r: r[5]):
    print("  %-9d %-9d %-8s %7d %6d %8.3f" % r)
b2 = [r[5] for r in rows if r[2] == 'SPLIT2']
bk = [r[5] for r in rows if r[2] != 'SPLIT2']
if b2:
    print("  SPLIT2  n=%d  median %.3f  mean %.3f  exact %d" % (len(b2), statistics.median(b2), sum(b2) / len(b2), sum(1 for x in b2 if x > 0.999)))
if bk:
    print("  k>=3    n=%d  median %.3f  mean %.3f  (capped by max_parts=2)" % (len(bk), statistics.median(bk), sum(bk) / len(bk)))

with open('docs/pr/pr139-scan-verdicts.tsv', 'w') as f:
    w = csv.writer(f, delimiter='\t')
    w.writerow(['event', 'node', 'owner_verdict', 'owner_k', 'cxx_fired', 'b_cm',
                'cxx_k', 'boundary_agree', 'p11_refuses'])
    ag = {(r[0], r[1]): r for r in rows}
    for k in sorted(J):
        r = ag.get(k)
        w.writerow([k[0], k[1], J[k], len(set(opart[k].values())) or 1, B[k]['fired'],
                    '%.2f' % B[k]['b'], (r[4] if r else ''), ('%.3f' % r[5]) if r else '',
                    int(k[0] in SH)])
print("\nwrote docs/pr/pr139-scan-verdicts.tsv (%d rows)" % len(J))
