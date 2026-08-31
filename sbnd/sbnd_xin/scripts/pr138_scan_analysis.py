#!/usr/bin/env python3
# doc pr/138 Phase A -- fold the owner's 172-object hand scan into numbers.
"""The Phase A analysis: what the owner's labels say, and what Phase B must beat.

Reads
  em_labels/splitscan-0901-owner/   172 hand labels (verdict + segment->part)
  em_labels/splitscan-0901-agent/    26 agent labels (the overlap)
  docs/pr/pr137-curated-set.tsv     the 25 offline features per object

and produces, in order:

  N1  agent-vs-owner agreement on the overlap        (sec A4 number 1)
  N2  false-split rate on the S1 random control      (sec A4 number 2)
  N3  the arm-difference proxy's confusion matrix    (sec A4 number 3)
  N4  the feature bake-off against REAL labels, fit/holdout split
  N5  the shipped proposal scored as a trigger -- the number Phase B must beat
  N6  the kernel test: does the proposal's boundary match the owner's

PRE-REGISTRATION.  The fit/holdout split is `event % 2`, fixed here and never
re-drawn.  Feature ranking and the 2-feature cut scan run on FIT ONLY; the
holdout is evaluated once, at the end, for the rule chosen on fit.  Anything
else would be choosing a threshold on the data used to certify it.

    python3 scripts/pr138_scan_analysis.py [--tsv out.tsv]
"""
import os, sys, json, glob, csv, math, argparse, collections, itertools

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(ROOT, 'split_display'))
sys.path.insert(0, HERE)
os.chdir(ROOT)
import numpy as np                                                # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--owner-tag', default='splitscan-0901-owner')
ap.add_argument('--agent-tag', default='splitscan-0901-agent')
ap.add_argument('--set', default='docs/pr/pr137-curated-set.tsv')
ap.add_argument('--tsv', default='docs/pr/pr138-scan-analysis.tsv')
ap.add_argument('--kernel', action='store_true',
                help='also run N6 (needs the arms; ~2 min)')
args = ap.parse_args()

SPLITS = ('SPLIT2', 'SPLIT3', 'SPLIT4+')


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
AGT = load_labels(args.agent_tag)
FEAT = {}
for r in csv.DictReader((l for l in open(args.set) if not l.startswith('#')),
                        delimiter='\t'):
    FEAT[(int(r['event']), int(r['node']))] = r

# the 8 track-typed objects of sec A1.6 -- pid != 11 in the arm's own record
TRACKS = {(99838, 14004), (389538, 19021), (292524, 9018), (176502, 109141),
          (286681, 72040), (122660, 54071), (415278, 23047), (278420, 18002)}

KEYS = [k for k in FEAT if k in OWN]
print("doc pr/138 Phase A -- %d owner labels, %d agent labels, %d feature rows"
      % (len(OWN), len(AGT), len(FEAT)))
print("scored objects: %d  (of which track-typed: %d)"
      % (len(KEYS), len([k for k in KEYS if k in TRACKS])))


def verdict(k, tab=OWN):
    return (tab.get(k) or {}).get('verdict')


def is_split(k, tab=OWN):
    return verdict(k, tab) in SPLITS


# --------------------------------------------------------------- N0 the census
print("\n=== N0. what the owner's 172 say ===")
byv = collections.Counter(verdict(k) for k in KEYS)
print("  all 172        :", dict(byv))
sci = [k for k in KEYS if k not in TRACKS]
print("  the 164 EM ones:", dict(collections.Counter(verdict(k) for k in sci)))
print("  track-typed 8  :", dict(collections.Counter(verdict(k) for k in KEYS
                                                     if k in TRACKS)))
for st in ('S1', 'S2', 'S3', 'S1+S2'):
    ks = [k for k in sci if FEAT[k]['stratum'] == st]
    c = collections.Counter(verdict(k) for k in ks)
    print("  %-6s n=%-4d %s" % (st, len(ks), dict(c)))

# ------------------------------------------------------- N1 agreement (A4 #1)
print("\n=== N1. agent-vs-owner agreement on the overlap (sec A4 number 1) ===")
ov = sorted(set(OWN) & set(AGT))
if not ov:
    print("  no overlap")
else:
    exact = sum(1 for k in ov if verdict(k, OWN) == verdict(k, AGT))
    coarse = sum(1 for k in ov if is_split(k, OWN) == is_split(k, AGT))
    print("  n=%d   exact verdict %d/%d = %.3f   SPLIT-vs-not %d/%d = %.3f"
          % (len(ov), exact, len(ov), exact / len(ov), coarse, len(ov),
             coarse / len(ov)))
    dis = [(k, verdict(k, OWN), verdict(k, AGT)) for k in ov
           if is_split(k, OWN) != is_split(k, AGT)]
    for k, a, b in dis:
        print("    DISAGREE evt%-8d node%-8d owner=%-8s agent=%s" % (k[0], k[1], a, b))

# ------------------------------------------- N2 false-split on S1 (A4 #2)
print("\n=== N2. false-split rate on the S1 random control (sec A4 number 2) ===")
s1 = [k for k in sci if FEAT[k]['stratum'].startswith('S1')]
s1pure = [k for k in sci if FEAT[k]['stratum'] == 'S1']
# ONLY the first row is a false-split budget.  S1+S2 is by construction the
# known-merge INTERSECTION, so pooling it with S1 destroys the very property
# (feature independence) that makes S1 quotable -- the pooled 11/96 is a
# stratum-contaminated number and must never be quoted as a prevalence.
for nm, ks in (('S1 only  <= USE THIS', s1pure),
               ('S1+S1+S2 (contaminated, do NOT quote)', s1)):
    nsp = sum(1 for k in ks if is_split(k))
    ntr = sum(1 for k in ks if verdict(k) == 'TRIM')
    print("  %-38s n=%-4d SPLIT %d (%.1f%%)  TRIM %d (%.1f%%)  KEEP %d"
          % (nm, len(ks), nsp, 100.0 * nsp / max(len(ks), 1),
             ntr, 100.0 * ntr / max(len(ks), 1), len(ks) - nsp - ntr))
print("  => S1 is the feature-INDEPENDENT stratum, so its SPLIT fraction is the"
      "\n     population prevalence of over-clustering, and 1 - (its KEEP fraction)"
      "\n     bounds what an always-fire splitter would break.")

# --------------------------------------------- N3 the proxy's truth (A4 #3)
print("\n=== N3. the arm-difference proxy vs the hand labels (sec A4 number 3) ===")
cm = collections.Counter()
for k in sci:
    cm[(FEAT[k]['proxy_cls'], 'SPLIT' if is_split(k) else verdict(k))] += 1
prox = sorted({p for p, _ in cm})
outs = sorted({o for _, o in cm})
print("  %-10s %s" % ('proxy', ''.join('%-9s' % o for o in outs)))
for p in prox:
    n = sum(cm[(p, o)] for o in outs)
    print("  %-10s %s  n=%d" % (p, ''.join('%-9d' % cm[(p, o)] for o in outs), n))
mg = [k for k in sci if FEAT[k]['proxy_cls'] == 'MERGED']
real = sum(1 for k in mg if is_split(k))
print("  proxy-MERGED that are REALLY splittable: %d / %d = %.3f"
      % (real, len(mg), real / max(len(mg), 1)))
sg = [k for k in sci if FEAT[k]['proxy_cls'] == 'SINGLE']
miss = sum(1 for k in sg if is_split(k))
print("  proxy-SINGLE that the owner SPLIT      : %d / %d = %.3f"
      % (miss, len(sg), miss / max(len(sg), 1)))

# ------------------------------------------------------------ N4 the bake-off
print("\n=== N4. feature bake-off against REAL labels (fit = even event id) ===")
FEATS = ['valley_best', 'd2_best', 'frac_best', 'angle_best', 'valley',
         'd2_over_d1', 'seed_frac', 'n_seed', 'angle', 'balance', 'gap_cm',
         'gap_scaled', 'w_pull', 'sep_scaled', 'vgap_min', 'dedx15_min',
         'n_2mip', 'r_ratio', 'q_ratio', 'm_pi0', 'nseg', 'npts', 'Q']


def fval(k, f):
    try:
        v = float(FEAT[k][f])
    except Exception:
        return float('nan')
    return v


def auc(pos, neg):
    """rank AUC, NaNs dropped; returns (auc, n_pos_used, n_neg_used)"""
    p = [x for x in pos if x == x]
    n = [x for x in neg if x == x]
    if not p or not n:
        return float('nan'), len(p), len(n)
    allv = sorted(p + n)
    rank = {}
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        r = 0.5 * (i + j) + 1.0
        rank[allv[i]] = r
        i = j + 1
    R = sum(rank[x] for x in p)
    a = (R - len(p) * (len(p) + 1) / 2.0) / (len(p) * len(n))
    return a, len(p), len(n)


def purity_at_eff(pos, neg, eff=0.5):
    """purity of the selected set at `eff` efficiency, scanning both directions"""
    best = (float('nan'), None, None)
    for sign in (+1, -1):
        p = sorted([sign * x for x in pos if x == x])
        n = [sign * x for x in neg if x == x]
        if len(p) < 3 or not n:
            continue
        need = max(1, int(round(eff * len(p))))
        thr = p[len(p) - need]                      # keep the `need` largest
        tp = sum(1 for x in p if x >= thr)
        fp = sum(1 for x in n if x >= thr)
        pur = tp / float(tp + fp)
        if not (best[0] == best[0]) or pur > best[0]:
            best = (pur, sign, sign * thr)
    return best


FIT = [k for k in sci if k[0] % 2 == 0]
HOLD = [k for k in sci if k[0] % 2 == 1]
print("  fit n=%d (SPLIT %d) | holdout n=%d (SPLIT %d)"
      % (len(FIT), sum(1 for k in FIT if is_split(k)),
         len(HOLD), sum(1 for k in HOLD if is_split(k))))

rank = []
for f in FEATS:
    pos = [fval(k, f) for k in FIT if is_split(k)]
    neg = [fval(k, f) for k in FIT if not is_split(k)]
    a, npos, nneg = auc(pos, neg)
    pur, sign, thr = purity_at_eff(pos, neg, 0.5)
    rank.append((f, a, pur, sign, thr, npos, nneg))
rank.sort(key=lambda t: -(abs(t[1] - 0.5) if t[1] == t[1] else -1))
print("  %-13s %6s %8s   %s" % ('feature', 'AUC', 'pur@50%', 'cut'))
for f, a, pur, sign, thr, npos, nneg in rank:
    if a != a:
        continue
    op = '>=' if sign > 0 else '<='
    print("  %-13s %6.3f %8s   %s %s %.4g"
          % (f, a, ('%.3f' % pur) if pur == pur else '  --', f, op, thr))

# 2-feature cut scan, FIT ONLY
print("\n  best 2-feature cuts on FIT (>= 40%% efficiency), by purity:")
cand = [t[0] for t in rank if t[1] == t[1]][:10]


def cuts_for(f, ks):
    vs = sorted({fval(k, f) for k in ks if fval(k, f) == fval(k, f)})
    if len(vs) < 3:
        return []
    q = [vs[int(x * (len(vs) - 1))] for x in np.linspace(0.05, 0.95, 19)]
    return sorted(set(q))


def rule_perf(rule, ks):
    tp = sum(1 for k in ks if rule(k) and is_split(k))
    fp = sum(1 for k in ks if rule(k) and not is_split(k))
    P = sum(1 for k in ks if is_split(k))
    eff = tp / float(P) if P else float('nan')
    pur = tp / float(tp + fp) if (tp + fp) else float('nan')
    return eff, pur, tp, fp


best2 = []
for f1, f2 in itertools.combinations(cand, 2):
    for t1 in cuts_for(f1, FIT):
        for s1_ in (+1, -1):
            for t2 in cuts_for(f2, FIT):
                for s2_ in (+1, -1):
                    def rule(k, f1=f1, t1=t1, s1_=s1_, f2=f2, t2=t2, s2_=s2_):
                        a = fval(k, f1); b = fval(k, f2)
                        if a != a or b != b:
                            return False
                        return ((a >= t1) if s1_ > 0 else (a <= t1)) and \
                               ((b >= t2) if s2_ > 0 else (b <= t2))
                    eff, pur, tp, fp = rule_perf(rule, FIT)
                    if eff >= 0.40 and tp >= 5:
                        best2.append((pur, eff, f1, s1_, t1, f2, s2_, t2, tp, fp))
best2.sort(key=lambda t: (-t[0], -t[1]))
seen = set()
shown = []
for b in best2:
    key = (b[2], b[5])
    if key in seen:
        continue
    seen.add(key)
    shown.append(b)
    if len(shown) >= 6:
        break
for pur, eff, f1, s1_, t1, f2, s2_, t2, tp, fp in shown:
    print("    pur %.3f eff %.3f  (%s %s %.4g) AND (%s %s %.4g)   tp=%d fp=%d"
          % (pur, eff, f1, '>=' if s1_ > 0 else '<=', t1,
             f2, '>=' if s2_ > 0 else '<=', t2, tp, fp))

# ------------------------------- N5 the shipped proposal, scored as a trigger
print("\n=== N5. the SHIPPED proposal scored as a trigger (the bar for Phase B) ===")


def shipped(k):
    """propose() fires when a second angular maximum exists and valley_best<=0.95"""
    v = fval(k, 'valley_best')
    ns = fval(k, 'n_seed')
    return (ns == ns and ns >= 2) and (v == v and v <= 0.95)


for nm, ks in (('all 164', sci), ('fit', FIT), ('holdout', HOLD)):
    eff, pur, tp, fp = rule_perf(shipped, ks)
    print("  %-9s eff %.3f  pur %.3f  (tp %d, fp %d, positives %d)"
          % (nm, eff, pur, tp, fp, sum(1 for k in ks if is_split(k))))
print("  and what the PROPOSAL actually pre-filled (n_groups>=2 in the label):")
pf = [k for k in sci if (OWN[k].get('proposal') or '').startswith('2 groups')]
tp = sum(1 for k in pf if is_split(k))
print("    fired on %d of %d;  of those %d were really SPLIT -> purity %.3f"
      % (len(pf), len(sci), tp, tp / max(len(pf), 1)))
miss = [k for k in sci if is_split(k) and k not in pf]
print("    MISSED %d of the %d real splits -> efficiency %.3f"
      % (len(miss), sum(1 for k in sci if is_split(k)),
         1 - len(miss) / max(sum(1 for k in sci if is_split(k)), 1)))

# ------------------------------------------------- write the machine-readable
with open(args.tsv, 'w') as fh:
    fh.write("# doc pr/138 Phase A -- per-object scan result + features\n")
    cols = ['event', 'node', 'stratum', 'proxy_cls', 'track_typed',
            'owner_verdict', 'owner_conf', 'owner_nparts', 'agent_verdict',
            'proposal_fired', 'half'] + FEATS
    fh.write('\t'.join(cols) + '\n')
    for k in sorted(KEYS):
        r = OWN[k]
        row = [k[0], k[1], FEAT[k]['stratum'], FEAT[k]['proxy_cls'],
               int(k in TRACKS), r.get('verdict'), r.get('confidence'),
               r.get('n_parts'), (AGT.get(k) or {}).get('verdict', ''),
               int((r.get('proposal') or '').startswith('2 groups')),
               'fit' if k[0] % 2 == 0 else 'holdout'] + \
              [FEAT[k].get(f, '') for f in FEATS]
        fh.write('\t'.join(str(x) for x in row) + '\n')
print("\nwrote %s" % args.tsv)


# =========================================================== N6 + N7
# These need the arms (they re-run propose()), so they are opt-in with --kernel.
if args.kernel:
    import split_model as SM
    import itertools as _it

    print("\n=== N6. did the owner EDIT the proposal?  (the anchoring probe) ===")
    print("  The scan tool PRE-FILLS the columns and prints the proposal in words,")
    print("  so these labels are NOT blind to it (feedback: blind-the-scan-sheet).")
    print("  The only defence is to measure how often the owner overrode it.")
    same = diff = nofile = 0
    edited_split = edited_keep = 0
    kern = []        # (key, charge purity of the proposal's boundary vs owner's)
    for k in sorted(sci):
        row = SM.load_object(k[0], k[1])
        if row is None:
            nofile += 1
            continue
        p = SM.object_payload(row)
        prop = {s['seg']: s['group'] for s in p['segs']}
        ownr = {int(a): int(b) for a, b in (OWN[k].get('groups') or {}).items()}
        q = {s['seg']: max(s['q'], 0.0) for s in p['segs']}
        segs = [s for s in prop if s in ownr]
        if not segs:
            nofile += 1
            continue
        identical = all(prop[s] == ownr[s] for s in segs)
        same += identical
        diff += (not identical)
        if not identical:
            if is_split(k):
                edited_split += 1
            else:
                edited_keep += 1
        # boundary quality: best label permutation, charge-weighted
        if is_split(k):
            pg = sorted({prop[s] for s in segs})
            og = sorted({ownr[s] for s in segs})
            Qt = sum(q[s] for s in segs) or 1.0
            best = 0.0
            for perm in _it.permutations(pg, min(len(pg), len(og))):
                m = dict(zip(perm, og))
                got = sum(q[s] for s in segs if m.get(prop[s]) == ownr[s])
                best = max(best, got / Qt)
            kern.append((k, best))
    print("  proposal accepted UNCHANGED: %d   overridden: %d   (unscored %d)"
          % (same, diff, nofile))
    print("    of the overrides, %d ended SPLIT and %d ended KEEP/TRIM"
          % (edited_split, edited_keep))
    fired = [k for k in sci if (OWN[k].get('proposal') or '').startswith('2 groups')]
    kept = [k for k in fired if not is_split(k)]
    print("  proposal fired but owner said KEEP/TRIM anyway: %d of %d fires"
          % (len(kept), len(fired)))
    nofire_split = [k for k in sci if is_split(k)
                    and not (OWN[k].get('proposal') or '').startswith('2 groups')]
    print("  proposal silent but owner SPLIT anyway         : %d" % len(nofire_split))
    print("  => the owner overrode the pre-fill in both directions, so the labels")
    print("     are not a rubber stamp; the residual anchoring is bounded by the")
    print("     agent-vs-owner comparison below, where the agent scanned BLIND.")

    if kern:
        vals = sorted(v for _, v in kern)
        print("\n=== N7. the KERNEL: does the proposal's boundary match the owner's? ===")
        print("  n=%d owner-SPLIT objects, charge-weighted agreement of the best"
              " label matching:" % len(kern))
        print("    median %.3f   mean %.3f   >=0.90 in %d   <0.60 in %d"
              % (vals[len(vals) // 2], sum(vals) / len(vals),
                 sum(1 for v in vals if v >= 0.90), sum(1 for v in vals if v < 0.60)))
        worst = sorted(kern, key=lambda t: t[1])[:6]
        for (e, n), v in worst:
            print("    worst  evt%-8d node%-8d agreement %.3f  (%s)"
                  % (e, n, v, verdict((e, n))))

    print("\n=== N8. blind agent vs proposal-anchored owner, on the 26 overlap ===")
    ovk = [k for k in ov if k in sci]
    fa = sum(1 for k in ovk
             if (OWN[k].get('proposal') or '').startswith('2 groups') == is_split(k, OWN))
    fb = sum(1 for k in ovk
             if (OWN[k].get('proposal') or '').startswith('2 groups') == is_split(k, AGT))
    print("  owner verdict agrees with the proposal on %d/%d = %.3f"
          % (fa, len(ovk), fa / max(len(ovk), 1)))
    print("  BLIND agent  agrees with the proposal on %d/%d = %.3f"
          % (fb, len(ovk), fb / max(len(ovk), 1)))
    print("  a large gap would be the anchoring signature; a small one bounds it.")
