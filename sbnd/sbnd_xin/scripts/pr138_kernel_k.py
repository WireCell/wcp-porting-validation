#!/usr/bin/env python3
# doc pr/138 Phase B stage B3 -- the kernel with k from the SEED COUNT.  READ-ONLY.
"""Does taking k = the number of surviving angular maxima fix the k>=3 boundary?

sec A5.6 measured the shipped kernel against the owner's 43 SPLIT labels:

    SPLIT2   n=33  median 1.000  mean 0.927     <- done
    SPLIT3   n=8   median 0.671  mean 0.620     <- broken
    SPLIT4+  n=2   median 0.467  mean 0.377     <- broken

and named the cause: `split_model.propose()` picks the single best PAIR of seeds
and assigns every bundle to one of two directions, so 10 of 43 splits (23%) are
unreachable no matter how good the seeding is.  This script is sec B3's
pre-registered test of the fix.

PRE-REGISTRATION, and it is the whole point of the design:

  * seed FINDING is FROZEN at the Phase A operating point -- profile_sigma_fn
    (w0=3.575, slope=0.0283 cm, floor 2 deg, ceil 60 deg), sep_scale=1.6,
    max_seeds=4.  If seed finding moved, `n_seed` would move, the trigger would
    move, and sec A5.4's 42-fire list would have to be re-scored.  It does not
    move here.
  * seed ACCEPTANCE introduces NO NEW PARAMETER.  The greedy rule reuses the two
    numbers the trigger already carries: valley <= 0.95 (sec A5.4, threshold
    scan says it is the knee, holdout spent) and charge fraction >= 0.03
    (`propose()`'s own pair filter).  So a win here is not a fit.

  Any variant that DOES introduce a parameter is printed under a FITTED banner
  and is not eligible to ship without its own holdout.

Repro:
    python3 scripts/pr138_kernel_k.py            # the pre-registered comparison
    python3 scripts/pr138_kernel_k.py --fitted   # + the parameter scans, flagged
"""
import os, sys, json, glob, csv, argparse, collections, itertools
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'split_display'))
import numpy as np
import pr137_lib as L
import split_model as SM

ap = argparse.ArgumentParser()
ap.add_argument('--owner-tag', default='splitscan-0901-owner')
ap.add_argument('--set', default='docs/pr/pr138-scan-analysis.tsv')
ap.add_argument('--fitted', action='store_true',
                help='also scan the acceptance parameters (FITTED, not eligible)')
ap.add_argument('--tsv', default='docs/pr/pr138-kernel-k.tsv')
args = ap.parse_args()

SPLITS = ('SPLIT2', 'SPLIT3', 'SPLIT4+')
# Phase A seed-finding constants.  FROZEN -- see the docstring.
SEP_SCALE, MAX_SEEDS = 1.6, 4
V_ACCEPT, F_ACCEPT = 0.95, 0.03


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
FEAT = {}
for r in csv.DictReader((l for l in open(args.set) if not l.startswith('#')),
                        delimiter='\t'):
    FEAT[(int(r['event']), int(r['node']))] = r
# The 8 track-typed objects of sec A1.6 are excluded, exactly as N7 did, so the
# baseline row here is comparable to sec A5.6's 43 (34 - 1 track-typed = 33).
TRACKS = {(99838, 14004), (389538, 19021), (292524, 9018), (176502, 109141),
          (286681, 72040), (122660, 54071), (415278, 23047), (278420, 18002)}
KEYS = [k for k in FEAT if k in OWN and (OWN[k].get('verdict') in SPLITS)
        and k not in TRACKS]
KEYS.sort()
print("doc pr/138 B3 -- kernel with k from the seed count")
print("owner-SPLIT objects scored: %d" % len(KEYS))


# ------------------------------------------------------------ the kernels
def seeds_of(row):
    """the FROZEN Phase A seed set: dirs, valley matrix, charge fractions."""
    pts, q, _ = L.pack(row['P'], row['segs'])
    if pts is None or len(pts) < 8:
        return None
    return L.angular_maxima(pts, q, row['v'], L.profile_sigma_fn(),
                            sep_scale=SEP_SCALE, max_seeds=MAX_SEEDS)


def accept_pair(M, vmax=V_ACCEPT, fmin=F_ACCEPT):
    """K0 -- the SHIPPED rule: the single best PAIR by minimum valley."""
    k = len(M['dirs'])
    if k < 2:
        return []
    V, fr = M['valley'], M['frac']
    best = None
    for i in range(k):
        for j in range(i + 1, k):
            if min(fr[i], fr[j]) < fmin:
                continue
            if best is None or V[i, j] < best[0]:
                best = (float(V[i, j]), i, j)
    if best is None or best[0] > vmax:
        return []
    return [best[1], best[2]]


def accept_greedy(M, vmax=V_ACCEPT, fmin=F_ACCEPT, nearest=False, cap=MAX_SEEDS):
    """K1 -- k from the seed count.  ATLAS/CMS/GARLIC shape (doc sec 10).

    Walk the maxima in DENSITY order (they already are).  The brightest is always
    a seed.  A later maximum joins iff it carries `fmin` of the charge AND a
    charge VALLEY at most `vmax` separates it from an already-accepted seed --
    i.e. it is a distinct core, not a bright patch inside one.  `nearest=True`
    tests the valley against the angularly CLOSEST accepted seed instead of the
    easiest one (the stricter reading of 'is there a dip between these two')."""
    k = len(M['dirs'])
    if k < 1:
        return []
    V, fr, S = M['valley'], M['frac'], M['dirs']
    acc = [0]
    for s in range(1, k):
        if len(acc) >= cap:
            break
        if fr[s] < fmin:
            continue
        if nearest == 'all':
            # STRICT reading: a new core must be separated from EVERY seed
            # already accepted, not just from the easiest one.  Same threshold,
            # no new parameter -- only the quantifier changes.
            ok = max(V[s, a] for a in acc) <= vmax
        elif nearest:
            a = min(acc, key=lambda t: -float(S[s] @ S[t]))
            ok = V[s, a] <= vmax
        else:
            ok = min(V[s, a] for a in acc) <= vmax
        if ok:
            acc.append(s)
    return acc if len(acc) >= 2 else []


def _unit_charge_share(row, b, D):
    """charge of bundle/segment `b` nearest to each accepted seed direction."""
    A = [row['P'][s] for s in b if s in row['P'] and len(row['P'][s])]
    if not A:
        return None
    A = np.concatenate(A, axis=0)
    U, _ = L.rays(A[:, :3], row['v'])
    w = L.qwt(A[:, 3])
    best = np.argmax(U @ D.T, axis=1)
    return np.array([w[best == j].sum() for j in range(len(D))])


def assign(row, M, acc, unit='bundle', mode='centroid', gap=4.0, snap=0.0):
    """{seg: part} -- assign each unit to one accepted seed.

    `unit`  bundle  : spatially connected component at `gap` cm (what propose()
                      uses, and the conservative choice -- a proposal that cuts
                      through a connected bundle is harder to fix by hand)
            segment : the finest thing the SPLITTER can move
                      (Shower::detach_member_set, PRShower.cxx:640-700)
            hier    : bundle, unless the bundle straddles -- see `snap`
    `mode`  centroid: the unit's charge-weighted centroid ray picks the seed.
                      This is what propose() does, and sec B3's measurement says
                      it is the thing that breaks: a minority lobe living INSIDE
                      a bundle that is mostly main body has no ray of its own, so
                      every bundle falls to seed 0 and prop_k collapses to 1.
            plurality: the seed holding the PLURALITY of the unit's own per-point
                      charge (doc pr/137 step 2 item 3, L.assign_segments).
    `snap`  hier only: a bundle whose winning seed holds at least this share of
                      its charge is assigned whole; below it, its segments are
                      assigned individually."""
    P, ms, segs = row['P'], row['ms'], row['segs']
    if not acc:
        return {s: 0 for s in segs}
    D = M['dirs'][acc]
    out = {s: 0 for s in segs}

    def place(b):
        if mode == 'plurality':
            sh = _unit_charge_share(row, b, D)
            if sh is None or sh.sum() <= 0:
                return None
            return int(np.argmax(sh)), float(sh.max() / sh.sum())
        bp, bq, _ = L.pack(P, b)
        if bp is None:
            return None
        c = L.qw_centroid(bp, bq) - row['v']
        n = np.linalg.norm(c)
        if n <= 0:
            return None
        return int(np.argmax(D @ (c / n))), 1.0

    if unit == 'segment':
        units = [[s] for s in sorted(segs)]
    else:
        units = SM.connected_bundles(P, segs, ms, gap=gap)
    for b in units:
        r = place(b)
        if r is None:
            continue
        g, dom = r
        if unit == 'hier' and dom < snap and len(b) > 1:
            for s in b:
                rs = place([s])
                if rs is not None:
                    out[s] = rs[0]
            continue
        for s in b:
            out[s] = g
    return out


def agreement(prop, own, qmap):
    """charge-weighted best-label-permutation agreement -- sec A5.6's metric,
    transcribed from pr138_scan_analysis.py N7 so the numbers are comparable."""
    segs = [s for s in prop if s in own]
    if not segs:
        return None
    pg = sorted({prop[s] for s in segs})
    og = sorted({own[s] for s in segs})
    Qt = sum(qmap.get(s, 0.0) for s in segs) or 1.0
    best = 0.0
    for perm in itertools.permutations(pg, min(len(pg), len(og))):
        m = dict(zip(perm, og))
        got = sum(qmap.get(s, 0.0) for s in segs if m.get(prop[s]) == own[s])
        best = max(best, got / Qt)
    return best


# ------------------------------------------------------------- run them
def accept_pair_then_grow(M, cap=MAX_SEEDS, vmax=V_ACCEPT, fmin=F_ACCEPT):
    """K7 -- the SHIPPED trigger, then grow.  This is the design that ships.

    The accept test is untouched: the best PAIR by minimum valley among the seeds
    carrying >= `fmin` of the charge, firing iff that valley is <= `vmax`.  So
    the fire list is byte-for-byte sec A5.4's -- the trigger does not move when
    `cap` moves, which is what makes `shower_split_max_parts` an orthogonal knob
    and keeps sec B2's "reproduce the offline fire list object for object"
    checkable at every setting.

    Only AFTER the object has fired does `cap` matter: further maxima join, in
    density order, on the same two numbers.  cap=2 is exactly the shipped
    kernel."""
    pair = accept_pair(M, vmax=vmax, fmin=fmin)
    if not pair or cap <= 2:
        return pair
    V, fr = M['valley'], M['frac']
    acc = list(pair)
    for s in range(len(M['dirs'])):
        if len(acc) >= cap:
            break
        if s in acc or fr[s] < fmin:
            continue
        if min(V[s, a] for a in acc) <= vmax:
            acc.append(s)
    return acc


def assign_k5(row, M, acc, gap=4.0, snap=0.80):
    """K5 -- the unit is chosen by k, and that is not a free parameter.

    Two accepted seeds: keep the shipped bundle/centroid rule.  Conditioned on
    the trigger firing it is essentially exact (median 1.000, mean 0.982, 25 of
    27 at >=0.90), so touching it can only cost.
    Three or more: a bundle CAN now straddle two of the parts, and a bundle
    assigned whole by its centroid ray sends the whole thing to one -- so a
    bundle whose winning seed holds < `snap` of its charge is subdivided at the
    segment level, which is the finest cut detach_member_set can make anyway."""
    if len(acc) <= 2:
        return assign(row, M, acc, unit='bundle', mode='centroid', gap=gap)
    return assign(row, M, acc, unit='hier', mode='plurality', gap=gap, snap=snap)


VARIANTS = [
    ("K0  PAIR   bundle/centroid  (shipped)", lambda M: accept_pair(M),   dict(unit='bundle',  mode='centroid')),
    ("K1  greedy bundle/centroid  (PRE-REG)", lambda M: accept_greedy(M), dict(unit='bundle',  mode='centroid')),
    ("K1n greedy bundle/centroid nearest-V ", lambda M: accept_greedy(M, nearest=True), dict(unit='bundle', mode='centroid')),
    ("K2  greedy bundle/PLURALITY          ", lambda M: accept_greedy(M), dict(unit='bundle',  mode='plurality')),
    ("K3  greedy segment/PLURALITY         ", lambda M: accept_greedy(M), dict(unit='segment', mode='plurality')),
    ("K4  greedy hier/PLURALITY snap=0.80  ", lambda M: accept_greedy(M), dict(unit='hier',    mode='plurality', snap=0.80)),
    ("K5  greedy k-conditional unit        ", lambda M: accept_greedy(M), 'K5'),
    ("SHIP2 cap=2 greedy/bundle/centroid  ", lambda M: accept_greedy(M, cap=2), dict(unit='bundle', mode='centroid')),
    ("SHIP3 cap=3 greedy/k-cond unit      ", lambda M: accept_greedy(M, cap=3), 'K5'),
    ("SHIP4 cap=4 greedy/k-cond unit      ", lambda M: accept_greedy(M, cap=4), 'K5'),
    ("K7c3 pair-then-grow k-cond cap=3     ", lambda M: accept_pair_then_grow(M, cap=3), 'K5'),
    ("K7c4 pair-then-grow k-cond cap=4     ", lambda M: accept_pair_then_grow(M, cap=4), 'K5'),
    ("K7b pair-then-grow, bundle/centroid  ", lambda M: accept_pair_then_grow(M, cap=4), dict(unit='bundle', mode='centroid')),
    ("K6  greedy-ALL k-conditional unit    ", lambda M: accept_greedy(M, nearest='all'), 'K5'),
    ("K6b greedy-ALL bundle/centroid       ", lambda M: accept_greedy(M, nearest='all'), dict(unit='bundle', mode='centroid')),
    ("K0p PAIR   bundle/PLURALITY          ", lambda M: accept_pair(M),   dict(unit='bundle',  mode='plurality')),
    ("K0h PAIR   hier/PLURALITY snap=0.80  ", lambda M: accept_pair(M),   dict(unit='hier',    mode='plurality', snap=0.80)),
]

rows = []          # per-object, per-variant
cache = {}
for k in KEYS:
    row = SM.load_object(k[0], k[1])
    if row is None:
        continue
    M = seeds_of(row)
    if M is None:
        continue
    own = {int(a): int(b) for a, b in (OWN[k].get('groups') or {}).items()}
    qmap = {s: max(row['ms'].get(s, 0.0), 0.0) for s in row['segs']}
    n_own = len(set(own.values()))
    rec = dict(event=k[0], node=k[1], verdict=OWN[k]['verdict'],
               n_own=n_own, n_seed=len(M['dirs']))
    for name, rule, kw in VARIANTS:
        acc = rule(M)
        prop = (assign_k5(row, M, acc) if kw == 'K5'
                else assign(row, M, acc, **kw))
        a = agreement(prop, own, qmap)
        rec[name] = a
        rec['k_' + name] = len(set(prop.values()))
        rec['f_' + name] = bool(acc)          # did the TRIGGER fire at all?
    rows.append(rec)
    cache[k] = (row, M, own, qmap)

print("scored: %d" % len(rows))


def report(rows, names, fired_only=False):
    if fired_only:
        print("\n### conditioned on the TRIGGER HAVING FIRED for that variant."
              "\n### sec A5.6 did not condition, so its k>=3 numbers mix two"
              "\n### different failures: a boundary the kernel drew badly, and a"
              "\n### boundary it never drew because the accept test said KEEP"
              "\n### (the sec A5.7 / sec B7 no-valley class).  Those are separate"
              "\n### problems and only the first one is stage B3's.")
    cls = collections.OrderedDict([('SPLIT2', []), ('SPLIT3', []), ('SPLIT4+', [])])
    print("\n%-38s %-9s %6s %6s %6s %6s" % ("variant", "class", "n", "med", "mean", ">=.90"))
    for name in names:
        for c in cls:
            v = sorted(r[name] for r in rows if r['verdict'] == c and r[name] is not None
                       and (r['f_' + name] or not fired_only))
            if not v:
                continue
            print("%-38s %-9s %6d %6.3f %6.3f %6d"
                  % (name if c == 'SPLIT2' else '', c, len(v),
                     v[len(v) // 2], sum(v) / len(v), sum(1 for x in v if x >= 0.90)))
        v3 = [r[name] for r in rows if r['verdict'] in ('SPLIT3', 'SPLIT4+')
              and r[name] is not None and (r['f_' + name] or not fired_only)]
        va = [r[name] for r in rows if r[name] is not None
              and (r['f_' + name] or not fired_only)]
        print("%-38s %-9s %6d %6s %6.3f" % ('', 'k>=3 all', len(v3), '',
                                            sum(v3) / max(len(v3), 1)))
        print("%-38s %-9s %6d %6s %6.3f" % ('', 'ALL', len(va), '',
                                            sum(va) / max(len(va), 1)))
        print()


print("\n########## AS SEC A5.6 SCORED IT -- every SPLIT object, fired or not ##########")
report(rows, [n for n, _, _ in VARIANTS])
print("\n########## THE KERNEL ALONE ##########")
report(rows, [n for n, _, _ in VARIANTS], fired_only=True)
print("=== how often did the trigger fire at all, per class? ===")
for name, _, _ in VARIANTS[:2]:
    for c in ('SPLIT2', 'SPLIT3', 'SPLIT4+'):
        ks = [r for r in rows if r['verdict'] == c]
        print("  %-38s %-8s fired %2d of %2d"
              % (name, c, sum(1 for r in ks if r['f_' + name]), len(ks)))

print("=== does the accepted seed count match the owner's part count? ===")
for name, _, _ in VARIANTS:
    ok = sum(1 for r in rows if r['k_' + name] == r['n_own'])
    lo = sum(1 for r in rows if r['k_' + name] < r['n_own'])
    hi = sum(1 for r in rows if r['k_' + name] > r['n_own'])
    print("  %-38s exact %2d  too few %2d  too many %2d" % (name, ok, lo, hi))

print("\n=== the six worst boundaries under the pre-registered rule ===")
nm = VARIANTS[1][0]
for r in sorted([r for r in rows if r[nm] is not None], key=lambda r: r[nm])[:8]:
    print("  evt%-8d node%-8d %-8s owner_k=%d n_seed=%d prop_k=%d  K0=%.3f -> K1=%.3f"
          % (r['event'], r['node'], r['verdict'], r['n_own'], r['n_seed'],
             r['k_' + nm], r[VARIANTS[0][0]], r[nm]))

if args.fitted:
    print("\n=== FITTED: acceptance-parameter scan.  NOT eligible to ship as-is ===")
    print("  (43 objects, 10 of them k>=3 -- any gain here needs its own holdout)")
    print("  %-8s %-8s %6s %6s %6s" % ("vmax", "fmin", "S2", "k>=3", "ALL"))
    for vmax in (0.60, 0.80, 0.90, 0.95, 0.99):
        for fmin in (0.02, 0.03, 0.05, 0.10):
            s2, k3, al = [], [], []
            for k, (row, M, own, qmap) in cache.items():
                acc = accept_greedy(M, vmax=vmax, fmin=fmin)
                a = agreement(assign(row, M, acc), own, qmap)
                if a is None:
                    continue
                al.append(a)
                (s2 if OWN[k]['verdict'] == 'SPLIT2' else k3).append(a)
            print("  %-8.2f %-8.2f %6.3f %6.3f %6.3f"
                  % (vmax, fmin, sum(s2) / max(len(s2), 1),
                     sum(k3) / max(len(k3), 1), sum(al) / max(len(al), 1)))

SHORT = {n: n.split()[0] for n, _, _ in VARIANTS}
with open(args.tsv, 'w') as f:
    w = csv.writer(f, delimiter='\t')
    f.write("# doc pr/138 B3 -- kernel with k from the seed count; "
            "seed finding FROZEN at Phase A (sep_scale=%.1f max_seeds=%d), "
            "acceptance reuses the trigger's valley<=%.2f frac>=%.2f\n"
            % (SEP_SCALE, MAX_SEEDS, V_ACCEPT, F_ACCEPT))
    for n, _, _ in VARIANTS:
        f.write("# %-6s %s\n" % (SHORT[n], n.strip()))
    w.writerow(['event', 'node', 'verdict', 'n_own', 'n_seed'] +
               [c for n, _, _ in VARIANTS for c in (SHORT[n], 'k_' + SHORT[n])])
    for r in rows:
        w.writerow([r['event'], r['node'], r['verdict'], r['n_own'], r['n_seed']] +
                   [x for n, _, _ in VARIANTS
                    for x in (('%.4f' % r[n]) if r.get(n) is not None else '',
                              r.get('k_' + n, ''))])
print("\nwrote %s" % args.tsv)
