#!/usr/bin/env python3
# doc pr/141 item 1 -- RECURSION as the third-boundary fix.  READ-ONLY.
"""Does cutting a PART only when that part itself has an accepted seed pair
place the third boundary better than asking for three seeds at once?

FORK of scripts/pr138_kernel_k.py (M10; that script keeps producing doc pr/138's
numbers and is untouched).  What the fork adds:

  * the RECURSIVE kernel.  The shipped kernel calls `angular_maxima` ONCE on the
    whole object and then picks seeds out of that one map; every k>=3 variant
    doc pr/138 tried is a different way of picking more seeds from the SAME map.
    doc pr/139 sec 22.4 measured the result: boundary agreement mean 0.739 and
    0 of 2 exact on k>=3, against median 1.000 on SPLIT2.  The hypothesis
    (sec 26.5 item 1) is that the map itself is the problem -- a third core is
    invisible in an angular density built around the whole object's centroid --
    so the fix is to re-run seed finding ON EACH PART and cut that part only if
    it has an accepted pair of its own.
  * the owner-label pool is all three owner split tags (0901-owner, 0902-pi0,
    0903-wide), later scans winning on conflict, so every k>=3 label the
    campaign holds is in the denominator instead of one tag's share.

NO NEW PARAMETER.  The recursion reuses the SAME two numbers the shipped
trigger uses at the top level -- valley <= 0.95 and charge fraction >= 0.03 --
and the same frozen seed-finding constants (profile_sigma_fn, sep_scale 1.6,
max_seeds 4).  A win here is therefore not a fit.  The only new quantity is the
recursion DEPTH, which is capped at 1 (so at most 3 parts) and reported.

Repro:
    python3 scripts/pr141_kernel_recurse.py --tsv docs/pr/pr141-kernel-recurse.tsv
"""
import os, sys, json, glob, csv, argparse, collections, itertools
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'split_display'))
import numpy as np
import pr137_lib as L
import split_model as SM

ap = argparse.ArgumentParser()
ap.add_argument('--owner-tags', default='splitscan-0901-owner,splitscan-0902-pi0,splitscan-0903-wide')
ap.add_argument('--set', default='docs/pr/pr138-scan-analysis.tsv')
ap.add_argument('--tsv', default='docs/pr/pr141-kernel-recurse.tsv')
args = ap.parse_args()

SPLITS = ('SPLIT2', 'SPLIT3', 'SPLIT4+')
SEP_SCALE, MAX_SEEDS = 1.6, 4
V_ACCEPT, F_ACCEPT = 0.95, 0.03
TRACKS = {(99838, 14004), (389538, 19021), (292524, 9018), (176502, 109141),
          (286681, 72040), (122660, 54071), (415278, 23047), (278420, 18002)}


def load_labels(tags):
    """later tags win on conflict -- the same precedence sec 22 used."""
    out = {}
    for tag in tags:
        for f in sorted(glob.glob('em_labels/%s/labels-evt*.json' % tag)):
            j = json.load(open(f))
            try:
                ev = int(str(j.get('event', '')).replace('evt', ''))
            except Exception:
                continue
            for nd, r in (j.get('split_labels') or {}).items():
                out[(ev, int(nd))] = r
    return out


OWN = load_labels(args.owner_tags.split(','))
FEAT = {}
for r in csv.DictReader((l for l in open(args.set) if not l.startswith('#')),
                        delimiter='\t'):
    FEAT[(int(r['event']), int(r['node']))] = r
KEYS = sorted(k for k in FEAT if k in OWN and OWN[k].get('verdict') in SPLITS
              and k not in TRACKS)
print("doc pr/141 item 1 -- recursion as the third-boundary fix")
print("owner tags: %s" % args.owner_tags)
print("owner-SPLIT objects scored: %d  (k>=3: %d)"
      % (len(KEYS), sum(1 for k in KEYS if OWN[k]['verdict'] != 'SPLIT2')))


def maxima(pts, q, v, max_seeds=MAX_SEEDS):
    if pts is None or len(pts) < 8:
        return None
    return L.angular_maxima(pts, q, v, L.profile_sigma_fn(),
                            sep_scale=SEP_SCALE, max_seeds=max_seeds)


def seeds_of(row, segset=None):
    """the FROZEN seed map, over the whole object or over one PART of it."""
    segs = sorted(row['segs']) if segset is None else sorted(segset)
    pts, q, _ = L.pack(row['P'], segs)
    return maxima(pts, q, row['v'])


def accept_pair(M, vmax=V_ACCEPT, fmin=F_ACCEPT):
    """the SHIPPED accept rule: the single best PAIR by minimum valley."""
    if M is None:
        return []
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


def _share(row, b, D):
    A = [row['P'][s] for s in b if s in row['P'] and len(row['P'][s])]
    if not A:
        return None
    A = np.concatenate(A, axis=0)
    U, _ = L.rays(A[:, :3], row['v'])
    w = L.qwt(A[:, 3])
    best = np.argmax(U @ D.T, axis=1)
    return np.array([w[best == j].sum() for j in range(len(D))])


def assign(row, M, acc, segset=None, unit='bundle', mode='centroid',
           gap=4.0, snap=0.0):
    """{seg: part} over `segset` (default: the whole object)."""
    P, ms = row['P'], row['ms']
    segs = sorted(row['segs']) if segset is None else sorted(segset)
    if not acc:
        return {s: 0 for s in segs}
    D = M['dirs'][acc]
    out = {s: 0 for s in segs}

    def place(b):
        if mode == 'plurality':
            sh = _share(row, b, D)
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

    units = ([[s] for s in segs] if unit == 'segment'
             else SM.connected_bundles(P, segs, ms, gap=gap))
    for b in units:
        b = [s for s in b if s in out]
        if not b:
            continue
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


def k5_unit(nacc):
    return (dict(unit='bundle', mode='centroid') if nacc <= 2
            else dict(unit='hier', mode='plurality', snap=0.80))


def kernel_shipped(row, cap=2):
    """the shipped kernel: one seed map, the best pair, bundle/centroid."""
    M = seeds_of(row)
    acc = accept_pair(M)
    if not acc:
        return {s: 0 for s in row['segs']}, 1
    return assign(row, M, acc, **k5_unit(len(acc))), 2


def kernel_recurse(row, depth=1, min_segs=2, min_pts=8):
    """RECURSION.  Cut the object; then cut a PART only if that part, seeded on
    its OWN points, has an accepted pair of its own."""
    M = seeds_of(row)
    acc = accept_pair(M)
    if not acc:
        return {s: 0 for s in row['segs']}, 1, 0
    base = assign(row, M, acc, **k5_unit(len(acc)))
    parts = collections.defaultdict(list)
    for s, g in base.items():
        parts[g].append(s)
    out, nxt, nrec = dict(base), max(parts) + 1, 0
    if depth <= 0:
        return out, len(parts), 0
    for g in sorted(parts):
        sub = parts[g]
        if len(sub) < min_segs:
            continue
        pts, q, _ = L.pack(row['P'], sub)
        if pts is None or len(pts) < min_pts:
            continue
        M2 = maxima(pts, q, row['v'])
        acc2 = accept_pair(M2)
        if not acc2:
            continue                     # this part is one core -- leave it
        sub_assign = assign(row, M2, acc2, segset=sub, **k5_unit(len(acc2)))
        # part 0 of the sub-split keeps the parent label; part 1 gets a new one
        moved = False
        for s, gg in sub_assign.items():
            if gg != 0:
                out[s] = nxt
                moved = True
        if moved:
            nxt += 1
            nrec += 1
    return out, len(set(out.values())), nrec


def agreement(prop, own, qmap):
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


rows = []
for k in KEYS:
    row = SM.load_object(k[0], k[1])
    if row is None:
        continue
    own = {int(a): int(b) for a, b in (OWN[k].get('groups') or {}).items()}
    if not own:
        continue
    qmap = {s: max(row['ms'].get(s, 0.0), 0.0) for s in row['segs']}
    n_own = len(set(own.values()))
    try:
        p_ship, k_ship = kernel_shipped(row)
        p_rec, k_rec, n_rec = kernel_recurse(row)
    except Exception as e:                       # noqa: BLE001
        print("  evt %d node %d: %s" % (k[0], k[1], e))
        continue
    rows.append(dict(event=k[0], node=k[1], verdict=OWN[k]['verdict'],
                     n_own=n_own, k_ship=k_ship, k_rec=k_rec, n_rec=n_rec,
                     a_ship=agreement(p_ship, own, qmap),
                     a_rec=agreement(p_rec, own, qmap)))

print("\n=== boundary agreement, shipped kernel vs recursion ===")
print("%-10s %4s   %-28s %-28s" % ("class", "n", "SHIPPED (one seed map)",
                                   "RECURSION (re-seed each part)"))
for cls in ("SPLIT2", "SPLIT3", "SPLIT4+", "ALL"):
    sel = [r for r in rows if cls == "ALL" or r["verdict"] == cls]
    sel = [r for r in sel if r["a_ship"] is not None and r["a_rec"] is not None]
    if not sel:
        continue
    def stat(key):
        v = sorted(r[key] for r in sel)
        med = v[len(v) // 2] if len(v) % 2 else 0.5 * (v[len(v) // 2 - 1] + v[len(v) // 2])
        return "med %.3f  mean %.3f  exact %d/%d" % (
            med, sum(v) / len(v), sum(1 for x in v if x > 0.999), len(v))
    print("%-10s %4d   %-28s %-28s" % (cls, len(sel), stat("a_ship"), stat("a_rec")))

print("\n=== per-object, k>=3 only (the 6+ the recursion was built for) ===")
print("%-9s %-7s %-8s %5s %5s %5s   %7s %7s" %
      ("event", "node", "verdict", "n_own", "kship", "krec", "a_ship", "a_rec"))
for r in sorted(rows, key=lambda x: (x["verdict"] == "SPLIT2", -(x["a_rec"] or 0))):
    if r["verdict"] == "SPLIT2":
        continue
    print("%-9d %-7d %-8s %5d %5d %5d   %7.3f %7.3f"
          % (r["event"], r["node"], r["verdict"], r["n_own"], r["k_ship"],
             r["k_rec"], r["a_ship"] or 0, r["a_rec"] or 0))

reg = [r for r in rows if r["verdict"] == "SPLIT2" and r["k_rec"] > 2]
print("\nSPLIT2 objects the recursion over-cuts (regression risk): %d of %d"
      % (len(reg), sum(1 for r in rows if r["verdict"] == "SPLIT2")))
for r in reg:
    print("   evt %d node %d: k %d -> %d, agreement %.3f -> %.3f"
          % (r["event"], r["node"], r["k_ship"], r["k_rec"],
             r["a_ship"] or 0, r["a_rec"] or 0))

if args.tsv:
    hdr = ["event", "node", "verdict", "n_own", "k_ship", "k_rec", "n_rec",
           "a_ship", "a_rec"]
    with open(args.tsv, "w") as fh:
        fh.write("\t".join(hdr) + "\n")
        for r in rows:
            fh.write("\t".join(
                ("%.4f" % r[h]) if isinstance(r[h], float) else str(r[h])
                for h in hdr) + "\n")
    print("\nwrote %s (%d rows)" % (args.tsv, len(rows)))
