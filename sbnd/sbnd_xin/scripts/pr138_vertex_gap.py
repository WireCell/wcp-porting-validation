#!/usr/bin/env python3
# doc pr/138 sec 5.1 -- is the vertex WRONG, or is this a photon with a real gap?
"""`vgap` alone cannot tell a bad nu vertex from a legitimate conversion gap.

sec 4 found every false fire on an object whose nu vertex sits 13-96 cm from its
own charge, and the population tape says 44.5% of ALL production EM candidates
are like that (median vgap 9.6 cm).  That number must NOT be read as "44.5% of
vertices are wrong": a photon converts a mean 9/7 X0 ~ 18 cm from its origin, so
a large `vgap` is exactly what a real gamma looks like.  That conflation is also
why `vgap` scored AUC 0.499 as a trigger feature (sec A5.3) and why a veto on it
costs 17 of 33 correct cuts (sec 4.3).

THE DISCRIMINATOR THIS SCRIPT TESTS.  A genuine conversion gap is EMPTY -- the
photon leaves no charge between the vertex and its conversion point.  A
mis-placed vertex is not: it sits beside or inside other activity, so the ray
from it to the candidate passes THROUGH charge that belongs to something else.

  void_frac  = fraction of sample points along the vertex -> candidate-centroid
               ray with NO charge (from ANY segment in the event) within `rad`.
               near 1 = a clean empty gap = a photon.  low = the vertex is
               embedded in activity it is not attached to.
  occ_other  = the same, counting only charge NOT belonging to the candidate.

Both are computed from the arm's own dump; no label is used to define them.

Repro:
    python3 scripts/pr138_vertex_gap.py
"""
import os, sys, json, glob, csv, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'split_display'))
import numpy as np
import pr137_lib as L
import split_model as SM

RAD, NSTEP = 5.0, 40
SPL = ('SPLIT2', 'SPLIT3', 'SPLIT4+')
TRACKS = {(99838, 14004), (389538, 19021), (292524, 9018), (176502, 109141),
          (286681, 72040), (122660, 54071), (415278, 23047), (278420, 18002)}


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


OWN = load_labels('splitscan-0901-owner')
CMP = {}
for r in csv.DictReader((l for l in open('docs/pr/pr138-probe-compare.tsv')
                         if not l.startswith('#')), delimiter='\t'):
    CMP[(int(r['event']), int(r['node']))] = r

from scipy.spatial import cKDTree
_EV = {}


def event_cloud(ev):
    if ev not in _EV:
        d = L.dump(ev, 'onV1c90')
        P = L.seg_pts(d) if d else {}
        allp = np.concatenate([A[:, :3] for A in P.values()]) if P else np.zeros((0, 3))
        _EV.clear()
        _EV[ev] = (P, cKDTree(allp) if len(allp) else None)
    return _EV[ev]


rows = []
for k in sorted(CMP):
    if k in TRACKS:
        continue
    row = SM.load_object(k[0], k[1])
    if row is None:
        continue
    P, tree = event_cloud(k[0])
    pts, q, _ = L.pack(row['P'], row['segs'])
    if pts is None or tree is None:
        continue
    v = np.asarray(row['v'], float)
    c = L.qw_centroid(pts, q)
    ts = np.linspace(0.0, 1.0, NSTEP)[:, None]
    ray = v[None, :] * (1 - ts) + c[None, :] * ts
    # own-charge tree, to separate "any charge" from "someone else's charge"
    own = cKDTree(pts)
    d_all, _ = tree.query(ray, k=1)
    d_own, _ = own.query(ray, k=1)
    # sample points strictly between the vertex and the object body
    mid = (ts[:, 0] > 0.05) & (ts[:, 0] < 0.80)
    void_all = float((d_all[mid] > RAD).mean())
    void_other = float(((d_all[mid] > RAD) | (d_own[mid] <= RAD)).mean())
    rows.append(dict(event=k[0], node=k[1], verdict=CMP[k]['owner'],
                     pos=CMP[k]['owner'] in SPL, fired=int(CMP[k]['cxx_fired']),
                     vgap=float(CMP[k]['vgap_cm']),
                     void=void_all, occ_other=void_other,
                     comment=(OWN.get(k, {}).get('comment') or '')))

print("doc pr/138 sec 5.1 -- separating a real conversion gap from a wrong vertex")
print("scored objects: %d   (ray sampled at %d points, charge radius %.0f cm)"
      % (len(rows), NSTEP, RAD))

tp = [r for r in rows if r['fired'] and r['pos']]
fp = [r for r in rows if r['fired'] and not r['pos']]
print("\n=== the fires, split by whether the owner agreed ===")
print("%-26s %4s %8s %8s %8s" % ('class', 'n', 'vgap med', 'void med', 'occ_other med'))
for nm, sel in (('CORRECT cuts', tp), ('FALSE fires', fp)):
    if not sel:
        continue
    print("%-26s %4d %8.1f %8.3f %8.3f"
          % (nm, len(sel), np.median([r['vgap'] for r in sel]),
             np.median([r['void'] for r in sel]), np.median([r['occ_other'] for r in sel])))

print("\n=== the 8 false fires, one line each ===")
print("%-9s %-9s %8s %7s %9s  %s" % ('event', 'node', 'vgap', 'void', 'occ_other', 'owner comment'))
for r in sorted(fp, key=lambda r: r['void']):
    print("%-9d %-9d %8.1f %7.3f %9.3f  %s"
          % (r['event'], r['node'], r['vgap'], r['void'], r['occ_other'], r['comment'][:44]))

print("\n=== the correct cuts with a LARGE vgap -- the photons a naive veto would kill ===")
big = sorted([r for r in tp if r['vgap'] > 13], key=lambda r: -r['void'])
print("  %d of %d correct cuts have vgap > 13 cm" % (len(big), len(tp)))
for r in big[:12]:
    print("%-9d %-9d %8.1f %7.3f %9.3f" % (r['event'], r['node'], r['vgap'], r['void'], r['occ_other']))


def auc(pos, neg):
    if not pos or not neg:
        return float('nan')
    n = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return n / (len(pos) * len(neg))


print("\n=== does void_frac separate the false fires from the correct ones? ===")
for nm, key in (('vgap', 'vgap'), ('void_frac', 'void'), ('occ_other', 'occ_other')):
    a = auc([r[key] for r in tp], [r[key] for r in fp])
    print("  %-12s AUC(correct > false) = %.3f" % (nm, a))
print("  (0.5 = no separation.  A HIGH void_frac on the correct cuts and a LOW one")
print("   on the false fires would mean: real gammas leave an EMPTY gap, wrong")
print("   vertices sit inside other activity.)")

with open('docs/pr/pr138-vertex-gap.tsv', 'w') as f:
    w = csv.writer(f, delimiter='\t')
    f.write("# doc pr/138 sec 5.1 -- conversion gap vs wrong vertex, rad=%.0f cm\n" % RAD)
    w.writerow(['event', 'node', 'owner', 'positive', 'fired', 'vgap_cm', 'void_frac', 'occ_other'])
    for r in rows:
        w.writerow([r['event'], r['node'], r['verdict'], int(r['pos']), r['fired'],
                    '%.2f' % r['vgap'], '%.4f' % r['void'], '%.4f' % r['occ_other']])
print("\nwrote docs/pr/pr138-vertex-gap.tsv")
