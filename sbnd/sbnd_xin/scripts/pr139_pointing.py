#!/usr/bin/env python3
"""Does the object's OWN axis aim back at the nu vertex?  (pr/129's discriminator)

doc pr/138 sec 4.2b killed `void_frac`: the false fires have the CLEANEST gaps,
so nothing measured ALONG the vertex ray separates them.  This tests a quantity
that is not measured from the vertex at all -- the object's own principal axis --
and asks how far that axis MISSES the vertex (the impact parameter b).

  a real gamma with a 20 cm conversion gap still POINTS BACK at the vertex: b small
  an object seen from a wrong vertex does not:                              b large

b is perpendicular; vgap is along the ray.  They are different questions and
sec 4.2b only answered the second.
"""
import os, sys, csv, glob, json
sys.path[:0] = ['scripts', 'split_display']
import numpy as np
import pr137_lib as L
import split_model as SM

SPL = ('SPLIT2', 'SPLIT3', 'SPLIT4+')
TRACKS = {(99838, 14004), (389538, 19021), (292524, 9018), (176502, 109141),
          (286681, 72040), (122660, 54071), (415278, 23047), (278420, 18002)}
CMP = {(int(r['event']), int(r['node'])): r
       for r in csv.DictReader((l for l in open('docs/pr/pr138-probe-compare.tsv')
                                if not l.startswith('#')), delimiter='\t')}
rows = []
for k in sorted(CMP):
    if k in TRACKS:
        continue
    row = SM.load_object(k[0], k[1])
    if row is None:
        continue
    pts, q, _ = L.pack(row['P'], row['segs'])
    if pts is None or len(pts) < 8:
        continue
    v = np.asarray(row['v'], float)
    w = L.qwt(q)
    c = (pts * w[:, None]).sum(0) / w.sum()
    X = (pts - c) * np.sqrt(w)[:, None]
    ax = np.linalg.svd(X, full_matrices=False)[2][0]
    ax = ax / np.linalg.norm(ax)
    d = c - v
    b = float(np.linalg.norm(d - np.dot(d, ax) * ax))          # impact parameter
    r = float(np.linalg.norm(d))
    # the same thing as an ANGLE, which is scale free
    ang = float(np.degrees(np.arcsin(min(b / max(r, 1e-9), 1.0))))
    rows.append(dict(event=k[0], node=k[1], owner=CMP[k]['owner'],
                     pos=CMP[k]['owner'] in SPL, fired=int(CMP[k]['cxx_fired']),
                     vgap=float(CMP[k]['vgap_cm']), b=b, r=r, ang=ang))

tp = [x for x in rows if x['fired'] and x['pos']]
fp = [x for x in rows if x['fired'] and not x['pos']]
print("objects %d   correct cuts %d   false fires %d" % (len(rows), len(tp), len(fp)))


def auc(a, b_):
    return sum((p > q) + 0.5 * (p == q) for p in a for q in b_) / (len(a) * len(b_))


print("\n%-14s %10s %10s %8s" % ('feature', 'median TP', 'median FP', 'AUC(FP>TP)'))
for nm, key in (('vgap [cm]', 'vgap'), ('impact b [cm]', 'b'), ('miss angle [deg]', 'ang')):
    print("%-14s %10.2f %10.2f %8.3f"
          % (nm, np.median([x[key] for x in tp]), np.median([x[key] for x in fp]),
             auc([x[key] for x in fp], [x[key] for x in tp])))

print("\nthe 8 false fires:")
print("%-9s %-9s %8s %8s %8s" % ('event', 'node', 'vgap', 'b', 'miss deg'))
for x in sorted(fp, key=lambda x: -x['b']):
    print("%-9d %-9d %8.1f %8.1f %8.1f" % (x['event'], x['node'], x['vgap'], x['b'], x['ang']))
print("\ncorrect cuts, worst 12 by b:")
for x in sorted(tp, key=lambda x: -x['b'])[:12]:
    print("%-9d %-9d %8.1f %8.1f %8.1f" % (x['event'], x['node'], x['vgap'], x['b'], x['ang']))

print("\n=== what a veto on b would cost, against sec 4.3's vgap dial ===")
print("%-10s %6s %6s %6s %7s %7s" % ('b <=', 'fires', 'right', 'wrong', 'eff', 'pur'))
pos = [x for x in rows if x['pos']]
for t in (5, 10, 15, 20, 30, 50, 1e9):
    ff = [x for x in rows if x['fired'] and x['b'] <= t]
    rr = [x for x in ff if x['pos']]
    print("%-10s %6d %6d %6d %7.3f %7.3f"
          % (('%g' % t) if t < 1e8 else 'none', len(ff), len(rr), len(ff) - len(rr),
             len(rr) / len(pos), len(rr) / max(len(ff), 1)))
with open('docs/pr/pr139-pointing.tsv', 'w') as f:
    w_ = csv.writer(f, delimiter='\t')
    w_.writerow(['event', 'node', 'owner', 'positive', 'fired', 'vgap_cm', 'b_cm', 'miss_deg'])
    for x in rows:
        w_.writerow([x['event'], x['node'], x['owner'], int(x['pos']), x['fired'],
                     '%.2f' % x['vgap'], '%.2f' % x['b'], '%.2f' % x['ang']])
