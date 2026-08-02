#!/usr/bin/env python3
"""Re-evaluate segment_search_kink's criteria at the cathode, offline.

The PR job has already broken evt 169824's trajectory into 18005 / 18007 /
18008, so the segment the kink finder actually saw no longer exists.  We
reconstruct it by concatenating the three segments' fit points in order and
running the exact arithmetic of
`PRSegmentFunctions.cxx:191-352` (segment_search_kink) over it, to see whether
the cathode step clears the four kink criteria.

This is a proxy: the pre-break fit is not bit-identical to the concatenation of
the post-break fits.  It answers "is the cathode step the shape this finder
calls a kink", not "this exact index fired".
"""
import json, zipfile, sys
import numpy as np

PR = ('/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin/'
      'work-cathdbg1pr/pr_evt169824/mabc-pr.zip')
ORDER = [18005, 18007, 18008]          # +x half, cathode bridge, -x half

z = zipfile.ZipFile(PR)
d = json.loads(z.read('data/0/0-track_fit-global.json'))
P = np.stack([np.asarray(d['x'], float), np.asarray(d['y'], float),
              np.asarray(d['z'], float)], 1)
sid = np.asarray(d['real_cluster_id'])

pts = []
marks = {}
for s in ORDER:
    Q = P[sid == s]
    # orient each piece so the concatenation runs +x -> -x
    if Q[0][0] < Q[-1][0]:
        Q = Q[::-1]
    marks[s] = (len(pts), len(pts) + len(Q) - 1)
    pts.extend(Q.tolist())
F = np.asarray(pts)
n = len(F)
print(f'concatenated trajectory: {n} fit points, x from {F[0][0]:.2f} to {F[-1][0]:.2f}')
print('  segment index ranges:', marks)

drift = np.array([1.0, 0.0, 0.0])
refl = np.zeros(n)
para = np.zeros(n)
for i in range(n):
    a1 = a2 = None
    for j in range(6):
        off = (j + 1) * 2
        v10 = F[i] - (F[i - off] if i >= off else F[0])
        v20 = (F[i + off] if i + off < n else F[-1]) - F[i]
        m1, m2 = np.linalg.norm(v10), np.linalg.norm(v20)
        if j == 0:
            a1 = (np.degrees(np.arccos(np.clip(np.dot(v10, v20) / (m1 * m2), -1, 1)))
                  if m1 * m2 > 0 else 0.0)
            d1 = np.degrees(np.arccos(np.clip(np.dot(v10, drift) / m1, -1, 1))) if m1 > 0 else 90.
            d2 = np.degrees(np.arccos(np.clip(np.dot(v20, drift) / m2, -1, 1))) if m2 > 0 else 90.
            a2 = max(abs(d1 - 90), abs(d2 - 90))
        elif m1 > 0 and m2 > 0:
            a1 = min(a1, np.degrees(np.arccos(np.clip(np.dot(v10, v20) / (m1 * m2), -1, 1))))
            d1 = np.degrees(np.arccos(np.clip(np.dot(v10, drift) / m1, -1, 1)))
            d2 = np.degrees(np.arccos(np.clip(np.dot(v20, drift) / m2, -1, 1)))
            a2 = min(a2, max(abs(d1 - 90), abs(d2 - 90)))
    refl[i], para[i] = a1, a2


def sums(i):
    s, ns, s1, ns1 = 0., 0, 0., 0
    for j in range(-2, 3):
        k = i + j
        if 0 <= k < n:
            if para[k] > 10:
                s += refl[k] ** 2; ns += 1
            if para[k] > 7.5:
                s1 += refl[k] ** 2; ns1 += 1
    return (np.sqrt(s / ns) if ns else 0.), (np.sqrt(s1 / ns1) if ns1 else 0.)


print(f'\n{"i":>4} {"x":>7} {"refl":>7} {"para":>7} {"sum":>7} {"sum1":>7}  criteria met')
lo, hi = marks[18007]
for i in range(max(0, lo - 4), min(n, hi + 5)):
    sa, sa1 = sums(i)
    c = []
    if para[i] > 10 and refl[i] > 30 and sa > 15:   c.append('C1')
    if para[i] > 7.5 and refl[i] > 45 and sa1 > 25: c.append('C2')
    if para[i] > 15 and refl[i] > 27 and sa > 12.5: c.append('C3')
    tag = ' <- bridge' if lo <= i <= hi else ''
    print(f'{i:>4} {F[i][0]:7.2f} {refl[i]:7.1f} {para[i]:7.1f} {sa:7.1f} {sa1:7.1f}  '
          f'{",".join(c) if c else "-":<10}{tag}')

fired = [i for i in range(n)
         if (para[i] > 10 and refl[i] > 30 and sums(i)[0] > 15)
         or (para[i] > 7.5 and refl[i] > 45 and sums(i)[1] > 25)
         or (para[i] > 15 and refl[i] > 27 and sums(i)[0] > 12.5)]
print(f'\nindices meeting a kink criterion anywhere on the trajectory: {fired}')
print(f'  ... of which inside the cathode bridge [{lo},{hi}]: '
      f'{[i for i in fired if lo <= i <= hi]}')
print(f'  ... x of those: {[round(float(F[i][0]),2) for i in fired]}')
