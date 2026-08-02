#!/usr/bin/env python3
"""Gate B0-3b: classify the knob-band vertices B0 did NOT remove.

B0-3 proves the firings are correct and collateral-free.  It does not prove
that every cathode-band vertex *should* have gone.  This script classifies the
survivors by the degree of the vertex in the fitted segment graph:

  degree 1  -- a track END that happens to sit near the cathode: out of scope
               by construction (no kink to veto).
  degree>=3 -- a real junction (a delta ray, a real vertex): out of scope.
  degree 2  -- an in-line break at |x| < XKNOB.  These are the interesting
               ones: `segment_search_kink` is not the only breaker
               (`examine_structure_3` / NeutrinoOtherSegments also split), so a
               degree-2 survivor is either a different breaker -- the deferred
               B1 scope -- or a hole in the veto.  Each is printed with the
               kink angle between its two incident segments so it can be read.

Read-only.  Repro:
  python3 pr20_b03_survivors.py work-b0pr300-on [work-b0nue48-on ...]
"""
import os, sys, json, glob, zipfile
import numpy as np

XKNOB = 5.0     # cm: the knob's band
JOINTOL = 0.5   # cm: endpoint-to-vertex match (stub_census.py)
RAD = 10.0      # cm: direction-averaging radius (examine_structure_3 style)


def load(path):
    z = zipfile.ZipFile(path)
    n = z.namelist()
    if 'data/0/0-vertices-global.json' not in n:
        return None, None
    dv = json.loads(z.read('data/0/0-vertices-global.json'))
    V = np.stack([np.asarray(dv['x'], float), np.asarray(dv['y'], float),
                  np.asarray(dv['z'], float)], 1)
    dt = json.loads(z.read('data/0/0-track_fit-global.json'))
    P = np.stack([np.asarray(dt['x'], float), np.asarray(dt['y'], float),
                  np.asarray(dt['z'], float)], 1)
    sid = np.asarray(dt['real_cluster_id'])
    S = {int(s): P[sid == s] for s in set(sid.tolist()) if s != -1}
    return V, S


def caldir(Q, p, r=RAD):
    m = np.linalg.norm(Q - p, axis=1) < r
    if m.sum() == 0:
        return None
    v = Q[m].mean(0) - p
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def main(*arms):
    tot = {1: 0, 2: 0, 3: 0}
    deg2 = []
    for arm in arms:
        for p in sorted(glob.glob(os.path.join(arm, 'pr_evt*', 'mabc-pr.zip'))):
            evt = os.path.basename(os.path.dirname(p)).replace('pr_evt', '')
            V, S = load(p)
            if V is None:
                continue
            for v in V:
                if abs(v[0]) >= XKNOB:
                    continue
                inc = [s for s, Q in S.items() if len(Q) >= 2 and
                       min(np.linalg.norm(Q[0] - v), np.linalg.norm(Q[-1] - v)) < JOINTOL]
                d = min(3, max(1, len(inc)))
                tot[d] += 1
                if len(inc) == 2:
                    a, b = S[inc[0]], S[inc[1]]
                    da, db = caldir(a, v), caldir(b, v)
                    ang = None
                    if da is not None and db is not None:
                        ang = round(180 - np.degrees(np.arccos(
                            float(np.clip(np.dot(da, db), -1, 1)))), 2)
                    La = round(float(np.linalg.norm(a[0] - a[-1])), 2)
                    Lb = round(float(np.linalg.norm(b[0] - b[-1])), 2)
                    deg2.append((arm, evt, tuple(np.round(v, 2)), inc, (La, Lb), ang))
    print(f'knob-band vertices (|x|<{XKNOB}cm) surviving in {arms}:')
    print(f'  degree 1 (track end, out of scope):        {tot[1]}')
    print(f'  degree >=3 (real junction, out of scope):  {tot[3]}')
    print(f'  degree 2 (in-line break, IN scope):        {tot[2]}')
    for r in deg2:
        print(f'    {os.path.basename(r[0])} evt {r[1]:>7} at x={r[2][0]:6.2f} '
              f'segs {r[3]} L={r[4]} kink{int(RAD)}={r[5]} deg')


if __name__ == '__main__':
    main(*sys.argv[1:])
