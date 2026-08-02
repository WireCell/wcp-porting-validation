#!/usr/bin/env python3
"""How often would B1 (cathode-aware stub absorption) actually fire?

B1 as specified in doc pr/20 Part II absorbs a SEGMENT: a short
cathode-straddling stub whose two graph vertices both have degree 2, when the
two NEIGHBOURS (skipping the stub) are collinear to within `cathode_stub_angle`
measured at radius `cathode_stub_radius`.

Since B0 removes the kink-built stubs, this counts the trigger population on an
arm with B0 already ON as well as OFF, and -- because the two are different
shapes -- also counts the VERTEX variant: a single degree-2 vertex in the
cathode band whose two incident segments are collinear.  That is the shape evt
63603 has, and B1-as-specified does not match it.

Read-only.  Repro:
  python3 pr20_b1_population.py work-b0pr300-off work-b0pr300-on \
                                work-b0nue48-off work-b0nue48-on
"""
import os, sys, json, glob, zipfile
import numpy as np

MAXLEN = 8.0     # cathode_stub_max_len, proposed SBND
XCUT   = 4.0     # cathode_stub_xcut,    proposed SBND
RAD    = 15.0    # cathode_stub_radius,  proposed SBND
ANGLE  = 15.0    # cathode_stub_angle,   proposed SBND
JOINTOL = 0.5


def load(path):
    z = zipfile.ZipFile(path)
    if 'data/0/0-track_fit-global.json' not in z.namelist():
        return None, None
    dt = json.loads(z.read('data/0/0-track_fit-global.json'))
    P = np.stack([np.asarray(dt['x'], float), np.asarray(dt['y'], float),
                  np.asarray(dt['z'], float)], 1)
    sid = np.asarray(dt['real_cluster_id'])
    S = {int(s): P[sid == s] for s in set(sid.tolist()) if s != -1}
    dv = json.loads(z.read('data/0/0-vertices-global.json'))
    V = np.stack([np.asarray(dv['x'], float), np.asarray(dv['y'], float),
                  np.asarray(dv['z'], float)], 1)
    return S, V


def caldir(Q, p, r):
    m = np.linalg.norm(Q - p, axis=1) < r
    if m.sum() == 0:
        return None
    v = Q[m].mean(0) - p
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def ang(Qa, pa, Qb, pb, r):
    da, db = caldir(Qa, pa, r), caldir(Qb, pb, r)
    if da is None or db is None:
        return None
    return 180 - np.degrees(np.arccos(float(np.clip(np.dot(da, db), -1, 1))))


def incident(S, v, skip=None):
    return [s for s, Q in S.items() if s != skip and len(Q) >= 2 and
            min(np.linalg.norm(Q[0] - v), np.linalg.norm(Q[-1] - v)) < JOINTOL]


def scan(arm):
    n_ev = n_graph = 0
    stub_hits, vtx_hits = [], []
    for p in sorted(glob.glob(os.path.join(arm, 'pr_evt*', 'mabc-pr.zip'))):
        evt = os.path.basename(os.path.dirname(p)).replace('pr_evt', '')
        n_ev += 1
        S, V = load(p)
        if S is None:
            continue
        n_graph += 1
        # --- B1 as specified: a short cathode-straddling STUB segment
        for s, Q in S.items():
            if len(Q) < 2 or not (Q[:, 0].min() < 0 < Q[:, 0].max()):
                continue
            L = float(np.linalg.norm(Q[0] - Q[-1]))
            if L > MAXLEN or abs(Q[0][0]) > XCUT or abs(Q[-1][0]) > XCUT:
                continue
            n0, n1 = incident(S, Q[0], s), incident(S, Q[-1], s)
            if len(n0) != 1 or len(n1) != 1:
                continue                              # degree-2 proxy fails
            a = ang(S[n0[0]], Q[0], S[n1[0]], Q[-1], RAD)
            stub_hits.append((evt, s, round(L, 2), (n0[0], n1[0]),
                              None if a is None else round(a, 2),
                              a is not None and a < ANGLE))
        # --- the VERTEX variant: a degree-2 in-line break in the cathode band
        for v in V:
            if abs(v[0]) >= XCUT:
                continue
            inc = incident(S, v)
            if len(inc) != 2:
                continue
            a = ang(S[inc[0]], v, S[inc[1]], v, RAD)
            La = round(float(np.linalg.norm(S[inc[0]][0] - S[inc[0]][-1])), 1)
            Lb = round(float(np.linalg.norm(S[inc[1]][0] - S[inc[1]][-1])), 1)
            vtx_hits.append((evt, round(float(v[0]), 2), tuple(inc), (La, Lb),
                             None if a is None else round(a, 2),
                             a is not None and a < ANGLE))
    return n_ev, n_graph, stub_hits, vtx_hits


def main(*arms):
    for arm in arms:
        n_ev, n_graph, sh, vh = scan(arm)
        sfire = [r for r in sh if r[-1]]
        vfire = [r for r in vh if r[-1]]
        print(f'\n=== {arm}: {n_graph}/{n_ev} events with a PR graph')
        print(f'  B1 as specified (stub segment, deg-2 both ends, nb angle < {ANGLE} deg @ R={RAD}):')
        print(f'    candidate stubs {len(sh)}, of which FIRING {len(sfire)}')
        for r in sh:
            print(f'      evt {r[0]:>7} seg {r[1]} L={r[2]} nb={r[3]} nb_angle={r[4]} fire={r[5]}')
        print(f'  VERTEX variant (deg-2 vertex, |x|<{XCUT}, incident angle < {ANGLE} deg @ R={RAD}):')
        print(f'    candidate vertices {len(vh)}, of which FIRING {len(vfire)}')
        for r in vh:
            print(f'      evt {r[0]:>7} x={r[1]:6.2f} segs={r[2]} L={r[3]} angle={r[4]} fire={r[5]}')


if __name__ == '__main__':
    main(*sys.argv[1:])
