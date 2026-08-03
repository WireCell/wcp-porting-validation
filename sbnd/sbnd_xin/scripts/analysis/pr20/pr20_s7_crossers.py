#!/usr/bin/env python3
"""Gate S7: do the crossers A1+A2 newly joined stay ONE segment in the PR graph?

The ordering worry the doc raises: A1/A2 join the two halves of a cathode
crosser into one cluster, and the PR chain then breaks them apart again at the
cathode -- a class-A fix turning into a class-B problem.  This checks it
directly on the merge list from gate A-2.

For each event that gained a cathode-straddling merge, report cathode stubs
(stub_census.py definition: L < 10 cm, both ends |x| < 6 cm) in the A1+A2-only
arm and in the A1+A2+B0 arm.

  python3 pr20_s7_crossers.py docs/pr/pr20-edges-mcp1k-20260802.tsv \
      "work-b0pr300-off work-b0pr700-off" "work-b0pr300-on work-b0pr700-on"
"""
import os, sys, json, zipfile
import numpy as np

MAXLEN, XCUT = 10.0, 6.0


def stubs(path):
    z = zipfile.ZipFile(path)
    if 'data/0/0-track_fit-global.json' not in z.namelist():
        return None                                  # no PR graph on this event
    d = json.loads(z.read('data/0/0-track_fit-global.json'))
    P = np.stack([np.asarray(d['x'], float), np.asarray(d['y'], float),
                  np.asarray(d['z'], float)], 1)
    sid = np.asarray(d['real_cluster_id'])
    out = []
    for s in sorted({int(k) for k in sid.tolist()} - {-1}):
        Q = P[sid == s]
        if len(Q) < 2 or not (Q[:, 0].min() < 0 < Q[:, 0].max()):
            continue
        L = float(np.linalg.norm(Q[0] - Q[-1]))
        if L > MAXLEN or abs(Q[0][0]) > XCUT or abs(Q[-1][0]) > XCUT:
            continue
        out.append((s, round(L, 2)))
    return out


def find(roots, evt):
    for r in roots:
        p = os.path.join(r, f'pr_evt{evt}', 'mabc-pr.zip')
        if os.path.exists(p):
            return p
    return None


def main(tsv, off_roots, on_roots):
    off_roots, on_roots = off_roots.split(), on_roots.split()
    evts, cath = [], {}
    for line in open(tsv):
        if line.startswith('#'):
            continue
        t = line.split()
        if t[0] not in cath:
            evts.append(t[0])
        cath[t[0]] = cath.get(t[0], 0) + (1 if t[-1] == '1' else 0)
    print(f'{len(evts)} events carrying a new cathode_connect merge')
    ngraph = nstub_off = nstub_on = nmissing = 0
    for e in evts:
        po, pn = find(off_roots, e), find(on_roots, e)
        if not po or not pn:
            nmissing += 1
            print(f'  evt {e:>7}: NOT in the arms')
            continue
        so, sn = stubs(po), stubs(pn)
        if so is None and sn is None:
            print(f'  evt {e:>7}: no PR graph (no in-beam nu candidate)')
            continue
        ngraph += 1
        nstub_off += len(so or []); nstub_on += len(sn or [])
        flag = '' if not (so or sn) else '   <-- STUB'
        print(f'  evt {e:>7}: {cath[e]} cathode merge(s); '
              f'cathode stubs A1+A2 {len(so or [])} -> A1+A2+B0 {len(sn or [])}{flag}')
        if so:
            print(f'            A1+A2 only: {so}')
        if sn:
            print(f'            A1+A2+B0  : {sn}')
    print(f'\nevents with a PR graph: {ngraph};  not in the arms: {nmissing}')
    print(f'cathode stubs on merge events: A1+A2 {nstub_off} -> A1+A2+B0 {nstub_on}')


if __name__ == '__main__':
    main(*sys.argv[1:4])
