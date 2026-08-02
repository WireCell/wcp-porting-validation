#!/usr/bin/env python3
"""Population census of CATHODE STUB segments in the SBND PR chain.

A "cathode stub" is a fitted segment of the pattern-recognition graph whose
whole extent straddles x=0 and is only a few cm long -- the tip-to-tip bridge
across the cathode gap.  It is a separate segment, hence a separate particle in
the Bee particle-flow tree, so a single through-going track reads as >=2
particles.

Read-only over an existing PR-chain arm (doc pr/11's work-mcp1kall-pr11v3).
For each stub we also measure the KINK between its two neighbour segments,
computed the way NeutrinoStructureExaminer::examine_structure_3 measures it
(mean of the segment's fit points within R of the shared vertex, minus the
vertex) but with the neighbours taken pairwise, skipping the stub.
"""
import os, sys, json, glob, zipfile
import numpy as np

ARM = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin/work-mcp1kall-pr11v3'
MAXLEN = 10.0     # cm: a stub is shorter than this end-to-end
XCUT = 6.0        # cm: both ends must sit within this of the cathode
JOINTOL = 0.5     # cm: two segments share a vertex if their ends are this close


def seg_points(zf, member='data/0/0-track_fit-global.json'):
    d = json.loads(zf.read(member))
    P = np.stack([np.asarray(d['x'], float), np.asarray(d['y'], float),
                  np.asarray(d['z'], float)], 1)
    sid = np.asarray(d['real_cluster_id'])
    out = {}
    for s in set(sid.tolist()):
        if s == -1:
            continue                    # vertex fit points, not a segment
        out[int(s)] = P[sid == s]
    return out


def caldir(Q, p, r):
    m = np.linalg.norm(Q - p, axis=1) < r
    if m.sum() == 0:
        return None
    v = Q[m].mean(0) - p
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def kink(Qa, pa, Qb, pb, r):
    da, db = caldir(Qa, pa, r), caldir(Qb, pb, r)
    if da is None or db is None:
        return None
    c = float(np.clip(np.dot(da, db), -1, 1))
    return 180 - np.degrees(np.arccos(c))


def census_event(path):
    evt = os.path.basename(os.path.dirname(path)).replace('pr_evt', '')
    try:
        z = zipfile.ZipFile(path)
        segs = seg_points(z)
    except Exception as e:
        return evt, None, []
    out = []
    for s, Q in segs.items():
        if len(Q) < 2:
            continue
        xs = Q[:, 0]
        if not (xs.min() < 0 < xs.max()):
            continue
        # end-to-end extent (the segment's fit points are ordered along it)
        L = float(np.linalg.norm(Q[0] - Q[-1]))
        if L > MAXLEN:
            continue
        if abs(Q[0][0]) > XCUT or abs(Q[-1][0]) > XCUT:
            continue
        # neighbours: any other segment with an endpoint at either of our ends
        nb = {0: [], 1: []}
        for t, R in segs.items():
            if t == s or len(R) < 2:
                continue
            for ei, end in enumerate((Q[0], Q[-1])):
                if min(np.linalg.norm(R[0] - end), np.linalg.norm(R[-1] - end)) < JOINTOL:
                    nb[ei].append(t)
        rec = dict(seg=s, npts=len(Q), L=round(L, 2),
                   x0=round(float(Q[0][0]), 2), x1=round(float(Q[-1][0]), 2),
                   n_nb0=len(nb[0]), n_nb1=len(nb[1]))
        if len(nb[0]) == 1 and len(nb[1]) == 1:
            a, b = segs[nb[0][0]], segs[nb[1][0]]
            for r in (10.0, 15.0):
                k = kink(a, Q[0], b, Q[-1], r)
                rec[f'kink{int(r)}'] = None if k is None else round(k, 2)
            # the stub's OWN kink against each neighbour (what ES3 actually tests)
            k0 = kink(a, Q[0], Q, Q[0], 10.0)
            k1 = kink(b, Q[-1], Q, Q[-1], 10.0)
            rec['stub_kink0'] = None if k0 is None else round(k0, 2)
            rec['stub_kink1'] = None if k1 is None else round(k1, 2)
            rec['nb'] = (nb[0][0], nb[1][0])
            rec['nb_len'] = (round(float(np.linalg.norm(a[0] - a[-1])), 1),
                             round(float(np.linalg.norm(b[0] - b[-1])), 1))
        out.append(rec)
    return evt, len(segs), out


def main(arm=ARM):
    paths = sorted(glob.glob(os.path.join(arm, 'pr_evt*', 'mabc-pr.zip')))
    print(f'{len(paths)} PR events under {arm}')
    nev_with = 0
    rows = []
    for p in paths:
        evt, nseg, st = census_event(p)
        if nseg is None:
            continue
        if st:
            nev_with += 1
            for r in st:
                r['evt'] = evt
                rows.append(r)
    print(f'events with >=1 cathode stub segment (L<{MAXLEN}cm, both ends |x|<{XCUT}cm): '
          f'{nev_with} / {len(paths)}')
    print(f'total stub segments: {len(rows)}')
    clean = [r for r in rows if r.get('nb')]
    print(f'  of these, exactly one neighbour at each end (a degree-2 chain): {len(clean)}')
    if clean:
        k10 = [r['kink10'] for r in clean if r.get('kink10') is not None]
        sk = [max(r['stub_kink0'] or 0, r['stub_kink1'] or 0) for r in clean]
        print(f'  neighbour-neighbour kink (10cm): median {np.median(k10):.1f} deg, '
              f'<18deg in {sum(1 for k in k10 if k < 18)}/{len(k10)}')
        print(f'  stub-vs-neighbour kink (10cm, what examine_structure_3 tests): '
              f'median {np.median(sk):.1f} deg, <18deg in {sum(1 for k in sk if k < 18)}/{len(sk)}')
        print('\n  worst 25 by neighbour-collinearity (these are the recoverable ones):')
        for r in sorted(clean, key=lambda r: (r.get('kink10') is None, r.get('kink10')))[:25]:
            print(f'    evt {r["evt"]:>7} seg {r["seg"]:>7} L={r["L"]:5.2f} n={r["npts"]:3d} '
                  f'x {r["x0"]:6.2f}->{r["x1"]:6.2f}  nb={r["nb"]} len={r["nb_len"]}  '
                  f'kink10={r.get("kink10")} kink15={r.get("kink15")}  '
                  f'stub_kink={r.get("stub_kink0")}/{r.get("stub_kink1")}')
    with open('/home/xqian/tmp/cath3/stub_census.json', 'w') as f:
        json.dump(rows, f)
    print('\nwrote /home/xqian/tmp/cath3/stub_census.json')


if __name__ == '__main__':
    main(*sys.argv[1:])
