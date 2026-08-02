#!/usr/bin/env python3
"""Gate B0-3 census: cathode-band vertices and cathode stubs, OFF vs ON.

Compares two PR-chain arms produced from the SAME Q/L input tree, differing
only in the doc pr/20 Part II B0 knob (`cathode_kink_xcut`).  Read-only.

Repro:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
  python3 pr20_b03_census.py work-b0pr300-off work-b0pr300-on

Reports, per the plan's B0-3 criterion:
  * cathode-band vertices (|x| < XBAND cm) OFF -> ON;
  * on events with NO cathode-band vertex OFF, the vertex set must be
    identical (that is the no-collateral half of the gate);
  * on events where a break was suppressed, off-cathode changes are allowed
    but are listed so each can be traced to the affected track;
  * cathode stubs (stub_census.py definition) OFF -> ON.
"""
import os, sys, json, glob, zipfile
import numpy as np

XBAND = 3.0     # cm: "within 3 cm of x = 0", the pr/12 §7 reporting statistic
XKNOB = 5.0     # cm: the knob's own band (`cathode_kink_xcut`) -- a change is
                # traceable if the event had a vertex inside THIS band, not the
                # narrower reporting band.
MAXLEN = 10.0   # cm: cathode-stub length bound (stub_census.py)
XCUT = 6.0      # cm: cathode-stub end bound (stub_census.py)
TOL = 1e-4      # cm: vertex identity tolerance


def load(path, member):
    z = zipfile.ZipFile(path)
    if member not in z.namelist():
        return None
    return json.loads(z.read(member))


def vtx(path):
    d = load(path, 'data/0/0-vertices-global.json')
    if d is None:
        return None
    P = np.stack([np.asarray(d['x'], float), np.asarray(d['y'], float),
                  np.asarray(d['z'], float)], 1)
    return P


def segs(path):
    d = load(path, 'data/0/0-track_fit-global.json')
    if d is None:
        return None
    P = np.stack([np.asarray(d['x'], float), np.asarray(d['y'], float),
                  np.asarray(d['z'], float)], 1)
    sid = np.asarray(d['real_cluster_id'])
    return {int(s): P[sid == s] for s in set(sid.tolist()) if s != -1}


def stubs(S):
    """Cathode stubs, stub_census.py definition."""
    out = []
    for s, Q in S.items():
        if len(Q) < 2:
            continue
        if not (Q[:, 0].min() < 0 < Q[:, 0].max()):
            continue
        L = float(np.linalg.norm(Q[0] - Q[-1]))
        if L > MAXLEN or abs(Q[0][0]) > XCUT or abs(Q[-1][0]) > XCUT:
            continue
        out.append((s, round(L, 2)))
    return sorted(out)


def setrows(P):
    """Vertex positions as a hashable multiset of rounded triples."""
    return sorted(tuple(np.round(p, 4)) for p in P)


def main(a_off, a_on):
    evts = sorted(os.path.basename(os.path.dirname(p)).replace('pr_evt', '')
                  for p in glob.glob(os.path.join(a_off, 'pr_evt*', 'mabc-pr.zip')))
    n_graph = n_ident = 0
    n_band_off = n_band_on = 0
    n_knob_off = n_knob_on = 0
    ev_band_off, ev_knob_off, ev_changed, ev_bad = [], [], [], []
    stub_off_tot = stub_on_tot = 0
    ev_stub_off, ev_stub_on = [], []
    for e in evts:
        po = os.path.join(a_off, f'pr_evt{e}', 'mabc-pr.zip')
        pn = os.path.join(a_on, f'pr_evt{e}', 'mabc-pr.zip')
        if not os.path.exists(pn):
            ev_bad.append((e, 'missing ON')); continue
        Vo, Vn = vtx(po), vtx(pn)
        if Vo is None and Vn is None:
            continue                      # no PR graph on either arm
        if (Vo is None) != (Vn is None):
            ev_bad.append((e, 'graph on one arm only')); continue
        n_graph += 1
        bo = int((np.abs(Vo[:, 0]) < XBAND).sum())
        bn = int((np.abs(Vn[:, 0]) < XBAND).sum())
        ko = int((np.abs(Vo[:, 0]) < XKNOB).sum())
        kn = int((np.abs(Vn[:, 0]) < XKNOB).sum())
        n_band_off += bo; n_band_on += bn
        n_knob_off += ko; n_knob_on += kn
        if ko:
            ev_knob_off.append(e)
        if bo:
            ev_band_off.append((e, bo, bn))
        So, Sn = stubs(segs(po) or {}), stubs(segs(pn) or {})
        stub_off_tot += len(So); stub_on_tot += len(Sn)
        if So:
            ev_stub_off.append((e, So))
        if Sn:
            ev_stub_on.append((e, Sn))
        same = setrows(Vo) == setrows(Vn)
        if same:
            n_ident += 1
        else:
            ev_changed.append((e, bo, bn, len(Vo), len(Vn), len(So), len(Sn)))

    print(f'arms: OFF={a_off}  ON={a_on}')
    print(f'events with a PR graph on both arms: {n_graph} / {len(evts)}')
    print(f'vertex sets identical: {n_ident} / {n_graph};  differing: {len(ev_changed)}')
    print(f'cathode-band vertices (|x|<{XBAND}cm): OFF {n_band_off} -> ON {n_band_on}')
    print(f'  events with >=1 cathode-band vertex OFF: {len(ev_band_off)}')
    print(f'knob-band vertices  (|x|<{XKNOB}cm): OFF {n_knob_off} -> ON {n_knob_on}')
    print(f'  events with >=1 knob-band vertex OFF: {len(ev_knob_off)}')
    print(f'cathode stubs (L<{MAXLEN}, ends |x|<{XCUT}): OFF {stub_off_tot} -> ON {stub_on_tot}')
    print(f'  events with >=1 stub: OFF {len(ev_stub_off)} -> ON {len(ev_stub_on)}')

    band_evts = set(ev_knob_off)
    stub_evts = {e for e, _ in ev_stub_off}
    collateral = [r for r in ev_changed if r[0] not in band_evts and r[0] not in stub_evts]
    print(f'\nCOLLATERAL (vertex set changed with NO knob-band vertex and NO stub OFF): '
          f'{len(collateral)}')
    for r in collateral:
        print(f'  evt {r[0]}  nvtx {r[3]}->{r[4]}  band {r[1]}->{r[2]}  stubs {r[5]}->{r[6]}')

    print(f'\nchanged events WITH a knob-band vertex or stub at baseline: '
          f'{len(ev_changed) - len(collateral)}')
    for r in ev_changed:
        if r[0] in band_evts or r[0] in stub_evts:
            print(f'  evt {r[0]}  nvtx {r[3]}->{r[4]}  band {r[1]}->{r[2]}  stubs {r[5]}->{r[6]}')

    print('\nstub detail OFF:')
    for e, S in ev_stub_off:
        print(f'  evt {e}: {S}')
    print('stub detail ON:')
    for e, S in ev_stub_on:
        print(f'  evt {e}: {S}')
    if ev_bad:
        print('\nanomalies:', ev_bad)


if __name__ == '__main__':
    main(*sys.argv[1:3])
