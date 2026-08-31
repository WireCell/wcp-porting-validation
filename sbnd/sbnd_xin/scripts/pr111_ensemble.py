#!/usr/bin/env python3
"""doc pr/111 sec 9 -- if the argmax is unstable, does AVERAGING it help?

sec 7/8: a 0.05 cm iid jitter relocates the net's global argmax > 2 cm in ~37 %
of draws, on BOTH detectors.  That is the scale fit_exclusion perturbs the cloud
by (0.277 cm median), so switching exclusion re-rolls the argmax rather than
systematically improving it.

If that reading is right, the lever is not the cloud but the READOUT: averaging
the score field over an ensemble of jittered evaluations should recover the
vertex more often than a single evaluation, with fit_exclusion left ON and the
cloud untouched.  This measures that ceiling offline -- no arm, no knob, no
change to any code path.

Scores are accumulated on the UNJITTERED 0.5 cm lattice, so the ensemble mean is
taken over the same voxels the live net would report.

Repro:  python3 scripts/pr111_ensemble.py [M]
"""
import os, sys, glob, json, csv, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pr111_scn_lib import load_cloud, infer, field, RES

H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARM = os.path.join(H, 'work-vtx106-harv-base-nuecc48')     # fit_exclusion ON = production
TSV = os.path.join(H, 'docs/pr/106_events-prod-orig.tsv')
SEED = 20260822
SIG = 0.05
HIT = 2.0     # cm; the argmax must land within this of the target vertex


def target(b, vid):
    for r in b.get('rows', []):
        if r['vertex_id'] == vid:
            return np.array([r['x'], r['y'], r['z']])
    c = b.get('hv_cloud') or {}
    for i, v in enumerate(c.get('vertex_ids', [])):
        if v == vid:
            return np.array([c['x'][i], c['y'][i], c['z'][i]])
    return None


def key(p, origin):
    return tuple(np.floor((p - origin) / RES).astype(np.int64))


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    rng = np.random.default_rng(SEED)
    tgts = {r['evt']: r for r in csv.DictReader(open(TSV), delimiter='\t')
            if r.get('sample') == 'nuecc48'}
    evts = sorted(tgts, key=int)
    raw_hit = ens_hit = n = 0
    rows = []
    for e in evts:
        try:
            cloud, b = load_cloud(ARM, e)
        except Exception:
            continue
        t = target(b, int(tgts[e]['target']))
        if t is None:
            continue
        x, y, z, q = cloud
        origin = np.array([x.min(), y.min(), z.min()], dtype=np.float64)
        f0 = field(cloud)
        p_raw = f0[int(np.argmax(f0[:, 3])), :3]
        acc = {}
        for _ in range(M):
            j = rng.normal(0, SIG, size=(len(x), 3)).astype(np.float32)
            fj = field((x + j[:, 0], y + j[:, 1], z + j[:, 2], q))
            for row in fj:
                k = key(row[:3], origin)
                a = acc.setdefault(k, [0.0, 0])
                a[0] += row[3]; a[1] += 1
        # mean over the draws in which that voxel was occupied, weighted by
        # occupancy so a voxel seen once cannot outrank one seen every draw
        best_k, best_v = None, -1e9
        for k, (s, c) in acc.items():
            v = (s / M)                      # missing draws count as 0 -> stability wins
            if v > best_v:
                best_v, best_k = v, k
        p_ens = origin + (np.array(best_k) + 0.5) * RES
        d_raw = float(np.linalg.norm(p_raw - t))
        d_ens = float(np.linalg.norm(p_ens - t))
        rh, eh = d_raw < HIT, d_ens < HIT
        raw_hit += rh; ens_hit += eh; n += 1
        rows.append((e, d_raw, d_ens, rh, eh))
    print(f"# M={M} jittered draws, sigma={SIG} cm, seed={SEED}; hit = argmax within {HIT} cm of target")
    print(f"# arm = {os.path.basename(ARM)}  (fit_exclusion ON, production)")
    print(f"\n{'evt':>7s} {'d_raw':>8s} {'d_ens':>8s}  change")
    for e, dr, de, rh, eh in rows:
        tag = 'RECOVERED' if (eh and not rh) else ('LOST' if (rh and not eh) else '')
        if tag or abs(dr - de) > 1.0:
            print(f"{e:>7s} {dr:8.2f} {de:8.2f}  {tag}")
    print(f"\nargmax within {HIT} cm of target:  raw {raw_hit}/{n}   ensemble-mean {ens_hit}/{n}")


if __name__ == '__main__':
    main()
