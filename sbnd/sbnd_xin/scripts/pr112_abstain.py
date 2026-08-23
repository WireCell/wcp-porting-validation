#!/usr/bin/env python3
'''doc pr/112 -- size an INSTABILITY-ABSTENTION gate for the DL main vertex.

pr/111 sec 7 measured the mechanism: the SCN's global argmax is chaotic at
1/10 of a voxel (an iid sigma = 0.05 cm jitter relocates it > 2 cm in 30 % of
draws) while fit_exclusion moves the cloud 0.277 cm median -- 5.5x that.  This
script asks the operational question that follows: if we can SEE that the net
is unstable on an event, is it better not to let it override the traditional
vertex there?

NOT test-time-augmentation AVERAGING.  That was measured twice and rejected --
pr/77 sec 8a (snap-hit 82/116 -> 82/116, tails to 357 cm, verdict "do NOT
deploy TTA averaging; keep TTA disagreement as a flag") and pr/111 sec 9
(ensemble +2/47 raw, with 2 lost).  Here the ensemble is used ONLY to score
the net's own stability; the deployed answer stays the single unjittered
inference.  The knob being sized is a ROUTE override, not a score change.

Rulers and instruments, all pre-existing:
  * TARGET metric of pr/106 -- target = the pre-DL candidate (hv_cloud vertex
    row) nearest the hand-scan click; hit = the code PICKS that candidate.
    Immune to the fit epoch, which a 1 cm click match is not (pr/105).
  * acceptance replay = calib_guard.replay_rerank, which reproduces
    NeutrinoVertexFinder.cxx:4396-4676 on the exact harvested cloud.  Imported,
    not copied, so it cannot drift from the screen ft2u was judged by.
  * the abstain outcome is the REAL DL-off arm (work-vtx105-trad-*, route
    dl-not-run), not hv_trad_main_vertex_id -- that field is the PRE-fallback
    state (pr/106 F16, 55 unresolved picks), so using it would understate the
    traditional route.  Events whose trad arm is missing are reported
    separately, never silently counted.

CLOSURE, mandatory before any verdict: the unjittered replay must reproduce
the live winner (CP24's voxels are bit-identical to the recording, pr/111
sec 3).  Events where it does not are excluded and counted.

Usage:
  ./pr112_abstain.py --samples nuecc48 --n 8 --tsv runs/pr112-abstain-nuecc48.tsv
  ./pr112_abstain.py --samples nuecc48 ncpi0 mcp1k mcp2k --n 8 --jobs 12 \
      --tsv runs/pr112-abstain-all.tsv
'''
import argparse
import collections
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(os.path.dirname(HERE), 'dl_vtx_training')
sys.path.insert(0, TRAIN)
sys.path.insert(0, '/nfs/data/1/xqian/toolkit-dev/toolkit/pyutil/python')
from scn_vtx import io as vio              # noqa: E402
from calib_guard import replay_rerank      # noqa: E402

WEIGHTS = ('/nfs/data/1/xqian/toolkit-dev/wire-cell-data/uboone/scn_vtx/'
           't48k-m16-l5-lr5d-res0.5-CP24.pth')
# pr/106 sec 0 repro: original labels only, so truth favours neither topology.
ORIG_TAGS = {'nuecc48': ['vtxscan-harv3-nuecc48'], 'ncpi0': ['vtxscan-harv3-ncpi0'],
             'mcp1k': ['vtxscan-harv3-mcp1k'],
             'mcp2k': ['vtxscan-mcp2k', 'vtxscan-mcp2k-auto', 'vtxscan-mcp2k-ragree']}
# Regenerated 2026-08-23 after the retire campaign; work-pr112-harv-* is proven
# bit-identical to the retired work-vtx106-harv-base-* by pr112_repro_gate.py.
HARV = 'work-pr112-harv-%s'            # exclusion ON, dl_vtx_harvest on
TRAD = 'work-pr112-trad-%s'            # exclusion ON, DL disabled (dl-not-run)
SIGMA = 0.05                           # cm; pr/111 sec 7 P1, one tenth of a voxel
SEED = 20260823


def net(x, y, z, q, top_k):
    import SCN_Vertex as sv
    raw = sv.SCN_Vertex(WEIGHTS, x.tobytes(), y.tobytes(), z.tobytes(),
                        q.tobytes(), dtype='float32', top_k=int(top_k))
    return np.frombuffer(raw, np.float32).reshape(-1, 4)


def one(job):
    evt, sample, src, truth, harv_p, trad_p, ndraw, sigma = job
    r = dict(evt=evt, sample=sample, src=src, skip='')
    hb = vio.load_calib(harv_p) if os.path.exists(harv_p) else None
    if hb is None:
        r['skip'] = 'no-harvest'; return r
    sb = (hb.get('vertex_scoreboard') or {})
    c = sb.get('hv_cloud')
    if not c or not int(c.get('n_vertex_rows', 0)):
        r['skip'] = 'empty-cloud'; return r
    nv = int(c['n_vertex_rows'])
    ids = c['vertex_ids'][:nv]
    xyz = np.array([c['x'][:nv], c['y'][:nv], c['z'][:nv]], float).T
    tr = np.asarray(truth, float)
    d = np.linalg.norm(xyz - tr, axis=1)
    it = int(np.argmin(d))
    r['target'], r['d_target'] = ids[it], float(d[it])

    x = np.array(c['x'], np.float32); y = np.array(c['y'], np.float32)
    z = np.array(c['z'], np.float32); q = np.array(c['q'], np.float32)
    top_k = int(sb.get('dl_top_k') or 5)

    # ---- closure: unjittered replay must reproduce the live winner ----------
    vox = net(x, y, z, q, top_k)
    rep = replay_rerank(sb, vox)
    if rep is None:
        r['skip'] = 'no-candidates'; return r
    live_win = next((w['vertex_id'] for w in (sb.get('rows') or [])
                     if w.get('dl_winner')), None)
    route = sb.get('route', '')
    accept = rep['best_tot'] >= float(sb['dl_min_accept_score'])
    r['route'] = route
    r['closure'] = int(live_win is None or rep['best_vid'] == live_win)
    r['pick_dl'] = rep['best_vid']

    # ---- the traditional answer the reject route would inherit --------------
    # PREFERRED source is a live DL-off arm (route dl-not-run).  The 2026-08-23
    # retire campaign deleted every one of them (work-vtx105-trad-*), so this
    # falls back to hv_trad_main_vertex_id, which pr/106 F16 records as the
    # PRE-fallback state, not the fallback outcome -- it UNDERSTATES the live
    # traditional route.  Every abstention number below is therefore a
    # STAND-IN and is labelled as such; it is not the live answer.
    r['trad_src'] = ''
    r['pick_trad'] = ''
    tj = vio.load_calib(trad_p) if os.path.exists(trad_p) else None
    if tj is not None and (tj.get('main_vertex') or {}).get('x') is not None:
        mv = tj['main_vertex']
        pt = np.array([mv['x'], mv['y'], mv['z']], float)
        dt = np.linalg.norm(xyz - pt, axis=1)
        r['pick_trad'] = ids[int(np.argmin(dt))]
        r['d_trad_snap'] = float(dt.min())
        r['trad_src'] = 'live-trad-arm'
    else:
        tv = int(sb.get('hv_trad_main_vertex_id', -1))
        if tv >= 0 and tv in ids:
            r['pick_trad'] = tv
            r['d_trad_snap'] = 0.0
            r['trad_src'] = 'hv_trad_standin'
    if r['pick_trad'] == '':
        r['skip'] = 'no-trad-answer'; return r

    # ---- stability of the net's OWN answer under a null perturbation -------
    # iid gaussian position jitter, charge untouched: pr/111 sec 7 P1 showed
    # the instability is geometric, not calorimetric (P2 moved the argmax on
    # 0/160 draws).  A RIGID translation is an exact no-op (x - x.min), so the
    # jitter must be per-point.
    rng = np.random.default_rng(SEED + int(evt))
    peaks = []
    for _ in range(ndraw):
        jx = (x + rng.normal(0, sigma, x.shape).astype(np.float32)).astype(np.float32)
        jy = (y + rng.normal(0, sigma, y.shape).astype(np.float32)).astype(np.float32)
        jz = (z + rng.normal(0, sigma, z.shape).astype(np.float32)).astype(np.float32)
        v = net(jx, jy, jz, q, 2)          # top_k=1 takes the legacy 3-float branch
        peaks.append(v[0, :3].astype(float))
    P = np.asarray(peaks)
    cen = P.mean(axis=0)
    r['spread'] = float(np.sqrt(((P - cen) ** 2).sum(axis=1).mean()))
    p0 = vox[0, :3].astype(float)
    r['frac_gt2'] = float(np.mean(np.linalg.norm(P - p0, axis=1) > 2.0))
    r['accept'] = int(accept)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', nargs='+', default=['nuecc48'])
    ap.add_argument('--n', type=int, default=8, help='stability draws per event')
    ap.add_argument('--sigma', type=float, default=SIGMA)
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--dmax', type=float, default=3.0,
                    help='primary universe: d_target <= dmax cm (pr/106)')
    ap.add_argument('--harv', default=HARV)
    ap.add_argument('--trad', default=TRAD)
    ap.add_argument('--tsv', default=None)
    a = ap.parse_args()

    root = vio.default_sbnd_root()
    jobs = []
    for s in a.samples:
        for lab in vio.iter_labels(root, ORIG_TAGS[s]):
            e = int(lab['eventNo'])
            jobs.append((e, s, lab.get('label_source', 'human'),
                         np.asarray(lab['truth_xyz'], float),
                         os.path.join(root, a.harv % s, 'pr_evt%d' % e,
                                      'calib-pr-evt%d.json' % e),
                         os.path.join(root, a.trad % s, 'pr_evt%d' % e,
                                      'calib-pr-evt%d.json' % e),
                         a.n, a.sigma))
    import multiprocessing as mp
    with mp.Pool(a.jobs) as pool:
        recs = pool.map(one, jobs, chunksize=1)

    skip = collections.Counter(r['skip'] for r in recs if r['skip'])
    good = [r for r in recs if not r['skip']]
    nclos = sum(1 for r in good if not r['closure'])
    good = [r for r in good if r['closure']]
    print('events %d ; skipped %s ; closure failures excluded %d'
          % (len(recs), dict(skip), nclos))

    prim = [r for r in good if r['d_target'] <= a.dmax]
    print('primary universe (d_target <= %g cm): %d of %d' % (a.dmax, len(prim), len(good)))

    def score(rs, th):
        hit = 0; nab = 0
        for r in rs:
            pick = r['pick_dl']
            if r['accept'] and th is not None and r['spread'] > th:
                pick = r['pick_trad']; nab += 1
            hit += int(pick == r['target'])
        return hit, nab

    for label, rs in (('primary', prim), ('all-labeled', good)):
        base, _ = score(rs, None)
        print('\n%s (n=%d)   baseline (production selection) = %d/%d'
              % (label, len(rs), base, len(rs)))
        print('  %-9s %-14s %-9s %s' % ('thresh', 'hit', 'abstained', 'delta'))
        for th in (1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 40.0):
            h, n = score(rs, th)
            print('  %-9.1f %-14s %-9d %+d' % (th, '%d/%d' % (h, len(rs)), n, h - base))

    if a.tsv:
        os.makedirs(os.path.dirname(a.tsv) or '.', exist_ok=True)
        cols = ['evt', 'sample', 'src', 'target', 'd_target', 'route', 'accept',
                'pick_dl', 'pick_trad', 'trad_src', 'd_trad_snap', 'spread', 'frac_gt2', 'closure']
        with open(a.tsv, 'w') as fh:
            fh.write('\t'.join(cols) + '\n')
            for r in sorted(good, key=lambda r: r['evt']):
                fh.write('\t'.join(str(r.get(c, '')) for c in cols) + '\n')
        print('\nwrote %s' % a.tsv)


if __name__ == '__main__':
    sys.exit(main())
