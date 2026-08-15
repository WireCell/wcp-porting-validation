#!/usr/bin/env python3
'''
doc pr/79 §11 -- calibration-replay guard: the offline screen ft2u needed.

pr/79 §3 root-caused ft2u's live -40/473 to score-scale inflation (median
dl_best_score x1.53): rank-based offline metrics (d_argmax, snap_hit) are
invariant to scale, but live acceptance consumes the ABSOLUTE composite total
against dl_vtx_min_accept_score.  This script replays the full rerank
acceptance stage (NeutrinoVertexFinder.cxx:4396-4676) for a candidate weights
file on the EXACT live input cloud recorded by dl_vtx_harvest:

  1. run the net on hv_cloud through the production pyutil path
     (SCN_Vertex, as verify_harvest.py does) -> top-K voxels+scores;
  2. snap each voxel to the nearest live candidate = the hv_cloud vertex
     block (leading n_vertex_rows points ARE cand_vertices in graph order,
     NeutrinoVertexFinder.cxx:4237-4247, so C++ argmin-first-tie and the
     graph_index iteration order are both reproduced by cloud-block order);
  3. rescore: s_dl = dl_score*dl_score_scale and s_snap, s_fwd_z recomputed
     (min_z_set is over the SNAPPED set, :4529 -- it moves when the snapped
     set moves); s_clen/s_isol/s_main/s_fv are cluster/vertex-static and
     read from the recorded row when it was scored live, else derived from
     harvest fields (counted as `approx`);
  4. route: accept iff best total >= dl_min_accept_score; anchored-replay
     outcome semantics as rank_sim.py (matching decision -> recorded live
     answer; changed accept -> row stand-in; new reject -> trad stand-in
     via hv_trad_main_vertex_id, which UNDERSTATES the live trad route).

Self-validation protocol (mandatory before trusting any verdict): CP24 must
reproduce the recorded routes ~exactly (its voxels are bit-identical to the
recording, doc pr/79 §10), and ft2u CP9 must reproduce the known failure
signature (score inflation ~x1.5, reject->accept flip mass, negative
predicted delta).  Only then is the guard a valid screen for a new candidate.

Usage:
  python3 calib_guard.py --name cp24                          # baseline anchor
  python3 calib_guard.py --name ft2u --weights /nfs/.../sbnd-vtx-ft2u-full473-e10-CP9.pth
  python3 calib_guard.py --name hft1 --weights runs/hft1-deploy/fold0/CP9.pth \
      --tsv runs/calibguard-hft1-<date>.tsv
'''
import argparse
import multiprocessing as mp
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from scn_vtx import io as vio
from taxonomy import ALL_TAGS

PYUTIL = '/nfs/data/1/xqian/toolkit-dev/toolkit/pyutil/python'

HARV_ARMS = ['vtxscan-prod0813=work-nuecc48-ma10k20-harv2',
             'vtxscan-prod0813-ncpi0=work-ncpi0-ma10k20-harv2',
             'vtxscan-prod0813-mcp1k=work-mcp1k-ma10k20-harv2']

# NeutrinoVertexFinder.cxx:4541-4550 (constexpr, verbatim)
W_MAIN, W_CLEN, W_CLEN_L = 2.0, 2.0, 60.0
W_ISOL, W_ISOL_L, W_FV = 2.0, 6.0, 0.5
W_SNAP_L, W_SNAP_MAX = 5.0, 2.0
W_FWD_Z, W_FWD_L = 0.25, 400.0

_G = {}


def _init(weights):
    sys.path.insert(0, PYUTIL)
    import SCN_Vertex as sv
    _G['sv'] = sv
    _G['weights'] = weights


def run_net(x, y, z, q, top_k):
    raw = _G['sv'].SCN_Vertex(_G['weights'], x.tobytes(), y.tobytes(),
                              z.tobytes(), q.tobytes(), dtype='float32',
                              top_k=top_k)
    return np.frombuffer(raw, np.float32).reshape(-1, 4)


def recorded_answer(calib, sb):
    mv = calib.get('main_vertex') or {}
    if mv.get('x') is not None:
        return np.array([mv['x'], mv['y'], mv['z']], float)
    if sb.get('filled'):
        return np.array([sb['final_x'], sb['final_y'], sb['final_z']], float)
    return None


def replay_rerank(sb, vox):
    """New-net rerank replay from the harvest payload.  Returns None when
    nothing snapped (C++ 'dl-no-candidates')."""
    c = sb['hv_cloud']
    nvr = int(c['n_vertex_rows'])
    cand = np.stack([np.array(c['x'][:nvr], np.float32),
                     np.array(c['y'][:nvr], np.float32),
                     np.array(c['z'][:nvr], np.float32)], axis=1).astype(float)
    ids = c['vertex_ids']
    rows = {r['vertex_id']: r for r in (sb.get('rows') or [])}
    tv = int(sb.get('hv_trad_main_vertex_id', -1))
    main_cid = tv // 1000 if tv >= 0 else None
    if main_cid is None:
        for r in rows.values():
            if r.get('dl_snapped') and r.get('s_main', 0) > 0:
                main_cid = r['cluster_id']
                break
    # cluster -> host track length, from any live-scored row of that cluster
    clen = {}
    for r in rows.values():
        if r.get('dl_snapped') and not r.get('skipped_by_swap_guard') \
                and r.get('host_length', 0) > 0:
            clen[r['cluster_id']] = float(r['host_length'])

    # snap: per voxel nearest candidate (argmin = first-min, matching C++
    # strict <); per candidate keep the higher dl_score (:4472-4474)
    best = {}
    for rank in range(len(vox)):
        d = np.linalg.norm(cand - vox[rank, :3].astype(float), axis=1)
        ci = int(np.argmin(d))
        s = float(vox[rank, 3])
        if ci not in best or s > best[ci][0]:
            best[ci] = (s, float(d[ci]), rank)
    if not best:
        return None

    scale = float(sb.get('dl_score_scale') or 1000.0)
    min_z = min(cand[ci][2] for ci in best)
    out = dict(best_tot=-np.inf, best_ci=-1, n_approx=0)
    for ci in sorted(best):          # cloud-block order == graph_index order
        s, snap_dis, _rank = best[ci]
        r = rows.get(ids[ci])
        s_dl = s * scale
        s_snap = -min(W_SNAP_MAX, snap_dis / W_SNAP_L)
        s_fwd_z = -W_FWD_Z * float(np.clip((cand[ci][2] - min_z) / W_FWD_L,
                                           0.0, 1.0))
        if r and r.get('dl_snapped') and not r.get('skipped_by_swap_guard'):
            s_stat = (float(r['s_clen']) + float(r['s_isol'])
                      + float(r['s_main']) + float(r['s_fv']))
        else:
            cid = r['cluster_id'] if r else ids[ci] // 1000
            L = clen.get(cid, 0.0)
            s_clen = W_CLEN * min(1.0, L / W_CLEN_L)
            s_isol = -W_ISOL if (L < W_ISOL_L and main_cid is not None
                                 and cid != main_cid) else 0.0
            s_main = W_MAIN if (main_cid is not None and cid == main_cid) else 0.0
            s_fv = W_FV if (r and r.get('hv_filled') and r.get('hv_in_fv')) else 0.0
            s_stat = s_clen + s_isol + s_main + s_fv
            out['n_approx'] += 1
        tot = s_dl + s_snap + s_fwd_z + s_stat
        out.setdefault('cands', []).append(
            (ids[ci], s, s_snap + s_fwd_z + s_stat, cand[ci]))
        if tot > out['best_tot']:
            out.update(best_tot=tot, best_ci=ci)
    out.update(best_vid=ids[out['best_ci']], best_xyz=cand[out['best_ci']],
               top1=float(vox[0, 3]))
    return out


def replay_one(job):
    evt, sample, corrective, calib_path, truth, tol = job
    calib = vio.load_calib(calib_path)
    sb = calib.get('vertex_scoreboard') or {}
    ans = recorded_answer(calib, sb)
    ok_rec = int(ans is not None
                 and np.linalg.norm(ans - np.asarray(truth, float)) <= tol)
    rec = dict(evt=evt, sample=sample, corrective=corrective, ok_rec=ok_rec,
               route_rec=sb.get('route', ''), best_rec=sb.get('dl_best_score'),
               top1_rec=(sb.get('voxels') or [{}])[0].get('dl_score'),
               ok_new=ok_rec, route_new=sb.get('route', ''), standin='',
               best_new='', top1_new='', winner_match='', n_approx=0,
               d_row='', d_trad='')
    if not sb.get('harvest') \
            or rec['route_rec'] not in ('dl-rerank-accept', 'dl-rerank-reject'):
        rec['standin'] = 'anchored'   # route the rerank stage cannot change
        return rec

    c = sb['hv_cloud']
    vox = run_net(np.array(c['x'], np.float32), np.array(c['y'], np.float32),
                  np.array(c['z'], np.float32), np.array(c['q'], np.float32),
                  int(sb['dl_top_k']))
    rep = replay_rerank(sb, vox)
    if rep is None:
        rec.update(route_new='dl-no-candidates', standin='anchored')
        return rec

    rows = sb.get('rows') or []
    rec_accept = rec['route_rec'] == 'dl-rerank-accept'
    rec_winner = next((r['vertex_id'] for r in rows if r.get('dl_winner')), None)
    accept = rep['best_tot'] >= float(sb['dl_min_accept_score'])
    truth = np.asarray(truth, float)
    d_row = float(np.linalg.norm(rep['best_xyz'] - truth))
    tv = int(sb.get('hv_trad_main_vertex_id', -1))
    d_trad = ''
    if tv >= 0 and tv in c['vertex_ids']:
        ci = c['vertex_ids'].index(tv)
        p = np.array([c['x'][ci], c['y'][ci], c['z'][ci]], float)
        d_trad = float(np.linalg.norm(p - truth))

    rec.update(best_new='%.4f' % rep['best_tot'], top1_new='%.6f' % rep['top1'],
               n_approx=rep['n_approx'], d_row='%.3f' % d_row,
               d_trad=('' if d_trad == '' else '%.3f' % d_trad),
               winner_match=int(rep['best_vid'] == rec_winner),
               route_new='dl-rerank-accept' if accept else 'dl-rerank-reject')
    # internal (not TSV): per-candidate table for --fit-scale re-routing
    rec['_cands'] = [(vid, dl, rest,
                      float(np.linalg.norm(np.asarray(p) - truth)))
                     for vid, dl, rest, p in rep['cands']]
    rec['_scale0'] = float(sb.get('dl_score_scale') or 1000.0)
    rec['_ma'] = float(sb['dl_min_accept_score'])
    rec['_rec_winner'] = rec_winner
    rec['_d_trad_f'] = d_trad if d_trad != '' else None
    if accept and rec_accept and rep['best_vid'] == rec_winner:
        pass                                    # full live answer (post-refit)
    elif accept:
        rec.update(ok_new=int(d_row <= tol), standin='row')
    elif not rec_accept:
        pass                                    # both reject: live trad answer
    else:                                       # accept -> reject
        rec.update(ok_new=int(d_trad != '' and d_trad <= tol), standin='trad')
    return rec


def pct(v, q):
    return float(np.percentile(np.asarray(v, float), q)) if len(v) else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True, help='label for the report')
    ap.add_argument('--weights', default=None,
                    help='candidate .pth (default: production CP24)')
    ap.add_argument('--arm-roots', nargs='+', default=HARV_ARMS)
    ap.add_argument('--sbnd-root', default=vio.default_sbnd_root())
    ap.add_argument('--tol', type=float, default=1.0)
    ap.add_argument('--jobs', type=int, default=16)
    ap.add_argument('--sweep-min-accepts', nargs='+', type=float,
                    default=[4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0])
    ap.add_argument('--events-file', default=None,
                    help='doc pr/81 B: file of eventNo (one per line) -- '
                         'replay only these (guard-in-loop fold selection)')
    ap.add_argument('--fit-scale', action='store_true',
                    help='doc pr/81 C: fit a multiplicative score scale on '
                         'CONFIRMING events matching CP24 top-1 calibration, '
                         'and report the replay again at the fitted scale')
    ap.add_argument('--tsv', default=None)
    args = ap.parse_args()

    keep_events = None
    if args.events_file:
        with open(args.events_file) as fh:
            keep_events = {int(l.split()[0]) for l in fh if l.strip()}

    weights = args.weights or vio.default_weights()
    roots = vio.parse_arm_roots(args.arm_roots, args.sbnd_root)
    jobs = []
    for label in vio.iter_labels(args.sbnd_root, ALL_TAGS):
        if keep_events is not None and label['eventNo'] not in keep_events:
            continue
        path = vio.calib_path_in_roots(roots, label)
        if not path or not os.path.exists(path):
            print('MISSING calib for evt%d' % label['eventNo'])
            return 1
        jobs.append((label['eventNo'], vio.sample_of_label(label),
                     int(vio.is_corrective(label)), path,
                     np.asarray(label['truth_xyz'], float), args.tol))
    print('calib_guard %s: %d labeled events, weights=%s'
          % (args.name, len(jobs), weights))

    with mp.Pool(args.jobs, initializer=_init, initargs=(weights,)) as pool:
        recs = list(pool.imap_unordered(replay_one, jobs, chunksize=4))
    recs.sort(key=lambda r: r['evt'])

    replayed = [r for r in recs if r['standin'] != 'anchored']
    anchored = len(recs) - len(replayed)
    n_aa = sum(1 for r in replayed if r['route_rec'] == 'dl-rerank-accept'
               and r['route_new'] == 'dl-rerank-accept')
    n_ar = sum(1 for r in replayed if r['route_rec'] == 'dl-rerank-accept'
               and r['route_new'] == 'dl-rerank-reject')
    n_ra = sum(1 for r in replayed if r['route_rec'] == 'dl-rerank-reject'
               and r['route_new'] == 'dl-rerank-accept')
    n_rr = sum(1 for r in replayed if r['route_rec'] == 'dl-rerank-reject'
               and r['route_new'] == 'dl-rerank-reject')
    print('\n== routes (replayed %d, anchored %d) ==' % (len(replayed), anchored))
    print('  accept->accept %d   accept->REJECT %d   reject->ACCEPT %d   '
          'reject->reject %d' % (n_aa, n_ar, n_ra, n_rr))

    for tagname, keep in (('confirming', 0), ('corrective', 1)):
        pairs = [(float(r['best_new']), float(r['best_rec'])) for r in replayed
                 if r['corrective'] == keep and r['best_rec'] is not None
                 and float(r['best_rec']) > 0 and float(r['best_new']) > 0]
        ratios = [a / b for a, b in pairs]
        higher = sum(1 for a, b in pairs if a > b)
        t_pairs = [(float(r['top1_new']), float(r['top1_rec'])) for r in replayed
                   if r['corrective'] == keep and r['top1_rec'] is not None
                   and float(r['top1_rec']) > 0 and float(r['top1_new']) > 0]
        t_ratios = [a / b for a, b in t_pairs]
        print('== score scale, %s ==' % tagname)
        print('  best composite total ratio new/rec: median %.3f  p10 %.3f  '
              'p90 %.3f  (n=%d, new higher %d)'
              % (pct(ratios, 50), pct(ratios, 10), pct(ratios, 90),
                 len(ratios), higher))
        print('  raw net top1 ratio new/rec:         median %.3f  p10 %.3f  '
              'p90 %.3f  (n=%d)'
              % (pct(t_ratios, 50), pct(t_ratios, 10), pct(t_ratios, 90),
                 len(t_ratios)))

    n_rec = sum(r['ok_rec'] for r in recs)
    n_new = sum(r['ok_new'] for r in recs)
    st_row = sum(1 for r in recs if r['standin'] == 'row')
    st_trad = sum(1 for r in recs if r['standin'] == 'trad')
    napx = sum(r['n_approx'] for r in recs)
    print('\n== predicted correct (anchored replay, tol %.1f cm) ==' % args.tol)
    print('  recorded %d/%d -> predicted %d/%d  (delta %+d) | stand-ins: '
          'row %d, trad %d | approx-term candidates %d'
          % (n_rec, len(recs), n_new, len(recs), n_new - n_rec,
             st_row, st_trad, napx))
    for s in ('nuecc', 'ncpi0', 'mcp1k'):
        sr = [r for r in recs if r['sample'] == s]
        print('    %-6s %3d -> %3d  (%+d) of %d'
              % (s, sum(r['ok_rec'] for r in sr), sum(r['ok_new'] for r in sr),
                 sum(r['ok_new'] - r['ok_rec'] for r in sr), len(sr)))

    print('\n== min_accept sweep (same replays, chooser fixed) ==')
    for ma in args.sweep_min_accepts:
        n = 0
        for r in recs:
            if r['standin'] == 'anchored' or r['best_new'] == '':
                n += r['ok_rec']
                continue
            rec_accept = r['route_rec'] == 'dl-rerank-accept'
            if float(r['best_new']) >= ma:
                if rec_accept and r['winner_match'] == 1:
                    n += r['ok_rec']
                else:
                    n += int(float(r['d_row']) <= args.tol)
            else:
                if not rec_accept:
                    n += r['ok_rec']
                else:
                    n += int(r['d_trad'] != '' and float(r['d_trad']) <= args.tol)
        print('  min_accept %5.1f : predicted %3d/%d  (%+d vs recorded)'
              % (ma, n, len(recs), n - n_rec))

    if args.fit_scale:
        t_pairs = [(float(r['top1_new']), float(r['top1_rec']))
                   for r in replayed if r['corrective'] == 0
                   and r['top1_rec'] is not None
                   and float(r['top1_rec']) > 0 and float(r['top1_new']) > 0]
        med = pct([a / b for a, b in t_pairs], 50)
        mult = 1.0 / med if med and np.isfinite(med) and med > 0 else 1.0
        print('\n== fitted scale (confirming top-1 median ratio %.3f -> '
              'scale multiplier %.3f, i.e. dl_vtx_score_scale %.0f) =='
              % (med, mult, 1000.0 * mult))
        n = n_flip = 0
        for r in recs:
            if '_cands' not in r:
                n += r['ok_rec']
                continue
            rec_accept = r['route_rec'] == 'dl-rerank-accept'
            best_tot, best = -np.inf, None
            for vid, dl, rest, d in r['_cands']:
                tot = dl * r['_scale0'] * mult + rest
                if tot > best_tot:
                    best_tot, best = tot, (vid, d)
            if best_tot >= r['_ma']:
                if rec_accept and best[0] == r['_rec_winner']:
                    n += r['ok_rec']
                else:
                    n += int(best[1] <= args.tol)
                n_flip += int(not rec_accept)
            else:
                if not rec_accept:
                    n += r['ok_rec']
                else:
                    n += int(r['_d_trad_f'] is not None
                             and r['_d_trad_f'] <= args.tol)
                    n_flip += 1
        ma0 = next((r['_ma'] for r in recs if '_ma' in r), 10.0)
        print('  at (scale x%.3f, min_accept %.1f): predicted %d/%d (%+d vs '
              'recorded), %d route flips'
              % (mult, ma0, n, len(recs), n - n_rec, n_flip))

    if args.tsv:
        cols = ['evt', 'sample', 'corrective', 'route_rec', 'route_new',
                'standin', 'ok_rec', 'ok_new', 'best_rec', 'best_new',
                'top1_rec', 'top1_new', 'winner_match', 'd_row', 'd_trad',
                'n_approx']
        with open(args.tsv, 'w') as fh:
            fh.write('\t'.join(cols) + '\n')
            for r in recs:
                fh.write('\t'.join(str(r[c]) for c in cols) + '\n')
        print('wrote %s' % args.tsv)
    return 0


if __name__ == '__main__':
    sys.exit(main())
