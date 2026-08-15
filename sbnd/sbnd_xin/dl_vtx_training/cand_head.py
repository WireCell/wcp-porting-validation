#!/usr/bin/env python3
'''
doc pr/81 Phase A (idea 4.iii) -- candidate-scoring head on EXACT live
features: the CP24 penultimate 16-dim vector at each candidate voxel
(extract_feats.py --harvest, doc pr/81 A1), the dl_vtx_harvest traditional
features (hv_*), and the recorded composite terms.  Per-event softmax
cross-entropy over candidates (truth candidate = label pick's row within
--tol), nested-OOF over the harv473 folds (seed 20260814), then ANCHORED
deploy replay in both deployable semantics:

  (i)  chooser-only: head picks among usable rows, acceptance frozen to the
       recorded route (this is the step-4 "frozen-route" slot, which was -8
       with a logistic on recorded features; the new elements are the exact
       16-dim features and the ranking objective);
  (ii) full: accept iff the head's best score clears a threshold chosen on
       the training folds (the step-4 "router" slot, -18..-25 before).

Anchored outcomes as calib_guard.py: unchanged decision -> recorded live
answer; changed accept -> row stand-in; new reject -> trad stand-in
(UNDERSTATES the live trad route).  Go/no-go per the pr/81 plan: no
formulation > 0 => confirmatory negative, close Phase A; >= +5 => stop for
owner (a head is NOT deployable without new C++).

Usage:
  python3 cand_head.py --feats data/k20feats-harv-20260815 \
      --tsv runs/candhead-20260815.tsv | tee runs/candhead-20260815.log
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
from dataset import load_manifest, kfold_split
from calib_guard import recorded_answer, HARV_ARMS

TERM_KEYS = ['s_dl', 's_snap', 's_fwd_z', 's_clen', 's_isol', 's_main',
             's_fv', 'snap_dis', 'host_length', 'dl_score']
HV_KEYS = ['hv_n_proton_in', 'hv_n_proton_out', 'hv_z_prior', 'hv_n_tracks',
           'hv_n_showers', 'hv_in_fv', 'hv_conflicts', 'hv_reduced_chi2']

_ROOTS = None


def _init(roots):
    global _ROOTS
    _ROOTS = roots


def load_event(job):
    """One event -> compact dict (features per usable row + replay facts)."""
    label_path, feats_path, tol = job
    label = vio.load_label(label_path)
    calib = vio.load_calib(vio.calib_path_in_roots(_ROOTS, label))
    sb = calib.get('vertex_scoreboard') or {}
    rows = sb.get('rows') or []
    usable = [r for r in rows
              if r.get('dl_snapped') and not r.get('skipped_by_swap_guard')]
    truth = np.asarray(label['truth_xyz'], float)
    ans = recorded_answer(calib, sb)
    ok_rec = int(ans is not None and np.linalg.norm(ans - truth) <= tol)
    ev = dict(evt=label['eventNo'], ok_rec=ok_rec,
              route=sb.get('route', ''), truth=truth)
    if not usable:
        return ev
    # trad stand-in position (calib_guard convention)
    c = sb.get('hv_cloud') or {}
    tv = int(sb.get('hv_trad_main_vertex_id', -1))
    ev['d_trad'] = None
    if tv >= 0 and c and tv in c.get('vertex_ids', []):
        ci = c['vertex_ids'].index(tv)
        p = np.array([c['x'][ci], c['y'][ci], c['z'][ci]], float)
        ev['d_trad'] = float(np.linalg.norm(p - truth))

    terms = np.array([[float(r.get(k, 0) or 0) for k in TERM_KEYS]
                      for r in usable], np.float32)
    hv = np.array([[float(r.get(k, 0) or 0) if r.get('hv_filled') else 0.0
                    for k in HV_KEYS] + [float(bool(r.get('hv_filled')))]
                   for r in usable], np.float32)
    with np.load(feats_path) as f:
        f16 = f['cand16'].astype(np.float32)
        cscore = f['cand_score'].astype(np.float32)
    assert len(f16) == len(usable), 'evt%d feats/rows mismatch' % ev['evt']
    pos = np.array([[r['x'], r['y'], r['z']] for r in usable], float)
    d = np.linalg.norm(pos - truth, axis=1)
    ti = int(np.argmin(d)) if len(d) else -1
    ev.update(terms=terms, hv=hv, f16=f16, cand_score=cscore,
              d_row=d.astype(np.float32),
              truth_i=(ti if (ti >= 0 and d[ti] <= tol) else -1),
              rec_winner_i=next((i for i, r in enumerate(usable)
                                 if r.get('dl_winner')), -1),
              comp_i=int(np.argmax([float(r['total']) for r in usable])))
    return ev


FEATSETS = {
    'T':    lambda ev: ev['terms'],
    'T+H':  lambda ev: np.concatenate([ev['terms'], ev['hv']], axis=1),
    'T+F':  lambda ev: np.concatenate([ev['terms'], ev['f16']], axis=1),
    'T+H+F': lambda ev: np.concatenate([ev['terms'], ev['hv'], ev['f16']],
                                       axis=1),
}


def fit_head(train_evs, featfn, hidden, seed, steps=400, lr=0.05, l2=1e-3):
    """Softmax-CE over candidates, full-batch Adam, best of 3 restarts by
    train loss.  Returns (predict(ev)->scores, norm stats)."""
    import torch
    X = [featfn(ev) for ev in train_evs]
    mu = np.mean(np.concatenate(X), axis=0)
    sd = np.std(np.concatenate(X), axis=0) + 1e-6
    Xn = [torch.tensor((x - mu) / sd) for x in X]
    yi = [ev['truth_i'] for ev in train_evs]
    nfeat = Xn[0].shape[1]
    best = (np.inf, None)
    for r in range(3):
        g = torch.Generator().manual_seed(seed * 10 + r)
        if hidden:
            W1 = torch.randn(nfeat, hidden, generator=g) * 0.3
            b1 = torch.zeros(hidden)
            W2 = torch.randn(hidden, 1, generator=g) * 0.3
            params = [W1.requires_grad_(), b1.requires_grad_(),
                      W2.requires_grad_()]
            def score(x): return torch.tanh(x @ params[0] + params[1]) @ params[2]
        else:
            W = torch.randn(nfeat, 1, generator=g) * 0.1
            params = [W.requires_grad_()]
            def score(x): return x @ params[0]
        opt = torch.optim.Adam(params, lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            loss = sum(torch.nn.functional.cross_entropy(
                score(x).T, torch.tensor([y]))
                for x, y in zip(Xn, yi)) / len(Xn)
            loss = loss + l2 * sum((p ** 2).sum() for p in params)
            loss.backward()
            opt.step()
        with torch.no_grad():
            fl = float(sum(torch.nn.functional.cross_entropy(
                score(x).T, torch.tensor([y]))
                for x, y in zip(Xn, yi)) / len(Xn))
        if fl < best[0]:
            best = (fl, [p.detach().clone() for p in params])
    params = best[1]

    def predict(ev):
        import torch
        with torch.no_grad():
            x = torch.tensor((featfn(ev) - mu) / sd)
            if hidden:
                s = torch.tanh(x @ params[0] + params[1]) @ params[2]
            else:
                s = x @ params[0]
        return s.numpy().ravel()
    return predict


def replay(ev, scores, tol, mode, tau=None):
    """Anchored outcome for one event under head `scores`."""
    if 'terms' not in ev:                       # nothing usable: recorded
        return ev['ok_rec'], ''
    rec_accept = ev['route'] == 'dl-rerank-accept'
    if ev['route'] not in ('dl-rerank-accept', 'dl-rerank-reject'):
        return ev['ok_rec'], ''                 # veto/no-cand routes anchored
    ci = int(np.argmax(scores))
    if mode == 'chooser':
        accept = rec_accept                     # route frozen
    else:
        accept = float(scores[ci]) >= tau
    if accept:
        if rec_accept and ci == ev['rec_winner_i']:
            return ev['ok_rec'], ''
        return int(ev['d_row'][ci] <= tol), 'row'
    if not rec_accept:
        return ev['ok_rec'], ''
    return int(ev['d_trad'] is not None and ev['d_trad'] <= tol), 'trad'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feats', default='data/k20feats-harv-20260815')
    ap.add_argument('--manifest', default='data/harv473/manifest.tsv')
    ap.add_argument('--arm-roots', nargs='+', default=HARV_ARMS)
    ap.add_argument('--sbnd-root', default=vio.default_sbnd_root())
    ap.add_argument('--tol', type=float, default=1.0)
    ap.add_argument('--kfold', type=int, default=6)
    ap.add_argument('--seed', type=int, default=20260814)
    ap.add_argument('--jobs', type=int, default=16)
    ap.add_argument('--tsv', default=None)
    args = ap.parse_args()

    feats_dir = args.feats if os.path.isabs(args.feats) \
        else os.path.join(HERE, args.feats)
    roots = vio.parse_arm_roots(args.arm_roots, args.sbnd_root)
    jobs = []
    for label in vio.iter_labels(args.sbnd_root, ALL_TAGS):
        fp = os.path.join(feats_dir, 'evt%d.npz' % label['eventNo'])
        jobs.append((label['label_path'],
                     fp if os.path.exists(fp) else None, args.tol))
    missing = [j for j in jobs if j[1] is None]
    jobs = [j for j in jobs if j[1] is not None]
    with mp.Pool(args.jobs, initializer=_init, initargs=(roots,)) as pool:
        evs = list(pool.imap_unordered(load_event, jobs, chunksize=8))
    # events with no feats npz (no usable rows) still count via recorded
    _init(roots)
    for label_path, _fp, tol in missing:
        evs.append(load_event((label_path, None, tol)))
    evs.sort(key=lambda e: e['evt'])
    by_evt = {e['evt']: e for e in evs}
    n_rec = sum(e['ok_rec'] for e in evs)
    print('loaded %d events (%d with usable rows; recorded correct %d)'
          % (len(evs), sum(1 for e in evs if 'terms' in e), n_rec))

    man = load_manifest(os.path.dirname(os.path.join(HERE, args.manifest))
                        if os.path.basename(args.manifest) == 'manifest.tsv'
                        else args.manifest)
    folds = list(kfold_split(man, args.kfold, args.seed))
    eligible = lambda e: ('terms' in e and e['truth_i'] >= 0
                          and len(e['d_row']) >= 2)
    n_elig = sum(1 for e in evs if eligible(e))
    print('eligible for training (>=2 usable rows, truth on a row): %d' % n_elig)

    results = []
    for fname, featfn in FEATSETS.items():
        for hidden in (0, 8):
            head = '%s/%s' % (fname, 'mlp8' if hidden else 'lin')
            tot = {('chooser',): [0, 0, 0], ('full',): [0, 0, 0]}
            chooser_hits = [0, 0]   # head vs composite chooser acc on eligible val
            for fi, (_, tr_rows, va_rows) in enumerate(folds):
                tr = [by_evt[r['evt']] for r in tr_rows
                      if eligible(by_evt[r['evt']])]
                predict = fit_head(tr, featfn, hidden, args.seed + fi)
                # nested threshold for the full semantics: best on TRAIN folds
                taus = sorted({float(np.max(predict(e))) for e in tr})
                taus = taus[:: max(1, len(taus) // 40)]
                best_tau, best_n = None, -1
                for tau in taus:
                    n = sum(replay(e, predict(e) if 'terms' in e else None,
                                   args.tol, 'full', tau)[0]
                            for e in tr)
                    if n > best_n:
                        best_n, best_tau = n, tau
                for r in va_rows:
                    e = by_evt[r['evt']]
                    s = predict(e) if 'terms' in e else None
                    for mode, tau in (('chooser', None), ('full', best_tau)):
                        ok, standin = replay(e, s, args.tol, mode, tau)
                        acc = tot[(mode,)]
                        acc[0] += ok
                        acc[1] += (standin == 'row')
                        acc[2] += (standin == 'trad')
                    if eligible(e):
                        chooser_hits[0] += int(np.argmax(s) == e['truth_i'])
                        chooser_hits[1] += int(e['comp_i'] == e['truth_i'])
            for mode in ('chooser', 'full'):
                n, srow, strad = tot[(mode,)]
                results.append(dict(head=head, mode=mode, ok=n,
                                    delta=n - n_rec, row=srow, trad=strad))
            print('%-10s chooser-acc head %d vs composite %d /%d elig | '
                  'chooser %+d (row %d)  full %+d (row %d trad %d)'
                  % (head, chooser_hits[0], chooser_hits[1], n_elig,
                     tot[('chooser',)][0] - n_rec, tot[('chooser',)][1],
                     tot[('full',)][0] - n_rec, tot[('full',)][1],
                     tot[('full',)][2]))

    print('\n== summary (recorded %d/%d) ==' % (n_rec, len(evs)))
    for r in sorted(results, key=lambda r: -r['delta']):
        print('  %-10s %-8s predicted %3d  (%+d)  stand-ins row %d trad %d'
              % (r['head'], r['mode'], r['ok'], r['delta'], r['row'], r['trad']))
    if args.tsv:
        with open(args.tsv, 'w') as fh:
            fh.write('head\tmode\tok\tdelta\trow\ttrad\n')
            for r in results:
                fh.write('%s\t%s\t%d\t%d\t%d\t%d\n'
                         % (r['head'], r['mode'], r['ok'], r['delta'],
                            r['row'], r['trad']))
        print('wrote %s' % args.tsv)
    return 0


if __name__ == '__main__':
    sys.exit(main())
